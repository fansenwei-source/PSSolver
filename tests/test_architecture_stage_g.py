"""Characterization tests for Stage G geometry-specific Stokes execution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from types import SimpleNamespace

import pytest
import torch

import pssolver
from pssolver.channel import ModalSaddleStokesCompute
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.execution import (
    AlgebraicRuntimeRestartState,
    AlgebraicSystemRestartState,
    INCOMPRESSIBLE_STOKES_CAPABILITY,
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)
from pssolver.experimental import (
    ChannelStokesSolverOptions,
    PlaneStokesSolverOptions,
    build_experimental_model_runtime,
    create_stokes_geometry_solver_registry,
)
from pssolver.geometries import PeriodicBox, PlaneSlab, RectangularChannel
from pssolver.models.canary import BodyForceStokesCanaryModel
from pssolver.transforms import FreeSlipModalStokesSolver


P = PeriodicBC()
N = HomogeneousNeumannBC()
D = HomogeneousDirichletBC()
PLANE_TANGENTIAL = BoundarySet((P, P, N))
PLANE_NORMAL = BoundarySet((P, P, D))
PLANE_PRESSURE = BoundarySet((P, P, N))
CHANNEL_VELOCITY = BoundarySet((P, D, D))
CHANNEL_PRESSURE = BoundarySet((P, N, N))


def _numerics():
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.NONE,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.FULL,
        spectral_storage=SpectralStorage.FULL_COMPLEX,
    )


def _stokes_spec(*, channel=False, friction=0.0):
    policy = (
        TangentialZeroModePolicy.NOT_APPLICABLE
        if channel
        else (
            TangentialZeroModePolicy.FRICTION
            if friction > 0.0
            else TangentialZeroModePolicy.ZERO_MEAN
        )
    )
    return IncompressibleStokesSystemSpec(
        name="flow",
        force_components=("force_x", "force_y", "force_z"),
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=0.73,
        friction=friction,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=policy,
    )


def _plane_model(*, friction=0.0, amplitudes=(0.7, -0.35, 0.2)):
    return BodyForceStokesCanaryModel(
        force_boundaries=(
            PLANE_TANGENTIAL,
            PLANE_TANGENTIAL,
            PLANE_NORMAL,
        ),
        velocity_boundaries=(
            PLANE_TANGENTIAL,
            PLANE_TANGENTIAL,
            PLANE_NORMAL,
        ),
        pressure_boundaries=PLANE_PRESSURE,
        stokes_system=_stokes_spec(friction=friction),
        force_diffusivity=0.025,
        initial_amplitudes=amplitudes,
        initial_modes=((1, 1, 1), (2, 1, 2), (1, 2, 1)),
    )


def _channel_model():
    return BodyForceStokesCanaryModel(
        force_boundaries=(CHANNEL_VELOCITY,) * 3,
        velocity_boundaries=(CHANNEL_VELOCITY,) * 3,
        pressure_boundaries=CHANNEL_PRESSURE,
        stokes_system=_stokes_spec(channel=True),
        force_diffusivity=0.018,
        initial_amplitudes=(0.55, -0.25, 0.3),
        initial_modes=((1, 1, 1), (2, 2, 1), (1, 1, 2)),
    )


def _build(model, geometry, *, numerics=None):
    if numerics is None:
        numerics = _numerics()
    return build_experimental_model_runtime(
        model,
        geometry,
        numerics,
        dt=0.01,
        device="cpu",
        batch_size=2,
        geometry_solver_registry=create_stokes_geometry_solver_registry(
            channel_options=ChannelStokesSolverOptions(
                pressure_relative_tolerance=1.0e-11,
                pressure_max_iterations=120,
            )
        ),
    )


def _direct_lower(runtime, *, channel):
    backend = runtime.solver.transform_backend
    spec = IncompressibleStokesSystemSpec.from_algebraic_system_spec(
        runtime.resolved_algebraic_systems[0].system
    )
    if channel:
        view = SimpleNamespace(
            transform_backend=backend,
            qx=SimpleNamespace(device=backend.device),
        )
        return ModalSaddleStokesCompute(
            view,
            beta=0.0,
            friction=spec.friction,
            viscosity=spec.viscosity,
            pressure_rel_tol=1.0e-11,
            pressure_max_iter=120,
        )
    return FreeSlipModalStokesSolver(
        backend,
        tangential_boundary_conditions=("periodic", "periodic", "neumann"),
        normal_boundary_conditions=("periodic", "periodic", "dirichlet"),
        pressure_boundary_conditions=("periodic", "periodic", "neumann"),
        friction=spec.friction,
        viscosity=spec.viscosity,
        zero_mode_policy=spec.tangential_zero_mode_policy.value,
        pressure_diagnostics=True,
    )


def _direct_solution(runtime, lower):
    system = runtime.resolved_algebraic_systems[0].system
    force_hats = tuple(
        runtime.projector.forward_transform(
            runtime.solver.fields[name],
            runtime.solver.fields[f"{name}.bc"],
        )
        for name in system.dependencies
    )
    return lower.solve_force_hats(*force_hats)


def _assert_solution_is_stored(runtime, solution):
    outputs = runtime.resolved_algebraic_systems[0].system.output_components
    for name, expected_hat in zip(outputs, solution, strict=True):
        torch.testing.assert_close(
            runtime.solver.fields[f"{name}.hat"],
            expected_hat,
            rtol=0.0,
            atol=0.0,
        )
        expected = runtime.projector.inverse_transform(
            expected_hat,
            runtime.solver.fields[f"{name}.bc"],
        )
        torch.testing.assert_close(
            runtime.solver.fields[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("channel", (False, True), ids=("plane", "channel"))
def test_stokes_runtime_matches_fixed_geometry_solver_trajectory(channel):
    if channel:
        model = _channel_model()
        geometry = RectangularChannel(
            DomainSpec((7, 6, 5), (4.0, 3.0, 2.5)),
            streamwise_axis=0,
        )
    else:
        model = _plane_model()
        geometry = PlaneSlab(
            DomainSpec((7, 6, 5), (4.0, 3.0, 2.5)),
            wall_normal_axis=2,
        )
    runtime = _build(model, geometry)
    reference = _direct_lower(runtime, channel=channel)

    _assert_solution_is_stored(runtime, _direct_solution(runtime, reference))
    for _ in range(4):
        old_state = {
            name: runtime.solver.fields[name].clone()
            for name in model.stokes_system.force_components
        }
        runtime.solver.run(1)
        old_force_hats = tuple(
            runtime.projector.forward_transform(
                old_state[name],
                runtime.solver.fields[f"{name}.bc"],
            )
            for name in model.stokes_system.force_components
        )
        _assert_solution_is_stored(
            runtime,
            reference.solve_force_hats(*old_force_hats),
        )
        runtime.synchronize_algebraic_for_observation()
        _assert_solution_is_stored(
            runtime,
            _direct_solution(runtime, reference),
        )


def test_plane_stokes_contract_supports_qualified_half_spectrum_defaults():
    numerics = NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.CUBIC_HALF,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.TRUNCATED,
        spectral_storage=SpectralStorage.HERMITIAN_HALF,
        hermitian_axis=1,
    )
    runtime = _build(
        _plane_model(),
        PlaneSlab(DomainSpec((8, 8, 6), (4.0, 3.0, 2.5))),
        numerics=numerics,
    )
    reference = _direct_lower(runtime, channel=False)

    _assert_solution_is_stored(runtime, _direct_solution(runtime, reference))
    runtime.solver.run(1)
    runtime.synchronize_algebraic_for_observation()
    _assert_solution_is_stored(runtime, _direct_solution(runtime, reference))


@pytest.mark.parametrize("channel", (False, True), ids=("plane", "channel"))
def test_stokes_outputs_obey_gauge_incompressibility_and_residual(channel):
    if channel:
        runtime = _build(
            _channel_model(),
            RectangularChannel(
                DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))
            ),
        )
    else:
        runtime = _build(
            _plane_model(),
            PlaneSlab(DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))),
        )
    adapter = runtime.resolved_algebraic_systems[0].solver
    lower = adapter.lower_solver
    ux_hat = runtime.solver.fields["ux.hat"]
    uy_hat = runtime.solver.fields["uy.hat"]
    uz_hat = runtime.solver.fields["uz.hat"]
    pressure_hat = runtime.solver.fields["p.hat"]
    force_hats = tuple(
        runtime.projector.forward_transform(
            runtime.solver.fields[name],
            runtime.solver.fields[f"{name}.bc"],
        )
        for name in ("force_x", "force_y", "force_z")
    )

    assert pressure_hat[0, 0, 0, 0].item() == 0j
    divergence = lower.divergence_hat(ux_hat, uy_hat, uz_hat)
    velocity_norm = max(
        torch.linalg.vector_norm(value.reshape(-1)).item()
        for value in (ux_hat, uy_hat, uz_hat)
    )
    assert (
        torch.linalg.vector_norm(divergence.reshape(-1)).item()
        / max(velocity_norm, 1.0)
        < 5.0e-10
    )

    pressure_gradients = lower.pressure_gradient_hats(pressure_hat)
    if channel:
        helmholtz = tuple(value / lower.a_inv for value in (ux_hat, uy_hat, uz_hat))
    else:
        helmholtz = (
            ux_hat / lower.a_tangential_inv.masked_fill(
                lower.tangential_null_mask,
                1.0,
            ),
            uy_hat / lower.a_tangential_inv.masked_fill(
                lower.tangential_null_mask,
                1.0,
            ),
            uz_hat / lower.a_normal_inv,
        )
    residuals = tuple(
        force_hat - gradient_hat - velocity_term
        for force_hat, gradient_hat, velocity_term in zip(
            force_hats,
            pressure_gradients,
            helmholtz,
            strict=True,
        )
    )
    residual_norm = torch.sqrt(
        sum(
            torch.linalg.vector_norm(value.reshape(-1)).square()
            for value in residuals
        )
    ).item()
    force_norm = torch.sqrt(
        sum(
            torch.linalg.vector_norm(value.reshape(-1)).square()
            for value in force_hats
        )
    ).item()
    assert residual_norm / force_norm < 5.0e-10


def test_pressure_diagnostics_are_observations_not_state_fields():
    runtime = _build(
        _plane_model(),
        PlaneSlab(DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))),
    )
    component_names = tuple(
        component.component_name
        for component in runtime.plan.stored_components
    )
    diagnostics = runtime.algebraic_diagnostics()

    assert component_names == (
        "force_x",
        "force_y",
        "force_z",
        "ux",
        "uy",
        "uz",
        "p",
    )
    assert not any("residual" in name for name in component_names)
    assert diagnostics["flow"]["pressure_diagnostics_enabled"] is True
    assert diagnostics["flow"]["last_pressure_iterations"] == 1
    assert diagnostics["flow"]["last_pressure_relative_residual"] < 1.0e-12
    json.dumps(diagnostics, allow_nan=False)
    metadata = runtime.to_metadata()
    assert (
        metadata["algebraic_lifecycle"]["systems"][0]["observability"]
        ["solver_options"]["pressure_diagnostics"]
        is True
    )
    json.dumps(metadata, allow_nan=False)

    disabled = build_experimental_model_runtime(
        _plane_model(),
        PlaneSlab(DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))),
        _numerics(),
        dt=0.01,
        geometry_solver_registry=create_stokes_geometry_solver_registry(
            plane_options=PlaneStokesSolverOptions(
                pressure_diagnostics=False
            )
        ),
    )
    value = disabled.algebraic_diagnostics()["flow"]
    assert value["last_pressure_residual"] is None
    assert value["last_pressure_relative_residual"] is None
    json.dumps(value, allow_nan=False)


def test_plane_zero_mode_policy_is_distinct_from_pressure_gauge():
    shape = (6, 6, 5)
    geometry = PlaneSlab(DomainSpec(shape, (4.0, 3.0, 2.5)))
    amplitudes = (1.25, -0.75, 0.0)
    modes = ((0, 0, 0), (0, 0, 0), (0, 0, 1))

    def model(friction):
        base = _plane_model(friction=friction, amplitudes=amplitudes)
        return BodyForceStokesCanaryModel(
            force_boundaries=base.force_boundaries,
            velocity_boundaries=base.velocity_boundaries,
            pressure_boundaries=base.pressure_boundaries,
            stokes_system=base.stokes_system,
            force_diffusivity=base.force_diffusivity,
            initial_amplitudes=base.initial_amplitudes,
            initial_modes=modes,
        )

    zero_mean = _build(model(0.0), geometry)
    dragged = _build(model(0.4), geometry)
    torch.testing.assert_close(
        zero_mean.solver.fields["ux"],
        torch.zeros_like(zero_mean.solver.fields["ux"]),
        rtol=0.0,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        dragged.solver.fields["ux"],
        torch.full_like(dragged.solver.fields["ux"], 1.25 / 0.4),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    assert zero_mean.solver.fields["p.hat"][0, 0, 0, 0].item() == 0j
    assert dragged.solver.fields["p.hat"][0, 0, 0, 0].item() == 0j


def test_channel_warm_start_restart_is_captured_and_identity_bound():
    model = _channel_model()
    geometry = RectangularChannel(
        DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))
    )
    source = _build(model, geometry)
    source.solver.run(2)
    source.synchronize_algebraic_for_observation()
    restart = source.capture_algebraic_restart_state()
    metadata = restart.to_metadata()

    assert metadata["format_version"] == 1
    assert set(metadata["systems"][0]["tensors"]) == {"pressure_guess"}
    assert len(
        metadata["systems"][0]["tensors"]["pressure_guess"]["sha256"]
    ) == 64
    json.dumps(metadata, allow_nan=False)

    target = _build(model, geometry)
    current_forces = {
        name: source.solver.fields[name].clone()
        for name in model.stokes_system.force_components
    }
    target.reset(current_forces)
    target.restore_algebraic_restart_state(restart)
    source.synchronize_algebraic_for_observation()
    target.synchronize_algebraic_for_observation()
    for name in model.stokes_system.velocity_components + ("p",):
        torch.testing.assert_close(
            target.solver.fields[f"{name}.hat"],
            source.solver.fields[f"{name}.hat"],
            rtol=0.0,
            atol=0.0,
        )
    assert target.algebraic_diagnostics()["flow"]["last_warm_start_used"]

    system = restart.systems[0]
    mismatched = AlgebraicRuntimeRestartState(
        1,
        (
            AlgebraicSystemRestartState(
                system_name=system.system_name,
                capability=system.capability,
                implementation_name="wrong_implementation",
                provenance_sha256=system.provenance_sha256,
                tensors=system.tensors,
            ),
        ),
    )
    with pytest.raises(ValueError, match="dispatch mismatch"):
        target.restore_algebraic_restart_state(mismatched)

    wrong_provenance = AlgebraicRuntimeRestartState(
        1,
        (
            AlgebraicSystemRestartState(
                system_name=system.system_name,
                capability=system.capability,
                implementation_name=system.implementation_name,
                provenance_sha256="0" * 64,
                tensors=system.tensors,
            ),
        ),
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        target.restore_algebraic_restart_state(wrong_provenance)


def test_plane_restart_is_explicitly_stateless():
    runtime = _build(
        _plane_model(),
        PlaneSlab(DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))),
    )
    restart = runtime.capture_algebraic_restart_state()
    assert restart.systems[0].tensors == {}
    assert (
        runtime.to_metadata()["algebraic_lifecycle"]["systems"][0]
        ["observability"]["restart"]["kind"]
        == "stateless"
    )
    runtime.restore_algebraic_restart_state(restart)


def test_dispatch_rejects_geometry_policy_and_boundary_crossovers():
    registry = create_stokes_geometry_solver_registry()
    plane_system = _plane_model().algebraic_system_specs()[0]
    channel_system = _channel_model().algebraic_system_specs()[0]
    periodic = PeriodicBox(DomainSpec((7, 6, 5), (4.0, 3.0, 2.5)))

    assert registry.resolve(
        PlaneSlab(DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))),
        plane_system,
    ).implementation_name == "plane_free_slip_modal_stokes"
    assert registry.resolve(
        RectangularChannel(DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))),
        channel_system,
    ).implementation_name == "channel_no_slip_modal_stokes"
    with pytest.raises(LookupError, match="implicit geometry fallback"):
        registry.resolve(periodic, plane_system)

    channel_boundaries_with_plane_policy = BodyForceStokesCanaryModel(
        force_boundaries=(CHANNEL_VELOCITY,) * 3,
        velocity_boundaries=(CHANNEL_VELOCITY,) * 3,
        pressure_boundaries=CHANNEL_PRESSURE,
        stokes_system=_stokes_spec(channel=False),
        force_diffusivity=0.018,
        initial_amplitudes=(0.55, -0.25, 0.3),
        initial_modes=((1, 1, 1), (2, 2, 1), (1, 1, 2)),
    )
    with pytest.raises(ValueError, match="not_applicable"):
        _build(
            channel_boundaries_with_plane_policy,
            RectangularChannel(
                DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))
            ),
        )
    plane_boundaries_with_channel_policy = BodyForceStokesCanaryModel(
        force_boundaries=(
            PLANE_TANGENTIAL,
            PLANE_TANGENTIAL,
            PLANE_NORMAL,
        ),
        velocity_boundaries=(
            PLANE_TANGENTIAL,
            PLANE_TANGENTIAL,
            PLANE_NORMAL,
        ),
        pressure_boundaries=PLANE_PRESSURE,
        stokes_system=_stokes_spec(channel=True),
        force_diffusivity=0.025,
        initial_amplitudes=(0.7, -0.35, 0.2),
        initial_modes=((1, 1, 1), (2, 1, 2), (1, 2, 1)),
    )
    with pytest.raises(ValueError, match="zero_mean or friction"):
        _build(
            plane_boundaries_with_channel_policy,
            PlaneSlab(DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))),
        )


def test_stokes_contract_is_typed_immutable_and_round_trips():
    value = _stokes_spec()
    generic = value.to_algebraic_system_spec()
    recovered = IncompressibleStokesSystemSpec.from_algebraic_system_spec(
        generic
    )

    assert generic.capability == INCOMPRESSIBLE_STOKES_CAPABILITY
    assert recovered == value
    with pytest.raises(FrozenInstanceError):
        value.viscosity = 2.0
    with pytest.raises(ValueError, match="friction == 0"):
        IncompressibleStokesSystemSpec(
            name="bad",
            force_components=("fx", "fy", "fz"),
            velocity_components=("ux", "uy", "uz"),
            pressure_component="p",
            viscosity=1.0,
            friction=0.1,
            tangential_zero_mode_policy=(
                TangentialZeroModePolicy.ZERO_MEAN
            ),
        )


def test_channel_public_force_solve_preserves_previous_operation_order():
    runtime = _build(
        _channel_model(),
        RectangularChannel(
            DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))
        ),
    )
    force_hats = tuple(
        runtime.projector.forward_transform(
            runtime.solver.fields[name],
            runtime.solver.fields[f"{name}.bc"],
        )
        for name in ("force_x", "force_y", "force_z")
    )
    public = _direct_lower(runtime, channel=True)
    legacy = _direct_lower(runtime, channel=True)
    observed = public.solve_force_hats(*force_hats)

    free_velocity = [legacy._helmholtz_inverse(value) for value in force_hats]
    provisional_divergence = sum(
        legacy._velocity_divergence_component(free_velocity[axis], axis)
        for axis in range(3)
    )
    pressure_hat = legacy._solve_pressure(
        legacy._project_pressure_gauge(-provisional_divergence)
    )
    velocity_hat = [
        free_velocity[axis]
        - legacy._helmholtz_inverse(
            legacy._pressure_gradient(pressure_hat, axis)
        )
        for axis in range(3)
    ]
    expected = (*velocity_hat, pressure_hat)
    for actual, reference in zip(observed, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)


def test_stage_g_stays_opt_in_and_benchmark_driver_is_unchanged():
    assert not hasattr(pssolver, "IncompressibleStokesSystemSpec")
    assert not hasattr(pssolver, "create_stokes_geometry_solver_registry")

    with open("Plane_beris_edwards_stokes.py", encoding="utf-8") as source:
        text = source.read()
    assert "pssolver.experimental" not in text
    assert "IncompressibleStokesSystemSpec" not in text
    assert "create_stokes_geometry_solver_registry" not in text
