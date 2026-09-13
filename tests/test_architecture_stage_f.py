"""Characterization tests for Stage F algebraic lifecycle and dispatch."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import json
import math

import pytest
import torch

import pssolver
from pssolver import SpectralSolver
from pssolver.core import (
    AxisTopology,
    BoundarySet,
    DealiasRule,
    DomainSpec,
    GeometrySpec,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.execution import (
    AlgebraicExecutableModelProtocol,
    AlgebraicSolverContext,
    AlgebraicSystemSpec,
    AlgebraicUpdatePhase,
    GeometrySolverRegistry,
)
from pssolver.experimental import (
    ProjectedSemiImplicitEulerIntegrator,
    build_experimental_model_runtime,
    create_canary_geometry_solver_registry,
)
from pssolver.geometries import PeriodicBox, PlaneSlab, RectangularChannel
from pssolver.models.canary import (
    DiffusionHelmholtzCouplingModel,
    ScalarDiffusionModel,
)
from pssolver.transforms import BasisAwareSpectralProjector


def _numerics(
    *,
    storage=SpectralStorage.FULL_COMPLEX,
    hermitian_axis=None,
    dealias_rule=DealiasRule.CUBIC_HALF,
    projected_execution=ProjectedTransformExecution.TRUNCATED,
):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=dealias_rule,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=projected_execution,
        spectral_storage=storage,
        hermitian_axis=hermitian_axis,
    )


def _coupled_cases():
    periodic_boundaries = BoundarySet((PeriodicBC(),))
    yield (
        DiffusionHelmholtzCouplingModel(
            periodic_boundaries,
            diffusivity=0.12,
            coupling=0.35,
            helmholtz_shift=1.1,
            helmholtz_length_sq=0.25,
            initial_amplitude=0.3,
            initial_modes=(2,),
        ),
        PeriodicBox(DomainSpec((9,), (5.0,))),
        _numerics(
            dealias_rule=DealiasRule.NONE,
            projected_execution=ProjectedTransformExecution.FULL,
        ),
    )

    plane_boundaries = BoundarySet(
        (PeriodicBC(), HomogeneousNeumannBC())
    )
    yield (
        DiffusionHelmholtzCouplingModel(
            plane_boundaries,
            diffusivity=0.08,
            coupling=-0.2,
            helmholtz_shift=0.9,
            helmholtz_length_sq=0.4,
            initial_amplitude=0.25,
            initial_modes=(1, 2),
        ),
        PlaneSlab(DomainSpec((8, 7), (4.0, 3.5))),
        _numerics(
            storage=SpectralStorage.HERMITIAN_HALF,
            hermitian_axis=0,
        ),
    )


def _legacy_boundaries(model):
    return tuple(condition.kind.value for condition in model.boundaries.axes)


def _manual_initial(solver, model):
    value = torch.full(
        solver.shape,
        model.initial_amplitude,
        dtype=solver.dtype,
        device=solver.device,
    )
    for axis, (coordinate, length, mode, condition) in enumerate(
        zip(
            solver.transform_backend.axes,
            solver.L,
            model.initial_modes,
            model.boundaries.axes,
        )
    ):
        if condition.kind.value == "periodic":
            factor = torch.cos(2.0 * math.pi * mode * coordinate / length)
        elif condition.kind.value == "neumann":
            factor = torch.cos(math.pi * mode * coordinate / length)
        else:
            factor = torch.sin(math.pi * mode * coordinate / length)
        shape = [1] * len(solver.shape)
        shape[axis] = coordinate.numel()
        value = value * factor.reshape(shape)
    return value


class _ManualHelmholtzStatic(torch.nn.Module):
    def __init__(self, model, solver, projector, boundaries):
        super().__init__()
        self.projector = projector
        self.boundaries = boundaries
        laplacian = solver.get_laplacian_eigs(boundaries)
        self.denominator = (
            model.helmholtz_shift
            - model.helmholtz_length_sq * laplacian
        )

    def forward(self, fields, parameters):
        del parameters
        source_hat = self.projector.forward_transform(
            fields["phi"],
            self.boundaries,
        )
        return (source_hat / self.denominator).unsqueeze(0)


class _ManualCoupledRHS(torch.nn.Module):
    def __init__(self, model, projector, boundaries):
        super().__init__()
        self.model = model
        self.projector = projector
        self.boundaries = boundaries

    def forward(self, fields, parameters):
        del parameters
        rhs = self.model.coupling * fields["response"]
        return self.projector.forward_transform(
            rhs,
            self.boundaries,
        ).unsqueeze(0)


def _manual_runtime(model, geometry, numerics, *, dt, batch_size):
    boundaries = _legacy_boundaries(model)
    solver = SpectralSolver(
        geometry.domain.shape,
        L=geometry.domain.lengths,
        dt=dt,
        batchsize=batch_size,
        device="cpu",
        dtype=torch.float64,
        transform_execution_order=numerics.transform_execution_order.value,
        spectral_storage=numerics.spectral_storage.value,
        hermitian_axis=numerics.hermitian_axis,
    )
    projector = BasisAwareSpectralProjector(
        solver,
        rule=numerics.dealias_rule.value,
        transform_execution=numerics.projected_transform_execution.value,
    )
    solver.model.spectral_projector = projector
    if projector.enabled:
        solver.integrator_cl = ProjectedSemiImplicitEulerIntegrator
    solver.model.add_dynamic_field(
        "phi",
        _manual_initial(solver, model),
        model.diffusivity * solver.get_laplacian_eigs(boundaries),
        boundary_conditions=boundaries,
    )
    solver.model.add_static_field(
        "response",
        boundary_conditions=boundaries,
    )
    solver.build()
    if projector.enabled:
        projector.project_dynamic_fields(solver.fields, sync_spatial=True)
    solver.model.set_static_compute_model(
        _ManualHelmholtzStatic(model, solver, projector, boundaries)
    )
    solver.model.set_static_inverse_transform(projector.inverse_transform)
    solver.refresh_static_fields()
    solver.model.set_nonlinear_model(
        _ManualCoupledRHS(model, projector, boundaries)
    )
    solver.integrator.restore_progress(0, static_fields_are_current=True)
    return solver, projector


@pytest.mark.parametrize(
    "model,geometry,numerics",
    tuple(_coupled_cases()),
    ids=("periodic", "plane_mixed_basis"),
)
def test_coupled_canary_matches_independent_manual_legacy_trajectory(
    model,
    geometry,
    numerics,
):
    manual, _ = _manual_runtime(
        model,
        geometry,
        numerics,
        dt=0.01,
        batch_size=2,
    )
    runtime = build_experimental_model_runtime(
        model,
        geometry,
        numerics,
        dt=0.01,
        batch_size=2,
        geometry_solver_registry=create_canary_geometry_solver_registry(),
    )

    assert torch.equal(manual.fields.spatial, runtime.solver.fields.spatial)
    assert torch.equal(manual.fields.spectral, runtime.solver.fields.spectral)
    assert torch.equal(manual.fields.L_hat, runtime.solver.fields.L_hat)

    for _ in range(8):
        manual.integrator.step()
        runtime.solver.integrator.step()
        assert torch.equal(manual.fields.spatial, runtime.solver.fields.spatial)
        assert torch.equal(manual.fields.spectral, runtime.solver.fields.spectral)


@pytest.mark.parametrize(
    "model,geometry,numerics",
    tuple(_coupled_cases()),
    ids=("periodic", "plane_mixed_basis"),
)
def test_algebraic_initialization_staleness_and_observation_sync(
    model,
    geometry,
    numerics,
):
    runtime = build_experimental_model_runtime(
        model,
        geometry,
        numerics,
        dt=0.01,
        geometry_solver_registry=create_canary_geometry_solver_registry(),
    )
    boundaries = _legacy_boundaries(model)
    initial_response = runtime.solver.fields["response"].clone()
    runtime.solver.run(1)

    assert torch.equal(runtime.solver.fields["response"], initial_response)
    current_source_hat = runtime.projector.forward_transform(
        runtime.solver.fields["phi"],
        boundaries,
    )
    denominator = (
        model.helmholtz_shift
        - model.helmholtz_length_sq
        * runtime.solver.get_laplacian_eigs(boundaries)
    )
    expected_current = runtime.projector.inverse_transform(
        current_source_hat / denominator,
        boundaries,
    )
    assert not torch.equal(runtime.solver.fields["response"], expected_current)

    before_step_count = runtime.solver.integrator.step_count
    runtime.synchronize_algebraic_for_observation()
    assert torch.equal(runtime.solver.fields["response"], expected_current)
    assert runtime.solver.integrator.step_count == before_step_count


def test_algebraic_contract_and_dispatch_metadata_are_auditable():
    model, geometry, numerics = next(_coupled_cases())
    registry = create_canary_geometry_solver_registry()
    runtime = build_experimental_model_runtime(
        model,
        geometry,
        numerics,
        dt=0.01,
        geometry_solver_registry=registry,
    )

    assert isinstance(model, AlgebraicExecutableModelProtocol)
    system = model.algebraic_system_specs()[0]
    assert system.update_phase is AlgebraicUpdatePhase.PRE_EXPLICIT_RHS
    assert system.output_components == ("response",)
    assert system.dependencies == ("phi",)
    assert isinstance(
        runtime.resolved_algebraic_systems[0].solver.context,
        AlgebraicSolverContext,
    )
    metadata = runtime.to_metadata()["algebraic_lifecycle"]
    assert metadata["post_step_state"] == "stale_until_next_pre_rhs_or_sync"
    assert metadata["systems"][0]["dispatch"] == {
        "geometry_type": "pssolver.geometries.tensor_product.PeriodicBox",
        "geometry_name": "periodic_box",
        "capability": "scalar_helmholtz",
        "implementation_name": "periodic_diagonal_helmholtz",
    }
    json.dumps(runtime.to_metadata(), allow_nan=False)


def test_runtime_reset_restores_initial_algebraic_freshness():
    model, geometry, numerics = next(_coupled_cases())
    runtime = build_experimental_model_runtime(
        model,
        geometry,
        numerics,
        dt=0.01,
        geometry_solver_registry=create_canary_geometry_solver_registry(),
    )
    initial = runtime.solver.fields.spatial.clone()
    runtime.solver.run(3)
    assert not torch.equal(runtime.solver.fields.spatial, initial)

    runtime.reset()
    assert torch.equal(runtime.solver.fields.spatial, initial)
    assert runtime.solver.integrator.step_count == 0
    assert runtime.solver.integrator._static_fields_are_current is True

    with pytest.raises(ValueError, match="evolved components only"):
        runtime.reset({"response": torch.zeros(9, dtype=torch.float64)})


def test_runtime_reset_also_preserves_stage_e_models():
    boundaries = BoundarySet((PeriodicBC(),))
    runtime = build_experimental_model_runtime(
        ScalarDiffusionModel(
            boundaries,
            diffusivity=0.2,
            initial_amplitude=0.4,
            initial_modes=(1,),
        ),
        PeriodicBox(DomainSpec((8,), (4.0,))),
        _numerics(
            dealias_rule=DealiasRule.NONE,
            projected_execution=ProjectedTransformExecution.FULL,
        ),
        dt=0.01,
    )
    initial = runtime.solver.fields.spatial.clone()
    runtime.solver.run(2)
    runtime.reset()

    assert torch.equal(runtime.solver.fields.spatial, initial)
    runtime.solver.run(1)


def test_plane_and_periodic_dispatch_are_distinct_and_channel_has_no_fallback():
    registry = create_canary_geometry_solver_registry()
    periodic_model, periodic_geometry, _ = next(_coupled_cases())
    system = periodic_model.algebraic_system_specs()[0]
    periodic = registry.resolve(periodic_geometry, system)
    plane = registry.resolve(
        PlaneSlab(DomainSpec((8, 6), (4.0, 3.0))),
        system,
    )
    channel = RectangularChannel(
        DomainSpec((8, 6), (4.0, 3.0)),
        streamwise_axis=0,
    )

    assert periodic.implementation_name == "periodic_diagonal_helmholtz"
    assert plane.implementation_name == "plane_mixed_basis_diagonal_helmholtz"
    with pytest.raises(LookupError, match="implicit geometry fallback is disabled"):
        registry.resolve(channel, system)
    spoofed_name = GeometrySpec(
        name="periodic_box",
        domain=DomainSpec((8,), (4.0,)),
        axis_topologies=(AxisTopology.PERIODIC,),
    )
    with pytest.raises(LookupError, match="GeometrySpec"):
        registry.resolve(spoofed_name, system)


def test_registry_rejects_duplicate_geometry_capability_registration():
    registry = GeometrySolverRegistry()

    def factory(context, system):
        del context, system
        raise AssertionError("factory should not run")

    registry.register(
        geometry_type=PeriodicBox,
        geometry_name="periodic_box",
        capability="scalar_helmholtz",
        implementation_name="first",
        factory=factory,
    )
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(
            geometry_type=PeriodicBox,
            geometry_name="periodic_box",
            capability="scalar_helmholtz",
            implementation_name="second",
            factory=factory,
        )


def test_algebraic_system_spec_is_immutable_and_finite():
    system = AlgebraicSystemSpec(
        name="response_system",
        capability="scalar_helmholtz",
        output_components=("response",),
        dependencies=("phi",),
        parameters={"shift": 1.0},
    )
    with pytest.raises(FrozenInstanceError):
        system.name = "changed"
    with pytest.raises(TypeError):
        system.parameters["shift"] = 2.0
    json.dumps(system.to_metadata(), allow_nan=False)

    with pytest.raises(ValueError, match="finite"):
        AlgebraicSystemSpec(
            name="bad",
            capability="scalar_helmholtz",
            output_components=("response",),
            dependencies=("phi",),
            parameters={"shift": float("nan")},
        )


@dataclass(frozen=True)
class _BadDependencyModel(DiffusionHelmholtzCouplingModel):
    def algebraic_system_specs(self):
        system = super().algebraic_system_specs()[0]
        return (
            AlgebraicSystemSpec(
                name=system.name,
                capability=system.capability,
                output_components=system.output_components,
                dependencies=("missing",),
                parameters=system.parameters,
            ),
        )


def test_builder_rejects_non_evolved_algebraic_dependencies():
    boundaries = BoundarySet((PeriodicBC(),))
    model = _BadDependencyModel(
        boundaries,
        diffusivity=0.1,
        coupling=0.2,
        helmholtz_shift=1.0,
        helmholtz_length_sq=0.3,
    )
    with pytest.raises(ValueError, match="dependencies must be evolved"):
        build_experimental_model_runtime(
            model,
            PeriodicBox(DomainSpec((8,), (4.0,))),
            _numerics(
                dealias_rule=DealiasRule.NONE,
                projected_execution=ProjectedTransformExecution.FULL,
            ),
            dt=0.01,
            geometry_solver_registry=create_canary_geometry_solver_registry(),
        )


def test_builder_requires_an_explicit_registry_for_algebraic_models():
    model, geometry, numerics = next(_coupled_cases())
    with pytest.raises(TypeError, match="explicit GeometrySolverRegistry"):
        build_experimental_model_runtime(
            model,
            geometry,
            numerics,
            dt=0.01,
        )


def test_stage_f_remains_outside_production_paths():
    assert not hasattr(pssolver, "GeometrySolverRegistry")
    assert not hasattr(pssolver, "AlgebraicSystemSpec")
    assert not hasattr(pssolver, "create_canary_geometry_solver_registry")

    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
        "pssolver/models/active_nematics/stokes.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "GeometrySolverRegistry" not in text
        assert "AlgebraicSystemSpec" not in text
        assert "pssolver.experimental" not in text
