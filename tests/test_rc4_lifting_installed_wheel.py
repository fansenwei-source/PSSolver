"""Source/installed-wheel CPU gates for the RC4.3 lifting boundary."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
)
from pssolver.boundaries import (
    HomogeneousBoundaryPolicy,
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    prescribed_dirichlet,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.lifting import build_plane_static_lifting_plan
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)
from pssolver.core.boundary import BoundarySemantic
from pssolver.core.domain import DomainSpec
from pssolver.core.fields import FieldRole
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    Q_COMPONENTS,
    strong_homeotropic_q,
    strong_planar_q,
)
from pssolver.operators import localized_quintic_wall_transition
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.runtime.plane_beris_edwards import PlaneRuntimeBuildRequest
from pssolver.runtime.static_lifting import PlaneStaticLiftingRuntime
from pssolver.systems.algebraic import AlgebraicSystemSpec
from pssolver.systems.equations import (
    EquationFieldSpec,
    EquationSystemSpec,
    EquationTermSpec,
)
from pssolver.workflows.plane_checkpoint import (
    capture_plane_checkpoint,
    load_plane_checkpoint,
    restore_plane_checkpoint,
    write_plane_checkpoint,
)


def _model() -> CompleteStressBerisEdwards:
    return CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
    )


def _public_simulation(
    tmp_path: Path,
    *,
    lifted: bool,
    runtime_path: str,
    linear_space: str,
) -> Simulation:
    model = _model()
    geometry = PlaneSlab(shape=(8, 8, 6), lengths=(100.0, 100.0, 20.0))
    q_policy = (
        strong_homeotropic_q(
            scalar_order=0.6,
            face_normals={
                (2, "lower"): (0.0, 0.0, -1.0),
                (2, "upper"): (0.0, 0.0, 1.0),
            },
        )
        if lifted
        else neumann_q()
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": q_policy,
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.005, refresh={"mode": "disabled"}),
        initial_condition=GeneratedInitialCondition(
            "extruded_defect_gas",
            parameters={
                "seed": 24,
                "num_defect_pairs": 1,
                "defect_min_separation": 10.0,
                "defect_core_radius": 1.5,
                "background_angle": 0.0,
                "twist_amplitude": 0.01,
                "twist_modes": [1, 2, 3],
                "initial_s": 1.0 / 3.0,
            },
        ),
        execution=TorchSpectralExecution(
            runtime_path=runtime_path,
            device="cpu",
            options={
                "tf32": "off",
                "fallback_allowed": False,
                "molecular_field_linear_space": linear_space,
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / f"{runtime_path}_{linear_space}_{lifted}",
            steps=1,
            save_start_step=0,
            save_interval=1,
            diagnostic_interval=1,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
    )


@pytest.mark.parametrize("runtime_path", ("legacy_production", "compiled_v2"))
@pytest.mark.parametrize("linear_space", ("physical", "spectral"))
def test_public_lifted_plane_remains_fail_closed_before_output(
    tmp_path,
    runtime_path,
    linear_space,
):
    simulation = _public_simulation(
        tmp_path,
        lifted=True,
        runtime_path=runtime_path,
        linear_space=linear_space,
    )
    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(simulation)
    assert caught.value.code is (
        PublicCompilationRejectionCode.UNSUPPORTED_LIFTED_WALL_MOLECULAR_FIELD
    )
    assert caught.value.context["supported_Q_boundary"] == "homogeneous_neumann"
    assert not Path(simulation.output.options["output_dir"]).exists()


@pytest.mark.parametrize("runtime_path", ("legacy_production", "compiled_v2"))
@pytest.mark.parametrize("linear_space", ("physical", "spectral"))
def test_public_homogeneous_plane_preserves_requested_effective_space(
    tmp_path,
    runtime_path,
    linear_space,
):
    compiled = compile_simulation(
        _public_simulation(
            tmp_path,
            lifted=False,
            runtime_path=runtime_path,
            linear_space=linear_space,
        )
    )
    assert compiled.lowering_plan.lifting_plan is None
    assert compiled.normalization["requested_molecular_field_linear_space"] == (
        linear_space
    )
    assert compiled.normalization["effective_molecular_field_linear_space"] == (
        linear_space
    )
    assert compiled.normalization["molecular_field_space_fallback_used"] is False


def _generic_system() -> EquationSystemSpec:
    return EquationSystemSpec(
        name="generic_transport",
        variant="generic_scalar_transport",
        fields=(
            EquationFieldSpec("concentration", FieldRole.EVOLVED, ("c",)),
            EquationFieldSpec("flux", FieldRole.TRANSIENT, ("j",)),
            EquationFieldSpec("multiplier", FieldRole.ALGEBRAIC, ("lambda_value",)),
        ),
        evolution_laws=(
            EquationTermSpec(
                "evolve_concentration", "generic_transport_rhs", ("c",), ("c", "j")
            ),
        ),
        constitutive_laws=(
            EquationTermSpec("construct_flux", "generic_flux", ("j",), ("c",)),
        ),
        algebraic_systems=(
            AlgebraicSystemSpec(
                "solve_multiplier", "generic_constraint", ("lambda_value",), ("c", "j")
            ),
        ),
        parameters={"diffusivity": 0.2, "reaction_rate": -0.4},
    )


def _lifting_runtime(wall_points: int, *, localized: bool) -> PlaneStaticLiftingRuntime:
    model = _generic_system()
    geometry = PlaneSlab(DomainSpec((4, wall_points), (4.0, 2.0)))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "concentration": prescribed_dirichlet(
                "concentration",
                {
                    ("c", 1, "lower"): 1.25,
                    ("c", 1, "upper"): -0.5,
                },
            ),
            "multiplier": HomogeneousBoundaryPolicy(
                "multiplier", BoundarySemantic.ALGEBRAIC_COMPATIBILITY, "neumann"
            ),
        },
    )
    plan = build_plane_static_lifting_plan(
        equation_system=model,
        geometry=geometry,
        boundaries=boundaries,
    )
    return PlaneStaticLiftingRuntime(
        plan,
        dtype=torch.float64,
        device="cpu",
        batch_size=1,
        fresh_initial_conditioning=(
            localized_quintic_wall_transition(0.5) if localized else None
        ),
        affected_field="concentration",
    )


def _homogeneous_second_derivative(value: torch.Tensor, spacing: float) -> torch.Tensor:
    result = torch.empty_like(value)
    result[..., 1:-1] = (
        value[..., 2:] - 2.0 * value[..., 1:-1] + value[..., :-2]
    ) / spacing**2
    result[..., 0] = (value[..., 1] - 3.0 * value[..., 0]) / spacing**2
    result[..., -1] = (value[..., -2] - 3.0 * value[..., -1]) / spacing**2
    return result


def _pipeline_error(wall_points: int) -> float:
    runtime = _lifting_runtime(wall_points, localized=False)
    length = runtime.plan.domain_lengths[runtime.plan.wall_normal_axis]
    spacing = length / wall_points
    z = (torch.arange(wall_points, dtype=torch.float64) + 0.5) * spacing
    remainder = torch.sin(math.pi * z / length).expand(runtime.plan.domain_shape).clone()
    physical = runtime.operator.reconstruct_physical("c", remainder)
    recovered = runtime.extract_initial_remainders(
        {"c": physical}, apply_fresh_conditioning=False
    )["c"]
    numerical = _homogeneous_second_derivative(recovered, spacing)
    exact = -((math.pi / length) ** 2) * remainder
    return float(torch.sqrt(torch.mean((numerical - exact).square())).item())


def test_field_neutral_lifting_pipeline_has_second_order_manufactured_convergence():
    errors = tuple(_pipeline_error(size) for size in (16, 32, 64))
    rates = tuple(math.log(errors[i] / errors[i + 1], 2.0) for i in (0, 1))
    assert errors[0] > errors[1] > errors[2]
    assert all(rate > 1.95 for rate in rates)


def test_localized_policy_is_fresh_only_deterministic_and_interior_exact():
    runtime = _lifting_runtime(16, localized=True)
    generator = torch.Generator(device="cpu").manual_seed(436)
    physical = {
        "c": torch.randn(
            runtime.plan.domain_shape, generator=generator, dtype=torch.float64
        )
    }
    raw = runtime.extract_initial_remainders(
        physical, apply_fresh_conditioning=False
    )["c"]
    fresh = runtime.extract_initial_remainders(
        physical, apply_fresh_conditioning=True
    )["c"]
    repeated = runtime.extract_initial_remainders(
        physical, apply_fresh_conditioning=True
    )["c"]
    count = runtime.plan.domain_shape[runtime.plan.wall_normal_axis]
    length = runtime.plan.domain_lengths[runtime.plan.wall_normal_axis]
    centers = (torch.arange(count, dtype=torch.float64) + 0.5) * length / count
    interior = torch.minimum(centers, length - centers) >= 0.5
    assert torch.equal(fresh[..., interior], raw[..., interior])
    assert torch.any(fresh[..., ~interior] != raw[..., ~interior])
    assert torch.equal(fresh, repeated)
    assert runtime.restart_metadata()["lifting"].get(
        "fresh_initial_remainder_conditioning"
    ) is None
    metadata = runtime.fresh_construction_provenance(conditioning_applied=True)
    assert metadata["conditioning"]["effective_policy"] == (
        "localized_quintic_wall_transition_v1"
    )
    assert metadata["checkpoint_compatibility_identity"] is False


def _internal_runtime(tmp_path: Path, *, apply_fresh: bool):
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "unused",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        runtime_path="legacy_production",
        nx=4,
        ny=4,
        nz=6,
        lx=100.0,
        ly=100.0,
        height=20.0,
        steps=4,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        defect_min_separation=10.0,
        defect_core_radius=1.5,
        twist_modes=(1, 2, 3),
    )
    base = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )
    normals = {
        (2, "lower"): (0.0, 0.0, -1.0),
        (2, "upper"): (0.0, 0.0, 1.0),
    }
    simulation = replace(
        base,
        boundaries=assign_boundaries(
            model=base.equation_system,
            geometry=base.geometry,
            policies={
                "Q": strong_planar_q(
                    scalar_order=0.6,
                    face_directors={
                        (2, "lower"): (1.0, 0.0, 0.0),
                        (2, "upper"): (0.0, 1.0, 0.0),
                    },
                    face_normals=normals,
                ),
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
    )
    generator = torch.Generator(device="cpu").manual_seed(436)
    initial = {
        name: 0.01
        * torch.randn((4, 4, 6), generator=generator, dtype=torch.float64)
        for name in Q_COMPONENTS
    }
    request = PlaneRuntimeBuildRequest(
        run_spec,
        {
            "configuration": run_spec.identity_metadata(),
            "runtime_selection": run_spec.runtime_selection_metadata(),
        },
        initial,
        "cpu",
        simulation,
        fresh_initial_conditioning=localized_quintic_wall_transition(2.0),
        apply_fresh_initial_conditioning=apply_fresh,
    )
    adapter = build_package_simulation_runtime(
        PackageRuntimeConstructionInput(
            plan_package_runtime_construction(simulation), request
        )
    )
    return run_spec, adapter


def test_internal_same_version_file_restart_is_byte_identical(tmp_path):
    identity, continuous = _internal_runtime(tmp_path / "continuous", apply_fresh=True)
    _, segment = _internal_runtime(tmp_path / "segment", apply_fresh=True)
    _, resumed = _internal_runtime(tmp_path / "resumed", apply_fresh=False)
    continuous.advance(4)
    segment.advance(2)
    persisted = load_plane_checkpoint(
        write_plane_checkpoint(
            tmp_path / "checkpoint",
            capture_plane_checkpoint(segment, run_spec=identity),
        )
    )
    assert persisted.fresh_construction_provenance["conditioning"][
        "effective_policy"
    ] == "localized_quintic_wall_transition_v1"
    assert restore_plane_checkpoint(resumed, persisted, run_spec=identity) == 2
    resumed.advance(2)
    for name in (*Q_COMPONENTS, "ux", "uy", "uz", "p"):
        assert torch.equal(continuous.fields[name], resumed.fields[name])
