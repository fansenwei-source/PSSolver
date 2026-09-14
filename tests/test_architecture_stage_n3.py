"""CPU qualification for Stage N.3 batched physical computation islands."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import pssolver
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
    IncompressibleStokesSystemSpec,
    TangentialZeroModePolicy,
)
from pssolver.experimental import (
    PlaneBerisEdwardsSolverOptions,
    build_experimental_model_runtime,
    create_beris_edwards_plane_geometry_solver_registry,
)
from pssolver.experimental.representations import (
    AlgebraicGenerationState,
    BatchedPhysicalMaterialization,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    NEMATIC_FORCE_COMPONENTS,
    VELOCITY_COMPONENTS,
    BerisEdwardsConstitutiveParameters,
    BerisEdwardsConstitutiveStokesCanaryModel,
    BerisEdwardsPlaneCoupledModel,
)


P = PeriodicBC()
N = HomogeneousNeumannBC()
PROJECT_ROOT = Path(__file__).parents[1]


def _runtime(*, batched: bool, batch_size: int = 1):
    even = BoundarySet((P, P, N))
    odd = BoundarySet((P, P, HomogeneousDirichletBC()))
    constitutive = BerisEdwardsConstitutiveStokesCanaryModel(
        q_boundaries=even,
        tangential_boundaries=even,
        normal_boundaries=odd,
        pressure_boundaries=even,
        parameters=BerisEdwardsConstitutiveParameters(
            ldg_a=0.0,
            ldg_b=-0.3,
            ldg_c=0.3,
            ldg_l1=0.04,
            flow_alignment=0.31,
            active_prefactor=-0.18,
        ),
        stokes_system=IncompressibleStokesSystemSpec(
            name="flow",
            force_components=NEMATIC_FORCE_COMPONENTS,
            velocity_components=VELOCITY_COMPONENTS,
            pressure_component="p",
            viscosity=2.0 / 3.0,
            friction=0.0,
            tangential_zero_mode_policy=TangentialZeroModePolicy.ZERO_MEAN,
        ),
        initial_amplitude=0.1,
    )
    model = BerisEdwardsPlaneCoupledModel(
        constitutive_model=constitutive,
        rotational_viscosity=2.94,
    )
    numerics = NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.CUBIC_HALF,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.TRUNCATED,
        spectral_storage=SpectralStorage.HERMITIAN_HALF,
        hermitian_axis=1,
    )
    return build_experimental_model_runtime(
        model,
        PlaneSlab(DomainSpec((8, 6, 5), (5.0, 4.0, 3.5))),
        numerics,
        dt=0.005,
        batch_size=batch_size,
        geometry_solver_registry=(
            create_beris_edwards_plane_geometry_solver_registry(
                constitutive_options=PlaneBerisEdwardsSolverOptions(
                    molecular_field_linear_space="spectral",
                    stress_divergence_sum_space="spectral",
                    pointwise_execution="eager",
                )
            )
        ),
        enable_performance_instrumentation=True,
        enable_algebraic_representation_reuse=True,
        enable_lazy_algebraic_materialization=True,
        enable_batched_physical_islands=batched,
    )


def test_generation_state_prefetches_one_declared_physical_island():
    physical = torch.zeros((1, 3), dtype=torch.float64)
    spectral = torch.complex(physical, torch.zeros_like(physical))

    def materialize(name, value):
        return value.real.clone()

    def materialize_many(names, values):
        return BatchedPhysicalMaterialization(
            {
                name: value.real.clone()
                for name, value in zip(names, values, strict=True)
            },
            (2, 1),
        )

    generation = AlgebraicGenerationState(
        1,
        {"q": physical},
        {"q": spectral},
        materialize=materialize,
        materialize_many=materialize_many,
    )
    for name, offset in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        generation.publish_spectral(name, spectral + offset)
    view = generation.view(("q", "a", "b", "c"))
    view.prefetch_physical(("a", "b", "c"))

    assert torch.equal(view["a"], (spectral + 1.0).real)
    snapshot = generation.snapshot()
    assert snapshot["physical_materializations"] == 3
    assert snapshot["physical_materialization_batches"] == 2
    assert snapshot["batched_physical_components"] == 2
    assert snapshot["singleton_materialization_batches"] == 1
    assert snapshot["maximum_materialization_batch_size"] == 2
    assert snapshot["physical_island_prefetches"] == 1
    assert snapshot["physical_island_requested_components"] == 3

    final = generation.invalidate()
    assert final["retained_physical_components"] == 0
    assert final["retained_spectral_components"] == 0


def test_batched_physical_islands_require_the_stage_n2_lifecycle():
    reference = _runtime(batched=False)
    kwargs = {
        "model": reference.problem.model,
        "geometry": PlaneSlab(DomainSpec((8, 6, 5), (5.0, 4.0, 3.5))),
        "numerics": reference.problem.numerics,
        "dt": 0.005,
        "geometry_solver_registry": (
            create_beris_edwards_plane_geometry_solver_registry()
        ),
        "enable_batched_physical_islands": True,
    }
    with pytest.raises(ValueError, match="require representation reuse"):
        build_experimental_model_runtime(**kwargs)
    with pytest.raises(ValueError, match="require representation reuse and lazy"):
        build_experimental_model_runtime(
            **kwargs,
            enable_algebraic_representation_reuse=True,
        )


def test_batched_plane_islands_preserve_results_and_reduce_both_transforms():
    control = _runtime(batched=False)
    candidate = _runtime(batched=True)
    control.performance_recorder.reset()
    candidate.performance_recorder.reset()

    control.solver.run(3)
    candidate.solver.run(3)

    torch.testing.assert_close(
        candidate.solver.fields.spatial,
        control.solver.fields.spatial,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        candidate.solver.fields.spectral,
        control.solver.fields.spectral,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    control_regions = control.performance_recorder.snapshot()["regions"]
    candidate_regions = candidate.performance_recorder.snapshot()["regions"]
    assert candidate_regions["transform.forward"]["calls"] < (
        control_regions["transform.forward"]["calls"]
    )
    assert candidate_regions["transform.inverse"]["calls"] < (
        control_regions["transform.inverse"]["calls"]
    )

    lifecycle = candidate.algebraic_physical_materialization_diagnostics()
    assert lifecycle is not None
    assert lifecycle["physical_materialization_batches"] < (
        lifecycle["physical_materializations"]
    )
    assert lifecycle["batched_physical_components"] >= 20
    assert lifecycle["maximum_materialization_batch_size"] >= 5
    assert lifecycle["on_demand_physical_materializations"] == 0
    assert lifecycle["retained_physical_components"] >= 0
    reuse = candidate.algebraic_representation_reuse_diagnostics()
    assert reuse is not None
    assert reuse["retained_pairs_after_generation"] == 0

    metadata = candidate.to_metadata()["algebraic_lifecycle"]
    assert metadata["physical_islands"] == {
        "mode": "boundary_signature_batched",
        "solver_dependencies_declared": True,
        "explicit_rhs_dependencies_declared": True,
        "grouping_key": "boundary_signature",
        "cross_timestep_reuse": False,
        "checkpointed": False,
    }
    assert control.to_metadata()["algebraic_lifecycle"]["physical_islands"][
        "mode"
    ] == "on_demand_componentwise"


def test_batched_physical_islands_preserve_the_runtime_batch_dimension():
    control = _runtime(batched=False, batch_size=2)
    candidate = _runtime(batched=True, batch_size=2)

    control.solver.run(2)
    candidate.solver.run(2)

    torch.testing.assert_close(
        candidate.solver.fields.spatial,
        control.solver.fields.spatial,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    torch.testing.assert_close(
        candidate.solver.fields.spectral,
        control.solver.fields.spectral,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_stage_n3_remains_outside_production_and_generic_solver_paths():
    assert not hasattr(pssolver, "BatchedPhysicalMaterialization")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "enable_batched_physical_islands" not in text
        assert "BatchedPhysicalMaterialization" not in text
