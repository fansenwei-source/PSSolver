"""CPU qualification for Stage N.2 lazy algebraic materialization."""

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
from pssolver.experimental import (
    PlaneBerisEdwardsSolverOptions,
    build_experimental_model_runtime,
    create_beris_edwards_plane_geometry_solver_registry,
)
from pssolver.experimental.representations import AlgebraicGenerationState
from pssolver.execution import (
    IncompressibleStokesSystemSpec,
    TangentialZeroModePolicy,
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


def _runtime(*, lazy: bool, sum_space: str = "spectral"):
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
        batch_size=1,
        geometry_solver_registry=(
            create_beris_edwards_plane_geometry_solver_registry(
                constitutive_options=PlaneBerisEdwardsSolverOptions(
                    molecular_field_linear_space="spectral",
                    stress_divergence_sum_space=sum_space,
                    pointwise_execution="eager",
                )
            )
        ),
        enable_performance_instrumentation=True,
        enable_algebraic_representation_reuse=True,
        enable_lazy_algebraic_materialization=lazy,
    )


def test_generation_state_materializes_once_and_releases_every_tensor():
    physical = torch.arange(6.0, dtype=torch.float64).reshape(2, 3)
    spectral = torch.complex(physical, torch.zeros_like(physical))
    calls = []

    def materialize(name, value):
        calls.append((name, value))
        return value.real.clone()

    generation = AlgebraicGenerationState(
        1,
        {"phi": physical},
        {"phi": spectral},
        materialize=materialize,
    )
    output_hat = spectral + 1.0
    generation.publish_spectral("psi", output_hat)
    view = generation.view(("phi", "psi"))

    assert view.spectral("psi") is output_hat
    assert calls == []
    first = view["psi"]
    assert view["psi"] is first
    assert len(calls) == 1
    snapshot = generation.snapshot()
    assert snapshot["physical_materializations"] == 1
    assert snapshot["spectral_dependency_hits"] == 1
    assert snapshot["unmaterialized_published_components"] == 0

    final = generation.invalidate()
    assert final["active"] is False
    assert final["retained_physical_components"] == 0
    assert final["retained_spectral_components"] == 0
    with pytest.raises(RuntimeError, match="no longer current"):
        view["phi"]


def test_lazy_materialization_is_opt_in_and_requires_representation_reuse():
    with pytest.raises(
        ValueError,
        match="requires representation reuse",
    ):
        build_experimental_model_runtime(
            _runtime(lazy=False).problem.model,
            PlaneSlab(DomainSpec((8, 6, 5), (5.0, 4.0, 3.5))),
            _runtime(lazy=False).problem.numerics,
            dt=0.005,
            geometry_solver_registry=(
                create_beris_edwards_plane_geometry_solver_registry()
            ),
            enable_lazy_algebraic_materialization=True,
        )


def test_lazy_plane_dag_preserves_results_and_reduces_inverse_transforms():
    control = _runtime(lazy=False)
    candidate = _runtime(lazy=True)
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
    assert candidate_regions["transform.inverse"]["calls"] <= (
        control_regions["transform.inverse"]["calls"] - 20
    )
    assert candidate_regions["transform.forward"]["calls"] <= (
        control_regions["transform.forward"]["calls"]
    )

    diagnostics = candidate.algebraic_physical_materialization_diagnostics()
    assert diagnostics is not None
    assert diagnostics["physical_materializations"] > 0
    assert diagnostics["spectral_dependency_hits"] >= 20
    assert diagnostics["unmaterialized_published_components"] > 0
    reuse = candidate.algebraic_representation_reuse_diagnostics()
    assert reuse is not None
    assert reuse["retained_pairs_after_generation"] == 0

    lifecycle = candidate.to_metadata()["algebraic_lifecycle"]
    assert lifecycle["physical_materialization"] == {
        "mode": "lazy_generation_local",
        "spectral_dependencies_may_bypass_physical": True,
        "lifetime": "one_synchronized_pre_rhs_state",
        "cross_timestep_reuse": False,
        "checkpointed": False,
    }
    assert control.to_metadata()["algebraic_lifecycle"][
        "physical_materialization"
    ]["mode"] == "eager"


def test_lazy_transient_view_expires_at_the_next_generation():
    runtime = _runtime(lazy=True)
    transient = runtime.transient_algebraic_state()
    assert torch.isfinite(transient["Hxx"]).all()

    runtime.synchronize_algebraic_for_observation()

    with pytest.raises(RuntimeError, match="no longer current"):
        transient["Hxx"]
    current = runtime.transient_algebraic_state()
    assert current is not transient
    assert torch.isfinite(current["Hxx"]).all()


def test_lazy_physical_sum_space_preserves_the_complete_plane_dag():
    control = _runtime(lazy=False, sum_space="physical")
    candidate = _runtime(lazy=True, sum_space="physical")

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


def test_stage_n2_remains_outside_production_and_generic_solver_paths():
    assert not hasattr(pssolver, "AlgebraicGenerationState")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "enable_lazy_algebraic_materialization" not in text
        assert "AlgebraicGenerationState" not in text
