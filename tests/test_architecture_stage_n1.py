"""CPU qualification for Stage N.1 generation-local representation reuse."""

from __future__ import annotations

import json
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
    AlgebraicRepresentationCache,
    PlaneBerisEdwardsSolverOptions,
    RuntimePerformanceRecorder,
    build_experimental_model_runtime,
    create_beris_edwards_plane_geometry_solver_registry,
    create_canary_geometry_solver_registry,
)
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
from pssolver.models.canary import DiffusionHelmholtzCouplingModel


P = PeriodicBC()
N = HomogeneousNeumannBC()
BOUNDARIES = BoundarySet((P, N))
PROJECT_ROOT = Path(__file__).parents[1]


def _runtime(*, reuse: bool):
    model = DiffusionHelmholtzCouplingModel(
        boundaries=BOUNDARIES,
        diffusivity=0.07,
        coupling=-0.3,
        helmholtz_shift=1.1,
        helmholtz_length_sq=0.2,
        initial_amplitude=0.25,
        initial_modes=(1, 2),
    )
    numerics = NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.CUBIC_HALF,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.TRUNCATED,
        spectral_storage=SpectralStorage.HERMITIAN_HALF,
        hermitian_axis=0,
    )
    return build_experimental_model_runtime(
        model,
        PlaneSlab(DomainSpec((8, 7), (4.0, 3.5))),
        numerics,
        dt=0.01,
        batch_size=2,
        geometry_solver_registry=create_canary_geometry_solver_registry(),
        enable_performance_instrumentation=True,
        enable_algebraic_representation_reuse=reuse,
    )


def _beris_edwards_runtime(*, reuse: bool):
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
                    stress_divergence_sum_space="spectral",
                    pointwise_execution="eager",
                )
            )
        ),
        enable_performance_instrumentation=True,
        enable_algebraic_representation_reuse=reuse,
    )


def test_cache_requires_exact_unmodified_aliases_and_drops_references():
    cache = AlgebraicRepresentationCache()
    physical = torch.arange(6.0, dtype=torch.float64).reshape(2, 3)
    spectral = torch.complex(physical, torch.zeros_like(physical))
    cache.begin(1, {"phi": physical}, {"phi": spectral})

    assert cache.spectral_for("phi", physical) is spectral
    assert cache.spectral_for("phi", physical.view_as(physical)) is None
    physical.add_(1.0)
    assert cache.spectral_for("phi", physical) is None

    snapshot = cache.end()
    assert snapshot["generation"] == 1
    assert snapshot["forward_hits"] == 1
    assert snapshot["forward_misses"] == 2
    assert snapshot["retained_pairs_after_generation"] == 0
    assert cache.active is False
    assert cache.spectral_for("phi", physical) is None
    json.dumps(dict(snapshot), allow_nan=False, sort_keys=True)


def test_cache_rejects_nested_generations_and_mismatched_seeds():
    cache = AlgebraicRepresentationCache()
    physical = torch.zeros((1, 3), dtype=torch.float64)
    spectral = torch.zeros((1, 3), dtype=torch.complex128)
    with pytest.raises(ValueError, match="seeds must match"):
        cache.begin(1, {"phi": physical}, {"psi": spectral})

    cache.begin(1, {"phi": physical}, {"phi": spectral})
    with pytest.raises(RuntimeError, match="generation is active"):
        cache.begin(2, {"phi": physical}, {"phi": spectral})
    cache.abort()
    assert cache.active is False


def test_representation_reuse_is_opt_in_generation_local_and_auditable():
    control = _runtime(reuse=False)
    candidate = _runtime(reuse=True)
    assert isinstance(control.performance_recorder, RuntimePerformanceRecorder)
    assert isinstance(candidate.performance_recorder, RuntimePerformanceRecorder)
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
    assert candidate_regions["transform.inverse"]["calls"] == (
        control_regions["transform.inverse"]["calls"]
    )

    assert control.algebraic_representation_reuse_diagnostics() is None
    diagnostics = candidate.algebraic_representation_reuse_diagnostics()
    assert diagnostics is not None
    assert diagnostics["generation"] >= 3
    assert diagnostics["seeded_pairs"] == 1
    assert diagnostics["forward_hits"] > 0
    assert diagnostics["retained_pairs_after_generation"] == 0
    assert candidate.to_metadata()["algebraic_lifecycle"][
        "representation_reuse"
    ] == {
        "enabled": True,
        "scope": "one_pre_explicit_rhs_generation",
        "identity_guard": "tensor_object_and_in_place_version",
        "cross_generation_reuse": False,
        "checkpointed": False,
    }
    assert control.to_metadata()["algebraic_lifecycle"][
        "representation_reuse"
    ]["enabled"] is False


def test_reset_discards_reuse_diagnostics_and_reseeds_current_spectra():
    runtime = _runtime(reuse=True)
    assert runtime.algebraic_representation_reuse_diagnostics() is not None
    initial = {
        "phi": torch.full(
            runtime.context.physical_shape,
            0.125,
            dtype=runtime.context.real_dtype,
        )
    }
    runtime.reset(initial)
    diagnostics = runtime.algebraic_representation_reuse_diagnostics()
    assert diagnostics is not None
    assert diagnostics["seeded_pairs"] == 1
    assert diagnostics["forward_hits"] > 0
    assert diagnostics["retained_pairs_after_generation"] == 0


def test_full_plane_beris_edwards_dag_reuses_evolved_force_and_velocity_spectra():
    control = _beris_edwards_runtime(reuse=False)
    candidate = _beris_edwards_runtime(reuse=True)
    control.performance_recorder.reset()
    candidate.performance_recorder.reset()

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
    control_regions = control.performance_recorder.snapshot()["regions"]
    candidate_regions = candidate.performance_recorder.snapshot()["regions"]
    avoided = control_regions["transform.forward"]["calls"] - (
        candidate_regions["transform.forward"]["calls"]
    )
    assert avoided >= 30
    assert candidate_regions["transform.inverse"]["calls"] == (
        control_regions["transform.inverse"]["calls"]
    )
    diagnostics = candidate.algebraic_representation_reuse_diagnostics()
    assert diagnostics is not None
    assert diagnostics["seeded_pairs"] == 5
    assert diagnostics["forward_hits"] >= 30
    assert diagnostics["retained_pairs_after_generation"] == 0


def test_invalid_reuse_builder_flag_is_rejected_before_assembly():
    with pytest.raises(TypeError, match="representation_reuse must be a bool"):
        build_experimental_model_runtime(
            DiffusionHelmholtzCouplingModel(
                boundaries=BOUNDARIES,
                diffusivity=0.07,
                coupling=-0.3,
                helmholtz_shift=1.1,
                helmholtz_length_sq=0.2,
                initial_amplitude=0.25,
                initial_modes=(1, 2),
            ),
            PlaneSlab(DomainSpec((8, 7), (4.0, 3.5))),
            NumericsConfig(
                precision=Precision.FLOAT64,
                dealias_rule=DealiasRule.NONE,
                transform_execution_order=TransformExecutionOrder.LEGACY,
                projected_transform_execution=ProjectedTransformExecution.FULL,
                spectral_storage=SpectralStorage.FULL_COMPLEX,
            ),
            dt=0.01,
            enable_algebraic_representation_reuse=1,
        )


def test_stage_n1_remains_outside_production_and_generic_solver_paths():
    assert not hasattr(pssolver, "AlgebraicRepresentationCache")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "enable_algebraic_representation_reuse" not in text
        assert "AlgebraicRepresentationCache" not in text
