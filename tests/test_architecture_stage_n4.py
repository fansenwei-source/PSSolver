"""CPU qualification for Stage N.4 execution-policy consolidation."""

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
    AlgebraicExecutionMode,
    AlgebraicExecutionPolicy,
    IncompressibleStokesSystemSpec,
    TangentialZeroModePolicy,
    resolve_algebraic_execution_policy,
)
from pssolver.experimental import (
    BoundarySignatureTransformScheduler,
    PlaneBerisEdwardsSolverOptions,
    ProjectedTransformDirection,
    build_experimental_model_runtime,
    create_beris_edwards_plane_geometry_solver_registry,
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


def _runtime(*, unified_policy: bool, batch_size: int = 1):
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
    options = dict(
        model=model,
        geometry=PlaneSlab(DomainSpec((8, 6, 5), (5.0, 4.0, 3.5))),
        numerics=numerics,
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
    )
    if unified_policy:
        options["algebraic_execution_policy"] = (
            AlgebraicExecutionPolicy.batched()
        )
    else:
        options.update(
            enable_algebraic_representation_reuse=True,
            enable_lazy_algebraic_materialization=True,
            enable_batched_physical_islands=True,
        )
    return build_experimental_model_runtime(**options)


def test_execution_policy_encodes_only_the_valid_monotone_modes():
    expected = (
        (AlgebraicExecutionPolicy.eager(), (False, False, False)),
        (AlgebraicExecutionPolicy.reuse(), (True, False, False)),
        (AlgebraicExecutionPolicy.lazy(), (True, True, False)),
        (AlgebraicExecutionPolicy.batched(), (True, True, True)),
    )
    for policy, flags in expected:
        assert (
            policy.representation_reuse,
            policy.lazy_physical_materialization,
            policy.batched_physical_islands,
        ) == flags
        assert policy.to_metadata()["mode"] == policy.mode.value
    assert AlgebraicExecutionPolicy.batched().mode is (
        AlgebraicExecutionMode.BATCHED_PHYSICAL_ISLANDS
    )
    with pytest.raises(ValueError, match="requires representation reuse"):
        AlgebraicExecutionPolicy.from_legacy_flags(
            lazy_physical_materialization=True
        )
    with pytest.raises(ValueError, match="require representation reuse and lazy"):
        AlgebraicExecutionPolicy.from_legacy_flags(
            representation_reuse=True,
            batched_physical_islands=True,
        )


def test_policy_and_legacy_flags_have_one_configuration_authority():
    policy = AlgebraicExecutionPolicy.batched()
    assert resolve_algebraic_execution_policy(policy) is policy
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_algebraic_execution_policy(
            policy,
            enable_algebraic_representation_reuse=False,
        )
    with pytest.raises(TypeError, match="must be a bool"):
        resolve_algebraic_execution_policy(
            None,
            enable_algebraic_representation_reuse=1,
        )


class _RecordingTransformContext:
    batch_size = 1
    physical_shape = (2, 2)
    spectral_shape = (2, 2)
    real_dtype = torch.float64
    spectral_dtype = torch.complex128
    device = torch.device("cpu")

    def __init__(self):
        self.packed_calls = []

    def boundary_conditions(self, name):
        return ("periodic", "periodic") if name != "c" else (
            "periodic",
            "neumann",
        )

    def forward_projected(self, name, value):
        del name
        self.packed_calls.append(("forward", 1))
        return torch.complex(value, torch.zeros_like(value))

    def inverse_projected(self, name, value):
        del name
        self.packed_calls.append(("inverse", 1))
        return value.real

    def transform_projected_packed(self, boundaries, value, *, direction):
        self.packed_calls.append((direction.value, value.shape[0], boundaries))
        if direction is ProjectedTransformDirection.FORWARD:
            return torch.complex(value, torch.zeros_like(value))
        return value.real


def test_scheduler_groups_only_complete_execution_keys_and_preserves_order():
    context = _RecordingTransformContext()
    scheduler = BoundarySignatureTransformScheduler(context)
    values = tuple(
        torch.full((1, 2, 2), float(index), dtype=torch.float64)
        for index in (1, 2, 3)
    )
    result = scheduler.forward_many(("a", "b", "c"), values)

    assert result.batch_sizes == (2, 1)
    assert [key.boundary_signature for key in result.batch_keys] == [
        ("periodic", "periodic"),
        ("periodic", "neumann"),
    ]
    assert all(
        key.direction is ProjectedTransformDirection.FORWARD
        for key in result.batch_keys
    )
    for actual, expected in zip(result.values, values, strict=True):
        assert torch.equal(actual.real, expected)
    assert context.packed_calls == [
        ("forward", 2, ("periodic", "periodic")),
        ("forward", 1),
    ]


def test_unified_policy_is_identical_to_the_frozen_stage_n3_adapter():
    control = _runtime(unified_policy=False)
    candidate = _runtime(unified_policy=True)
    control.performance_recorder.reset()
    candidate.performance_recorder.reset()

    control.solver.run(3)
    candidate.solver.run(3)

    assert torch.equal(
        candidate.solver.fields.spatial,
        control.solver.fields.spatial,
    )
    assert torch.equal(
        candidate.solver.fields.spectral,
        control.solver.fields.spectral,
    )
    control_regions = control.performance_recorder.snapshot()["regions"]
    candidate_regions = candidate.performance_recorder.snapshot()["regions"]
    assert {
        name: value["calls"] for name, value in candidate_regions.items()
    } == {name: value["calls"] for name, value in control_regions.items()}
    assert candidate.algebraic_representation_reuse_diagnostics() == (
        control.algebraic_representation_reuse_diagnostics()
    )
    assert candidate.algebraic_physical_materialization_diagnostics() == (
        control.algebraic_physical_materialization_diagnostics()
    )
    metadata = candidate.to_metadata()
    assert metadata["algebraic_execution_policy"]["mode"] == (
        "batched_physical_islands"
    )
    scheduler = metadata["algebraic_lifecycle"]["materialization_scheduler"]
    assert scheduler["lifecycle_owner"] == "algebraic_generation_state"
    assert "boundary_signature" in scheduler["grouping_key_fields"]


def test_unified_policy_preserves_the_runtime_batch_dimension():
    control = _runtime(unified_policy=False, batch_size=2)
    candidate = _runtime(unified_policy=True, batch_size=2)
    control.solver.run(2)
    candidate.solver.run(2)
    assert torch.equal(
        candidate.solver.fields.spatial,
        control.solver.fields.spatial,
    )
    assert torch.equal(
        candidate.solver.fields.spectral,
        control.solver.fields.spectral,
    )


def test_stage_n4_remains_outside_production_and_generic_solver_paths():
    assert not hasattr(pssolver, "AlgebraicExecutionPolicy")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "AlgebraicExecutionPolicy" not in text
        assert "BoundarySignatureTransformScheduler" not in text
