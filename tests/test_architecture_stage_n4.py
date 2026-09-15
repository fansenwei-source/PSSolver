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
    AlgebraicOutputPublicationPolicy,
    IncompressibleStokesSystemSpec,
    TangentialZeroModePolicy,
    resolve_algebraic_execution_policy,
)
from pssolver.experimental import (
    BoundarySignatureTransformScheduler,
    CompiledProjectedTransformPlan,
    PlaneBerisEdwardsSolverOptions,
    ProjectedBatchAssemblyPolicy,
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


def _runtime(
    *,
    unified_policy: bool,
    batch_size: int = 1,
    batch_assembly_policy: ProjectedBatchAssemblyPolicy | None = None,
    output_publication_policy: AlgebraicOutputPublicationPolicy | None = None,
):
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
        options["projected_batch_assembly_policy"] = batch_assembly_policy
        options["algebraic_output_publication_policy"] = (
            output_publication_policy
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
        self.boundary_queries = []

    def boundary_conditions(self, name):
        self.boundary_queries.append(name)
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


def test_scheduler_compiles_each_stable_signature_once_and_reuses_its_plan():
    context = _RecordingTransformContext()
    scheduler = BoundarySignatureTransformScheduler(context)
    names = ("a", "b", "c")
    first_values = tuple(
        torch.full((1, 2, 2), float(index), dtype=torch.float64)
        for index in (1, 2, 3)
    )
    second_values = tuple(value + 4.0 for value in first_values)

    first_plan = scheduler.compile_plan(
        names,
        direction=ProjectedTransformDirection.FORWARD,
    )
    assert isinstance(first_plan, CompiledProjectedTransformPlan)
    assert first_plan.batch_sizes == (2, 1)
    assert context.boundary_queries == ["a", "b", "c"]

    second_plan = scheduler.compile_plan(
        names,
        direction=ProjectedTransformDirection.FORWARD,
    )
    assert second_plan is first_plan
    transformed = scheduler.forward_values_many(names, second_values)
    assert scheduler.compiled_plan_count == 1
    assert context.boundary_queries == ["a", "b", "c"]
    for actual, expected in zip(transformed, second_values, strict=True):
        assert torch.equal(actual.real, expected)

    inverse_values = tuple(
        torch.complex(value, torch.zeros_like(value)) for value in second_values
    )
    inverse = scheduler.inverse_values_many(names, inverse_values)
    assert scheduler.compiled_plan_count == 2
    assert context.boundary_queries == ["a", "b", "c"] * 2
    for actual, expected in zip(inverse, second_values, strict=True):
        assert torch.equal(actual, expected)


def test_compiled_plan_cannot_cross_scheduler_contexts():
    first = BoundarySignatureTransformScheduler(_RecordingTransformContext())
    second = BoundarySignatureTransformScheduler(_RecordingTransformContext())
    plan = first.compile_plan(
        ("a", "b"),
        direction=ProjectedTransformDirection.FORWARD,
    )
    values = (
        torch.ones((1, 2, 2), dtype=torch.float64),
        torch.ones((1, 2, 2), dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="another scheduler"):
        second.execute_plan(plan, values)


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
    assert metadata["algebraic_output_publication_policy"]["mode"] == (
        "deferred_stack"
    )
    assert metadata["projected_batch_assembly_policy"]["mode"] == "copy_cat"
    scheduler = metadata["algebraic_lifecycle"]["materialization_scheduler"]
    assert scheduler["lifecycle_owner"] == "algebraic_generation_state"
    assert "boundary_signature" in scheduler["grouping_key_fields"]
    assert scheduler["plan_compilation"] == "runtime_local_cached"
    assert scheduler["plan_cache_key"] == [
        "direction",
        "ordered_component_names",
    ]
    assert scheduler["compiled_plans_retain_tensors"] is False


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


def test_runtime_reuses_one_compiled_scheduler_across_generations():
    runtime = _runtime(unified_policy=True)
    algebraic = runtime.algebraic_fields_adapter
    assert algebraic is not None
    scheduler = algebraic._context.transform_scheduler
    assert algebraic._physical_island_scheduler._transforms is scheduler
    assert runtime.explicit_rhs_adapter._physical_island_scheduler._transforms is (
        scheduler
    )

    runtime.solver.run(1)
    compiled_after_first_step = scheduler.compiled_plan_count
    assert compiled_after_first_step > 0
    runtime.solver.run(2)
    assert scheduler.compiled_plan_count == compiled_after_first_step


def test_contiguous_view_batch_assembly_preserves_complete_cpu_timestep():
    control = _runtime(unified_policy=True)
    candidate = _runtime(
        unified_policy=True,
        batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.contiguous_storage_view()
        ),
        output_publication_policy=(
            AlgebraicOutputPublicationPolicy.preallocated()
        ),
    )
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
    metadata = candidate.to_metadata()
    assert metadata["projected_batch_assembly_policy"]["mode"] == (
        "contiguous_storage_view"
    )
    diagnostics = candidate.projected_batch_assembly_diagnostics()
    assert diagnostics["retained_tensor_references"] == 0
    assert diagnostics["contiguous_view_batches"] > 0
    generation = candidate.algebraic_fields_adapter._generation_state
    assert generation is not None
    stored = generation._spectral["ux"]
    transient = generation._spectral["Hxx"]
    assert stored.untyped_storage().data_ptr() != (
        transient.untyped_storage().data_ptr()
    )
    assert stored.untyped_storage().nbytes() == (
        4 * stored.numel() * stored.element_size()
    )
    assert transient.untyped_storage().nbytes() == (
        len(candidate.assembly.omitted_transient_components)
        * transient.numel()
        * transient.element_size()
    )


def test_natural_storage_views_do_not_require_packed_republication():
    control = _runtime(unified_policy=True)
    candidate = _runtime(
        unified_policy=True,
        batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.contiguous_storage_view()
        ),
        output_publication_policy=(
            AlgebraicOutputPublicationPolicy.deferred_stack()
        ),
    )
    control.reset_projected_batch_assembly_diagnostics()
    candidate.reset_projected_batch_assembly_diagnostics()
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
    diagnostics = candidate.projected_batch_assembly_diagnostics()
    control_diagnostics = control.projected_batch_assembly_diagnostics()
    assert diagnostics["contiguous_view_batches"] > 0
    assert diagnostics["copy_cat_batches"] < (
        control_diagnostics["copy_cat_batches"]
    )
    assert diagnostics["retained_tensor_references"] == 0
    assert candidate.algebraic_output_publication_policy.to_metadata()[
        "mode"
    ] == "deferred_stack"


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
