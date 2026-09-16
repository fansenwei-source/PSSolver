"""CPU contracts for the bounded Stage Q.4 batch-workspace candidate."""

from __future__ import annotations

import gc
from pathlib import Path
import weakref

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
    AlgebraicExecutionPolicy,
    IncompressibleStokesSystemSpec,
    TangentialZeroModePolicy,
)
from pssolver.experimental import (
    BoundarySignatureTransformScheduler,
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


PROJECT_ROOT = Path(__file__).parents[1]
P = PeriodicBC()
N = HomogeneousNeumannBC()


class _WorkspaceTransformContext:
    batch_size = 1
    physical_shape = (2, 3)
    spectral_shape = (2, 3)
    real_dtype = torch.float64
    spectral_dtype = torch.complex128
    device = torch.device("cpu")

    def boundary_conditions(self, name: str) -> tuple[str, ...]:
        del name
        return ("periodic", "neumann")

    def forward_projected(
        self,
        name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del name
        return torch.complex(value, torch.zeros_like(value))

    def inverse_projected(
        self,
        name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del name
        return value.real.clone()

    def transform_projected_packed(
        self,
        boundaries: tuple[str, ...],
        value: torch.Tensor,
        *,
        direction: ProjectedTransformDirection,
    ) -> torch.Tensor:
        assert boundaries == ("periodic", "neumann")
        if direction is ProjectedTransformDirection.FORWARD:
            return torch.complex(value, torch.zeros_like(value))
        return value.real.clone()


def _workspace_scheduler(
    context: _WorkspaceTransformContext | None = None,
) -> BoundarySignatureTransformScheduler:
    return BoundarySignatureTransformScheduler(
        context or _WorkspaceTransformContext(),
        batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.preallocated_workspace(
                source_prefixes=("test.",)
            )
        ),
        enable_batch_assembly_diagnostics=True,
    )


def _runtime(
    *,
    workspace: bool,
    device: str = "cpu",
    shape: tuple[int, int, int] = (8, 6, 5),
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
            tangential_zero_mode_policy=(
                TangentialZeroModePolicy.ZERO_MEAN
            ),
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
        PlaneSlab(DomainSpec(shape, (5.0, 4.0, 3.5))),
        numerics,
        dt=0.005,
        device=device,
        geometry_solver_registry=(
            create_beris_edwards_plane_geometry_solver_registry(
                constitutive_options=PlaneBerisEdwardsSolverOptions(
                    molecular_field_linear_space="spectral",
                    stress_divergence_sum_space="spectral",
                    pointwise_execution="eager",
                )
            )
        ),
        algebraic_execution_policy=AlgebraicExecutionPolicy.batched(),
        projected_batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.preallocated_workspace()
            if workspace
            else ProjectedBatchAssemblyPolicy.copy_cat()
        ),
        enable_performance_instrumentation=True,
    )


def test_workspace_policy_reuses_one_plan_owned_buffer_without_input_retention():
    scheduler = _workspace_scheduler()
    names = ("a", "b", "c")
    first = tuple(
        torch.full((1, 2, 3), float(index), dtype=torch.float64)
        for index in range(3)
    )
    weak_inputs = tuple(weakref.ref(value) for value in first)
    first_result = scheduler.forward_values_many(
        names,
        first,
        attribution_source="test.forward",
    )
    assert scheduler.compiled_plan_count == 1
    assert scheduler.compiled_workspace_count == 1
    workspace = next(iter(scheduler._workspace_cache.values()))
    workspace_pointer = workspace.data_ptr()
    assert all(value.data_ptr() != workspace_pointer for value in first)
    for actual, expected in zip(first_result, first, strict=True):
        assert torch.equal(actual.real, expected)
    del actual, expected

    del first_result, first
    gc.collect()
    assert all(reference() is None for reference in weak_inputs)

    second = tuple(
        torch.full((1, 2, 3), float(index + 4), dtype=torch.float64)
        for index in range(3)
    )
    second_result = scheduler.forward_values_many(
        names,
        second,
        attribution_source="test.forward",
    )
    assert next(iter(scheduler._workspace_cache.values())).data_ptr() == (
        workspace_pointer
    )
    for actual, expected in zip(second_result, second, strict=True):
        assert torch.equal(actual.real, expected)

    diagnostics = scheduler.batch_assembly_diagnostics()
    assert diagnostics["preallocated_workspace_batches"] == 2
    assert diagnostics["copy_cat_batches"] == 0
    assert diagnostics["workspace_count"] == 1
    assert diagnostics["workspace_active_count"] == 0
    assert diagnostics["workspace_retains_timestep_inputs"] is False
    assert diagnostics["retained_tensor_references"] == 0


def test_workspace_policy_preserves_inverse_results_and_autograd_falls_back():
    scheduler = _workspace_scheduler()
    names = ("a", "b")
    real = tuple(
        torch.full((1, 2, 3), float(index), dtype=torch.float64)
        for index in (1, 2)
    )
    spectral = tuple(
        torch.complex(value, torch.zeros_like(value)) for value in real
    )
    observed = scheduler.inverse_values_many(
        names,
        spectral,
        attribution_source="test.inverse",
    )
    for actual, expected in zip(observed, real, strict=True):
        assert torch.equal(actual, expected)

    differentiable = tuple(value.requires_grad_() for value in real)
    scheduler.forward_values_many(
        names,
        differentiable,
        attribution_source="test.forward",
    )
    diagnostics = scheduler.batch_assembly_diagnostics()
    assert diagnostics["preallocated_workspace_batches"] == 1
    assert diagnostics["copy_cat_batches"] == 1
    assert diagnostics["fallback_reasons"] == {"autograd_enabled": 1}


def test_workspace_policy_rejects_transform_output_aliasing():
    class _AliasingContext(_WorkspaceTransformContext):
        def transform_projected_packed(
            self,
            boundaries: tuple[str, ...],
            value: torch.Tensor,
            *,
            direction: ProjectedTransformDirection,
        ) -> torch.Tensor:
            del boundaries, direction
            return value

    scheduler = _workspace_scheduler(_AliasingContext())
    values = tuple(
        torch.ones((1, 2, 3), dtype=torch.float64) for _ in range(2)
    )
    with pytest.raises(RuntimeError, match="aliases its input workspace"):
        scheduler.forward_values_many(
            ("a", "b"),
            values,
            attribution_source="test.forward",
        )
    assert scheduler.batch_assembly_diagnostics()["workspace_active_count"] == 0


def test_workspace_candidate_preserves_complete_cpu_timesteps():
    control = _runtime(workspace=False)
    candidate = _runtime(workspace=True)
    control.solver.run(1)
    candidate.solver.run(1)
    scheduler = candidate.algebraic_fields_adapter._context.transform_scheduler
    workspace_count = scheduler.compiled_workspace_count
    assert workspace_count > 0
    control.solver.run(2)
    candidate.solver.run(2)
    assert scheduler.compiled_workspace_count == workspace_count

    assert torch.equal(
        candidate.solver.fields.spatial,
        control.solver.fields.spatial,
    )
    assert torch.equal(
        candidate.solver.fields.spectral,
        control.solver.fields.spectral,
    )
    diagnostics = candidate.projected_batch_assembly_diagnostics()
    assert diagnostics["preallocated_workspace_batches"] > 0
    assert diagnostics["workspace_count"] > 0
    assert diagnostics["workspace_active_count"] == 0
    for source, counters in diagnostics["source_attribution"].items():
        if source.startswith("algebraic."):
            assert counters["preallocated_workspace_batches"] > 0
            assert counters["copy_cat_batches"] == 0
        else:
            assert counters["preallocated_workspace_batches"] == 0
            assert counters["copy_cat_batches"] > 0
    metadata = candidate.to_metadata()["projected_batch_assembly_policy"]
    assert metadata["mode"] == "preallocated_workspace"
    assert metadata["owns_runtime_workspace"] is True
    assert metadata["workspace_retains_timestep_inputs"] is False


def test_workspace_requires_batched_policy_and_remains_experimental():
    control = _runtime(workspace=False)
    with pytest.raises(ValueError, match="requires batched physical islands"):
        build_experimental_model_runtime(
            control.problem.model,
            control.problem.geometry,
            control.problem.numerics,
            dt=0.005,
            projected_batch_assembly_policy=(
                ProjectedBatchAssemblyPolicy.preallocated_workspace()
            ),
        )
    with pytest.raises(ValueError, match="at least one source prefix"):
        ProjectedBatchAssemblyPolicy.preallocated_workspace(
            source_prefixes=()
        )

    assert not hasattr(pssolver, "ProjectedBatchAssemblyPolicy")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "preallocated_workspace" not in source
