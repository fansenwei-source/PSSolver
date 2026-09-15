"""CPU contracts for Stage O.4.3.3 producer-owned H/stress packing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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
    AlgebraicOutputPublicationPolicy,
    IncompressibleStokesSystemSpec,
    TangentialZeroModePolicy,
)
from pssolver.experimental import (
    PlaneBerisEdwardsSolverOptions,
    ProjectedBatchAssemblyPolicy,
    build_experimental_model_runtime,
    create_beris_edwards_plane_geometry_solver_registry,
)
from pssolver.experimental.beris_edwards import (
    _STRESS_BOUNDARY_STORAGE_ORDER,
    _plane_boundary_packed_stress,
    _plane_packed_bulk_molecular_field,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    ALGEBRAIC_STRESS_COMPONENTS,
    DISTORTION_STRESS_COMPONENTS,
    H_COMPONENTS,
    NEMATIC_FORCE_COMPONENTS,
    Q_COMPONENTS,
    BerisEdwardsConstitutiveParameters,
    BerisEdwardsConstitutiveStokesCanaryModel,
    BerisEdwardsPlaneCoupledModel,
    beris_edwards_algebraic_stress_components,
    beris_edwards_bulk_molecular_field_components,
    beris_edwards_distortion_stress_components,
)


PROJECT_ROOT = Path(__file__).parents[1]
P = PeriodicBC()
N = HomogeneousNeumannBC()
D = HomogeneousDirichletBC()
EVEN = BoundarySet((P, P, N))
ODD = BoundarySet((P, P, D))


def _random_components(count: int, *, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn((1, 6, 5, 4), generator=generator, dtype=torch.float64)
        for _ in range(count)
    )


def _build_runtime(producer_output_layout: str):
    parameters = BerisEdwardsConstitutiveParameters(
        ldg_a=0.07,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=0.04,
        flow_alignment=0.31,
        active_prefactor=-0.18,
    )
    constitutive = BerisEdwardsConstitutiveStokesCanaryModel(
        q_boundaries=EVEN,
        tangential_boundaries=EVEN,
        normal_boundaries=ODD,
        pressure_boundaries=EVEN,
        parameters=parameters,
        stokes_system=IncompressibleStokesSystemSpec(
            name="flow",
            force_components=NEMATIC_FORCE_COMPONENTS,
            velocity_components=("ux", "uy", "uz"),
            pressure_component="p",
            viscosity=2.0 / 3.0,
            friction=0.0,
            tangential_zero_mode_policy=(
                TangentialZeroModePolicy.ZERO_MEAN
            ),
        ),
    )
    model = BerisEdwardsPlaneCoupledModel(
        constitutive_model=constitutive,
        rotational_viscosity=2.94,
    )
    return build_experimental_model_runtime(
        model,
        PlaneSlab(DomainSpec((10, 8, 7), (5.0, 4.0, 3.5))),
        NumericsConfig(
            precision=Precision.FLOAT64,
            dealias_rule=DealiasRule.TWO_THIRDS,
            transform_execution_order=TransformExecutionOrder.REAL_FIRST,
            projected_transform_execution=ProjectedTransformExecution.FULL,
            spectral_storage=SpectralStorage.FULL_COMPLEX,
        ),
        dt=0.01,
        batch_size=1,
        enable_performance_instrumentation=True,
        geometry_solver_registry=(
            create_beris_edwards_plane_geometry_solver_registry(
                constitutive_options=PlaneBerisEdwardsSolverOptions(
                    molecular_field_linear_space="spectral",
                    stress_divergence_sum_space="spectral",
                    pointwise_execution="eager",
                    producer_output_layout=producer_output_layout,
                )
            )
        ),
        algebraic_execution_policy=AlgebraicExecutionPolicy.batched(),
        algebraic_output_publication_policy=(
            AlgebraicOutputPublicationPolicy.deferred_stack()
        ),
        projected_batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.contiguous_storage_view()
        ),
    )


def test_packed_pointwise_producers_match_component_helpers():
    q = _random_components(5, seed=1)
    h = _random_components(5, seed=2)
    flat_gradients = _random_components(15, seed=3)
    gradients = tuple(
        flat_gradients[axis * 5 : (axis + 1) * 5] for axis in range(3)
    )

    packed_h = _plane_packed_bulk_molecular_field(
        q,
        ldg_a=0.07,
        ldg_b=-0.3,
        ldg_c=0.3,
    )
    expected_h = beris_edwards_bulk_molecular_field_components(
        q,
        ldg_a=0.07,
        ldg_b=-0.3,
        ldg_c=0.3,
    )
    assert packed_h.is_contiguous()
    for actual, expected in zip(packed_h, expected_h, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    packed_stress = _plane_boundary_packed_stress(
        q,
        h,
        gradients,
        flow_alignment=0.31,
        active_prefactor=-0.18,
        ldg_l1=0.04,
    )
    algebraic = beris_edwards_algebraic_stress_components(
        q,
        h,
        flow_alignment=0.31,
        active_prefactor=-0.18,
    )
    distortion = beris_edwards_distortion_stress_components(
        gradients,
        ldg_l1=0.04,
    )
    expected_by_name = dict(
        zip(
            (*ALGEBRAIC_STRESS_COMPONENTS, *DISTORTION_STRESS_COMPONENTS),
            (*algebraic, *distortion),
            strict=True,
        )
    )
    assert packed_stress.is_contiguous()
    for name, actual in zip(
        _STRESS_BOUNDARY_STORAGE_ORDER,
        packed_stress,
        strict=True,
    ):
        torch.testing.assert_close(
            actual,
            expected_by_name[name],
            rtol=0.0,
            atol=0.0,
        )


def test_boundary_packed_runtime_is_equivalent_and_removes_three_copy_batches():
    baseline = _build_runtime("component_mapping")
    candidate = _build_runtime("boundary_packed")
    baseline.reset_projected_batch_assembly_diagnostics()
    candidate.reset_projected_batch_assembly_diagnostics()

    baseline.solver.run(2)
    candidate.solver.run(2)

    torch.testing.assert_close(
        candidate.solver.fields.spatial,
        baseline.solver.fields.spatial,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        candidate.solver.fields.spectral,
        baseline.solver.fields.spectral,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    baseline_diagnostics = baseline.projected_batch_assembly_diagnostics()
    candidate_diagnostics = candidate.projected_batch_assembly_diagnostics()
    assert candidate_diagnostics["contiguous_view_batches"] >= (
        baseline_diagnostics["contiguous_view_batches"] + 3
    )
    assert candidate_diagnostics["copy_cat_batches"] <= (
        baseline_diagnostics["copy_cat_batches"] - 3
    )
    assert candidate_diagnostics["retained_tensor_references"] == 0

    systems = {
        item["system"]["name"]: item
        for item in candidate.to_metadata()["algebraic_lifecycle"]["systems"]
    }
    for name in ("molecular_field", "nematic_stress"):
        policy = systems[name]["observability"]["numerical_policy"]
        assert policy["producer_output_layout"] == "boundary_packed"
        packing = policy["producer_packing"]
        assert packing["ownership"] == "producer"
        assert packing["packing_site"] == "inside_pointwise_kernel"
        assert packing["post_kernel_stack"] is False
        assert packing["post_kernel_cat"] is False


def test_producer_layout_is_opt_in_and_remains_geometry_isolated():
    assert PlaneBerisEdwardsSolverOptions().producer_output_layout == (
        "component_mapping"
    )
    with pytest.raises(ValueError, match="producer_output_layout"):
        PlaneBerisEdwardsSolverOptions(producer_output_layout="packed")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "producer_output_layout" not in source
        assert "boundary_packed" not in source
