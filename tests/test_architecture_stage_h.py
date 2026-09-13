"""Characterization tests for Stage H algebraic DAGs and field lifetimes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math

import pytest
import torch

import pssolver
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    FieldRole,
    FieldSpec,
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
    AlgebraicSystemSpec,
    MathematicalOperatorContext,
    build_algebraic_execution_plan,
)
from pssolver.experimental import (
    build_experimental_model_runtime,
    create_canary_geometry_solver_registry,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.canary import (
    DifferentialAlgebraicChainModel,
    ScalarDiffusionModel,
)


P = PeriodicBC()
N = HomogeneousNeumannBC()
D = HomogeneousDirichletBC()
SCALAR_BOUNDARIES = BoundarySet((P, N))
GRADIENT_BOUNDARIES = BoundarySet((P, D))


def _numerics(*, half_spectrum=False):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=(
            DealiasRule.CUBIC_HALF
            if half_spectrum
            else DealiasRule.NONE
        ),
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=(
            ProjectedTransformExecution.TRUNCATED
            if half_spectrum
            else ProjectedTransformExecution.FULL
        ),
        spectral_storage=(
            SpectralStorage.HERMITIAN_HALF
            if half_spectrum
            else SpectralStorage.FULL_COMPLEX
        ),
        hermitian_axis=0 if half_spectrum else None,
    )


def _model():
    return DifferentialAlgebraicChainModel(
        boundaries=SCALAR_BOUNDARIES,
        gradient_boundaries=GRADIENT_BOUNDARIES,
        gradient_axis=1,
        diffusivity=0.07,
        coupling=-0.3,
        helmholtz_shift=1.1,
        helmholtz_length_sq=0.2,
        initial_amplitude=0.25,
        initial_modes=(1, 2),
    )


def _build(*, half_spectrum=False, batch_size=2):
    return build_experimental_model_runtime(
        _model(),
        PlaneSlab(DomainSpec((8, 7), (4.0, 3.5))),
        _numerics(half_spectrum=half_spectrum),
        dt=0.01,
        batch_size=batch_size,
        geometry_solver_registry=create_canary_geometry_solver_registry(),
    )


@pytest.mark.parametrize("half_spectrum", (False, True))
def test_reverse_declared_algebraic_chain_is_frozen_and_evaluated(
    half_spectrum,
):
    runtime = _build(half_spectrum=half_spectrum)
    execution = runtime.algebraic_execution_plan

    assert tuple(
        system.name for system in execution.declared_systems
    ) == ("screened_gradient", "wall_normal_gradient")
    assert execution.execution_order == (
        "wall_normal_gradient",
        "screened_gradient",
    )
    assert tuple(
        (edge.producer, edge.consumer, edge.component)
        for edge in execution.dependency_edges
    ) == (
        ("wall_normal_gradient", "screened_gradient", "gradient_phi"),
    )
    assert tuple(
        resolved.system.name for resolved in runtime.resolved_algebraic_systems
    ) == execution.execution_order

    phi = runtime.solver.fields["phi"]
    phi_hat = runtime.projector.forward_transform(
        phi,
        ("periodic", "neumann"),
    )
    raw_gradient_hat, gradient_boundaries = (
        runtime.solver.transform_backend.gradient_hat(
            phi_hat,
            ("periodic", "neumann"),
            1,
        )
    )
    assert gradient_boundaries == ("periodic", "dirichlet")
    raw_gradient = runtime.projector.inverse_transform(
        raw_gradient_hat,
        gradient_boundaries,
    )
    gradient_hat = runtime.projector.forward_transform(
        raw_gradient,
        gradient_boundaries,
    )
    expected_gradient = runtime.projector.inverse_transform(
        gradient_hat,
        gradient_boundaries,
    )
    torch.testing.assert_close(
        runtime.transient_algebraic_state()["gradient_phi"],
        expected_gradient,
        rtol=0.0,
        atol=0.0,
    )

    denominator = (
        _model().helmholtz_shift
        - _model().helmholtz_length_sq
        * runtime.solver.get_laplacian_eigs(gradient_boundaries)
    )
    response_input_hat = runtime.projector.forward_transform(
        expected_gradient,
        gradient_boundaries,
    )
    torch.testing.assert_close(
        runtime.solver.fields["response.hat"],
        response_input_hat / denominator,
        rtol=0.0,
        atol=0.0,
    )


def test_transient_component_is_planned_but_not_runtime_or_restart_storage():
    runtime = _build()
    stored = tuple(
        component.component_name for component in runtime.plan.stored_components
    )
    transient = tuple(
        component.component_name
        for component in runtime.plan.transient_components
    )

    assert stored == ("phi", "response")
    assert transient == ("gradient_phi",)
    assert runtime.plan.transient_component_count == 1
    assert runtime.assembly.omitted_transient_components == transient
    assert "gradient_phi" not in runtime.solver.fields.name_to_idx
    assert runtime.solver.fields.spatial.shape[0] == 2
    assert all(
        "gradient_phi" not in system.tensors
        for system in runtime.capture_algebraic_restart_state().systems
    )

    metadata = runtime.to_metadata()
    lifecycle = metadata["algebraic_lifecycle"]
    assert lifecycle["stored_components"] == ["response"]
    assert lifecycle["transient_components"] == ["gradient_phi"]
    assert lifecycle["transient_cache"] == {
        "lifetime": "one_synchronized_pre_rhs_state",
        "stored_in_legacy_fields": False,
        "checkpointed": False,
    }
    assert metadata["spectral_plan"]["transient"] == {
        "component_names": ["gradient_phi"],
        "component_count": 1,
        "persistent_storage": False,
    }
    json.dumps(metadata, allow_nan=False)


def test_transient_cache_is_read_only_stale_after_step_and_refreshed_on_sync():
    runtime = _build(batch_size=1)
    adapter = runtime.algebraic_fields_adapter
    initial_generation = adapter.cache_generation
    cached = runtime.transient_algebraic_state()["gradient_phi"].clone()
    with pytest.raises(TypeError):
        runtime.transient_algebraic_state()["gradient_phi"] = cached

    runtime.solver.run(1)
    assert adapter.cache_generation == initial_generation
    assert torch.equal(
        runtime.transient_algebraic_state()["gradient_phi"],
        cached,
    )

    runtime.synchronize_algebraic_for_observation()
    assert adapter.cache_generation == initial_generation + 1
    assert not torch.equal(
        runtime.transient_algebraic_state()["gradient_phi"],
        cached,
    )
    before_reset = adapter.cache_generation
    runtime.reset()
    assert adapter.cache_generation == before_reset + 1
    assert runtime.solver.integrator._static_fields_are_current is True


def test_explicit_rhs_consumes_both_stored_and_transient_outputs():
    runtime = _build(batch_size=1)
    model = _model()
    expected = runtime.projector.forward_transform(
        model.coupling
        * runtime.transient_algebraic_state()["gradient_phi"]
        * runtime.solver.fields["response"],
        ("periodic", "neumann"),
    ).unsqueeze(0)
    actual = runtime.explicit_rhs_adapter(
        runtime.solver.fields,
        runtime.solver.parameters,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    runtime.algebraic_fields_adapter.clear_cached_outputs()
    with pytest.raises(RuntimeError, match="not synchronized"):
        runtime.explicit_rhs_adapter(
            runtime.solver.fields,
            runtime.solver.parameters,
        )


def test_mathematical_operator_context_hides_basis_choices_and_is_correct():
    runtime = _build(batch_size=1)
    context = runtime.context
    phi = runtime.solver.fields["phi"]
    x, z = runtime.context.axis_coordinates
    kx = 2.0 * math.pi / runtime.context.lengths[0]
    kz = 2.0 * math.pi / runtime.context.lengths[1]
    shape_x = (1, x.numel(), 1)
    shape_z = (1, 1, z.numel())
    expected_dx = (
        -_model().initial_amplitude
        * kx
        * torch.sin(kx * x).reshape(shape_x)
        * torch.cos(kz * z).reshape(shape_z)
    )
    expected_dz = (
        -_model().initial_amplitude
        * kz
        * torch.cos(kx * x).reshape(shape_x)
        * torch.sin(kz * z).reshape(shape_z)
    )

    assert isinstance(context, MathematicalOperatorContext)
    torch.testing.assert_close(
        context.gradient("phi", "phi", phi, 0),
        expected_dx,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        context.gradient("phi", "gradient_phi", phi, 1),
        expected_dz,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        context.laplacian("phi", phi),
        -(kx * kx + kz * kz) * phi,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    torch.testing.assert_close(
        context.divergence(
            ("phi", "gradient_phi"),
            "phi",
            (phi, expected_dz),
        ),
        expected_dx - kz * kz * phi,
        rtol=3.0e-13,
        atol=3.0e-13,
    )
    with pytest.raises(ValueError, match="boundary space"):
        context.gradient("phi", "phi", phi, 1)
    assert not hasattr(context, "forward_projected")
    assert not hasattr(context, "legacy_transform_backend")


@dataclass(frozen=True)
class _OperatorRHSModel(ScalarDiffusionModel):
    def explicit_rhs(self, state, context):
        return {"phi": context.laplacian("phi", state["phi"])}


def test_explicit_rhs_receives_the_mathematical_operator_context():
    model = _OperatorRHSModel(
        SCALAR_BOUNDARIES,
        diffusivity=0.07,
        initial_amplitude=0.25,
        initial_modes=(1, 2),
    )
    runtime = build_experimental_model_runtime(
        model,
        PlaneSlab(DomainSpec((8, 7), (4.0, 3.5))),
        _numerics(),
        dt=0.01,
        batch_size=1,
    )
    expected = runtime.projector.forward_transform(
        runtime.context.laplacian("phi", runtime.solver.fields["phi"]),
        ("periodic", "neumann"),
    ).unsqueeze(0)
    actual = runtime.explicit_rhs_adapter(
        runtime.solver.fields,
        runtime.solver.parameters,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_dependency_plan_rejects_missing_owners_duplicate_owners_and_cycles():
    first = AlgebraicSystemSpec(
        name="first",
        capability="first_capability",
        output_components=("a",),
        dependencies=("phi",),
    )
    missing = AlgebraicSystemSpec(
        name="missing",
        capability="missing_capability",
        output_components=("b",),
        dependencies=("unknown",),
    )
    with pytest.raises(ValueError, match="must be evolved or algebraic"):
        build_algebraic_execution_plan(
            (first, missing),
            initial_components=("phi",),
            output_components=("a", "b"),
        )

    duplicate = AlgebraicSystemSpec(
        name="duplicate",
        capability="duplicate_capability",
        output_components=("a",),
        dependencies=("phi",),
    )
    with pytest.raises(ValueError, match="multiple owners"):
        build_algebraic_execution_plan(
            (first, duplicate),
            initial_components=("phi",),
            output_components=("a",),
        )

    cyclic_a = AlgebraicSystemSpec(
        name="cyclic_a",
        capability="capability_a",
        output_components=("a",),
        dependencies=("b",),
    )
    cyclic_b = AlgebraicSystemSpec(
        name="cyclic_b",
        capability="capability_b",
        output_components=("b",),
        dependencies=("a",),
    )
    with pytest.raises(ValueError, match="contains a cycle"):
        build_algebraic_execution_plan(
            (cyclic_a, cyclic_b),
            initial_components=("phi",),
            output_components=("a", "b"),
        )


@dataclass(frozen=True)
class _TransientOnlyModel:
    def field_specs(self):
        return (
            FieldSpec.scalar("phi", FieldRole.EVOLVED, SCALAR_BOUNDARIES),
            FieldSpec.scalar(
                "gradient_phi",
                FieldRole.TRANSIENT,
                GRADIENT_BOUNDARIES,
            ),
        )

    @property
    def name(self):
        return "transient_only"

    def parameter_metadata(self):
        return {}

    def algebraic_system_specs(self):
        return (
            AlgebraicSystemSpec(
                name="gradient",
                capability="component_gradient",
                output_components=("gradient_phi",),
                dependencies=("phi",),
                parameters={"axis": 1},
            ),
        )

    def initial_values(self, context):
        return {
            "phi": torch.zeros(
                context.physical_shape,
                dtype=context.real_dtype,
                device=context.device,
            )
        }

    def linear_operators(self, context):
        return {"phi": torch.zeros_like(context.laplacian_eigenvalues("phi"))}

    def explicit_rhs(self, state, context):
        del context
        return {"phi": state["gradient_phi"]}


def test_legacy_bridge_rejects_a_transient_only_graph_without_storage_driver():
    with pytest.raises(ValueError, match="at least one stored algebraic"):
        build_experimental_model_runtime(
            _TransientOnlyModel(),
            PlaneSlab(DomainSpec((8, 7), (4.0, 3.5))),
            _numerics(),
            dt=0.01,
            geometry_solver_registry=create_canary_geometry_solver_registry(),
        )


def test_stage_h_remains_isolated_from_production_and_benchmark_paths():
    assert not hasattr(pssolver, "AlgebraicExecutionPlan")
    assert not hasattr(pssolver, "MathematicalOperatorContext")
    assert not hasattr(pssolver, "DifferentialAlgebraicChainModel")
    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
        "pssolver/models/active_nematics/stokes.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "DifferentialAlgebraicChainModel" not in text
        assert "AlgebraicExecutionPlan" not in text
        assert "FieldRole.TRANSIENT" not in text
