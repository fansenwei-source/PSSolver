"""Characterization tests for read-only Stage C spectral planning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import json
from types import SimpleNamespace

import pytest
import torch

from pssolver.Field import Fields
from pssolver.adapters import compare_spectral_plan_to_runtime
from pssolver.channel import Q_BC as LEGACY_CHANNEL_Q_BC
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    FieldComponentSpec,
    FieldRole,
    FieldSpec,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProblemSpec,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.geometries import PeriodicBox, PlaneSlab, RectangularChannel
from pssolver.models.active_nematics.stokes import (
    PLANE_Q_BOUNDARY_CONDITIONS,
)
from pssolver.plane import (
    DEFAULT_PLANE_SPECTRAL_STORAGE,
    PLANE_HERMITIAN_AXIS,
)
from pssolver.planning import TransformKind, assemble_spectral_plan
from pssolver.transforms import (
    BasisAwareSpectralProjector,
    TensorProductTransformBackend,
)


def _boundaries(*conditions):
    return BoundarySet(conditions)


def _plane_boundaries(normal_condition):
    return _boundaries(PeriodicBC(), PeriodicBC(), normal_condition)


@dataclass(frozen=True)
class DeclaredModel:
    specs: tuple[FieldSpec, ...]
    name: str = "declared_model"

    def field_specs(self):
        return self.specs

    def parameter_metadata(self):
        return {"coefficient": 2.0, "nested": {"enabled": True}}


def _active_like_plane_model(*, include_diagnostic=True):
    q_boundaries = _plane_boundaries(HomogeneousNeumannBC())
    normal_velocity_boundaries = _plane_boundaries(
        HomogeneousDirichletBC()
    )
    specs = [
        FieldSpec(
            name="Q",
            role=FieldRole.EVOLVED,
            components=tuple(
                FieldComponentSpec(name, q_boundaries)
                for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
            ),
        ),
        FieldSpec(
            name="velocity",
            role=FieldRole.ALGEBRAIC,
            components=(
                FieldComponentSpec("ux", q_boundaries),
                FieldComponentSpec("uy", q_boundaries),
                FieldComponentSpec("uz", normal_velocity_boundaries),
            ),
        ),
        FieldSpec.scalar("p", FieldRole.ALGEBRAIC, q_boundaries),
    ]
    if include_diagnostic:
        specs.append(
            FieldSpec.scalar(
                "energy",
                FieldRole.DIAGNOSTIC,
                q_boundaries,
            )
        )
    return DeclaredModel(tuple(specs), name="active_like_plane")


def _numerics(
    *,
    dealias_rule=DealiasRule.CUBIC_HALF,
    projected_execution=ProjectedTransformExecution.TRUNCATED,
    storage=SpectralStorage.HERMITIAN_HALF,
    hermitian_axis=1,
):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=dealias_rule,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=projected_execution,
        spectral_storage=storage,
        hermitian_axis=hermitian_axis,
    )


def _plane_problem(shape=(12, 10, 8), **numerics_kwargs):
    domain = DomainSpec(shape, (100.0, 90.0, 20.0))
    return ProblemSpec(
        _active_like_plane_model(),
        PlaneSlab(domain),
        _numerics(**numerics_kwargs),
    )


def _backend_for_plan(plan):
    dtype = {
        Precision.FLOAT32: torch.float32,
        Precision.FLOAT64: torch.float64,
    }[plan.numerics.precision]
    return TensorProductTransformBackend(
        shape=plan.physical_shape,
        lengths=plan.lengths,
        device="cpu",
        dtype=dtype,
        execution_order=plan.numerics.transform_execution_order.value,
        spectral_storage=plan.numerics.spectral_storage.value,
        hermitian_axis=plan.numerics.hermitian_axis,
    )


def _projector_for_plan(plan, backend):
    solver_view = SimpleNamespace(
        shape=plan.physical_shape,
        transform_backend=backend,
    )
    return BasisAwareSpectralProjector(
        solver_view,
        rule=plan.numerics.dealias_rule.value,
        transform_execution=plan.numerics.projected_transform_execution.value,
    )


def _fields_for_plan(plan, backend):
    fields = Fields(
        plan.physical_shape,
        device="cpu",
        dtype=backend.real_dtype,
    )
    fields.set_transform_backend(backend)
    fields.name_to_idx = {
        component.component_name: component.storage_index
        for component in plan.stored_components
    }
    fields.boundary_conditions = [
        tuple(condition.kind.value for condition in component.boundaries.axes)
        for component in plan.stored_components
    ]
    fields.dyn_count = plan.evolved_component_count
    fields.stat_count = plan.algebraic_component_count
    fields._refresh_metadata()
    return fields


def test_plane_plan_resolves_layout_storage_and_transform_families():
    plan = assemble_spectral_plan(_plane_problem())

    assert plan.model_name == "active_like_plane"
    assert plan.geometry_name == "plane_slab"
    assert plan.physical_shape == (12, 10, 8)
    assert plan.spectral_shape == (12, 6, 8)
    assert plan.numerics.spectral_storage.value == (
        DEFAULT_PLANE_SPECTRAL_STORAGE
    )
    assert plan.numerics.hermitian_axis == PLANE_HERMITIAN_AXIS
    assert tuple(axis.hermitian_packed for axis in plan.axes) == (
        False,
        True,
        False,
    )
    assert tuple(
        component.component_name for component in plan.stored_components
    ) == ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz", "ux", "uy", "uz", "p")
    assert tuple(
        component.storage_index for component in plan.stored_components
    ) == tuple(range(9))

    qxx = plan.stored_components[0]
    uz = plan.stored_components[7]
    assert qxx.transform_kinds == (
        TransformKind.FFT,
        TransformKind.FFT,
        TransformKind.DCT,
    )
    assert uz.transform_kinds == (
        TransformKind.FFT,
        TransformKind.FFT,
        TransformKind.DST,
    )
    assert qxx.retained_mode_counts == (5, 3, 4)
    assert qxx.computed_axis_sizes == (12, 6, 4)
    assert uz.retained_mode_counts == (5, 3, 3)
    assert uz.computed_axis_sizes == (12, 6, 3)
    assert tuple(
        condition.kind.value for condition in qxx.boundaries.axes
    ) == PLANE_Q_BOUNDARY_CONDITIONS


def test_diagnostic_components_are_declared_but_not_runtime_stored():
    plan = assemble_spectral_plan(_plane_problem())
    diagnostic = plan.fields[-1].components[0]

    assert diagnostic.role is FieldRole.DIAGNOSTIC
    assert diagnostic.storage_index is None
    assert diagnostic not in plan.stored_components
    assert plan.evolved_component_count == 5
    assert plan.algebraic_component_count == 4


def test_runtime_storage_is_role_ordered_without_reordering_field_declarations():
    boundaries = _plane_boundaries(HomogeneousNeumannBC())
    model = DeclaredModel(
        (
            FieldSpec.scalar("pressure", FieldRole.ALGEBRAIC, boundaries),
            FieldSpec.scalar("phase", FieldRole.EVOLVED, boundaries),
            FieldSpec.scalar("measure", FieldRole.DIAGNOSTIC, boundaries),
        )
    )
    domain = DomainSpec((8, 6, 4), (8.0, 6.0, 4.0))
    plan = assemble_spectral_plan(
        ProblemSpec(model, PlaneSlab(domain), _numerics())
    )

    assert tuple(field.name for field in plan.fields) == (
        "pressure",
        "phase",
        "measure",
    )
    assert tuple(
        component.component_name for component in plan.stored_components
    ) == ("phase", "pressure")


def test_plan_is_immutable_and_metadata_is_an_independent_json_value():
    plan = assemble_spectral_plan(_plane_problem())
    metadata = plan.to_metadata()
    metadata["model"]["parameters"]["coefficient"] = 99.0

    assert plan.to_metadata()["model"]["parameters"]["coefficient"] == 2.0
    assert plan.to_metadata()["schema_version"] == 1
    json.dumps(plan.to_metadata(), allow_nan=False)
    with pytest.raises(FrozenInstanceError):
        plan.geometry_name = "changed"


def test_assembly_uses_the_problem_snapshot_not_a_later_model_mutation():
    class MutableModel:
        def __init__(self):
            self.name = "before"
            self.coefficient = 2.0
            self.specs = _active_like_plane_model(
                include_diagnostic=False
            ).field_specs()

        def field_specs(self):
            return self.specs

        def parameter_metadata(self):
            return {"coefficient": self.coefficient}

    model = MutableModel()
    domain = DomainSpec((8, 6, 4), (8.0, 6.0, 4.0))
    problem = ProblemSpec(model, PlaneSlab(domain), _numerics())
    model.name = "after"
    model.coefficient = 99.0

    plan = assemble_spectral_plan(problem)

    assert plan.model_name == "before"
    assert plan.to_metadata()["model"]["parameters"] == {"coefficient": 2.0}


def test_plane_shadow_comparison_matches_backend_projector_and_fields():
    plan = assemble_spectral_plan(_plane_problem())
    backend = _backend_for_plan(plan)
    projector = _projector_for_plan(plan, backend)
    fields = _fields_for_plan(plan, backend)
    before = (
        dict(fields.name_to_idx),
        list(fields.boundary_conditions),
        fields.dyn_count,
        fields.stat_count,
    )

    comparison = compare_spectral_plan_to_runtime(
        plan,
        backend,
        fields=fields,
        projector=projector,
    )

    assert comparison.matches
    assert comparison.mismatches == ()
    assert comparison.checked_components == 9
    assert comparison.checked_fields
    assert comparison.checked_projector
    comparison.require_match()
    json.dumps(comparison.to_metadata())
    assert before == (
        dict(fields.name_to_idx),
        list(fields.boundary_conditions),
        fields.dyn_count,
        fields.stat_count,
    )


def test_channel_plan_matches_current_periodic_neumann_transform_metadata():
    boundaries = _boundaries(
        PeriodicBC(),
        HomogeneousNeumannBC(),
        HomogeneousNeumannBC(),
    )
    model = DeclaredModel(
        (FieldSpec.scalar("Qxx", FieldRole.EVOLVED, boundaries),),
        name="channel_scalar",
    )
    domain = DomainSpec((8, 6, 4), (8.0, 6.0, 4.0))
    problem = ProblemSpec(
        model,
        RectangularChannel(domain),
        _numerics(
            storage=SpectralStorage.FULL_COMPLEX,
            hermitian_axis=None,
        ),
    )
    plan = assemble_spectral_plan(problem)
    backend = _backend_for_plan(plan)
    projector = _projector_for_plan(plan, backend)

    assert plan.stored_components[0].transform_kinds == (
        TransformKind.FFT,
        TransformKind.DCT,
        TransformKind.DCT,
    )
    assert tuple(
        condition.kind.value
        for condition in plan.stored_components[0].boundaries.axes
    ) == LEGACY_CHANNEL_Q_BC
    assert plan.stored_components[0].computed_axis_sizes == (8, 3, 2)
    assert compare_spectral_plan_to_runtime(
        plan,
        backend,
        projector=projector,
    ).matches


def test_periodic_box_plan_matches_a_full_complex_backend():
    boundaries = _boundaries(PeriodicBC(), PeriodicBC())
    model = DeclaredModel(
        (FieldSpec.scalar("phase", FieldRole.EVOLVED, boundaries),)
    )
    domain = DomainSpec((9, 7), (9.0, 7.0))
    problem = ProblemSpec(
        model,
        PeriodicBox(domain),
        _numerics(
            dealias_rule=DealiasRule.NONE,
            projected_execution=ProjectedTransformExecution.FULL,
            storage=SpectralStorage.FULL_COMPLEX,
            hermitian_axis=None,
        ),
    )
    plan = assemble_spectral_plan(problem)
    backend = _backend_for_plan(plan)
    projector = _projector_for_plan(plan, backend)

    assert plan.spectral_shape == (9, 7)
    assert plan.stored_components[0].retained_mode_counts == (9, 7)
    assert compare_spectral_plan_to_runtime(
        plan,
        backend,
        projector=projector,
    ).matches


@pytest.mark.parametrize(
    "shape, rule, execution, storage, hermitian_axis",
    (
        (
            (7, 8, 5),
            DealiasRule.CUBIC_HALF,
            ProjectedTransformExecution.TRUNCATED,
            SpectralStorage.FULL_COMPLEX,
            None,
        ),
        (
            (8, 7, 6),
            DealiasRule.TWO_THIRDS,
            ProjectedTransformExecution.TRUNCATED,
            SpectralStorage.HERMITIAN_HALF,
            1,
        ),
        (
            (9, 8, 7),
            DealiasRule.CUBIC_HALF,
            ProjectedTransformExecution.FULL,
            SpectralStorage.HERMITIAN_HALF,
            0,
        ),
        (
            (7, 9, 5),
            DealiasRule.NONE,
            ProjectedTransformExecution.FULL,
            SpectralStorage.FULL_COMPLEX,
            None,
        ),
    ),
)
def test_planned_dealias_counts_match_the_qualified_runtime_projector(
    shape,
    rule,
    execution,
    storage,
    hermitian_axis,
):
    boundaries = _plane_boundaries(HomogeneousDirichletBC())
    model = DeclaredModel(
        (FieldSpec.scalar("phase", FieldRole.EVOLVED, boundaries),)
    )
    domain = DomainSpec(shape, tuple(float(size) for size in shape))
    problem = ProblemSpec(
        model,
        PlaneSlab(domain),
        _numerics(
            dealias_rule=rule,
            projected_execution=execution,
            storage=storage,
            hermitian_axis=hermitian_axis,
        ),
    )
    plan = assemble_spectral_plan(problem)
    backend = _backend_for_plan(plan)
    projector = _projector_for_plan(plan, backend)

    comparison = compare_spectral_plan_to_runtime(
        plan,
        backend,
        projector=projector,
    )

    assert comparison.matches, comparison.to_metadata()


def test_shadow_comparison_reports_structural_mismatches_without_execution():
    plan = assemble_spectral_plan(_plane_problem())
    wrong_backend = TensorProductTransformBackend(
        shape=plan.physical_shape,
        lengths=plan.lengths,
        device="cpu",
        dtype=torch.float32,
        execution_order="legacy",
        spectral_storage="full_complex",
        hermitian_axis=None,
    )

    comparison = compare_spectral_plan_to_runtime(plan, wrong_backend)
    paths = {mismatch.path for mismatch in comparison.mismatches}

    assert not comparison.matches
    assert "backend.spectral_shape" in paths
    assert "backend.execution_order" in paths
    assert "backend.spectral_storage" in paths
    assert "backend.hermitian_axis" in paths
    assert "backend.real_dtype" in paths
    with pytest.raises(RuntimeError, match="spectral-plan shadow mismatch"):
        comparison.require_match()


def test_shadow_comparison_reports_runtime_metadata_rejection_as_a_mismatch():
    plan = assemble_spectral_plan(_plane_problem())
    wrong_backend = TensorProductTransformBackend(
        shape=plan.physical_shape,
        lengths=plan.lengths,
        device="cpu",
        dtype=torch.float64,
        execution_order="real_first",
        spectral_storage="hermitian_half",
        hermitian_axis=2,
    )

    comparison = compare_spectral_plan_to_runtime(plan, wrong_backend)
    paths = {mismatch.path for mismatch in comparison.mismatches}

    assert not comparison.matches
    assert "components.Qxx.runtime_metadata" in paths


def test_shadow_comparison_reports_field_order_and_boundary_mismatches():
    plan = assemble_spectral_plan(_plane_problem())
    backend = _backend_for_plan(plan)
    fields = _fields_for_plan(plan, backend)
    names = tuple(fields.name_to_idx)
    fields.name_to_idx = {
        name: index for index, name in enumerate(reversed(names))
    }
    fields.boundary_conditions[0] = (
        "periodic",
        "periodic",
        "dirichlet",
    )

    comparison = compare_spectral_plan_to_runtime(
        plan,
        backend,
        fields=fields,
    )
    paths = {mismatch.path for mismatch in comparison.mismatches}

    assert "fields.component_names" in paths
    assert "fields.Qxx.boundary_conditions" in paths


@pytest.mark.parametrize(
    "value, description",
    (
        (object(), "plan"),
        (None, "backend"),
    ),
)
def test_shadow_comparison_rejects_untyped_inputs(value, description):
    plan = assemble_spectral_plan(_plane_problem())

    if description == "plan":
        with pytest.raises(TypeError, match="plan"):
            compare_spectral_plan_to_runtime(value, _backend_for_plan(plan))
    else:
        with pytest.raises(TypeError, match="backend"):
            compare_spectral_plan_to_runtime(plan, value)
