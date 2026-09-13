"""Pure assembly from a validated ProblemSpec to a frozen SpectralPlan."""

from __future__ import annotations

from fractions import Fraction
import json

from pssolver.core.boundary import BoundaryKind
from pssolver.core.fields import FieldRole
from pssolver.core.numerics import (
    DealiasRule,
    ProjectedTransformExecution,
    SpectralStorage,
)
from pssolver.core.problem import ProblemSpec

from .plan import (
    ComponentTransformPlan,
    DomainAxisPlan,
    FieldPlan,
    SpectralPlan,
    TransformKind,
)


_TRANSFORM_KIND_BY_BOUNDARY = {
    BoundaryKind.PERIODIC: TransformKind.FFT,
    BoundaryKind.DIRICHLET: TransformKind.DST,
    BoundaryKind.NEUMANN: TransformKind.DCT,
}

_DEALIAS_FRACTIONS = {
    DealiasRule.NONE: None,
    DealiasRule.TWO_THIRDS: Fraction(2, 3),
    DealiasRule.CUBIC_HALF: Fraction(1, 2),
}


def _periodic_mode_numbers(size: int, hermitian_packed: bool) -> tuple[int, ...]:
    if hermitian_packed:
        return tuple(range(size // 2 + 1))
    midpoint = (size - 1) // 2
    return tuple(
        index if index <= midpoint else index - size
        for index in range(size)
    )


def _axis_mode_numbers(
    boundary_kind: BoundaryKind,
    size: int,
    hermitian_packed: bool,
) -> tuple[int, ...]:
    if boundary_kind is BoundaryKind.PERIODIC:
        return _periodic_mode_numbers(size, hermitian_packed)
    if boundary_kind is BoundaryKind.NEUMANN:
        return tuple(range(size))
    if boundary_kind is BoundaryKind.DIRICHLET:
        return tuple(range(1, size + 1))
    raise ValueError(f"unsupported boundary kind {boundary_kind!r}")


def _retained_mode_count(
    rule: DealiasRule,
    boundary_kind: BoundaryKind,
    size: int,
    hermitian_packed: bool,
) -> int:
    modes = _axis_mode_numbers(boundary_kind, size, hermitian_packed)
    fraction = _DEALIAS_FRACTIONS[rule]
    if fraction is None:
        return len(modes)
    nyquist = Fraction(size, 2 if boundary_kind is BoundaryKind.PERIODIC else 1)
    cutoff = fraction * nyquist
    return sum(Fraction(abs(mode), 1) < cutoff for mode in modes)


def _computed_axis_size(
    execution: ProjectedTransformExecution,
    boundary_kind: BoundaryKind,
    spectral_size: int,
    retained_count: int,
) -> int:
    if execution is ProjectedTransformExecution.FULL:
        return spectral_size
    if boundary_kind is BoundaryKind.PERIODIC:
        return spectral_size
    return retained_count


def assemble_spectral_plan(problem: ProblemSpec) -> SpectralPlan:
    """Resolve immutable layout and transform metadata without allocating tensors."""

    if not isinstance(problem, ProblemSpec):
        raise TypeError("problem must be a ProblemSpec")

    domain = problem.geometry.domain
    numerics = problem.numerics
    axes = []
    for index, (
        name,
        size,
        length,
        topology,
    ) in enumerate(
        zip(
            domain.axis_names,
            domain.shape,
            domain.lengths,
            problem.geometry.axis_topologies,
        )
    ):
        hermitian_packed = (
            numerics.spectral_storage is SpectralStorage.HERMITIAN_HALF
            and numerics.hermitian_axis == index
        )
        spectral_size = size // 2 + 1 if hermitian_packed else size
        axes.append(
            DomainAxisPlan(
                index=index,
                name=name,
                topology=topology,
                physical_size=size,
                spectral_size=spectral_size,
                length=length,
                hermitian_packed=hermitian_packed,
            )
        )
    axes = tuple(axes)

    storage_index_by_component = {}
    next_storage_index = 0
    for role in (FieldRole.EVOLVED, FieldRole.ALGEBRAIC):
        for field_spec in problem.field_specs:
            if field_spec.role is not role:
                continue
            for component in field_spec.components:
                storage_index_by_component[component.name] = next_storage_index
                next_storage_index += 1

    field_plans = []
    for field_spec in problem.field_specs:
        component_plans = []
        for component in field_spec.components:
            transform_kinds = tuple(
                _TRANSFORM_KIND_BY_BOUNDARY[condition.kind]
                for condition in component.boundaries.axes
            )
            retained_counts = tuple(
                _retained_mode_count(
                    numerics.dealias_rule,
                    condition.kind,
                    axis.physical_size,
                    axis.hermitian_packed,
                )
                for condition, axis in zip(component.boundaries.axes, axes)
            )
            computed_sizes = tuple(
                _computed_axis_size(
                    numerics.projected_transform_execution,
                    condition.kind,
                    axis.spectral_size,
                    retained_count,
                )
                for condition, axis, retained_count in zip(
                    component.boundaries.axes,
                    axes,
                    retained_counts,
                )
            )
            component_plans.append(
                ComponentTransformPlan(
                    field_name=field_spec.name,
                    component_name=component.name,
                    role=field_spec.role,
                    boundaries=component.boundaries,
                    transform_kinds=transform_kinds,
                    retained_mode_counts=retained_counts,
                    computed_axis_sizes=computed_sizes,
                    storage_index=storage_index_by_component.get(
                        component.name
                    ),
                )
            )
        field_plans.append(
            FieldPlan(
                name=field_spec.name,
                role=field_spec.role,
                components=tuple(component_plans),
            )
        )

    problem_metadata = problem.to_metadata()
    model_metadata = problem_metadata["model"]
    parameters = model_metadata["parameters"]
    parameters_json = json.dumps(
        parameters,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SpectralPlan(
        model_name=model_metadata["name"],
        geometry_name=problem.geometry.name,
        axes=axes,
        fields=tuple(field_plans),
        numerics=numerics,
        model_parameters_json=parameters_json,
    )
