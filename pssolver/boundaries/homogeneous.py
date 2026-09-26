"""Typed homogeneous boundary policies and tensor-free assignment helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pssolver.core.boundary import (
    BoundaryAssignment,
    BoundaryCondition,
    BoundarySemantic,
    BoundarySide,
    ComponentBoundaryAssignment,
    FaceBoundaryCondition,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    PeriodicBC,
    PrescribedDirichletBC,
)
from pssolver.core.fields import FieldRole
from pssolver.core.geometry import AxisTopology, GeometrySpec
from pssolver.systems.equations import EquationSystemSpec

from .prescribed import StaticPrescribedDirichletPolicy


@dataclass(frozen=True, slots=True)
class HomogeneousBoundaryPolicy:
    """A typed field-level policy expanded only during composition."""

    field_name: str
    semantic: BoundarySemantic
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name.isidentifier():
            raise ValueError("boundary-policy field_name must be a Python identifier")
        if not isinstance(self.semantic, BoundarySemantic):
            raise TypeError("boundary-policy semantic must be a BoundarySemantic")
        if self.kind not in {
            "neumann",
            "free_slip_velocity",
            "no_slip_velocity",
        }:
            raise ValueError("unsupported homogeneous boundary policy")


BoundaryPolicy = HomogeneousBoundaryPolicy | StaticPrescribedDirichletPolicy


def neumann_q() -> HomogeneousBoundaryPolicy:
    """Return homogeneous Neumann anchoring for the logical Q field."""

    return HomogeneousBoundaryPolicy(
        field_name="Q",
        semantic=BoundarySemantic.PHYSICAL,
        kind="neumann",
    )


def free_slip_velocity() -> HomogeneousBoundaryPolicy:
    """Return no-penetration/free-slip conditions for vector velocity."""

    return HomogeneousBoundaryPolicy(
        field_name="velocity",
        semantic=BoundarySemantic.PHYSICAL,
        kind="free_slip_velocity",
    )


def no_slip_velocity() -> HomogeneousBoundaryPolicy:
    """Return zero velocity for every component on bounded faces."""

    return HomogeneousBoundaryPolicy(
        field_name="velocity",
        semantic=BoundarySemantic.PHYSICAL,
        kind="no_slip_velocity",
    )


def neumann_pressure_compatibility() -> HomogeneousBoundaryPolicy:
    """Return Neumann modal compatibility for the pressure multiplier."""

    return HomogeneousBoundaryPolicy(
        field_name="pressure",
        semantic=BoundarySemantic.ALGEBRAIC_COMPATIBILITY,
        kind="neumann",
    )


def _axis_condition(
    policy: HomogeneousBoundaryPolicy,
    *,
    topology: AxisTopology,
    axis: int,
    component_index: int,
) -> BoundaryCondition:
    if topology is AxisTopology.PERIODIC:
        return PeriodicBC()
    if policy.kind == "neumann":
        return HomogeneousNeumannBC()
    if policy.kind == "free_slip_velocity":
        if component_index == axis:
            return HomogeneousDirichletBC()
        return HomogeneousNeumannBC()
    if policy.kind == "no_slip_velocity":
        return HomogeneousDirichletBC()
    raise AssertionError("validated boundary policy was not handled")


def _component_assignment(
    *,
    component: str,
    component_index: int,
    geometry: GeometrySpec,
    policy: HomogeneousBoundaryPolicy,
) -> ComponentBoundaryAssignment:
    faces = []
    for axis, topology in enumerate(geometry.axis_topologies):
        condition = _axis_condition(
            policy,
            topology=topology,
            axis=axis,
            component_index=component_index,
        )
        faces.extend(
            FaceBoundaryCondition(axis, side, condition)
            for side in BoundarySide
        )
    return ComponentBoundaryAssignment(
        component=component,
        semantic=policy.semantic,
        faces=tuple(faces),
    )


def _prescribed_component_assignments(
    *,
    components: tuple[str, ...],
    geometry: GeometrySpec,
    policy: StaticPrescribedDirichletPolicy,
) -> tuple[ComponentBoundaryAssignment, ...]:
    declared = {value.key: value.value for value in policy.face_values}
    expected = {
        (component, axis, side)
        for component in components
        for axis in geometry.bounded_axes
        for side in BoundarySide
    }
    observed = set(declared)
    if observed != expected:
        missing = tuple(
            (component, axis, side.value)
            for component, axis, side in sorted(
                expected - observed,
                key=lambda value: (value[0], value[1], value[2].value),
            )
        )
        extra = tuple(
            (component, axis, side.value)
            for component, axis, side in sorted(
                observed - expected,
                key=lambda value: (value[0], value[1], value[2].value),
            )
        )
        raise ValueError(
            "prescribed Dirichlet values must cover exactly every bounded "
            f"component-face; missing={missing!r}, extra={extra!r}"
        )

    assignments = []
    for component in components:
        faces = []
        for axis, topology in enumerate(geometry.axis_topologies):
            for side in BoundarySide:
                if topology is AxisTopology.PERIODIC:
                    condition = PeriodicBC()
                else:
                    condition = PrescribedDirichletBC(
                        declared[(component, axis, side)]
                    )
                faces.append(FaceBoundaryCondition(axis, side, condition))
        assignments.append(
            ComponentBoundaryAssignment(
                component=component,
                semantic=BoundarySemantic.PHYSICAL,
                faces=tuple(faces),
            )
        )
    return tuple(assignments)


def assign_boundaries(
    *,
    model: EquationSystemSpec,
    geometry: GeometrySpec,
    policies: Mapping[str, BoundaryPolicy],
    name: str = "public_boundary_assignment",
) -> BoundaryAssignment:
    """Expand typed logical-field policies into a canonical assignment.

    Every evolved or algebraic logical field must be present explicitly.
    Transient constitutive fields cannot receive physical boundaries.  This
    helper therefore does not infer a pressure condition from a velocity
    condition or silently supply a model-specific wall law.
    """

    if not isinstance(model, EquationSystemSpec):
        raise TypeError("model must be an EquationSystemSpec")
    if not isinstance(geometry, GeometrySpec):
        raise TypeError("geometry must be a GeometrySpec")
    if not isinstance(policies, Mapping):
        raise TypeError("policies must be a mapping")

    required_fields = tuple(
        field
        for field in model.fields
        if field.role in {FieldRole.EVOLVED, FieldRole.ALGEBRAIC}
    )
    required_names = {field.name for field in required_fields}
    observed_names = set(policies)
    if observed_names != required_names:
        missing = tuple(sorted(required_names - observed_names))
        extra = tuple(sorted(observed_names - required_names))
        raise ValueError(
            "boundary policies must cover exactly evolved and algebraic "
            f"logical fields; missing={missing!r}, extra={extra!r}"
        )

    assignments = []
    for field in required_fields:
        policy = policies[field.name]
        if not isinstance(
            policy,
            (HomogeneousBoundaryPolicy, StaticPrescribedDirichletPolicy),
        ):
            raise TypeError(
                f"boundary policy for '{field.name}' must be a "
                "HomogeneousBoundaryPolicy or "
                "StaticPrescribedDirichletPolicy"
            )
        if policy.field_name != field.name:
            raise ValueError(
                f"boundary policy for '{field.name}' declares "
                f"'{policy.field_name}'"
            )
        if isinstance(policy, StaticPrescribedDirichletPolicy):
            if field.role is not FieldRole.EVOLVED:
                raise ValueError(
                    "static prescribed Dirichlet data are supported only "
                    f"for evolved fields; '{field.name}' has role "
                    f"'{field.role.value}'"
                )
            assignments.extend(
                _prescribed_component_assignments(
                    components=field.components,
                    geometry=geometry,
                    policy=policy,
                )
            )
            continue
        if (
            field.role is FieldRole.EVOLVED
            and policy.semantic is not BoundarySemantic.PHYSICAL
        ):
            raise ValueError(
                f"evolved field '{field.name}' requires physical boundaries"
            )
        if policy.kind in {"free_slip_velocity", "no_slip_velocity"} and (
            len(field.components) != geometry.domain.ndim
        ):
            raise ValueError(
                "velocity boundary policy requires one ordered component "
                "per axis"
            )
        assignments.extend(
            _component_assignment(
                component=component,
                component_index=index,
                geometry=geometry,
                policy=policy,
            )
            for index, component in enumerate(field.components)
        )

    return BoundaryAssignment(
        name=name,
        ndim=geometry.domain.ndim,
        components=tuple(assignments),
    )


__all__ = [
    "BoundaryPolicy",
    "HomogeneousBoundaryPolicy",
    "assign_boundaries",
    "free_slip_velocity",
    "no_slip_velocity",
    "neumann_pressure_compatibility",
    "neumann_q",
]
