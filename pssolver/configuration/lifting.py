"""Fail-closed lowering from composed Plane declarations to lifting plans."""

from __future__ import annotations

import hashlib
import json

from pssolver.core.boundary import (
    BoundaryAssignment,
    BoundarySemantic,
    BoundarySide,
    PeriodicBC,
    PrescribedDirichletBC,
)
from pssolver.core.domain import GridPlacement
from pssolver.core.fields import FieldRole
from pssolver.core.geometry import AxisTopology, GeometrySpec
from pssolver.planning.lifting import (
    StaticLiftingComponentPlan,
    StaticLiftingPlan,
)
from pssolver.systems.equations import EquationSystemSpec


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_plane_static_lifting_plan(
    *,
    equation_system: EquationSystemSpec,
    geometry: GeometrySpec,
    boundaries: BoundaryAssignment,
) -> StaticLiftingPlan:
    """Lower prescribed Plane faces to an immutable affine-lifting plan."""

    if not isinstance(equation_system, EquationSystemSpec):
        raise TypeError("equation_system must be an EquationSystemSpec")
    if not isinstance(geometry, GeometrySpec):
        raise TypeError("geometry must be a GeometrySpec")
    if not isinstance(boundaries, BoundaryAssignment):
        raise TypeError("boundaries must be a BoundaryAssignment")
    if geometry.name != "plane_slab" or len(geometry.bounded_axes) != 1:
        raise ValueError(
            "P8.4.2 static lifting supports one-bounded-axis plane_slab only"
        )
    if boundaries.ndim != geometry.domain.ndim:
        raise ValueError("lifting boundary and geometry dimensions differ")
    if geometry.domain.grid_placement is not GridPlacement.CELL_CENTERED:
        raise ValueError("P8.4.2 supports cell-centered lifting only")

    component_owners = {
        component: (field.name, field.role)
        for field in equation_system.fields
        for component in field.components
    }
    wall_axis = geometry.bounded_axes[0]
    plans = []
    for assignment in boundaries.components:
        if assignment.component not in component_owners:
            raise ValueError(
                f"lifting component '{assignment.component}' is not registered"
            )
        prescribed = tuple(
            face
            for face in assignment.faces
            if not face.condition.is_homogeneous
        )
        if not prescribed:
            continue
        if any(
            not isinstance(face.condition, PrescribedDirichletBC)
            for face in prescribed
        ):
            raise TypeError("lifting requires typed prescribed Dirichlet data")
        field_name, role = component_owners[assignment.component]
        if role is not FieldRole.EVOLVED:
            raise ValueError("lifting is supported only for evolved fields")
        if assignment.semantic is not BoundarySemantic.PHYSICAL:
            raise ValueError("lifting requires a physical boundary semantic")

        for face in assignment.faces:
            topology = geometry.axis_topologies[face.axis]
            if topology is AxisTopology.PERIODIC:
                if not isinstance(face.condition, PeriodicBC):
                    raise ValueError(
                        "periodic lifting axes require periodic conditions"
                    )
            elif face.axis != wall_axis:
                raise ValueError("lifting face does not match the Plane wall axis")

        wall_faces = {
            face.side: face.condition
            for face in assignment.faces
            if face.axis == wall_axis
        }
        if set(wall_faces) != set(BoundarySide) or any(
            not isinstance(wall_faces[side], PrescribedDirichletBC)
            for side in BoundarySide
        ):
            raise ValueError(
                "both Plane wall faces require prescribed Dirichlet data"
            )
        lower = wall_faces[BoundarySide.LOWER]
        upper = wall_faces[BoundarySide.UPPER]
        assert isinstance(lower, PrescribedDirichletBC)
        assert isinstance(upper, PrescribedDirichletBC)
        plans.append(
            StaticLiftingComponentPlan(
                field_name=field_name,
                component=assignment.component,
                wall_normal_axis=wall_axis,
                lower_value=lower.value,
                upper_value=upper.value,
            )
        )

    if not plans:
        raise ValueError("no prescribed Dirichlet components require lifting")
    boundary_sha256 = _canonical_sha256(boundaries.to_metadata())
    domain = geometry.domain
    return StaticLiftingPlan(
        geometry_name=geometry.name,
        domain_shape=domain.shape,
        domain_lengths=domain.lengths,
        axis_names=domain.axis_names,
        grid_placement=domain.grid_placement,
        wall_normal_axis=wall_axis,
        components=tuple(plans),
        source_equation_sha256=equation_system.canonical_sha256(),
        source_boundary_sha256=boundary_sha256,
    )


__all__ = ["build_plane_static_lifting_plan"]
