"""Fail-closed lowering for the P8.5.3 generic scalar Robin pilot."""

from __future__ import annotations

import hashlib
import json

from pssolver.configuration.simulation import SimulationSpec
from pssolver.core.boundary import (
    BoundaryKind,
    BoundarySemantic,
    BoundarySide,
    StaticRobinBC,
)
from pssolver.core.domain import GridPlacement
from pssolver.core.fields import FieldRole
from pssolver.core.geometry import AxisTopology
from pssolver.core.numerics import Precision
from pssolver.planning.robin import (
    PlaneRobinScalarLoweringPlan,
    build_cell_centered_robin_eigenbasis_plan,
)


def _sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def lower_plane_robin_component_pilot(
    simulation: SimulationSpec,
    *,
    field_name: str,
    component: str,
) -> PlaneRobinScalarLoweringPlan:
    """Resolve one component of a registered evolved field onto P8.5.2.

    This deliberately separate entry point cannot make an unsupported Robin
    law executable through the production simulation compiler.
    """

    if not isinstance(simulation, SimulationSpec):
        raise TypeError("simulation must be a SimulationSpec")
    if not isinstance(field_name, str) or not field_name.isidentifier():
        raise ValueError("field_name must be a Python identifier")
    if not isinstance(component, str) or not component.isidentifier():
        raise ValueError("component must be a Python identifier")
    geometry = simulation.geometry
    domain = geometry.domain
    if geometry.name != "plane_slab":
        raise ValueError("P8.5.3 Robin lowering requires plane_slab geometry")
    if geometry.axis_topologies != (
        AxisTopology.PERIODIC,
        AxisTopology.PERIODIC,
        AxisTopology.BOUNDED,
    ):
        raise ValueError("P8.5.3 requires periodic-periodic-bounded topology")
    if geometry.bounded_axes != (2,):
        raise ValueError("P8.5.3 requires wall-normal axis 2")
    if domain.grid_placement is not GridPlacement.CELL_CENTERED:
        raise ValueError("P8.5.3 requires a cell-centered grid")
    if simulation.numerics.precision is not Precision.FLOAT64:
        raise ValueError("P8.5.3 CPU pilot requires float64 numerics")
    if simulation.execution.backend != "torch_spectral":
        raise ValueError("P8.5.3 requires the torch_spectral backend intent")

    matches = tuple(
        value
        for value in simulation.equation_system.fields
        if value.name == field_name
    )
    if len(matches) != 1:
        raise ValueError("Robin pilot field must be registered exactly once")
    field = matches[0]
    if field.role is not FieldRole.EVOLVED:
        raise ValueError("Robin pilot field must have the evolved role")
    if component not in field.components:
        raise ValueError("Robin pilot component is not registered by the field")
    assignment = simulation.boundaries.for_component(component)
    if assignment.semantic is not BoundarySemantic.PHYSICAL:
        raise ValueError("Robin pilot field requires physical boundary semantics")

    face_lookup = {
        (face.axis, face.side): face.condition for face in assignment.faces
    }
    for axis in (0, 1):
        for side in BoundarySide:
            if face_lookup[(axis, side)].kind is not BoundaryKind.PERIODIC:
                raise ValueError("Robin pilot periodic axes require periodic laws")
    lower = face_lookup[(2, BoundarySide.LOWER)]
    upper = face_lookup[(2, BoundarySide.UPPER)]
    if not isinstance(lower, StaticRobinBC) or not isinstance(
        upper,
        StaticRobinBC,
    ):
        raise ValueError("both Plane wall faces require static Robin laws")
    robin_plan = build_cell_centered_robin_eigenbasis_plan(
        size=domain.shape[2],
        length=domain.lengths[2],
        lower=lower.coefficients,
        upper=upper.coefficients,
    )
    return PlaneRobinScalarLoweringPlan(
        source_simulation_sha256=simulation.canonical_sha256(),
        source_boundary_sha256=_sha256(assignment.to_metadata()),
        field_name=field.name,
        component=component,
        geometry_name=geometry.name,
        domain_shape=domain.shape,
        domain_lengths=domain.lengths,
        axis_names=domain.axis_names,
        periodic_axes=geometry.periodic_axes,
        wall_normal_axis=geometry.bounded_axes[0],
        robin_plan=robin_plan,
    )


def lower_plane_robin_scalar_pilot(
    simulation: SimulationSpec,
    *,
    field_name: str,
) -> PlaneRobinScalarLoweringPlan:
    """Resolve a registered one-component evolved field onto P8.5.2."""

    matches = tuple(
        value
        for value in simulation.equation_system.fields
        if value.name == field_name
    )
    if len(matches) != 1:
        raise ValueError("Robin pilot field must be registered exactly once")
    if len(matches[0].components) != 1:
        raise ValueError("P8.5.3 supports one scalar component only")
    return lower_plane_robin_component_pilot(
        simulation,
        field_name=field_name,
        component=matches[0].components[0],
    )


__all__ = [
    "lower_plane_robin_component_pilot",
    "lower_plane_robin_scalar_pilot",
]
