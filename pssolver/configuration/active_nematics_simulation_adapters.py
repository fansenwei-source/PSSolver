"""P7.7.2 compatibility composition for qualified active-nematic runs.

The adapters translate existing ownership-separated Plane and Channel
component graphs into a common tensor-free ``SimulationSpec``.  They do not
lower boundaries to bases, choose geometry solvers, allocate tensors, import
Torch, or connect either declaration to a runtime.
"""

from __future__ import annotations

from pssolver.core.boundary import (
    BoundaryAssignment,
    BoundarySemantic,
    BoundarySet,
    BoundarySide,
    ComponentBoundaryAssignment,
    FaceBoundaryCondition,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    PeriodicBC,
)
from pssolver.core.integrators import IntegratorSpec

from .active_nematics_equation_adapters import (
    declare_channel_active_force_equation_system,
    declare_plane_complete_stress_equation_system,
)
from .channel_active_nematics_declarations import ChannelRunComponents
from .plane_beris_edwards_component_graph import (
    PlaneBerisEdwardsRunComponents,
)
from .simulation import (
    ExecutionSpec,
    InitialConditionSource,
    InitialConditionSpec,
    InvocationSpec,
    SimulationSpec,
    TimeIntegrationSpec,
    WorkflowSpec,
)


def _component_faces(
    component: str,
    boundaries: BoundarySet,
    *,
    semantic: BoundarySemantic,
) -> ComponentBoundaryAssignment:
    faces = tuple(
        FaceBoundaryCondition(axis=axis, side=side, condition=condition)
        for axis, condition in enumerate(boundaries.axes)
        for side in BoundarySide
    )
    return ComponentBoundaryAssignment(
        component=component,
        semantic=semantic,
        faces=faces,
    )


def _legacy_boundary_set(values: tuple[str, ...]) -> BoundarySet:
    constructors = {
        "periodic": PeriodicBC,
        "dirichlet": HomogeneousDirichletBC,
        "neumann": HomogeneousNeumannBC,
    }
    try:
        conditions = tuple(constructors[value]() for value in values)
    except KeyError as exc:
        raise ValueError(f"unsupported legacy boundary kind {exc.args[0]!r}") from exc
    return BoundarySet(conditions)


def _field_components(equation_system, name: str) -> tuple[str, ...]:
    for field_spec in equation_system.fields:
        if field_spec.name == name:
            return field_spec.components
    raise ValueError(f"equation system has no logical field {name!r}")


def _without(mapping: dict[str, object], *names: str) -> dict[str, object]:
    result = dict(mapping)
    for name in names:
        result.pop(name)
    return result


def compose_plane_beris_edwards_simulation(
    components: PlaneBerisEdwardsRunComponents,
) -> SimulationSpec:
    """Compose the qualified complete-stress Plane declaration."""

    if not isinstance(components, PlaneBerisEdwardsRunComponents):
        raise TypeError(
            "components must be a PlaneBerisEdwardsRunComponents"
        )
    equation_system = declare_plane_complete_stress_equation_system(
        components
    )
    effective = components.effective_boundaries
    assignments = []
    for component in _field_components(equation_system, "Q"):
        assignments.append(
            _component_faces(
                component,
                effective.q,
                semantic=BoundarySemantic.PHYSICAL,
            )
        )
    for component in ("ux", "uy"):
        assignments.append(
            _component_faces(
                component,
                effective.tangential_velocity,
                semantic=BoundarySemantic.PHYSICAL,
            )
        )
    assignments.append(
        _component_faces(
            "uz",
            effective.normal_velocity,
            semantic=BoundarySemantic.PHYSICAL,
        )
    )
    assignments.append(
        _component_faces(
            "p",
            effective.pressure_modal,
            semantic=BoundarySemantic.ALGEBRAIC_COMPATIBILITY,
        )
    )

    execution_metadata = components.execution.to_metadata()
    workflow_metadata = components.workflow.to_metadata()
    return SimulationSpec(
        equation_system=equation_system,
        geometry=components.geometry,
        boundaries=BoundaryAssignment(
            name="qualified_plane_free_slip",
            ndim=components.geometry.domain.ndim,
            components=tuple(assignments),
        ),
        numerics=components.numerics,
        time_integration=TimeIntegrationSpec(
            integrator=IntegratorSpec.projected_semi_implicit_euler(
                dt=components.time_stepping.dt
            ),
            refresh=components.time_stepping.spectral_refresh.to_metadata(),
        ),
        discretization_parameters={
            "legacy_derived_boundary_spaces": {
                "distortion_odd_z": (
                    effective.distortion_odd_z.to_metadata()
                ),
            },
        },
        initial_condition=InitialConditionSpec(
            family="extruded_defect_gas",
            source=InitialConditionSource.GENERATED,
            parameters=components.initial_condition.to_metadata(),
        ),
        execution=ExecutionSpec(
            backend="torch_spectral",
            runtime_path=components.execution.runtime_path.value,
            options=_without(execution_metadata, "runtime_path"),
        ),
        workflow=WorkflowSpec(
            steps=components.workflow.steps,
            options=_without(workflow_metadata, "steps"),
        ),
        invocation=InvocationSpec(components.invocation.to_metadata()),
        compatibility_metadata={
            "source_adapter": "PlaneBerisEdwardsRunComponents",
            "source_components": components.to_metadata(),
        },
    )


def compose_channel_active_nematics_simulation(
    components: ChannelRunComponents,
) -> SimulationSpec:
    """Compose the qualified legacy active-force Channel declaration."""

    if not isinstance(components, ChannelRunComponents):
        raise TypeError("components must be a ChannelRunComponents")
    equation_system = declare_channel_active_force_equation_system(components)
    q_boundaries = _legacy_boundary_set(components.boundaries.q)
    velocity_boundaries = _legacy_boundary_set(
        components.boundaries.velocity
    )
    pressure_boundaries = _legacy_boundary_set(
        components.boundaries.pressure
    )
    assignments = [
        _component_faces(
            component,
            q_boundaries,
            semantic=BoundarySemantic.PHYSICAL,
        )
        for component in _field_components(equation_system, "Q")
    ]
    assignments.extend(
        _component_faces(
            component,
            velocity_boundaries,
            semantic=BoundarySemantic.PHYSICAL,
        )
        for component in _field_components(equation_system, "velocity")
    )
    assignments.append(
        _component_faces(
            "p",
            pressure_boundaries,
            semantic=BoundarySemantic.ALGEBRAIC_COMPATIBILITY,
        )
    )

    initial_metadata = components.initial_condition.to_metadata()
    if components.initial_condition.mode == "generated":
        initial_family = "aligned_x_smooth_noise"
        initial_source = InitialConditionSource.GENERATED
    else:
        initial_family = "external_snapshot"
        initial_source = InitialConditionSource.SNAPSHOT
    execution_metadata = components.execution.to_metadata()
    workflow_metadata = components.workflow.to_metadata()
    return SimulationSpec(
        equation_system=equation_system,
        geometry=components.geometry,
        boundaries=BoundaryAssignment(
            name="qualified_channel_no_slip",
            ndim=components.geometry.domain.ndim,
            components=tuple(assignments),
        ),
        numerics=components.numerics,
        time_integration=TimeIntegrationSpec(
            integrator=IntegratorSpec.projected_semi_implicit_euler(
                dt=components.dt
            ),
        ),
        discretization_parameters={
            "pressure_solver": components.pressure_solver.to_metadata(),
        },
        initial_condition=InitialConditionSpec(
            family=initial_family,
            source=initial_source,
            parameters=initial_metadata,
        ),
        execution=ExecutionSpec(
            backend="torch_spectral",
            runtime_path=components.execution.runtime_path.value,
            options=_without(execution_metadata, "runtime_path"),
        ),
        workflow=WorkflowSpec(
            steps=components.workflow.steps,
            options=_without(workflow_metadata, "steps"),
        ),
        invocation=InvocationSpec(),
        compatibility_metadata={
            "source_adapter": "ChannelRunComponents",
            "source_components": components.to_metadata(),
        },
    )


__all__ = [
    "compose_channel_active_nematics_simulation",
    "compose_plane_beris_edwards_simulation",
]
