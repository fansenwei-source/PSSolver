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
)
from pssolver.core.integrators import IntegratorSpec
from pssolver.functional.legacy_declarations.active_nematics_equation_adapters import (
    declare_plane_complete_stress_equation_system,
)
from pssolver.functional.legacy_declarations.plane_beris_edwards_component_graph import (
    PlaneBerisEdwardsRunComponents,
)
from pssolver.functional.legacy_declarations.simulation import (
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


__all__ = [
    "compose_plane_beris_edwards_simulation",
]
