"""P8.5.4 lowering for quadratic finite-Q anchoring on a Plane slab."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

from pssolver.configuration.robin import lower_plane_robin_component_pilot
from pssolver.configuration.simulation import SimulationSpec
from pssolver.core.boundary import (
    BoundaryAssignment,
    BoundarySemantic,
    BoundarySide,
    ComponentBoundaryAssignment,
    FaceBoundaryCondition,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    PeriodicBC,
    StaticRobinBC,
)
from pssolver.models.active_nematics.boundaries import (
    QUADRATIC_FINITE_Q_SURFACE_LAW_ID,
    QuadraticFiniteQAnchoring,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.planning.robin import PlaneRobinScalarLoweringPlan


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_free_slip_pressure_boundaries(simulation: SimulationSpec) -> None:
    wall_types = {
        "ux": HomogeneousNeumannBC,
        "uy": HomogeneousNeumannBC,
        "uz": HomogeneousDirichletBC,
        "p": HomogeneousNeumannBC,
    }
    for component, wall_type in wall_types.items():
        assignment = simulation.boundaries.for_component(component)
        expected_semantic = (
            BoundarySemantic.ALGEBRAIC_COMPATIBILITY
            if component == "p"
            else BoundarySemantic.PHYSICAL
        )
        if assignment.semantic is not expected_semantic:
            raise ValueError("P8.5.4 hydrodynamic boundary semantic mismatch")
        for face in assignment.faces:
            expected_type = PeriodicBC if face.axis in (0, 1) else wall_type
            if not isinstance(face.condition, expected_type):
                raise ValueError(
                    "P8.5.4 requires free-slip velocity and Neumann pressure"
                )


def _finite_q_boundary_assignment(
    simulation: SimulationSpec,
    anchoring: QuadraticFiniteQAnchoring,
) -> BoundaryAssignment:
    policy = anchoring.to_robin_policy()
    declared = {value.key: value.coefficients for value in policy.face_laws}
    q_assignments = []
    for component in Q_COMPONENTS:
        faces = []
        for axis in range(3):
            for side in BoundarySide:
                if axis in (0, 1):
                    condition = PeriodicBC()
                else:
                    coefficients = declared[(component, axis, side)]
                    condition = StaticRobinBC(
                        coefficients.alpha.value,
                        coefficients.beta.value,
                        coefficients.gamma.value,
                    )
                faces.append(FaceBoundaryCondition(axis, side, condition))
        q_assignments.append(
            ComponentBoundaryAssignment(
                component,
                BoundarySemantic.PHYSICAL,
                tuple(faces),
            )
        )
    hydrodynamic = tuple(
        simulation.boundaries.for_component(component)
        for component in ("ux", "uy", "uz", "p")
    )
    return BoundaryAssignment(
        "plane_quadratic_finite_q_anchoring_pilot",
        simulation.geometry.domain.ndim,
        tuple(q_assignments) + hydrodynamic,
    )


@dataclass(frozen=True, slots=True)
class PlaneFiniteQAnchoringLoweringPlan:
    """Five scalar Robin plans plus the model surface-law identity."""

    source_simulation_sha256: str
    anchoring_json: str
    anchoring_sha256: str
    component_plans: tuple[PlaneRobinScalarLoweringPlan, ...]

    def __post_init__(self) -> None:
        for name in ("source_simulation_sha256", "anchoring_sha256"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(value not in "0123456789abcdef" for value in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        try:
            anchoring = json.loads(self.anchoring_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("anchoring_json must be canonical JSON") from exc
        if self.anchoring_json != _canonical_json(anchoring):
            raise ValueError("anchoring_json must be canonical JSON")
        if _sha256(anchoring) != self.anchoring_sha256:
            raise ValueError("anchoring identity does not match metadata")
        plans = tuple(self.component_plans)
        if len(plans) != len(Q_COMPONENTS) or not all(
            isinstance(value, PlaneRobinScalarLoweringPlan) for value in plans
        ):
            raise ValueError("finite-Q lowering requires five scalar plans")
        if tuple(value.component for value in plans) != Q_COMPONENTS:
            raise ValueError("finite-Q scalar plans use the wrong component order")
        if len({value.source_simulation_sha256 for value in plans}) != 1:
            raise ValueError("finite-Q scalar plans disagree on source simulation")
        if plans[0].source_simulation_sha256 != self.source_simulation_sha256:
            raise ValueError("finite-Q source simulation identity does not match")
        object.__setattr__(self, "component_plans", plans)

    @property
    def anchoring_metadata(self) -> dict[str, object]:
        return json.loads(self.anchoring_json)

    def for_component(self, component: str) -> PlaneRobinScalarLoweringPlan:
        for value in self.component_plans:
            if value.component == component:
                return value
        raise KeyError(component)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan_kind": "plane_quadratic_finite_q_anchoring_pilot",
            "source_simulation_sha256": self.source_simulation_sha256,
            "surface_law_id": QUADRATIC_FINITE_Q_SURFACE_LAW_ID,
            "anchoring": self.anchoring_metadata,
            "anchoring_sha256": self.anchoring_sha256,
            "component_order": list(Q_COMPONENTS),
            "component_plans": [
                value.to_metadata() for value in self.component_plans
            ],
            "complete_q_timestep_connected": False,
            "public_runner_connected": False,
            "workflow_connected": False,
        }

    def canonical_sha256(self) -> str:
        return _sha256(self.to_metadata())


def _require_first_application(
    simulation: SimulationSpec,
    anchoring: QuadraticFiniteQAnchoring,
) -> None:
    if not isinstance(simulation, SimulationSpec):
        raise TypeError("simulation must be a SimulationSpec")
    if not isinstance(anchoring, QuadraticFiniteQAnchoring):
        raise TypeError("anchoring must be a QuadraticFiniteQAnchoring")
    if simulation.equation_system.variant != "complete_stress_beris_edwards":
        raise ValueError(
            "P8.5.4 requires complete_stress_beris_edwards equations"
        )
    if simulation.geometry.name != "plane_slab":
        raise ValueError("P8.5.4 requires plane_slab geometry")
    if simulation.execution.runtime_path != "legacy_production":
        raise ValueError("P8.5.4 first application requires legacy_production")
    _require_free_slip_pressure_boundaries(simulation)
    if tuple(value.key for value in anchoring.faces) != (
        (2, BoundarySide.LOWER),
        (2, BoundarySide.UPPER),
    ):
        expected = ((2, "lower"), (2, "upper"))
        observed = tuple(
            (value.axis, value.side.value) for value in anchoring.faces
        )
        raise ValueError(
            "P8.5.4 requires both Plane z faces exactly; "
            f"expected={expected!r}, observed={observed!r}"
        )

    candidates = []
    parameters = simulation.equation_system.parameters
    if "ldg_l1" in parameters:
        candidates.append(("equation_system.ldg_l1", parameters["ldg_l1"]))
    for group_name, terms in (
        ("evolution", simulation.equation_system.evolution_laws),
        ("constitutive", simulation.equation_system.constitutive_laws),
    ):
        for term in terms:
            if "ldg_l1" in term.parameters:
                candidates.append(
                    (f"{group_name}.{term.name}.ldg_l1", term.parameters["ldg_l1"])
                )
    if not candidates:
        raise ValueError("complete-stress model does not declare ldg_l1")
    mismatches = tuple(
        (name, value)
        for name, value in candidates
        if not math.isclose(
            float(value),
            anchoring.k_q,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    )
    if mismatches:
        raise ValueError(
            "finite-Q k_q must exactly equal every qualified ldg_l1; "
            f"mismatches={mismatches!r}"
        )


def apply_plane_finite_q_anchoring_pilot(
    simulation: SimulationSpec,
    anchoring: QuadraticFiniteQAnchoring,
) -> SimulationSpec:
    """Attach the first qualified Q Robin policy to an existing Plane spec."""

    _require_first_application(simulation, anchoring)
    boundaries = _finite_q_boundary_assignment(simulation, anchoring)
    compatibility = dict(simulation.compatibility_metadata)
    compatibility["finite_q_anchoring"] = anchoring.to_metadata()
    compatibility["finite_q_anchoring_sha256"] = anchoring.canonical_sha256()
    return replace(
        simulation,
        boundaries=boundaries,
        compatibility_metadata=compatibility,
    )


def lower_plane_finite_q_anchoring_pilot(
    simulation: SimulationSpec,
    anchoring: QuadraticFiniteQAnchoring,
) -> PlaneFiniteQAnchoringLoweringPlan:
    """Lower five Q components after exact first-application validation."""

    _require_first_application(simulation, anchoring)
    expected = _finite_q_boundary_assignment(simulation, anchoring)
    if simulation.boundaries.to_metadata() != expected.to_metadata():
        raise ValueError(
            "simulation boundaries do not match the finite-Q anchoring identity"
        )
    compatibility = dict(simulation.compatibility_metadata)
    if compatibility.get("finite_q_anchoring_sha256") != (
        anchoring.canonical_sha256()
    ):
        raise ValueError("simulation finite-Q anchoring provenance is missing")
    metadata = anchoring.to_metadata()
    return PlaneFiniteQAnchoringLoweringPlan(
        source_simulation_sha256=simulation.canonical_sha256(),
        anchoring_json=_canonical_json(metadata),
        anchoring_sha256=anchoring.canonical_sha256(),
        component_plans=tuple(
            lower_plane_robin_component_pilot(
                simulation,
                field_name="Q",
                component=component,
            )
            for component in Q_COMPONENTS
        ),
    )


__all__ = [
    "PlaneFiniteQAnchoringLoweringPlan",
    "apply_plane_finite_q_anchoring_pilot",
    "lower_plane_finite_q_anchoring_pilot",
]
