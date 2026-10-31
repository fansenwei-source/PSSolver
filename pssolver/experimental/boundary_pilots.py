"""Explicit opt-in namespace for Robin, finite-Q, and lifting pilots.

These objects are not qualified through the public :class:`Simulation` API.
Lifting execution remains closed until B8/S2-07 is resolved and independently
qualified.  The original module paths remain available during the v0.4
deprecation window and as read-only compatibility owners for released
checkpoints.
"""

from __future__ import annotations

from pssolver.boundaries import (
    RobinFaceLaw,
    StaticRobinBoundaryPolicy,
    robin,
)
from pssolver.configuration.finite_q_anchoring import (
    PlaneFiniteQAnchoringLoweringPlan,
    apply_plane_finite_q_anchoring_pilot,
    lower_plane_finite_q_anchoring_pilot,
)
from pssolver.configuration.lifting import build_plane_static_lifting_plan
from pssolver.configuration.robin import (
    RobinImpedanceSupportError,
    lower_plane_robin_component_pilot,
    lower_plane_robin_scalar_pilot,
    validate_public_robin_impedance_support,
)
from pssolver.models.active_nematics.boundaries import (
    FiniteQAnchoringFace,
    QuadraticFiniteQAnchoring,
    finite_homeotropic_q_anchoring,
    finite_planar_q_anchoring,
    finite_q_variational_residual,
    quadratic_finite_q_anchoring,
)
from pssolver.operators.lifting import (
    FRESH_INITIAL_CONDITIONING_SCHEMA_VERSION,
    FreshInitialConditioningPolicy,
    LEGACY_WHOLE_SLAB_SINE,
    LOCALIZED_QUINTIC_WALL_TRANSITION,
    MaterializedLinearLiftCorrection,
    PlaneStaticLiftingOperator,
    STRICT_NO_AUTOMATIC_CONDITIONING,
    legacy_whole_slab_sine_conditioning,
    localized_quintic_wall_transition,
    materialize_plane_static_lifting,
    strict_no_automatic_conditioning,
)
from pssolver.operators.robin import (
    CellCenteredRobinEigenbasisOperator,
    materialize_cell_centered_robin_eigenbasis,
)
from pssolver.planning.lifting import (
    STATIC_LIFTING_PLAN_SCHEMA_VERSION,
    StaticLiftExtension,
    StaticLiftingComponentPlan,
    StaticLiftingPlan,
)
from pssolver.planning.robin import (
    MAX_QUALIFIED_DIMENSIONLESS_ROBIN_IMPEDANCE,
    PLANE_ROBIN_SCALAR_LOWERING_SCHEMA_VERSION,
    ROBIN_EIGENBASIS_PLAN_SCHEMA_VERSION,
    CellCenteredRobinEigenbasisPlan,
    PlaneRobinScalarLoweringPlan,
    build_cell_centered_robin_eigenbasis_plan,
    validate_dimensionless_robin_impedance,
)
from pssolver.runtime.finite_q_anchoring import (
    FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION,
    FINITE_Q_ANCHORING_LEGACY_CHECKPOINT_FORMAT_VERSION,
    FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION,
    PlaneFiniteQAnchoringCheckpoint,
    PlaneFiniteQAnchoringRuntime,
)
from pssolver.runtime.robin_scalar import (
    PlaneRobinOperatorCache,
    PlaneRobinOperatorCacheKey,
    PlaneRobinScalarCheckpoint,
    PlaneRobinScalarRuntime,
)
from pssolver.runtime.static_lifting import (
    FRESH_CONSTRUCTION_PROVENANCE_SCHEMA_VERSION,
    PlaneStaticLiftingRuntime,
)
from pssolver.workflows.finite_q_anchoring import (
    FINITE_Q_ANCHORING_WORKFLOW_SCHEMA_VERSION,
    PlaneFiniteQAnchoringWorkflow,
    PlaneFiniteQAnchoringWorkflowResult,
    load_finite_q_anchoring_checkpoint,
    upgrade_finite_q_anchoring_checkpoint_directory,
    write_finite_q_anchoring_checkpoint,
)


def boundary_pilot_status() -> dict[str, object]:
    """Return the frozen v0.4 qualification and compatibility boundary."""

    return {
        "namespace": "pssolver.experimental.boundary_pilots",
        "families": ["robin", "finite_q", "lifting"],
        "public_simulation_execution_qualified": False,
        "lifting_execution_open": False,
        "lifting_blocked_by": ["B8", "S2-07"],
        "released_checkpoint_readers_preserved": True,
        "legacy_imports_available_through": "all v0.4 releases",
    }


__all__ = [
    "FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION",
    "FINITE_Q_ANCHORING_LEGACY_CHECKPOINT_FORMAT_VERSION",
    "FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION",
    "FINITE_Q_ANCHORING_WORKFLOW_SCHEMA_VERSION",
    "FRESH_CONSTRUCTION_PROVENANCE_SCHEMA_VERSION",
    "FRESH_INITIAL_CONDITIONING_SCHEMA_VERSION",
    "LEGACY_WHOLE_SLAB_SINE",
    "LOCALIZED_QUINTIC_WALL_TRANSITION",
    "MAX_QUALIFIED_DIMENSIONLESS_ROBIN_IMPEDANCE",
    "PLANE_ROBIN_SCALAR_LOWERING_SCHEMA_VERSION",
    "ROBIN_EIGENBASIS_PLAN_SCHEMA_VERSION",
    "STATIC_LIFTING_PLAN_SCHEMA_VERSION",
    "STRICT_NO_AUTOMATIC_CONDITIONING",
    "CellCenteredRobinEigenbasisOperator",
    "CellCenteredRobinEigenbasisPlan",
    "FiniteQAnchoringFace",
    "FreshInitialConditioningPolicy",
    "MaterializedLinearLiftCorrection",
    "PlaneFiniteQAnchoringCheckpoint",
    "PlaneFiniteQAnchoringLoweringPlan",
    "PlaneFiniteQAnchoringRuntime",
    "PlaneFiniteQAnchoringWorkflow",
    "PlaneFiniteQAnchoringWorkflowResult",
    "PlaneRobinOperatorCache",
    "PlaneRobinOperatorCacheKey",
    "PlaneRobinScalarCheckpoint",
    "PlaneRobinScalarLoweringPlan",
    "PlaneRobinScalarRuntime",
    "PlaneStaticLiftingOperator",
    "PlaneStaticLiftingRuntime",
    "QuadraticFiniteQAnchoring",
    "RobinFaceLaw",
    "RobinImpedanceSupportError",
    "StaticLiftExtension",
    "StaticLiftingComponentPlan",
    "StaticLiftingPlan",
    "StaticRobinBoundaryPolicy",
    "apply_plane_finite_q_anchoring_pilot",
    "boundary_pilot_status",
    "build_cell_centered_robin_eigenbasis_plan",
    "build_plane_static_lifting_plan",
    "finite_homeotropic_q_anchoring",
    "finite_planar_q_anchoring",
    "finite_q_variational_residual",
    "legacy_whole_slab_sine_conditioning",
    "load_finite_q_anchoring_checkpoint",
    "localized_quintic_wall_transition",
    "lower_plane_finite_q_anchoring_pilot",
    "lower_plane_robin_component_pilot",
    "lower_plane_robin_scalar_pilot",
    "materialize_cell_centered_robin_eigenbasis",
    "materialize_plane_static_lifting",
    "quadratic_finite_q_anchoring",
    "robin",
    "strict_no_automatic_conditioning",
    "upgrade_finite_q_anchoring_checkpoint_directory",
    "validate_dimensionless_robin_impedance",
    "validate_public_robin_impedance_support",
    "write_finite_q_anchoring_checkpoint",
]
