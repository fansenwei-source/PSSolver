"""Explicit opt-in APIs under architectural migration.

Nothing in this package is imported by the production solver entry points.
"""

from ._shadow_support import (
    SHADOW_CHECKPOINT_FORMAT_VERSION,
    SHADOW_RUN_SCHEMA_VERSION,
    SHADOW_SCRIPT_ID,
    shadow_runtime_identity_sha256,
)
from .beris_edwards import (
    PlaneBerisEdwardsSolverOptions,
    create_beris_edwards_plane_geometry_solver_registry,
)
from .canary_algebraic import create_canary_geometry_solver_registry
from .checkpointing import (
    ShadowRunCheckpoint,
    capture_shadow_checkpoint,
    load_shadow_checkpoint,
    restore_shadow_checkpoint,
    write_shadow_checkpoint,
)
from .legacy_assembly import (
    LegacyAssemblySpec,
    LegacyBackendSpec,
    LegacyFieldDeclaration,
    LegacyProjectorSpec,
    create_legacy_projector,
    create_legacy_solver,
    declare_legacy_fields,
    materialize_legacy_assembly,
)
from .integrators import ProjectedSemiImplicitEulerIntegrator
from .model_execution import (
    ExperimentalModelRuntime,
    LegacyAlgebraicFieldsAdapter,
    LegacyAlgebraicSolverContext,
    LegacyExplicitRHSAdapter,
    LegacyModelExecutionContext,
    ResolvedAlgebraicSystem,
    build_experimental_model_runtime,
)
from .plane_shadow_driver import (
    ProductionPlaneReference,
    build_plane_shadow_runtime_from_production_metadata,
    load_production_plane_reference,
    run_plane_shadow_from_production_reference,
)
from .shadow_comparison import (
    PlaneShadowTrajectoryComparison,
    ShadowArrayComparison,
    compare_plane_shadow_trajectories,
    write_plane_shadow_comparison,
)
from .shadow_metadata import (
    ShadowMetadataComparison,
    compare_saved_shadow_to_production_metadata,
    compare_shadow_to_production_metadata,
    plane_beris_edwards_production_signature,
    plane_beris_edwards_saved_shadow_signature,
    plane_beris_edwards_shadow_signature,
)
from .shadow_run import (
    ExperimentalPlaneShadowRun,
    ShadowObservation,
    capture_shadow_observation,
    write_shadow_observation,
)
from .h100_shadow_qualification import (
    analyze_stage_m_h100_qualification,
    profile_h100_plane_shadow_from_production_reference,
    run_h100_plane_shadow_from_production_reference,
)
from .stage_m_plan import build_stage_m_h100_plan
from .stokes import (
    ChannelStokesSolverOptions,
    PlaneStokesSolverOptions,
    create_stokes_geometry_solver_registry,
)

__all__ = [
    "ExperimentalModelRuntime",
    "ExperimentalPlaneShadowRun",
    "ChannelStokesSolverOptions",
    "LegacyAlgebraicFieldsAdapter",
    "LegacyAlgebraicSolverContext",
    "LegacyAssemblySpec",
    "LegacyBackendSpec",
    "LegacyExplicitRHSAdapter",
    "LegacyFieldDeclaration",
    "LegacyModelExecutionContext",
    "LegacyProjectorSpec",
    "PlaneStokesSolverOptions",
    "PlaneBerisEdwardsSolverOptions",
    "PlaneShadowTrajectoryComparison",
    "ProductionPlaneReference",
    "ProjectedSemiImplicitEulerIntegrator",
    "ResolvedAlgebraicSystem",
    "SHADOW_CHECKPOINT_FORMAT_VERSION",
    "SHADOW_RUN_SCHEMA_VERSION",
    "SHADOW_SCRIPT_ID",
    "ShadowMetadataComparison",
    "ShadowObservation",
    "ShadowRunCheckpoint",
    "ShadowArrayComparison",
    "build_experimental_model_runtime",
    "build_stage_m_h100_plan",
    "build_plane_shadow_runtime_from_production_metadata",
    "capture_shadow_checkpoint",
    "capture_shadow_observation",
    "compare_plane_shadow_trajectories",
    "compare_saved_shadow_to_production_metadata",
    "compare_shadow_to_production_metadata",
    "create_canary_geometry_solver_registry",
    "create_beris_edwards_plane_geometry_solver_registry",
    "create_legacy_projector",
    "create_legacy_solver",
    "create_stokes_geometry_solver_registry",
    "declare_legacy_fields",
    "materialize_legacy_assembly",
    "load_shadow_checkpoint",
    "load_production_plane_reference",
    "plane_beris_edwards_production_signature",
    "plane_beris_edwards_saved_shadow_signature",
    "plane_beris_edwards_shadow_signature",
    "restore_shadow_checkpoint",
    "run_plane_shadow_from_production_reference",
    "shadow_runtime_identity_sha256",
    "write_shadow_checkpoint",
    "write_shadow_observation",
    "analyze_stage_m_h100_qualification",
    "profile_h100_plane_shadow_from_production_reference",
    "run_h100_plane_shadow_from_production_reference",
    "write_plane_shadow_comparison",
]
