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
    BerisEdwardsPlaneExplicitRHSExecutor,
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
from .performance import RuntimePerformanceRecorder
from .producer_packing import ProducerPackedComponentValues
from .projected_scheduler import (
    AlgebraicPhysicalIslandScheduler,
    BoundarySignatureTransformScheduler,
    CompiledProjectedTransformPlan,
    ProjectedBatchAssemblyMode,
    ProjectedBatchAssemblyPolicy,
    ProjectedTransformBatchKey,
    ProjectedTransformDirection,
    ScheduledProjectedTransforms,
)
from .representations import (
    AlgebraicGenerationState,
    AlgebraicPhysicalStateView,
    AlgebraicRepresentationCache,
    BatchedPhysicalMaterialization,
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
from .stage_n_diagnostics import (
    analyze_stage_n_diagnostics,
    profile_stage_n_h100_shadow,
)
from .stage_n_plan import build_stage_n_h100_plan
from .stage_n1_plan import build_stage_n1_h100_plan
from .stage_n1_qualification import (
    analyze_stage_n1_qualification,
    profile_stage_n1_h100_shadow,
    run_stage_n1_h100_candidate_trajectory,
)
from .stage_n2_plan import build_stage_n2_h100_plan
from .stage_n2_qualification import (
    analyze_stage_n2_qualification,
    profile_stage_n2_h100_shadow,
    run_stage_n2_h100_candidate_trajectory,
)
from .stage_n3_plan import build_stage_n3_h100_plan
from .stage_n3_qualification import (
    analyze_stage_n3_qualification,
    profile_stage_n3_h100_shadow,
    run_stage_n3_h100_candidate_trajectory,
)
from .stage_n4_plan import build_stage_n4_h100_plan
from .stage_n4_qualification import (
    analyze_stage_n4_qualification,
    profile_stage_n4_h100_shadow,
    run_stage_n4_h100_candidate_trajectory,
)
from .stage_o_migration import (
    MigrationDisposition,
    MigrationPhase,
    MigrationResponsibility,
    PlaneRuntimePath,
    STAGE_O_SCHEMA_VERSION,
    StageOMigrationDesign,
    StageOQualificationEvidence,
    build_stage_o_migration_design,
)
from .stage_o4_plan import build_stage_o4_h100_plan
from .stage_o4_qualification import (
    analyze_stage_o4_qualification,
    compare_plane_stage_o4_workflows,
)
from .stage_o41_plan import build_stage_o41_h100_plan
from .stage_o41_qualification import analyze_stage_o41_qualification
from .stage_o42_plan import build_stage_o42_h100_plan
from .stage_o43_plan import build_stage_o43_h100_plan
from .stage_o431_plan import build_stage_o431_h100_plan
from .stage_o433_plan import build_stage_o433_h100_plan
from .stage_o434_diagnostics import (
    analyze_stage_o434_diagnostics,
    profile_stage_o434_h100_runtime,
    summarize_projected_batch_source_attribution,
)
from .stage_o434_plan import build_stage_o434_h100_plan
from .stage_o_closure import (
    STAGE_O_CLOSURE_SCHEMA_VERSION,
    StageOClosureDecision,
    StageOClosureEvidence,
    StageOPathDisposition,
    build_stage_o_closure_decision,
    stage_o_closure_identity_sha256,
)
from .stage_p_diagnostics import (
    analyze_stage_p_diagnostics,
    summarize_operator_kernel_events,
)
from .stage_p_plan import build_stage_p_h100_plan
from .stage_q2_diagnostics import analyze_stage_q2_residual_gap
from .stage_o43_qualification import (
    analyze_stage_o431_qualification,
    analyze_stage_o433_qualification,
    analyze_stage_o43_qualification,
    profile_stage_o431_h100_runtime,
    profile_stage_o433_h100_runtime,
    profile_stage_o43_h100_runtime,
    run_stage_o431_h100_trajectory,
    run_stage_o433_h100_trajectory,
    run_stage_o43_h100_trajectory,
)
from .stokes import (
    ChannelStokesSolverOptions,
    PlaneStokesSolverOptions,
    create_stokes_geometry_solver_registry,
)

__all__ = [
    "BerisEdwardsPlaneExplicitRHSExecutor",
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
    "ProducerPackedComponentValues",
    "ProjectedSemiImplicitEulerIntegrator",
    "RuntimePerformanceRecorder",
    "AlgebraicPhysicalIslandScheduler",
    "BoundarySignatureTransformScheduler",
    "CompiledProjectedTransformPlan",
    "ProjectedBatchAssemblyMode",
    "ProjectedBatchAssemblyPolicy",
    "ProjectedTransformBatchKey",
    "ProjectedTransformDirection",
    "ScheduledProjectedTransforms",
    "AlgebraicGenerationState",
    "AlgebraicPhysicalStateView",
    "AlgebraicRepresentationCache",
    "BatchedPhysicalMaterialization",
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
    "build_stage_n_h100_plan",
    "build_stage_n1_h100_plan",
    "build_stage_n2_h100_plan",
    "build_stage_n3_h100_plan",
    "build_stage_n4_h100_plan",
    "build_stage_o_migration_design",
    "build_stage_o431_h100_plan",
    "build_stage_o433_h100_plan",
    "build_stage_o434_h100_plan",
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
    "analyze_stage_n_diagnostics",
    "analyze_stage_n1_qualification",
    "analyze_stage_n2_qualification",
    "analyze_stage_n3_qualification",
    "analyze_stage_n4_qualification",
    "analyze_stage_q2_residual_gap",
    "analyze_stage_o4_qualification",
    "analyze_stage_o41_qualification",
    "profile_h100_plane_shadow_from_production_reference",
    "profile_stage_n_h100_shadow",
    "profile_stage_n1_h100_shadow",
    "profile_stage_n2_h100_shadow",
    "profile_stage_n3_h100_shadow",
    "profile_stage_n4_h100_shadow",
    "run_h100_plane_shadow_from_production_reference",
    "run_stage_n1_h100_candidate_trajectory",
    "run_stage_n2_h100_candidate_trajectory",
    "run_stage_n3_h100_candidate_trajectory",
    "run_stage_n4_h100_candidate_trajectory",
    "build_stage_o4_h100_plan",
    "build_stage_o41_h100_plan",
    "build_stage_o42_h100_plan",
    "build_stage_o43_h100_plan",
    "compare_plane_stage_o4_workflows",
    "analyze_stage_o431_qualification",
    "analyze_stage_o433_qualification",
    "analyze_stage_o434_diagnostics",
    "analyze_stage_o43_qualification",
    "profile_stage_o431_h100_runtime",
    "profile_stage_o433_h100_runtime",
    "profile_stage_o434_h100_runtime",
    "summarize_projected_batch_source_attribution",
    "profile_stage_o43_h100_runtime",
    "run_stage_o431_h100_trajectory",
    "run_stage_o433_h100_trajectory",
    "run_stage_o43_h100_trajectory",
    "MigrationDisposition",
    "MigrationPhase",
    "MigrationResponsibility",
    "PlaneRuntimePath",
    "STAGE_O_SCHEMA_VERSION",
    "STAGE_O_CLOSURE_SCHEMA_VERSION",
    "StageOMigrationDesign",
    "StageOClosureDecision",
    "StageOClosureEvidence",
    "StageOQualificationEvidence",
    "StageOPathDisposition",
    "build_stage_o_closure_decision",
    "stage_o_closure_identity_sha256",
    "analyze_stage_p_diagnostics",
    "summarize_operator_kernel_events",
    "build_stage_p_h100_plan",
    "write_plane_shadow_comparison",
]
