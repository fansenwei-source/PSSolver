"""Simulation workflows above geometry-specific runtime adapters."""

from importlib import import_module as _import_module

from pssolver.core._compatibility import (
    facade_dir as _facade_dir,
    resolve_deprecated_facade_name as _resolve_deprecated_facade_name,
)

from .plane_beris_edwards import (
    PLANE_WORKFLOW_SCHEMA_VERSION,
    PlaneBerisEdwardsWorkflow,
    PlaneWorkflowResult,
)
from .plane_checkpoint import (
    PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PLANE_UNSEALED_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PlaneCheckpointHeader,
    PlaneWorkflowCheckpoint,
    build_plane_checkpoint_compatibility_identity,
    capture_plane_checkpoint,
    load_plane_checkpoint,
    plane_forward_dynamics_identity,
    plane_run_spec_dynamics_identity,
    read_plane_checkpoint_header,
    restore_plane_checkpoint,
    upgrade_plane_checkpoint_directory_v1,
    upgrade_plane_checkpoint_directory_v2,
    upgrade_plane_checkpoint_v1,
    upgrade_plane_checkpoint_v2,
    write_plane_checkpoint,
)
from .plane_observation import (
    DIAGNOSTIC_DTYPE,
    DIAGNOSTIC_HEADER,
    PlaneDiagnostic,
    PlaneObservation,
    capture_plane_diagnostic,
    capture_plane_observation,
    write_plane_diagnostics,
    write_plane_observation,
)
from .periodic_checkpoint import (
    PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1,
    PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PeriodicCheckpointHeader,
    PeriodicWorkflowCheckpoint,
    build_periodic_checkpoint_compatibility_identity,
    capture_periodic_checkpoint,
    load_periodic_checkpoint,
    periodic_forward_dynamics_identity,
    read_periodic_checkpoint_header,
    restore_periodic_checkpoint,
    upgrade_periodic_checkpoint,
    upgrade_periodic_checkpoint_directory,
    write_periodic_checkpoint,
)
from .channel_beris_edwards import (
    CHECKPOINT_VERSION as CHANNEL_BERIS_EDWARDS_CHECKPOINT_VERSION,
    LEGACY_CHECKPOINT_VERSION as CHANNEL_BERIS_EDWARDS_LEGACY_CHECKPOINT_VERSION,
    LEGACY_CHECKPOINT_VERSION_V1 as CHANNEL_BERIS_EDWARDS_LEGACY_CHECKPOINT_VERSION_V1,
    ChannelBerisEdwardsCheckpointHeader,
    build_channel_checkpoint_compatibility_identity,
    channel_forward_dynamics_identity,
    read_channel_beris_edwards_checkpoint_header,
    upgrade_channel_beris_edwards_checkpoint_directory,
)


_PILOT_EXPORTS = {
    name: ("pssolver.experimental.boundary_pilots", name)
    for name in (
        "FINITE_Q_ANCHORING_WORKFLOW_SCHEMA_VERSION",
        "PlaneFiniteQAnchoringWorkflow",
        "PlaneFiniteQAnchoringWorkflowResult",
        "load_finite_q_anchoring_checkpoint",
        "upgrade_finite_q_anchoring_checkpoint_directory",
        "write_finite_q_anchoring_checkpoint",
    )
}

_IMPLICIT_SUBMODULE_NAMES = frozenset({"finite_q_anchoring", "channel_checkpoint_observation"})


def __getattr__(name: str) -> object:
    if name in _IMPLICIT_SUBMODULE_NAMES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    return _resolve_deprecated_facade_name(
        name,
        facade=__name__,
        replacement="pssolver.experimental.boundary_pilots",
        exports=_PILOT_EXPORTS,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    return sorted(set(_facade_dir(globals(), _PILOT_EXPORTS)) | _IMPLICIT_SUBMODULE_NAMES)


__all__ = [
    "DIAGNOSTIC_DTYPE",
    "DIAGNOSTIC_HEADER",
    "PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PLANE_UNSEALED_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PLANE_WORKFLOW_SCHEMA_VERSION",
    "PlaneBerisEdwardsWorkflow",
    "PlaneCheckpointHeader",
    "PlaneDiagnostic",
    "PlaneObservation",
    "PlaneWorkflowCheckpoint",
    "PlaneWorkflowResult",
    "build_plane_checkpoint_compatibility_identity",
    "capture_plane_checkpoint",
    "capture_plane_diagnostic",
    "capture_plane_observation",
    "load_plane_checkpoint",
    "plane_forward_dynamics_identity",
    "plane_run_spec_dynamics_identity",
    "read_plane_checkpoint_header",
    "restore_plane_checkpoint",
    "upgrade_plane_checkpoint_directory_v1",
    "upgrade_plane_checkpoint_directory_v2",
    "upgrade_plane_checkpoint_v1",
    "upgrade_plane_checkpoint_v2",
    "write_plane_checkpoint",
    "write_plane_diagnostics",
    "write_plane_observation",
    "PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1",
    "PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PeriodicCheckpointHeader",
    "PeriodicWorkflowCheckpoint",
    "build_periodic_checkpoint_compatibility_identity",
    "capture_periodic_checkpoint",
    "load_periodic_checkpoint",
    "periodic_forward_dynamics_identity",
    "read_periodic_checkpoint_header",
    "restore_periodic_checkpoint",
    "upgrade_periodic_checkpoint",
    "upgrade_periodic_checkpoint_directory",
    "write_periodic_checkpoint",
    "CHANNEL_BERIS_EDWARDS_CHECKPOINT_VERSION",
    "CHANNEL_BERIS_EDWARDS_LEGACY_CHECKPOINT_VERSION",
    "CHANNEL_BERIS_EDWARDS_LEGACY_CHECKPOINT_VERSION_V1",
    "ChannelBerisEdwardsCheckpointHeader",
    "build_channel_checkpoint_compatibility_identity",
    "channel_forward_dynamics_identity",
    "read_channel_beris_edwards_checkpoint_header",
    "upgrade_channel_beris_edwards_checkpoint_directory",
    "FINITE_Q_ANCHORING_WORKFLOW_SCHEMA_VERSION",
    "PlaneFiniteQAnchoringWorkflow",
    "PlaneFiniteQAnchoringWorkflowResult",
    "load_finite_q_anchoring_checkpoint",
    "upgrade_finite_q_anchoring_checkpoint_directory",
    "write_finite_q_anchoring_checkpoint",
]
