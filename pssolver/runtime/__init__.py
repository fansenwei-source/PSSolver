"""Geometry-specific runtime selection boundaries.

Runtime adapters are intentionally not re-exported from :mod:`pssolver`.
Production drivers must select a geometry-specific factory explicitly.
"""

from .plane_beris_edwards import (
    LegacyPlaneRuntimeAdapter,
    PlaneRuntimeAdapterProtocol,
    PlaneRuntimeBuildRequest,
    SeparatedCanaryPlaneRuntimeAdapter,
    build_plane_beris_edwards_runtime,
    plane_lifting_restart_metadata,
    plane_physical_component,
    verify_plane_lifting_identity,
)
from .static_lifting import PlaneStaticLiftingRuntime
from .plane_legacy import (
    DealiasedSemiImplicitEulerIntegrator,
    build_legacy_plane_runtime,
)
from .channel_active_nematics import (
    ChannelOutputViews,
    ChannelRuntimeAdapterProtocol,
    ChannelRuntimeBuildRequest,
    LegacyChannelRuntimeAdapter,
    build_channel_active_nematic_runtime,
)
from .periodic_beris_edwards import (
    PeriodicRuntimeAdapter,
    PeriodicRuntimeAdapterProtocol,
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)
from .channel_beris_edwards import (
    ChannelBerisEdwardsRuntimeAdapter,
    ChannelBerisEdwardsRuntimeBuildRequest,
    build_channel_beris_edwards_runtime,
)
from .robin_scalar import (
    PlaneRobinOperatorCache,
    PlaneRobinOperatorCacheKey,
    PlaneRobinScalarCheckpoint,
    PlaneRobinScalarRuntime,
)
from .finite_q_anchoring import (
    FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION,
    FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION,
    PlaneFiniteQAnchoringCheckpoint,
    PlaneFiniteQAnchoringRuntime,
)

__all__ = [
    "LegacyPlaneRuntimeAdapter",
    "PlaneRuntimeAdapterProtocol",
    "PlaneRuntimeBuildRequest",
    "SeparatedCanaryPlaneRuntimeAdapter",
    "DealiasedSemiImplicitEulerIntegrator",
    "build_legacy_plane_runtime",
    "build_plane_beris_edwards_runtime",
    "plane_lifting_restart_metadata",
    "plane_physical_component",
    "verify_plane_lifting_identity",
    "PlaneStaticLiftingRuntime",
    "ChannelOutputViews",
    "ChannelRuntimeAdapterProtocol",
    "ChannelRuntimeBuildRequest",
    "LegacyChannelRuntimeAdapter",
    "build_channel_active_nematic_runtime",
    "PeriodicRuntimeAdapter",
    "PeriodicRuntimeAdapterProtocol",
    "PeriodicRuntimeBuildRequest",
    "build_periodic_beris_edwards_runtime",
    "ChannelBerisEdwardsRuntimeAdapter",
    "ChannelBerisEdwardsRuntimeBuildRequest",
    "build_channel_beris_edwards_runtime",
    "PlaneRobinOperatorCache",
    "PlaneRobinOperatorCacheKey",
    "PlaneRobinScalarCheckpoint",
    "PlaneRobinScalarRuntime",
    "FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION",
    "FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION",
    "PlaneFiniteQAnchoringCheckpoint",
    "PlaneFiniteQAnchoringRuntime",
]
