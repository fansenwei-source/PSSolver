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
)
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

__all__ = [
    "LegacyPlaneRuntimeAdapter",
    "PlaneRuntimeAdapterProtocol",
    "PlaneRuntimeBuildRequest",
    "SeparatedCanaryPlaneRuntimeAdapter",
    "DealiasedSemiImplicitEulerIntegrator",
    "build_legacy_plane_runtime",
    "build_plane_beris_edwards_runtime",
    "ChannelOutputViews",
    "ChannelRuntimeAdapterProtocol",
    "ChannelRuntimeBuildRequest",
    "LegacyChannelRuntimeAdapter",
    "build_channel_active_nematic_runtime",
]
