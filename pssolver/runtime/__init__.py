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

__all__ = [
    "LegacyPlaneRuntimeAdapter",
    "PlaneRuntimeAdapterProtocol",
    "PlaneRuntimeBuildRequest",
    "SeparatedCanaryPlaneRuntimeAdapter",
    "build_plane_beris_edwards_runtime",
]
