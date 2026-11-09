"""Geometry-specific runtime selection boundaries.

Runtime adapters are intentionally not re-exported from :mod:`pssolver`.
Production drivers must select a geometry-specific factory explicitly.
"""

from importlib import import_module as _import_module

from pssolver.core._compatibility import (
    facade_dir as _facade_dir,
    resolve_deprecated_facade_name as _resolve_deprecated_facade_name,
)

from .plane_beris_edwards import LegacyPlaneRuntimeAdapter, PlaneRuntimeAdapterProtocol, PlaneRuntimeBuildRequest, build_plane_beris_edwards_runtime, plane_lifting_fresh_construction_provenance, plane_lifting_restart_metadata, plane_physical_component, verify_plane_lifting_identity
from .plane_legacy import (
    DealiasedSemiImplicitEulerIntegrator,
    build_legacy_plane_runtime,
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


_PILOT_EXPORTS = {
    name: ("pssolver.experimental.boundary_pilots", name)
    for name in (
        "FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION",
        "FINITE_Q_ANCHORING_LEGACY_CHECKPOINT_FORMAT_VERSION",
        "FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION",
        "PlaneFiniteQAnchoringCheckpoint",
        "PlaneFiniteQAnchoringRuntime",
        "PlaneRobinOperatorCache",
        "PlaneRobinOperatorCacheKey",
        "PlaneRobinScalarCheckpoint",
        "PlaneRobinScalarRuntime",
        "PlaneStaticLiftingRuntime",
    )
}

_IMPLICIT_SUBMODULE_NAMES = frozenset({"finite_q_anchoring", "robin_scalar"})


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
    "LegacyPlaneRuntimeAdapter",
    "PlaneRuntimeAdapterProtocol",
    "PlaneRuntimeBuildRequest",
    "DealiasedSemiImplicitEulerIntegrator",
    "build_legacy_plane_runtime",
    "build_plane_beris_edwards_runtime",
    "plane_lifting_fresh_construction_provenance",
    "plane_lifting_restart_metadata",
    "plane_physical_component",
    "verify_plane_lifting_identity",
    "PlaneStaticLiftingRuntime",
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
    "FINITE_Q_ANCHORING_LEGACY_CHECKPOINT_FORMAT_VERSION",
    "FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION",
    "PlaneFiniteQAnchoringCheckpoint",
    "PlaneFiniteQAnchoringRuntime",
]
