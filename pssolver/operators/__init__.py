"""Reusable numerical operators owned independently of physical models."""

from importlib import import_module as _import_module

from pssolver.core._compatibility import (
    facade_dir as _facade_dir,
    resolve_deprecated_facade_name as _resolve_deprecated_facade_name,
)


_PILOT_EXPORTS = {
    name: ("pssolver.experimental.boundary_pilots", name)
    for name in (
        "CellCenteredRobinEigenbasisOperator",
        "FRESH_INITIAL_CONDITIONING_SCHEMA_VERSION",
        "FreshInitialConditioningPolicy",
        "LEGACY_WHOLE_SLAB_SINE",
        "LOCALIZED_QUINTIC_WALL_TRANSITION",
        "MaterializedLinearLiftCorrection",
        "PlaneStaticLiftingOperator",
        "STRICT_NO_AUTOMATIC_CONDITIONING",
        "legacy_whole_slab_sine_conditioning",
        "localized_quintic_wall_transition",
        "materialize_cell_centered_robin_eigenbasis",
        "materialize_plane_static_lifting",
        "strict_no_automatic_conditioning",
    )
}

_IMPLICIT_SUBMODULE_NAMES = frozenset({"lifting", "robin"})


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
    "FRESH_INITIAL_CONDITIONING_SCHEMA_VERSION",
    "FreshInitialConditioningPolicy",
    "LEGACY_WHOLE_SLAB_SINE",
    "LOCALIZED_QUINTIC_WALL_TRANSITION",
    "MaterializedLinearLiftCorrection",
    "PlaneStaticLiftingOperator",
    "STRICT_NO_AUTOMATIC_CONDITIONING",
    "legacy_whole_slab_sine_conditioning",
    "localized_quintic_wall_transition",
    "materialize_plane_static_lifting",
    "strict_no_automatic_conditioning",
    "CellCenteredRobinEigenbasisOperator",
    "materialize_cell_centered_robin_eigenbasis",
]
