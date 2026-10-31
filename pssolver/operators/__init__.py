"""Reusable numerical operators owned independently of physical models."""

from pssolver.core._compatibility import (
    facade_dir,
    resolve_deprecated_facade_name,
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


def __getattr__(name: str) -> object:
    return resolve_deprecated_facade_name(
        name,
        facade=__name__,
        replacement="pssolver.experimental.boundary_pilots",
        exports=_PILOT_EXPORTS,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    return facade_dir(globals(), _PILOT_EXPORTS)

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
