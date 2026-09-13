"""One-way adapters from new contracts to qualified legacy interfaces."""

from .legacy_boundaries import (
    boundary_condition_to_legacy,
    boundary_set_to_legacy,
)

__all__ = [
    "boundary_condition_to_legacy",
    "boundary_set_to_legacy",
]
