"""One-way adapters from new contracts to qualified legacy interfaces."""

from .legacy_boundaries import (
    boundary_condition_to_legacy,
    boundary_set_to_legacy,
)
from .runtime_shadow import (
    ShadowComparison,
    ShadowMismatch,
    compare_spectral_plan_to_runtime,
)

__all__ = [
    "boundary_condition_to_legacy",
    "boundary_set_to_legacy",
    "compare_spectral_plan_to_runtime",
    "ShadowComparison",
    "ShadowMismatch",
]
