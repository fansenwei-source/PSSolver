"""Concrete tensor-product geometry specifications."""

from .public import PlaneSlab
from .tensor_product import PeriodicBox, RectangularChannel

__all__ = [
    "PeriodicBox",
    "PlaneSlab",
    "RectangularChannel",
]
