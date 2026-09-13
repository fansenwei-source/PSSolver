"""Tensor-product computational-domain contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class GridPlacement(str, Enum):
    """Location of degrees of freedom along each coordinate axis."""

    CELL_CENTERED = "cell_centered"
    NODE_CENTERED = "node_centered"


@dataclass(frozen=True, slots=True)
class DomainSpec:
    """Immutable shape and coordinate metadata for a tensor-product domain."""

    shape: tuple[int, ...]
    lengths: tuple[float, ...]
    axis_names: tuple[str, ...] | None = None
    grid_placement: GridPlacement = GridPlacement.CELL_CENTERED

    def __post_init__(self) -> None:
        try:
            shape = tuple(self.shape)
            lengths = tuple(float(length) for length in self.lengths)
        except TypeError as exc:
            raise TypeError("shape and lengths must be iterable") from exc

        if not 1 <= len(shape) <= 3:
            raise ValueError("DomainSpec supports one, two, or three dimensions")
        if len(lengths) != len(shape):
            raise ValueError("shape and lengths must have the same dimension")
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in shape
        ):
            raise ValueError("every shape entry must be a positive integer")
        if any(not math.isfinite(length) or length <= 0.0 for length in lengths):
            raise ValueError("every physical length must be positive and finite")
        if not isinstance(self.grid_placement, GridPlacement):
            raise TypeError("grid_placement must be a GridPlacement")

        if self.axis_names is None:
            axis_names = ("x", "y", "z")[: len(shape)]
        else:
            try:
                axis_names = tuple(self.axis_names)
            except TypeError as exc:
                raise TypeError("axis_names must be an iterable of strings") from exc
        if len(axis_names) != len(shape):
            raise ValueError("axis_names must match the domain dimension")
        if any(
            not isinstance(name, str) or not name or not name.isidentifier()
            for name in axis_names
        ):
            raise ValueError("axis names must be non-empty Python identifiers")
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("axis names must be unique")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "lengths", lengths)
        object.__setattr__(self, "axis_names", axis_names)

    @property
    def ndim(self) -> int:
        """Number of coordinate dimensions."""

        return len(self.shape)

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible domain description."""

        return {
            "shape": list(self.shape),
            "lengths": list(self.lengths),
            "axis_names": list(self.axis_names),
            "grid_placement": self.grid_placement.value,
        }
