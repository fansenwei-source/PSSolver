"""Physical boundary-condition contracts.

The contracts describe physical semantics, not transform implementations.
Mapping these objects to FFT, DCT, DST, lifting, or tau methods belongs to a
later spectral-plan assembly stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BoundaryKind(str, Enum):
    """Physical boundary-condition families supported by Stage A."""

    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


@dataclass(frozen=True, slots=True)
class BoundaryCondition:
    """A homogeneous physical condition applied along one coordinate axis."""

    kind: BoundaryKind
    is_homogeneous: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryKind):
            raise TypeError("kind must be a BoundaryKind")
        if not isinstance(self.is_homogeneous, bool):
            raise TypeError("is_homogeneous must be a bool")
        if not self.is_homogeneous:
            raise ValueError(
                "Stage A supports homogeneous boundary contracts only"
            )

    @property
    def legacy_label(self) -> str:
        """Return the current string label for a compatibility adapter."""

        return self.kind.value

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description of the condition."""

        return {
            "kind": self.kind.value,
            "is_homogeneous": self.is_homogeneous,
        }


@dataclass(frozen=True, slots=True)
class PeriodicBC(BoundaryCondition):
    """Periodic continuation along one coordinate axis."""

    kind: BoundaryKind = field(default=BoundaryKind.PERIODIC, init=False)
    is_homogeneous: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class HomogeneousDirichletBC(BoundaryCondition):
    """Zero field value at both ends of a bounded coordinate axis."""

    kind: BoundaryKind = field(default=BoundaryKind.DIRICHLET, init=False)
    is_homogeneous: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class HomogeneousNeumannBC(BoundaryCondition):
    """Zero normal derivative at both ends of a bounded coordinate axis."""

    kind: BoundaryKind = field(default=BoundaryKind.NEUMANN, init=False)
    is_homogeneous: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class BoundarySet:
    """Per-axis physical boundary conditions for one field component."""

    axes: tuple[BoundaryCondition, ...]

    def __post_init__(self) -> None:
        try:
            axes = tuple(self.axes)
        except TypeError as exc:
            raise TypeError("axes must be an iterable of boundary conditions") from exc
        if not axes:
            raise ValueError("at least one axis boundary is required")
        if not all(isinstance(condition, BoundaryCondition) for condition in axes):
            raise TypeError("every axis entry must be a BoundaryCondition")
        object.__setattr__(self, "axes", axes)

    @property
    def ndim(self) -> int:
        """Number of coordinate axes described by this set."""

        return len(self.axes)

    @property
    def legacy_labels(self) -> tuple[str, ...]:
        """Return labels understood by the current transform backend."""

        return tuple(condition.legacy_label for condition in self.axes)

    def to_metadata(self) -> list[dict[str, object]]:
        """Return JSON-compatible per-axis boundary metadata."""

        return [condition.to_metadata() for condition in self.axes]
