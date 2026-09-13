"""Geometry contracts independent of physical models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .domain import DomainSpec


class AxisTopology(str, Enum):
    """Topological type of one tensor-product coordinate axis."""

    PERIODIC = "periodic"
    BOUNDED = "bounded"


@dataclass(frozen=True, slots=True)
class GeometrySpec:
    """Immutable geometric facts used when assembling a spectral plan."""

    name: str
    domain: DomainSpec
    axis_topologies: tuple[AxisTopology, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("geometry name must be a non-empty string")
        if not isinstance(self.domain, DomainSpec):
            raise TypeError("domain must be a DomainSpec")
        try:
            topologies = tuple(self.axis_topologies)
        except TypeError as exc:
            raise TypeError("axis_topologies must be iterable") from exc
        if len(topologies) != self.domain.ndim:
            raise ValueError("axis_topologies must match the domain dimension")
        if not all(isinstance(topology, AxisTopology) for topology in topologies):
            raise TypeError("every axis topology must be an AxisTopology")
        object.__setattr__(self, "axis_topologies", topologies)

    @property
    def periodic_axes(self) -> tuple[int, ...]:
        """Indices of periodic coordinate axes."""

        return tuple(
            index
            for index, topology in enumerate(self.axis_topologies)
            if topology is AxisTopology.PERIODIC
        )

    @property
    def bounded_axes(self) -> tuple[int, ...]:
        """Indices of wall-bounded coordinate axes."""

        return tuple(
            index
            for index, topology in enumerate(self.axis_topologies)
            if topology is AxisTopology.BOUNDED
        )

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible geometry description."""

        return {
            "name": self.name,
            "domain": self.domain.to_metadata(),
            "axis_topologies": [
                topology.value for topology in self.axis_topologies
            ],
            "periodic_axes": list(self.periodic_axes),
            "bounded_axes": list(self.bounded_axes),
        }
