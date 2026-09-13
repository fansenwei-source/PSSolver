"""Named tensor-product geometries built from core topology contracts.

These classes describe domain topology only.  They do not select physical
boundary conditions, transform bases, Stokes solvers, or numerical defaults.
"""

from __future__ import annotations

from pssolver.core.domain import DomainSpec
from pssolver.core.geometry import AxisTopology, GeometrySpec


def _require_domain(domain: DomainSpec) -> DomainSpec:
    if not isinstance(domain, DomainSpec):
        raise TypeError("domain must be a DomainSpec")
    return domain


def _normalize_axis(axis: int, ndim: int, description: str) -> int:
    if not isinstance(axis, int) or isinstance(axis, bool):
        raise TypeError(f"{description} must be an integer")
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"{description} must be in [0, {ndim}), got {axis}"
        )
    return axis


class PeriodicBox(GeometrySpec):
    """A one-, two-, or three-dimensional domain periodic on every axis."""

    __slots__ = ()

    def __init__(self, domain: DomainSpec) -> None:
        domain = _require_domain(domain)
        super().__init__(
            name="periodic_box",
            domain=domain,
            axis_topologies=(AxisTopology.PERIODIC,) * domain.ndim,
        )


class PlaneSlab(GeometrySpec):
    """A slab with one bounded wall-normal axis and periodic tangent axes."""

    __slots__ = ()

    def __init__(
        self,
        domain: DomainSpec,
        *,
        wall_normal_axis: int | None = None,
    ) -> None:
        domain = _require_domain(domain)
        if domain.ndim < 2:
            raise ValueError("a PlaneSlab requires at least two dimensions")
        if wall_normal_axis is None:
            wall_normal_axis = domain.ndim - 1
        wall_normal_axis = _normalize_axis(
            wall_normal_axis,
            domain.ndim,
            "wall_normal_axis",
        )
        topologies = [AxisTopology.PERIODIC] * domain.ndim
        topologies[wall_normal_axis] = AxisTopology.BOUNDED
        super().__init__(
            name="plane_slab",
            domain=domain,
            axis_topologies=tuple(topologies),
        )


class RectangularChannel(GeometrySpec):
    """A channel periodic along one streamwise axis and otherwise bounded."""

    __slots__ = ()

    def __init__(
        self,
        domain: DomainSpec,
        *,
        streamwise_axis: int = 0,
    ) -> None:
        domain = _require_domain(domain)
        if domain.ndim < 2:
            raise ValueError(
                "a RectangularChannel requires at least two dimensions"
            )
        streamwise_axis = _normalize_axis(
            streamwise_axis,
            domain.ndim,
            "streamwise_axis",
        )
        topologies = [AxisTopology.BOUNDED] * domain.ndim
        topologies[streamwise_axis] = AxisTopology.PERIODIC
        super().__init__(
            name="rectangular_channel",
            domain=domain,
            axis_topologies=tuple(topologies),
        )
