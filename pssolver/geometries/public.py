"""Ergonomic public geometry constructors over canonical implementations."""

from __future__ import annotations

from pssolver.core.domain import DomainSpec, GridPlacement

from .tensor_product import PeriodicBox as _CanonicalPeriodicBox
from .tensor_product import PlaneSlab as _CanonicalPlaneSlab
from .tensor_product import RectangularChannel as _CanonicalRectangularChannel


def _resolve_domain(
    domain: DomainSpec | None,
    *,
    shape: tuple[int, ...] | None,
    lengths: tuple[float, ...] | None,
    axis_names: tuple[str, ...] | None,
    grid_placement: GridPlacement,
) -> DomainSpec:
    if domain is not None:
        if shape is not None or lengths is not None or axis_names is not None:
            raise ValueError(
                "domain cannot be combined with shape, lengths, or axis_names"
            )
        if grid_placement is not GridPlacement.CELL_CENTERED:
            raise ValueError(
                "domain cannot be combined with a grid_placement override"
            )
        return domain
    if shape is None or lengths is None:
        raise TypeError("either domain or both shape and lengths are required")
    return DomainSpec(
        shape=shape,
        lengths=lengths,
        axis_names=axis_names,
        grid_placement=grid_placement,
    )


class _CanonicalFacade(type):
    _canonical_type: type

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, cls._canonical_type)

    def __eq__(cls, other: object) -> bool:
        return other is cls or other is cls._canonical_type

    def __hash__(cls) -> int:
        return hash(cls._canonical_type)


class _PeriodicBoxConstructor(_CanonicalFacade):
    _canonical_type = _CanonicalPeriodicBox

    def __call__(
        cls,
        domain: DomainSpec | None = None,
        *,
        shape: tuple[int, ...] | None = None,
        lengths: tuple[float, ...] | None = None,
        axis_names: tuple[str, ...] | None = None,
        grid_placement: GridPlacement = GridPlacement.CELL_CENTERED,
    ) -> _CanonicalPeriodicBox:
        resolved = _resolve_domain(
            domain,
            shape=shape,
            lengths=lengths,
            axis_names=axis_names,
            grid_placement=grid_placement,
        )
        return _CanonicalPeriodicBox(resolved)


class _PlaneSlabConstructor(_CanonicalFacade):
    _canonical_type = _CanonicalPlaneSlab

    def __call__(
        cls,
        domain: DomainSpec | None = None,
        *,
        shape: tuple[int, ...] | None = None,
        lengths: tuple[float, ...] | None = None,
        axis_names: tuple[str, ...] | None = None,
        grid_placement: GridPlacement = GridPlacement.CELL_CENTERED,
        wall_normal_axis: int | None = None,
    ) -> _CanonicalPlaneSlab:
        resolved = _resolve_domain(
            domain,
            shape=shape,
            lengths=lengths,
            axis_names=axis_names,
            grid_placement=grid_placement,
        )
        return _CanonicalPlaneSlab(
            resolved,
            wall_normal_axis=wall_normal_axis,
        )


class _RectangularChannelConstructor(_CanonicalFacade):
    _canonical_type = _CanonicalRectangularChannel

    def __call__(
        cls,
        domain: DomainSpec | None = None,
        *,
        shape: tuple[int, ...] | None = None,
        lengths: tuple[float, ...] | None = None,
        axis_names: tuple[str, ...] | None = None,
        grid_placement: GridPlacement = GridPlacement.CELL_CENTERED,
        streamwise_axis: int = 0,
    ) -> _CanonicalRectangularChannel:
        resolved = _resolve_domain(
            domain,
            shape=shape,
            lengths=lengths,
            axis_names=axis_names,
            grid_placement=grid_placement,
        )
        return _CanonicalRectangularChannel(
            resolved,
            streamwise_axis=streamwise_axis,
        )


class PeriodicBox(_CanonicalPeriodicBox, metaclass=_PeriodicBoxConstructor):
    """Compatibility constructor returning the canonical periodic-box type."""

    __slots__ = ()


class PlaneSlab(_CanonicalPlaneSlab, metaclass=_PlaneSlabConstructor):
    """Compatibility constructor returning the canonical Plane-slab type."""

    __slots__ = ()


class RectangularChannel(
    _CanonicalRectangularChannel,
    metaclass=_RectangularChannelConstructor,
):
    """Compatibility constructor returning the canonical Channel type."""

    __slots__ = ()


# Registries and restart provenance retain the historical canonical type
# names.  Every facade constructs the canonical object rather than a subtype.
for _facade, _canonical in (
    (PeriodicBox, _CanonicalPeriodicBox),
    (PlaneSlab, _CanonicalPlaneSlab),
    (RectangularChannel, _CanonicalRectangularChannel),
):
    _facade.__module__ = _canonical.__module__
    _facade.__qualname__ = _canonical.__qualname__


__all__ = ["PeriodicBox", "PlaneSlab", "RectangularChannel"]
