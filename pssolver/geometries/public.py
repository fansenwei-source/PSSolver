"""Ergonomic public geometry constructors over canonical implementations."""

from __future__ import annotations

from pssolver.core.domain import DomainSpec, GridPlacement

from .tensor_product import PlaneSlab as _CanonicalPlaneSlab


class _PlaneSlabConstructor(type):
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
        if domain is not None:
            if shape is not None or lengths is not None or axis_names is not None:
                raise ValueError(
                    "domain cannot be combined with shape, lengths, or axis_names"
                )
            if grid_placement is not GridPlacement.CELL_CENTERED:
                raise ValueError(
                    "domain cannot be combined with a grid_placement override"
                )
            resolved = domain
        else:
            if shape is None or lengths is None:
                raise TypeError(
                    "either domain or both shape and lengths are required"
                )
            resolved = DomainSpec(
                shape=shape,
                lengths=lengths,
                axis_names=axis_names,
                grid_placement=grid_placement,
            )
        return _CanonicalPlaneSlab(
            resolved,
            wall_normal_axis=wall_normal_axis,
        )

    def __instancecheck__(cls, instance: object) -> bool:
        return isinstance(instance, _CanonicalPlaneSlab)

    def __eq__(cls, other: object) -> bool:
        return other is cls or other is _CanonicalPlaneSlab

    def __hash__(cls) -> int:
        return hash(_CanonicalPlaneSlab)


class PlaneSlab(_CanonicalPlaneSlab, metaclass=_PlaneSlabConstructor):
    """Compatibility constructor returning the canonical Plane slab type.

    The class-like facade keeps ``isinstance(value, PlaneSlab)`` useful while
    every constructed value retains the historical canonical type and pickle
    identity from :mod:`pssolver.geometries.tensor_product`.
    """

    __slots__ = ()


# Registries record the historical canonical type name in restart provenance.
# The facade constructs canonical instances and retains that nominal identity.
PlaneSlab.__module__ = _CanonicalPlaneSlab.__module__
PlaneSlab.__qualname__ = _CanonicalPlaneSlab.__qualname__


__all__ = ["PlaneSlab"]
