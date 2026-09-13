"""Translate physical boundary contracts into current runtime labels.

This module is intentionally one-way.  The legacy strings are an execution
compatibility format, not the source of truth for a physical problem.
"""

from __future__ import annotations

from pssolver.core.boundary import (
    BoundaryCondition,
    BoundaryKind,
    BoundarySet,
)


_LEGACY_LABEL_BY_KIND = {
    BoundaryKind.PERIODIC: "periodic",
    BoundaryKind.DIRICHLET: "dirichlet",
    BoundaryKind.NEUMANN: "neumann",
}


def boundary_condition_to_legacy(condition: BoundaryCondition) -> str:
    """Return the label accepted by the current transform backend."""

    if not isinstance(condition, BoundaryCondition):
        raise TypeError("condition must be a BoundaryCondition")
    try:
        return _LEGACY_LABEL_BY_KIND[condition.kind]
    except KeyError as exc:
        raise ValueError(
            f"unsupported boundary kind {condition.kind!r}"
        ) from exc


def boundary_set_to_legacy(boundaries: BoundarySet) -> tuple[str, ...]:
    """Return one current-runtime boundary label per spatial axis."""

    if not isinstance(boundaries, BoundarySet):
        raise TypeError("boundaries must be a BoundarySet")
    return tuple(
        boundary_condition_to_legacy(condition)
        for condition in boundaries.axes
    )
