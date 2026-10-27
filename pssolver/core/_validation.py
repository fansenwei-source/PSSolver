"""Small standard-library-only validation primitives.

This module contains validation mechanics rather than public error policy.
Callers retain ownership of their exception type and structured error code by
providing the rejection callback.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Set
from typing import NoReturn


def require_exact_keys(
    value: Mapping[str, object],
    expected: Set[str],
    description: str,
    *,
    reject: Callable[[str], NoReturn],
) -> dict[str, object]:
    """Return ``value`` as a dict after enforcing one exact key set."""

    observed = set(value)
    if observed != set(expected):
        reject(
            f"{description} must declare every qualified option explicitly; "
            f"missing={tuple(sorted(set(expected) - observed))!r}, "
            f"extra={tuple(sorted(observed - set(expected)))!r}"
        )
    return dict(value)


__all__ = ["require_exact_keys"]
