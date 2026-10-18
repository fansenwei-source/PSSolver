"""Dependency-neutral pressure-policy constants.

Configuration and solver implementations both consume these declarations;
neither layer needs to import the other to determine the public tolerance
floor.
"""

from __future__ import annotations


_REAL_DTYPE_EPSILON = {
    "float32": 2.0**-23,
    "complex64": 2.0**-23,
    "float64": 2.0**-52,
    "complex128": 2.0**-52,
}


def channel_pressure_minimum_relative_tolerance_for_precision(
    precision: str,
) -> float:
    """Return the conservative Channel tolerance-mode floor."""

    try:
        epsilon = _REAL_DTYPE_EPSILON[precision]
    except KeyError as exc:
        raise TypeError(
            f"unsupported Channel pressure precision: {precision}"
        ) from exc
    return 8.0 * epsilon


__all__ = ["channel_pressure_minimum_relative_tolerance_for_precision"]
