"""Shared fail-closed validation for public application options.

The public compilers use these tensor-free validators before constructing a
run specification, lowering plan, output directory, or runtime.  Keeping the
rules here prevents application adapters from accepting subtly different
spellings of the same public contract.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Integral, Real

from pssolver.configuration.execution_device import (
    execution_device_normalization,
)


ITERATIVE_PRESSURE_OPTION_KEYS = frozenset(
    {
        "algorithm",
        "fixed_iterations",
        "max_iterations",
        "relative_tolerance",
        "warm_start",
    }
)


def validate_public_execution_device(requested: object) -> dict[str, str]:
    """Validate and record one public device request without allocation."""

    return execution_device_normalization(requested)


def validate_iterative_pressure_options(
    value: object,
    *,
    description: str,
    required_algorithm: str = "preconditioned_conjugate_gradient",
    minimum_relative_tolerance: float | None = None,
) -> dict[str, object]:
    """Validate the common public iterative-pressure declaration."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    observed = set(value)
    if observed != ITERATIVE_PRESSURE_OPTION_KEYS:
        raise ValueError(
            f"{description} must declare every qualified option explicitly; "
            f"missing={tuple(sorted(ITERATIVE_PRESSURE_OPTION_KEYS - observed))!r}, "
            f"extra={tuple(sorted(observed - ITERATIVE_PRESSURE_OPTION_KEYS))!r}"
        )
    pressure = dict(value)
    if pressure["algorithm"] != required_algorithm:
        raise ValueError(
            f"{description} algorithm must be {required_algorithm!r}"
        )
    tolerance = pressure["relative_tolerance"]
    if (
        not isinstance(tolerance, Real)
        or isinstance(tolerance, bool)
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise ValueError(
            f"{description} relative_tolerance must be finite and positive"
        )
    maximum = pressure["max_iterations"]
    if (
        not isinstance(maximum, Integral)
        or isinstance(maximum, bool)
        or int(maximum) <= 0
    ):
        raise ValueError(
            f"{description} max_iterations must be a positive integer"
        )
    fixed = pressure["fixed_iterations"]
    if fixed is not None:
        if (
            not isinstance(fixed, Integral)
            or isinstance(fixed, bool)
            or int(fixed) <= 0
        ):
            raise ValueError(
                f"{description} fixed_iterations must be null or a positive integer"
            )
        if int(fixed) > int(maximum):
            raise ValueError(
                f"{description} fixed_iterations must not exceed max_iterations"
            )
    if minimum_relative_tolerance is not None:
        if (
            not isinstance(minimum_relative_tolerance, Real)
            or isinstance(minimum_relative_tolerance, bool)
            or not math.isfinite(float(minimum_relative_tolerance))
            or float(minimum_relative_tolerance) <= 0.0
        ):
            raise ValueError(
                "minimum_relative_tolerance must be finite and positive"
            )
        if fixed is None and float(tolerance) < float(
            minimum_relative_tolerance
        ):
            raise ValueError(
                f"{description} relative_tolerance is below the supported "
                f"tolerance-mode floor ({float(minimum_relative_tolerance)!r})"
            )
    if not isinstance(pressure["warm_start"], bool):
        raise ValueError(f"{description} warm_start must be a bool")
    return pressure


__all__ = [
    "ITERATIVE_PRESSURE_OPTION_KEYS",
    "validate_iterative_pressure_options",
    "validate_public_execution_device",
]
