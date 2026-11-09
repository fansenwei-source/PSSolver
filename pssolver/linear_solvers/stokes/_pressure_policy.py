"""Frozen tensor-free Channel pressure policy shared by production and Functional API."""

from __future__ import annotations

import math
from numbers import Integral, Real


CHANNEL_PRESSURE_POLICY_SCHEMA_VERSION = 1


CHANNEL_PRESSURE_RHS_NORM_DEFINITION = "active_mode_root_mean_square.v1"


CHANNEL_PRESSURE_WORKING_VECTOR_SCALING = (
    "largest_power_of_two_not_greater_than_active_component_amax"
)


CHANNEL_PRESSURE_BREAKDOWN_DEFINITION = (
    "abs(mean_active(conj(left)*right))<=eps*active_rms(left)*active_rms(right)"
)


_CHANNEL_PRESSURE_INITIAL_GUESS_POLICIES = {
    "persistent_production_warm_start",
    "zero_every_call",
}


def _finite(value: object, description: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{description} must be finite")
    return float(value)


def _positive(value: object, description: str) -> float:
    normalized = _finite(value, description)
    if normalized <= 0.0:
        raise ValueError(f"{description} must be positive")
    return normalized


def _integer(value: object, description: str, *, positive: bool) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer")
    normalized = int(value)
    if positive and normalized <= 0:
        raise ValueError(f"{description} must be positive")
    if not positive and normalized < 0:
        raise ValueError(f"{description} must be non-negative")
    return normalized


def channel_pressure_policy_metadata(
    *,
    relative_tolerance: object,
    max_iterations: object,
    fixed_iterations: object,
    initial_guess_policy: str,
) -> dict[str, object]:
    """Return the frozen tensor-free Channel pressure policy."""

    _positive(relative_tolerance, "relative_tolerance")
    maximum = _integer(max_iterations, "max_iterations", positive=True)
    if fixed_iterations is None:
        fixed = None
    else:
        fixed = _integer(
            fixed_iterations,
            "fixed_iterations",
            positive=True,
        )
    if initial_guess_policy not in _CHANNEL_PRESSURE_INITIAL_GUESS_POLICIES:
        raise ValueError("unsupported Channel pressure initial-guess policy")
    if fixed is None:
        return {
            "schema_version": CHANNEL_PRESSURE_POLICY_SCHEMA_VERSION,
            "convergence_mode": "relative_tolerance",
            "authoritative_iteration_limit": maximum,
            "authoritative_iteration_limit_source": "max_iterations",
            "max_iterations_role": "authoritative_iteration_cap",
            "relative_tolerance_role": "required_postcondition",
            "initial_guess_policy": initial_guess_policy,
            "rhs_norm_definition": CHANNEL_PRESSURE_RHS_NORM_DEFINITION,
            "working_vector_scaling": CHANNEL_PRESSURE_WORKING_VECTOR_SCALING,
            "breakdown_definition": CHANNEL_PRESSURE_BREAKDOWN_DEFINITION,
        }
    return {
        "schema_version": CHANNEL_PRESSURE_POLICY_SCHEMA_VERSION,
        "convergence_mode": "fixed_work",
        "authoritative_iteration_limit": fixed,
        "authoritative_iteration_limit_source": "fixed_iterations",
        "max_iterations_role": "inactive_compatibility_field",
        "relative_tolerance_role": "diagnostic_only_not_early_exit",
        "initial_guess_policy": initial_guess_policy,
        "rhs_norm_definition": CHANNEL_PRESSURE_RHS_NORM_DEFINITION,
        "working_vector_scaling": CHANNEL_PRESSURE_WORKING_VECTOR_SCALING,
        "breakdown_definition": CHANNEL_PRESSURE_BREAKDOWN_DEFINITION,
    }
