"""Tensor-free plans for a cell-centered Robin eigenbasis.

The public boundary law remains method independent.  This module owns the
first qualified bounded-axis lowering selected by P8.5.2: a coefficient-
specific continuum Robin eigenbasis sampled on the existing uniform,
cell-centered Plane grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

from pssolver.core.boundary import StaticRobinCoefficients
from pssolver.core.domain import GridPlacement


ROBIN_EIGENBASIS_PLAN_SCHEMA_VERSION = 1


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_positive(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{description} must be positive and finite")
    return float(value)


def _characteristic(value: float, lower: float, upper: float) -> float:
    product = lower * upper
    total = lower + upper
    scale = max(1.0, abs(product), abs(total * value), value * value)
    return (
        (product - value * value) * math.sin(value)
        + value * total * math.cos(value)
    ) / scale


def _positive_robin_roots(
    count: int,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    roots: list[float] = []
    for index in range(count):
        left_endpoint = index * math.pi
        right_endpoint = (index + 1) * math.pi
        offset = 64.0 * math.ulp(max(1.0, right_endpoint))
        left = left_endpoint + offset
        right = right_endpoint - offset
        left_value = _characteristic(left, lower, upper)
        right_value = _characteristic(right, lower, upper)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise ValueError("Robin characteristic is not finite")
        if left_value == 0.0:
            roots.append(left)
            continue
        if right_value == 0.0:
            roots.append(right)
            continue
        if left_value * right_value >= 0.0:
            raise ValueError(
                "Robin characteristic does not bracket one root in the "
                f"expected interval {index}"
            )
        for _ in range(128):
            midpoint = 0.5 * (left + right)
            midpoint_value = _characteristic(midpoint, lower, upper)
            if midpoint_value == 0.0:
                left = midpoint
                right = midpoint
                break
            if left_value * midpoint_value < 0.0:
                right = midpoint
            else:
                left = midpoint
                left_value = midpoint_value
            if right - left <= 8.0 * math.ulp(max(1.0, midpoint)):
                break
        roots.append(0.5 * (left + right))
    return tuple(roots)


@dataclass(frozen=True, slots=True)
class CellCenteredRobinEigenbasisPlan:
    """Immutable coefficient-specific plan for one bounded Plane axis."""

    size: int
    length: float
    lower: StaticRobinCoefficients
    upper: StaticRobinCoefficients
    dimensionless_roots: tuple[float, ...]
    grid_placement: GridPlacement = GridPlacement.CELL_CENTERED
    operator_kind: str = "cell_centered_robin_eigenbasis"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or self.size < 2
        ):
            raise ValueError("Robin eigenbasis size must be an integer >= 2")
        object.__setattr__(
            self,
            "length",
            _finite_positive(self.length, "Robin axis length"),
        )
        for name in ("lower", "upper"):
            value = getattr(self, name)
            if not isinstance(value, StaticRobinCoefficients):
                raise TypeError(
                    f"{name} must be StaticRobinCoefficients"
                )
            if value.beta.value <= 0.0:
                raise ValueError(
                    "P8.5.2 Robin eigenbasis requires beta > 0 on both faces"
                )
            if value.alpha.value < 0.0:
                raise ValueError(
                    "P8.5.2 Robin eigenbasis requires alpha >= 0 on both faces"
                )
        if not isinstance(self.grid_placement, GridPlacement):
            raise TypeError("grid_placement must be a GridPlacement")
        if self.grid_placement is not GridPlacement.CELL_CENTERED:
            raise ValueError("P8.5.2 supports cell-centered grids only")
        roots = tuple(float(value) for value in self.dimensionless_roots)
        if len(roots) != self.size:
            raise ValueError("Robin root count must equal the axis size")
        if any(not math.isfinite(value) or value < 0.0 for value in roots):
            raise ValueError("Robin roots must be finite and non-negative")
        if any(right <= left for left, right in zip(roots, roots[1:])):
            raise ValueError("Robin roots must be strictly increasing")
        pure_neumann = (
            self.lower.alpha.value == 0.0
            and self.upper.alpha.value == 0.0
        )
        if pure_neumann != (roots[0] == 0.0):
            raise ValueError("Robin zero-mode identity is inconsistent")
        if self.operator_kind != "cell_centered_robin_eigenbasis":
            raise ValueError("unsupported Robin operator kind")
        object.__setattr__(self, "dimensionless_roots", roots)

    @property
    def wavenumbers(self) -> tuple[float, ...]:
        return tuple(value / self.length for value in self.dimensionless_roots)

    @property
    def laplacian_eigenvalues(self) -> tuple[float, ...]:
        return tuple(-(value * value) for value in self.wavenumbers)

    @property
    def is_pure_neumann(self) -> bool:
        return self.dimensionless_roots[0] == 0.0

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": ROBIN_EIGENBASIS_PLAN_SCHEMA_VERSION,
            "operator_kind": self.operator_kind,
            "size": self.size,
            "length": self.length,
            "grid_placement": self.grid_placement.value,
            "normal_derivative_convention": "outward_unit_normal",
            "lower": self.lower.to_metadata(),
            "lower_sha256": self.lower.canonical_sha256(),
            "upper": self.upper.to_metadata(),
            "upper_sha256": self.upper.canonical_sha256(),
            "normalized_impedance": {
                "lower_alpha_over_beta": (
                    self.lower.alpha.value / self.lower.beta.value
                ),
                "upper_alpha_over_beta": (
                    self.upper.alpha.value / self.upper.beta.value
                ),
            },
            "dimensionless_roots": list(self.dimensionless_roots),
            "wavenumbers": list(self.wavenumbers),
            "laplacian_eigenvalues": list(self.laplacian_eigenvalues),
            "root_solver": "deterministic_bracketed_bisection",
            "root_solve_location": "plan_construction_only",
            "physical_reconstruction": "homogeneous_remainder_plus_affine_lift",
        }

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_metadata())


def build_cell_centered_robin_eigenbasis_plan(
    *,
    size: int,
    length: float,
    lower: StaticRobinCoefficients,
    upper: StaticRobinCoefficients,
) -> CellCenteredRobinEigenbasisPlan:
    """Build the first qualified constant-coefficient Robin root plan."""

    if not isinstance(lower, StaticRobinCoefficients):
        raise TypeError("lower must be StaticRobinCoefficients")
    if not isinstance(upper, StaticRobinCoefficients):
        raise TypeError("upper must be StaticRobinCoefficients")
    normalized_length = _finite_positive(length, "Robin axis length")
    for name, value in (("lower", lower), ("upper", upper)):
        if value.beta.value <= 0.0:
            raise ValueError(
                f"P8.5.2 {name} Robin beta must be positive"
            )
        if value.alpha.value < 0.0:
            raise ValueError(
                f"P8.5.2 {name} Robin alpha must be non-negative"
            )
    if (
        lower.alpha.value == 0.0
        and upper.alpha.value == 0.0
    ):
        roots = tuple(index * math.pi for index in range(size))
    else:
        lower_impedance = (
            normalized_length * lower.alpha.value / lower.beta.value
        )
        upper_impedance = (
            normalized_length * upper.alpha.value / upper.beta.value
        )
        roots = _positive_robin_roots(
            size,
            lower_impedance,
            upper_impedance,
        )
    return CellCenteredRobinEigenbasisPlan(
        size=size,
        length=normalized_length,
        lower=lower,
        upper=upper,
        dimensionless_roots=roots,
    )


__all__ = [
    "ROBIN_EIGENBASIS_PLAN_SCHEMA_VERSION",
    "CellCenteredRobinEigenbasisPlan",
    "build_cell_centered_robin_eigenbasis_plan",
]
