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
PLANE_ROBIN_SCALAR_LOWERING_SCHEMA_VERSION = 1


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
    product = lower * upper
    total = lower + upper
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
        if left_value * right_value >= 0.0:
            # The normal bracket is intentionally separated from n*pi to
            # suppress endpoint roundoff.  A weak finite impedance can place
            # the root inside that offset.  Retry around its continuum
            # asymptotic position without changing the robust common path.
            if index == 0:
                estimate = math.sqrt(product + total)
                if 0.0 < estimate < 0.5 * math.pi:
                    left = max(
                        math.nextafter(left_endpoint, right_endpoint),
                        0.25 * estimate,
                    )
                    right = min(
                        math.nextafter(right_endpoint, left_endpoint),
                        4.0 * estimate,
                    )
            else:
                base = left_endpoint
                denominator = base * base - product
                estimate = (
                    base + base * total / denominator
                    if denominator > 0.0
                    else math.nan
                )
                if left_endpoint < estimate < right_endpoint:
                    distance = estimate - left_endpoint
                    left = max(
                        math.nextafter(left_endpoint, right_endpoint),
                        left_endpoint + 0.25 * distance,
                    )
                    right = min(
                        math.nextafter(right_endpoint, left_endpoint),
                        left_endpoint + 4.0 * distance,
                    )
            left_value = _characteristic(left, lower, upper)
            right_value = _characteristic(right, lower, upper)
        if left_value == 0.0:
            roots.append(left)
            continue
        if right_value == 0.0:
            roots.append(right)
            continue
        if left_value * right_value >= 0.0:
            if index == 0:
                estimate = math.sqrt(product + total)
            else:
                base = left_endpoint
                denominator = base * base - product
                estimate = (
                    base + base * total / denominator
                    if denominator > 0.0
                    else math.nan
                )
            estimate = min(
                max(
                    estimate,
                    math.nextafter(left_endpoint, right_endpoint),
                ),
                math.nextafter(right_endpoint, left_endpoint),
            )
            estimate_value = _characteristic(estimate, lower, upper)
            if (
                not math.isfinite(estimate_value)
                or abs(estimate_value) > 512.0 * math.ulp(1.0)
            ):
                raise ValueError(
                    "Robin characteristic does not bracket one root in the "
                    f"expected interval {index}"
                )
            roots.append(estimate)
            continue
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


@dataclass(frozen=True, slots=True)
class PlaneRobinScalarLoweringPlan:
    """Field-neutral lowering for one registered scalar Plane component.

    The plan resolves only the P8.5.3 bounded-axis runtime pilot.  Periodic
    axes are declared as Fourier axes but are not materialized by this slice.
    """

    source_simulation_sha256: str
    source_boundary_sha256: str
    field_name: str
    component: str
    geometry_name: str
    domain_shape: tuple[int, ...]
    domain_lengths: tuple[float, ...]
    axis_names: tuple[str, ...]
    periodic_axes: tuple[int, ...]
    wall_normal_axis: int
    robin_plan: CellCenteredRobinEigenbasisPlan
    evolved_representation: str = "homogeneous_remainder"

    def __post_init__(self) -> None:
        for name in ("source_simulation_sha256", "source_boundary_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("field_name", "component"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.isidentifier():
                raise ValueError(f"{name} must be a Python identifier")
        if self.geometry_name != "plane_slab":
            raise ValueError("Robin scalar lowering requires plane_slab geometry")
        shape = tuple(self.domain_shape)
        lengths = tuple(float(value) for value in self.domain_lengths)
        axis_names = tuple(self.axis_names)
        if len(shape) != 3 or len(lengths) != 3 or len(axis_names) != 3:
            raise ValueError("P8.5.3 requires a three-dimensional Plane domain")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in shape
        ):
            raise ValueError("domain_shape entries must be positive integers")
        if any(not math.isfinite(value) or value <= 0.0 for value in lengths):
            raise ValueError("domain_lengths entries must be positive and finite")
        if any(
            not isinstance(value, str) or not value.isidentifier()
            for value in axis_names
        ) or len(set(axis_names)) != 3:
            raise ValueError("axis_names must contain three unique identifiers")
        periodic_axes = tuple(self.periodic_axes)
        if periodic_axes != (0, 1) or self.wall_normal_axis != 2:
            raise ValueError(
                "P8.5.3 requires periodic axes (0, 1) and wall axis 2"
            )
        if not isinstance(self.robin_plan, CellCenteredRobinEigenbasisPlan):
            raise TypeError("robin_plan must be a Robin eigenbasis plan")
        if (
            self.robin_plan.size != shape[self.wall_normal_axis]
            or self.robin_plan.length != lengths[self.wall_normal_axis]
        ):
            raise ValueError("Robin plan and Plane wall axis disagree")
        if self.evolved_representation != "homogeneous_remainder":
            raise ValueError("unsupported Robin evolved representation")
        object.__setattr__(self, "domain_shape", shape)
        object.__setattr__(self, "domain_lengths", lengths)
        object.__setattr__(self, "axis_names", axis_names)
        object.__setattr__(self, "periodic_axes", periodic_axes)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": PLANE_ROBIN_SCALAR_LOWERING_SCHEMA_VERSION,
            "kind": "plane_robin_scalar_bounded_axis_pilot",
            "source_simulation_sha256": self.source_simulation_sha256,
            "source_boundary_sha256": self.source_boundary_sha256,
            "field_name": self.field_name,
            "component": self.component,
            "field_role": "evolved",
            "geometry_name": self.geometry_name,
            "domain_shape": list(self.domain_shape),
            "domain_lengths": list(self.domain_lengths),
            "axis_names": list(self.axis_names),
            "periodic_axes": list(self.periodic_axes),
            "periodic_axis_method": "fourier_declared_not_materialized_p8_5_3",
            "wall_normal_axis": self.wall_normal_axis,
            "bounded_axis_operator": self.robin_plan.to_metadata(),
            "bounded_axis_plan_sha256": self.robin_plan.canonical_sha256(),
            "evolved_representation": self.evolved_representation,
            "physical_observation": "homogeneous_remainder_plus_affine_lift",
            "model_specialization": None,
            "complete_timestep_connected": False,
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
    "PLANE_ROBIN_SCALAR_LOWERING_SCHEMA_VERSION",
    "ROBIN_EIGENBASIS_PLAN_SCHEMA_VERSION",
    "CellCenteredRobinEigenbasisPlan",
    "PlaneRobinScalarLoweringPlan",
    "build_cell_centered_robin_eigenbasis_plan",
]
