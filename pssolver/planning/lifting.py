"""Tensor-free plans for field-neutral static Plane lifting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Real

from pssolver.core.boundary import (
    BoundaryKind,
    StaticConstantBoundaryValue,
)
from pssolver.core.domain import GridPlacement


STATIC_LIFTING_PLAN_SCHEMA_VERSION = 1


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


class StaticLiftExtension(str, Enum):
    """Qualified spatial extension used to construct a static lift."""

    AFFINE_WALL_NORMAL = "affine_wall_normal"


@dataclass(frozen=True, slots=True)
class StaticLiftingComponentPlan:
    """Static wall data and homogeneous-remainder contract for one component."""

    field_name: str
    component: str
    wall_normal_axis: int
    lower_value: StaticConstantBoundaryValue
    upper_value: StaticConstantBoundaryValue

    def __post_init__(self) -> None:
        for value, description in (
            (self.field_name, "lifting field name"),
            (self.component, "lifting component"),
        ):
            if not isinstance(value, str) or not value.isidentifier():
                raise ValueError(f"{description} must be a Python identifier")
        if (
            not isinstance(self.wall_normal_axis, int)
            or isinstance(self.wall_normal_axis, bool)
            or self.wall_normal_axis < 0
        ):
            raise ValueError("wall_normal_axis must be a non-negative integer")
        for name in ("lower_value", "upper_value"):
            value = getattr(self, name)
            if not isinstance(value, StaticConstantBoundaryValue):
                raise TypeError(
                    f"{name} must be a StaticConstantBoundaryValue"
                )

    def value_at_fraction(self, fraction: Real) -> float:
        """Evaluate the affine extension at normalized wall coordinate."""

        if isinstance(fraction, bool) or not isinstance(fraction, Real):
            raise TypeError("lifting coordinate fraction must be real")
        normalized = float(fraction)
        if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError("lifting coordinate fraction must lie in [0, 1]")
        return self.lower_value.value + normalized * (
            self.upper_value.value - self.lower_value.value
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "component": self.component,
            "wall_normal_axis": self.wall_normal_axis,
            "lower_value": self.lower_value.to_metadata(),
            "lower_value_sha256": self.lower_value.canonical_sha256(),
            "upper_value": self.upper_value.to_metadata(),
            "upper_value_sha256": self.upper_value.canonical_sha256(),
            "homogeneous_remainder_boundary_kind": (
                BoundaryKind.DIRICHLET.value
            ),
        }


@dataclass(frozen=True, slots=True)
class StaticLiftingPlan:
    """Complete tensor-free plan for a cell-centered Plane affine lift."""

    geometry_name: str
    domain_shape: tuple[int, ...]
    domain_lengths: tuple[float, ...]
    axis_names: tuple[str, ...]
    grid_placement: GridPlacement
    wall_normal_axis: int
    components: tuple[StaticLiftingComponentPlan, ...]
    source_equation_sha256: str
    source_boundary_sha256: str
    extension: StaticLiftExtension = StaticLiftExtension.AFFINE_WALL_NORMAL

    def __post_init__(self) -> None:
        if self.geometry_name != "plane_slab":
            raise ValueError("static affine lifting requires plane_slab geometry")
        shape = tuple(self.domain_shape)
        lengths = tuple(float(value) for value in self.domain_lengths)
        names = tuple(self.axis_names)
        if not shape or len(shape) != len(lengths) or len(shape) != len(names):
            raise ValueError("lifting domain metadata dimensions must agree")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in shape
        ):
            raise ValueError("lifting domain shape entries must be positive")
        if any(not math.isfinite(value) or value <= 0.0 for value in lengths):
            raise ValueError("lifting domain lengths must be positive and finite")
        if any(
            not isinstance(value, str) or not value.isidentifier()
            for value in names
        ):
            raise ValueError("lifting axis names must be Python identifiers")
        if len(set(names)) != len(names):
            raise ValueError("lifting axis names must be unique")
        if not isinstance(self.grid_placement, GridPlacement):
            raise TypeError("grid_placement must be a GridPlacement")
        if self.grid_placement is not GridPlacement.CELL_CENTERED:
            raise ValueError("P8.4.2 supports cell-centered lifting only")
        if (
            not isinstance(self.wall_normal_axis, int)
            or isinstance(self.wall_normal_axis, bool)
            or self.wall_normal_axis < 0
            or self.wall_normal_axis >= len(shape)
        ):
            raise ValueError("wall_normal_axis is outside the lifting domain")
        components = tuple(self.components)
        if not components or not all(
            isinstance(value, StaticLiftingComponentPlan)
            for value in components
        ):
            raise TypeError(
                "components must contain StaticLiftingComponentPlan objects"
            )
        if len({value.component for value in components}) != len(components):
            raise ValueError("lifting component names must be unique")
        if any(
            value.wall_normal_axis != self.wall_normal_axis
            for value in components
        ):
            raise ValueError("component and plan wall-normal axes must agree")
        if not isinstance(self.extension, StaticLiftExtension):
            raise TypeError("extension must be a StaticLiftExtension")
        object.__setattr__(self, "domain_shape", shape)
        object.__setattr__(self, "domain_lengths", lengths)
        object.__setattr__(self, "axis_names", names)
        object.__setattr__(
            self,
            "components",
            tuple(sorted(components, key=lambda value: value.component)),
        )
        object.__setattr__(
            self,
            "source_equation_sha256",
            _digest(self.source_equation_sha256, "source equation identity"),
        )
        object.__setattr__(
            self,
            "source_boundary_sha256",
            _digest(self.source_boundary_sha256, "source boundary identity"),
        )

    @property
    def component_order(self) -> tuple[str, ...]:
        return tuple(value.component for value in self.components)

    def for_component(self, component: str) -> StaticLiftingComponentPlan:
        for value in self.components:
            if value.component == component:
                return value
        raise KeyError(component)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": STATIC_LIFTING_PLAN_SCHEMA_VERSION,
            "geometry_name": self.geometry_name,
            "domain_shape": list(self.domain_shape),
            "domain_lengths": list(self.domain_lengths),
            "axis_names": list(self.axis_names),
            "grid_placement": self.grid_placement.value,
            "wall_normal_axis": self.wall_normal_axis,
            "components": [value.to_metadata() for value in self.components],
            "source_equation_sha256": self.source_equation_sha256,
            "source_boundary_sha256": self.source_boundary_sha256,
            "extension": self.extension.value,
            "time_dependence": "static",
            "evolved_representation": "homogeneous_remainder",
            "physical_reconstruction": "homogeneous_remainder_plus_lift",
        }

    def canonical_sha256(self) -> str:
        return _canonical_sha256(self.to_metadata())


__all__ = [
    "STATIC_LIFTING_PLAN_SCHEMA_VERSION",
    "StaticLiftExtension",
    "StaticLiftingComponentPlan",
    "StaticLiftingPlan",
]
