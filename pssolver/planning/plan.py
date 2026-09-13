"""Immutable, tensor-free plans produced from declarative problem specs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math

from pssolver.core.boundary import BoundarySet
from pssolver.core.fields import FieldRole
from pssolver.core.geometry import AxisTopology
from pssolver.core.numerics import NumericsConfig, SpectralStorage


SPECTRAL_PLAN_SCHEMA_VERSION = 1


class TransformKind(str, Enum):
    """Resolved transform family along one component axis."""

    FFT = "fft"
    DCT = "dct"
    DST = "dst"


@dataclass(frozen=True, slots=True)
class DomainAxisPlan:
    """Resolved topology and storage facts for one domain axis."""

    index: int
    name: str
    topology: AxisTopology
    physical_size: int
    spectral_size: int
    length: float
    hermitian_packed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.index, int) or isinstance(self.index, bool):
            raise TypeError("axis index must be an integer")
        if self.index < 0:
            raise ValueError("axis index must be non-negative")
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("axis name must be a Python identifier")
        if not isinstance(self.topology, AxisTopology):
            raise TypeError("axis topology must be an AxisTopology")
        for label, value in (
            ("physical_size", self.physical_size),
            ("spectral_size", self.spectral_size),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{label} must be a positive integer")
        if not isinstance(self.length, (int, float)) or isinstance(
            self.length,
            bool,
        ):
            raise TypeError("axis length must be a real number")
        if not math.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("axis length must be positive and finite")
        if not isinstance(self.hermitian_packed, bool):
            raise TypeError("hermitian_packed must be a bool")
        expected_spectral_size = (
            self.physical_size // 2 + 1
            if self.hermitian_packed
            else self.physical_size
        )
        if self.spectral_size != expected_spectral_size:
            raise ValueError(
                "spectral_size is inconsistent with Hermitian packing"
            )
        if self.hermitian_packed and self.topology is not AxisTopology.PERIODIC:
            raise ValueError("a Hermitian-packed axis must be periodic")

    def to_metadata(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "topology": self.topology.value,
            "physical_size": self.physical_size,
            "spectral_size": self.spectral_size,
            "length": self.length,
            "hermitian_packed": self.hermitian_packed,
        }


@dataclass(frozen=True, slots=True)
class ComponentTransformPlan:
    """Resolved transform and storage layout for one scalar component."""

    field_name: str
    component_name: str
    role: FieldRole
    boundaries: BoundarySet
    transform_kinds: tuple[TransformKind, ...]
    retained_mode_counts: tuple[int, ...]
    computed_axis_sizes: tuple[int, ...]
    storage_index: int | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.field_name, str)
            or not self.field_name.isidentifier()
        ):
            raise ValueError("field_name must be a Python identifier")
        if (
            not isinstance(self.component_name, str)
            or not self.component_name.isidentifier()
        ):
            raise ValueError("component_name must be a Python identifier")
        if not isinstance(self.role, FieldRole):
            raise TypeError("role must be a FieldRole")
        if not isinstance(self.boundaries, BoundarySet):
            raise TypeError("boundaries must be a BoundarySet")

        transform_kinds = tuple(self.transform_kinds)
        retained_counts = tuple(self.retained_mode_counts)
        computed_sizes = tuple(self.computed_axis_sizes)
        ndim = self.boundaries.ndim
        if not (
            len(transform_kinds)
            == len(retained_counts)
            == len(computed_sizes)
            == ndim
        ):
            raise ValueError("component transform metadata must match its dimension")
        if not all(
            isinstance(kind, TransformKind) for kind in transform_kinds
        ):
            raise TypeError("transform_kinds must contain TransformKind values")
        for values, description in (
            (retained_counts, "retained mode counts"),
            (computed_sizes, "computed axis sizes"),
        ):
            if any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in values
            ):
                raise ValueError(
                    f"{description} must be non-negative integers"
                )
        if self.role in (FieldRole.TRANSIENT, FieldRole.DIAGNOSTIC):
            if self.storage_index is not None:
                raise ValueError(
                    "transient and diagnostic components cannot have "
                    "storage indices"
                )
        elif (
            not isinstance(self.storage_index, int)
            or isinstance(self.storage_index, bool)
            or self.storage_index < 0
        ):
            raise ValueError(
                "evolved and algebraic components need storage indices"
            )

        object.__setattr__(self, "transform_kinds", transform_kinds)
        object.__setattr__(self, "retained_mode_counts", retained_counts)
        object.__setattr__(self, "computed_axis_sizes", computed_sizes)

    def to_metadata(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "component_name": self.component_name,
            "role": self.role.value,
            "boundaries": self.boundaries.to_metadata(),
            "transform_kinds": [kind.value for kind in self.transform_kinds],
            "retained_mode_counts": list(self.retained_mode_counts),
            "computed_axis_sizes": list(self.computed_axis_sizes),
            "storage_index": self.storage_index,
        }


@dataclass(frozen=True, slots=True)
class FieldPlan:
    """Logical field and its resolved scalar-component plans."""

    name: str
    role: FieldRole
    components: tuple[ComponentTransformPlan, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("field plan name must be a non-empty string")
        if not isinstance(self.role, FieldRole):
            raise TypeError("field plan role must be a FieldRole")
        components = tuple(self.components)
        if not components:
            raise ValueError("a field plan must contain components")
        if not all(
            isinstance(component, ComponentTransformPlan)
            for component in components
        ):
            raise TypeError("field plan components have an invalid type")
        if any(component.field_name != self.name for component in components):
            raise ValueError("component field names must match their field plan")
        if any(component.role is not self.role for component in components):
            raise ValueError("component roles must match their field plan")
        object.__setattr__(self, "components", components)

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role.value,
            "components": [
                component.to_metadata() for component in self.components
            ],
        }


@dataclass(frozen=True, slots=True)
class SpectralPlan:
    """A complete read-only plan with no runtime tensors or executors."""

    model_name: str
    geometry_name: str
    axes: tuple[DomainAxisPlan, ...]
    fields: tuple[FieldPlan, ...]
    numerics: NumericsConfig
    model_parameters_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.geometry_name, str) or not self.geometry_name:
            raise ValueError("geometry_name must be a non-empty string")
        if not isinstance(self.numerics, NumericsConfig):
            raise TypeError("numerics must be a NumericsConfig")
        axes = tuple(self.axes)
        fields = tuple(self.fields)
        if not axes or not all(isinstance(axis, DomainAxisPlan) for axis in axes):
            raise TypeError("axes must contain DomainAxisPlan objects")
        if tuple(axis.index for axis in axes) != tuple(range(len(axes))):
            raise ValueError("axis plan indices must be contiguous and ordered")
        if not fields or not all(isinstance(field, FieldPlan) for field in fields):
            raise TypeError("fields must contain FieldPlan objects")
        if len({field.name for field in fields}) != len(fields):
            raise ValueError("field plan names must be unique")
        components = tuple(
            component for field in fields for component in field.components
        )
        if len({component.component_name for component in components}) != len(
            components
        ):
            raise ValueError("component plan names must be globally unique")
        if any(component.boundaries.ndim != len(axes) for component in components):
            raise ValueError("component plan dimensions must match plan axes")
        for component in components:
            for axis, kind, retained, computed in zip(
                axes,
                component.transform_kinds,
                component.retained_mode_counts,
                component.computed_axis_sizes,
            ):
                is_periodic_transform = kind is TransformKind.FFT
                is_periodic_axis = axis.topology is AxisTopology.PERIODIC
                if is_periodic_transform != is_periodic_axis:
                    raise ValueError(
                        "component transform kind conflicts with axis topology"
                    )
                if retained > axis.spectral_size:
                    raise ValueError(
                        "retained mode count exceeds spectral axis size"
                    )
                if computed > axis.spectral_size:
                    raise ValueError(
                        "computed axis size exceeds spectral axis size"
                    )
        storage_indices = sorted(
            component.storage_index
            for component in components
            if component.storage_index is not None
        )
        if storage_indices != list(range(len(storage_indices))):
            raise ValueError("storage indices must be contiguous and unique")
        packed_axes = tuple(axis.index for axis in axes if axis.hermitian_packed)
        expected_packed_axes = (
            (self.numerics.hermitian_axis,)
            if self.numerics.spectral_storage is SpectralStorage.HERMITIAN_HALF
            else ()
        )
        if packed_axes != expected_packed_axes:
            raise ValueError(
                "axis packing is inconsistent with the numerical configuration"
            )

        def reject_nonstandard_constant(value: str) -> None:
            raise ValueError(f"non-finite JSON constant {value!r}")

        try:
            parameters = json.loads(
                self.model_parameters_json,
                parse_constant=reject_nonstandard_constant,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("model_parameters_json must contain valid JSON") from exc
        if not isinstance(parameters, dict):
            raise ValueError("model_parameters_json must describe an object")

        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "fields", fields)

    @property
    def physical_shape(self) -> tuple[int, ...]:
        return tuple(axis.physical_size for axis in self.axes)

    @property
    def spectral_shape(self) -> tuple[int, ...]:
        return tuple(axis.spectral_size for axis in self.axes)

    @property
    def lengths(self) -> tuple[float, ...]:
        return tuple(axis.length for axis in self.axes)

    @property
    def components(self) -> tuple[ComponentTransformPlan, ...]:
        return tuple(
            component for field in self.fields for component in field.components
        )

    @property
    def stored_components(self) -> tuple[ComponentTransformPlan, ...]:
        return tuple(
            sorted(
                (
                    component
                    for component in self.components
                    if component.storage_index is not None
                ),
                key=lambda component: component.storage_index,
            )
        )

    @property
    def transient_components(self) -> tuple[ComponentTransformPlan, ...]:
        """Derived components cached only for one algebraic evaluation."""

        return tuple(
            component
            for component in self.components
            if component.role is FieldRole.TRANSIENT
        )

    @property
    def execution_components(self) -> tuple[ComponentTransformPlan, ...]:
        """Components whose transform spaces participate in execution."""

        return tuple(
            component
            for component in self.components
            if component.role is not FieldRole.DIAGNOSTIC
        )

    @property
    def evolved_component_count(self) -> int:
        return sum(
            component.role is FieldRole.EVOLVED
            for component in self.stored_components
        )

    @property
    def algebraic_component_count(self) -> int:
        return sum(
            component.role is FieldRole.ALGEBRAIC
            for component in self.stored_components
        )

    @property
    def transient_component_count(self) -> int:
        return len(self.transient_components)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": SPECTRAL_PLAN_SCHEMA_VERSION,
            "model": {
                "name": self.model_name,
                "parameters": json.loads(self.model_parameters_json),
            },
            "geometry": {
                "name": self.geometry_name,
                "physical_shape": list(self.physical_shape),
                "spectral_shape": list(self.spectral_shape),
                "axes": [axis.to_metadata() for axis in self.axes],
            },
            "numerics": self.numerics.to_metadata(),
            "fields": [field.to_metadata() for field in self.fields],
            "storage": {
                "component_names": [
                    component.component_name
                    for component in self.stored_components
                ],
                "evolved_component_count": self.evolved_component_count,
                "algebraic_component_count": self.algebraic_component_count,
            },
            "transient": {
                "component_names": [
                    component.component_name
                    for component in self.transient_components
                ],
                "component_count": self.transient_component_count,
                "persistent_storage": False,
            },
        }
