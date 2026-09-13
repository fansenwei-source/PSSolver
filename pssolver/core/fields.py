"""Declarative field contracts, separate from runtime tensor storage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .boundary import BoundarySet


class FieldRole(str, Enum):
    """How a field participates in the evolution problem."""

    EVOLVED = "evolved"
    ALGEBRAIC = "algebraic"
    TRANSIENT = "transient"
    DIAGNOSTIC = "diagnostic"


def _validate_name(name: str, description: str) -> None:
    if not isinstance(name, str) or not name or not name.isidentifier():
        raise ValueError(f"{description} must be a non-empty Python identifier")


@dataclass(frozen=True, slots=True)
class FieldComponentSpec:
    """One scalar component and its physical boundary conditions."""

    name: str
    boundaries: BoundarySet

    def __post_init__(self) -> None:
        _validate_name(self.name, "component name")
        if not isinstance(self.boundaries, BoundarySet):
            raise TypeError("boundaries must be a BoundarySet")

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible component description."""

        return {
            "name": self.name,
            "boundaries": self.boundaries.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Logical field declaration with component-wise boundary semantics."""

    name: str
    role: FieldRole
    components: tuple[FieldComponentSpec, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name, "field name")
        if not isinstance(self.role, FieldRole):
            raise TypeError("role must be a FieldRole")
        try:
            components = tuple(self.components)
        except TypeError as exc:
            raise TypeError("components must be iterable") from exc
        if not components:
            raise ValueError("a field must contain at least one component")
        if not all(isinstance(component, FieldComponentSpec) for component in components):
            raise TypeError("every component must be a FieldComponentSpec")
        names = tuple(component.name for component in components)
        if len(set(names)) != len(names):
            raise ValueError("component names must be unique within a field")
        dimensions = {component.boundaries.ndim for component in components}
        if len(dimensions) != 1:
            raise ValueError("all field components must have the same dimension")
        object.__setattr__(self, "components", components)

    @classmethod
    def scalar(
        cls,
        name: str,
        role: FieldRole,
        boundaries: BoundarySet,
    ) -> "FieldSpec":
        """Construct a one-component field using its logical name."""

        return cls(
            name=name,
            role=role,
            components=(FieldComponentSpec(name, boundaries),),
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        """Component names in deterministic declaration order."""

        return tuple(component.name for component in self.components)

    @property
    def ndim(self) -> int:
        """Spatial dimension shared by all components."""

        return self.components[0].boundaries.ndim

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible field description."""

        return {
            "name": self.name,
            "role": self.role.value,
            "components": [
                component.to_metadata() for component in self.components
            ],
        }
