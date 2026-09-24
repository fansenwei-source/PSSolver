"""Physical boundary-condition contracts.

The contracts describe physical semantics, not transform implementations.
Mapping these objects to FFT, DCT, DST, lifting, or tau methods belongs to a
later spectral-plan assembly stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BoundaryKind(str, Enum):
    """Physical boundary-condition families supported by Stage A."""

    PERIODIC = "periodic"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


class BoundarySide(str, Enum):
    """Oriented side of one tensor-product coordinate axis."""

    LOWER = "lower"
    UPPER = "upper"


class BoundarySemantic(str, Enum):
    """Meaning of a component-to-face boundary declaration.

    ``PHYSICAL`` records a prescribed physical field law.  Pressure-like
    Lagrange multipliers may instead use ``ALGEBRAIC_COMPATIBILITY`` to make
    clear that their modal boundary space is induced by the constrained
    system rather than independently prescribed wall physics.
    """

    PHYSICAL = "physical"
    ALGEBRAIC_COMPATIBILITY = "algebraic_compatibility"


@dataclass(frozen=True, slots=True)
class BoundaryCondition:
    """A homogeneous physical law assignable by coordinate axis or face."""

    kind: BoundaryKind
    is_homogeneous: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryKind):
            raise TypeError("kind must be a BoundaryKind")
        if not isinstance(self.is_homogeneous, bool):
            raise TypeError("is_homogeneous must be a bool")
        if not self.is_homogeneous:
            raise ValueError(
                "Stage A supports homogeneous boundary contracts only"
            )

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description of the condition."""

        return {
            "kind": self.kind.value,
            "is_homogeneous": self.is_homogeneous,
        }


@dataclass(frozen=True, slots=True)
class PeriodicBC(BoundaryCondition):
    """Periodic continuation along one coordinate axis."""

    kind: BoundaryKind = field(default=BoundaryKind.PERIODIC, init=False)
    is_homogeneous: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class HomogeneousDirichletBC(BoundaryCondition):
    """Zero field value on the assigned bounded face or axis ends."""

    kind: BoundaryKind = field(default=BoundaryKind.DIRICHLET, init=False)
    is_homogeneous: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class HomogeneousNeumannBC(BoundaryCondition):
    """Zero normal derivative on the assigned bounded face or axis ends."""

    kind: BoundaryKind = field(default=BoundaryKind.NEUMANN, init=False)
    is_homogeneous: bool = field(default=True, init=False)


@dataclass(frozen=True, slots=True)
class BoundarySet:
    """Per-axis physical boundary conditions for one field component."""

    axes: tuple[BoundaryCondition, ...]

    def __post_init__(self) -> None:
        try:
            axes = tuple(self.axes)
        except TypeError as exc:
            raise TypeError("axes must be an iterable of boundary conditions") from exc
        if not axes:
            raise ValueError("at least one axis boundary is required")
        if not all(isinstance(condition, BoundaryCondition) for condition in axes):
            raise TypeError("every axis entry must be a BoundaryCondition")
        object.__setattr__(self, "axes", axes)

    @property
    def ndim(self) -> int:
        """Number of coordinate axes described by this set."""

        return len(self.axes)

    def to_metadata(self) -> list[dict[str, object]]:
        """Return JSON-compatible per-axis boundary metadata."""

        return [condition.to_metadata() for condition in self.axes]


@dataclass(frozen=True, slots=True)
class FaceBoundaryCondition:
    """One physical or compatibility law on an oriented geometric face."""

    axis: int
    side: BoundarySide
    condition: BoundaryCondition

    def __post_init__(self) -> None:
        if (
            not isinstance(self.axis, int)
            or isinstance(self.axis, bool)
            or self.axis < 0
        ):
            raise ValueError("face axis must be a non-negative integer")
        if not isinstance(self.side, BoundarySide):
            raise TypeError("face side must be a BoundarySide")
        if not isinstance(self.condition, BoundaryCondition):
            raise TypeError("face condition must be a BoundaryCondition")

    def to_metadata(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "side": self.side.value,
            "condition": self.condition.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class ComponentBoundaryAssignment:
    """Boundary laws for one logical scalar field component."""

    component: str
    semantic: BoundarySemantic
    faces: tuple[FaceBoundaryCondition, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.isidentifier():
            raise ValueError("boundary component must be a Python identifier")
        if not isinstance(self.semantic, BoundarySemantic):
            raise TypeError("boundary semantic must be a BoundarySemantic")
        if isinstance(self.faces, (str, bytes)):
            raise TypeError("faces must be an iterable, not a string")
        try:
            faces = tuple(self.faces)
        except TypeError as exc:
            raise TypeError("faces must be iterable") from exc
        if not faces:
            raise ValueError("component boundary faces must not be empty")
        if not all(isinstance(face, FaceBoundaryCondition) for face in faces):
            raise TypeError(
                "component boundary faces must be FaceBoundaryCondition objects"
            )
        face_keys = tuple((face.axis, face.side) for face in faces)
        if len(set(face_keys)) != len(face_keys):
            raise ValueError("component boundary faces must be unique")
        ordered = tuple(
            sorted(faces, key=lambda face: (face.axis, face.side.value))
        )
        object.__setattr__(self, "faces", ordered)

    def to_metadata(self) -> dict[str, object]:
        return {
            "component": self.component,
            "semantic": self.semantic.value,
            "faces": [face.to_metadata() for face in self.faces],
        }


@dataclass(frozen=True, slots=True)
class BoundaryAssignment:
    """Face-aware boundary assignment for a complete logical field set.

    This object remains independent of a particular geometry or equation
    system.  ``SimulationSpec`` performs the cross-object validation before a
    later lowering stage chooses bases, lifting, tau rows, or solvers.
    """

    name: str
    ndim: int
    components: tuple[ComponentBoundaryAssignment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("boundary-assignment name must be a Python identifier")
        if (
            not isinstance(self.ndim, int)
            or isinstance(self.ndim, bool)
            or self.ndim <= 0
        ):
            raise ValueError("boundary-assignment ndim must be positive")
        if isinstance(self.components, (str, bytes)):
            raise TypeError("components must be an iterable, not a string")
        try:
            components = tuple(self.components)
        except TypeError as exc:
            raise TypeError("components must be iterable") from exc
        if not components:
            raise ValueError("boundary assignment must contain components")
        if not all(
            isinstance(value, ComponentBoundaryAssignment)
            for value in components
        ):
            raise TypeError(
                "components must be ComponentBoundaryAssignment objects"
            )
        names = tuple(value.component for value in components)
        if len(set(names)) != len(names):
            raise ValueError("boundary-assignment components must be unique")

        required_faces = {
            (axis, side)
            for axis in range(self.ndim)
            for side in BoundarySide
        }
        for component in components:
            observed = {(face.axis, face.side) for face in component.faces}
            if observed != required_faces:
                raise ValueError(
                    f"component '{component.component}' must assign both faces "
                    f"of every axis; expected={required_faces!r}, "
                    f"observed={observed!r}"
                )
        object.__setattr__(
            self,
            "components",
            tuple(sorted(components, key=lambda value: value.component)),
        )

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(value.component for value in self.components)

    def for_component(self, name: str) -> ComponentBoundaryAssignment:
        for value in self.components:
            if value.component == name:
                return value
        raise KeyError(name)

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ndim": self.ndim,
            "components": [value.to_metadata() for value in self.components],
        }
