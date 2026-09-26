"""Field-neutral declarations for static prescribed Dirichlet data."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
from numbers import Real

from pssolver.core.boundary import (
    BoundarySemantic,
    BoundarySide,
    StaticConstantBoundaryValue,
)


def _side(value: BoundarySide | str) -> BoundarySide:
    if isinstance(value, BoundarySide):
        return value
    try:
        return BoundarySide(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("prescribed face side must be 'lower' or 'upper'") from exc


@dataclass(frozen=True, slots=True)
class PrescribedFaceValue:
    """One static scalar value for one component on one oriented face."""

    component: str
    axis: int
    side: BoundarySide
    value: StaticConstantBoundaryValue

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component.isidentifier():
            raise ValueError("prescribed component must be a Python identifier")
        if (
            not isinstance(self.axis, int)
            or isinstance(self.axis, bool)
            or self.axis < 0
        ):
            raise ValueError("prescribed face axis must be a non-negative integer")
        object.__setattr__(self, "side", _side(self.side))
        if not isinstance(self.value, StaticConstantBoundaryValue):
            object.__setattr__(
                self,
                "value",
                StaticConstantBoundaryValue(self.value),
            )

    @property
    def key(self) -> tuple[str, int, BoundarySide]:
        return (self.component, self.axis, self.side)

    def to_metadata(self) -> dict[str, object]:
        return {
            "component": self.component,
            "axis": self.axis,
            "side": self.side.value,
            "value": self.value.to_metadata(),
            "value_sha256": self.value.canonical_sha256(),
        }


@dataclass(frozen=True, slots=True)
class StaticPrescribedDirichletPolicy:
    """Static prescribed values for every bounded face of one evolved field."""

    field_name: str
    face_values: tuple[PrescribedFaceValue, ...]
    semantic: BoundarySemantic = field(
        default=BoundarySemantic.PHYSICAL,
        init=False,
    )
    kind: str = field(default="prescribed_dirichlet", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name.isidentifier():
            raise ValueError("boundary-policy field_name must be a Python identifier")
        if isinstance(self.face_values, (str, bytes)):
            raise TypeError("face_values must be an iterable, not a string")
        try:
            values = tuple(self.face_values)
        except TypeError as exc:
            raise TypeError("face_values must be iterable") from exc
        if not values:
            raise ValueError("prescribed Dirichlet policy requires face values")
        if not all(isinstance(value, PrescribedFaceValue) for value in values):
            raise TypeError("face_values must contain PrescribedFaceValue objects")
        keys = tuple(value.key for value in values)
        if len(set(keys)) != len(keys):
            raise ValueError("prescribed component-face values must be unique")
        object.__setattr__(
            self,
            "face_values",
            tuple(
                sorted(
                    values,
                    key=lambda value: (
                        value.component,
                        value.axis,
                        value.side.value,
                    ),
                )
            ),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "kind": self.kind,
            "semantic": self.semantic.value,
            "face_values": [value.to_metadata() for value in self.face_values],
        }

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.to_metadata(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def prescribed_dirichlet(
    field_name: str,
    face_values: (
        Mapping[tuple[str, int, BoundarySide | str], Real]
        | Iterable[PrescribedFaceValue]
    ),
) -> StaticPrescribedDirichletPolicy:
    """Construct a static prescribed-Dirichlet field policy.

    Mapping keys are ``(component, axis, side)`` tuples.  Only bounded-face
    entries are supplied; periodic faces are added during composition.
    """

    if isinstance(face_values, Mapping):
        values = tuple(
            PrescribedFaceValue(component, axis, _side(side), value)
            for (component, axis, side), value in face_values.items()
        )
    else:
        values = tuple(face_values)
    return StaticPrescribedDirichletPolicy(field_name, values)


__all__ = [
    "PrescribedFaceValue",
    "StaticPrescribedDirichletPolicy",
    "prescribed_dirichlet",
]
