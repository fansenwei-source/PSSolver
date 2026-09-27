"""Field-neutral declarations for static per-face Robin laws."""

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
    StaticRobinCoefficients,
)


RobinCoefficientInput = (
    StaticRobinCoefficients | tuple[Real, Real, Real]
)


def _side(value: BoundarySide | str) -> BoundarySide:
    if isinstance(value, BoundarySide):
        return value
    try:
        return BoundarySide(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Robin face side must be 'lower' or 'upper'") from exc


def _coefficients(value: object) -> StaticRobinCoefficients:
    if isinstance(value, StaticRobinCoefficients):
        return value
    if isinstance(value, (str, bytes)):
        raise TypeError(
            "Robin coefficients must be a three-item real sequence"
        )
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(
            "Robin coefficients must be a three-item real sequence"
        ) from exc
    if len(values) != 3:
        raise ValueError(
            "Robin coefficients must contain alpha, beta, and gamma"
        )
    return StaticRobinCoefficients(*values)


@dataclass(frozen=True, slots=True, init=False)
class RobinFaceLaw:
    """One static Robin coefficient triple on one component and face."""

    component: str
    axis: int
    side: BoundarySide
    coefficients: StaticRobinCoefficients

    def __init__(
        self,
        component: str,
        axis: int,
        side: BoundarySide | str,
        coefficients: RobinCoefficientInput,
    ) -> None:
        if not isinstance(component, str) or not component.isidentifier():
            raise ValueError("Robin component must be a Python identifier")
        if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0:
            raise ValueError("Robin face axis must be a non-negative integer")
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "side", _side(side))
        object.__setattr__(self, "coefficients", _coefficients(coefficients))

    @property
    def key(self) -> tuple[str, int, BoundarySide]:
        return (self.component, self.axis, self.side)

    def to_metadata(self) -> dict[str, object]:
        return {
            "component": self.component,
            "axis": self.axis,
            "side": self.side.value,
            "coefficients": self.coefficients.to_metadata(),
            "coefficients_sha256": self.coefficients.canonical_sha256(),
        }


@dataclass(frozen=True, slots=True)
class StaticRobinBoundaryPolicy:
    """Static Robin laws for every bounded face of one evolved field."""

    field_name: str
    face_laws: tuple[RobinFaceLaw, ...]
    semantic: BoundarySemantic = field(
        default=BoundarySemantic.PHYSICAL,
        init=False,
    )
    kind: str = field(default="robin", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name.isidentifier():
            raise ValueError("boundary-policy field_name must be a Python identifier")
        if isinstance(self.face_laws, (str, bytes)):
            raise TypeError("face_laws must be an iterable, not a string")
        try:
            laws = tuple(self.face_laws)
        except TypeError as exc:
            raise TypeError("face_laws must be iterable") from exc
        if not laws:
            raise ValueError("static Robin policy requires face laws")
        if not all(isinstance(value, RobinFaceLaw) for value in laws):
            raise TypeError("face_laws must contain RobinFaceLaw objects")
        keys = tuple(value.key for value in laws)
        if len(set(keys)) != len(keys):
            raise ValueError("Robin component-face laws must be unique")
        object.__setattr__(
            self,
            "face_laws",
            tuple(
                sorted(
                    laws,
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
            "canonical_form": "alpha*phi+beta*(n_dot_grad_phi)=gamma",
            "normal_derivative_convention": "outward_unit_normal",
            "face_laws": [value.to_metadata() for value in self.face_laws],
        }

    def canonical_sha256(self) -> str:
        payload = json.dumps(
            self.to_metadata(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def robin(
    field_name: str,
    face_laws: (
        Mapping[
            tuple[str, int, BoundarySide | str],
            RobinCoefficientInput,
        ]
        | Iterable[RobinFaceLaw]
    ),
) -> StaticRobinBoundaryPolicy:
    """Construct a static field-neutral Robin policy.

    Mapping keys are ``(component, axis, side)`` and values are either
    :class:`StaticRobinCoefficients` or ``(alpha, beta, gamma)`` tuples.
    Only bounded faces are declared; periodic faces are topology-induced
    during composition.
    """

    if isinstance(face_laws, Mapping):
        values = tuple(
            RobinFaceLaw(component, axis, side, coefficients)
            for (component, axis, side), coefficients in face_laws.items()
        )
    else:
        values = tuple(face_laws)
    return StaticRobinBoundaryPolicy(field_name, values)


__all__ = [
    "RobinFaceLaw",
    "StaticRobinBoundaryPolicy",
    "robin",
]
