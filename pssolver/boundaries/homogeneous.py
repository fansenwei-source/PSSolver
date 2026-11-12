"""Typed homogeneous physical and compatibility boundary policies."""

from dataclasses import dataclass

from pssolver.core.boundary import BoundarySemantic

from .prescribed import StaticPrescribedDirichletPolicy
from .robin import StaticRobinBoundaryPolicy


@dataclass(frozen=True, slots=True)
class HomogeneousBoundaryPolicy:
    """A typed field-level policy expanded only during composition."""

    field_name: str
    semantic: BoundarySemantic
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.field_name, str) or not self.field_name.isidentifier():
            raise ValueError("boundary-policy field_name must be a Python identifier")
        if not isinstance(self.semantic, BoundarySemantic):
            raise TypeError("boundary-policy semantic must be a BoundarySemantic")
        if self.kind not in {
            "neumann",
            "free_slip_velocity",
            "no_slip_velocity",
        }:
            raise ValueError("unsupported homogeneous boundary policy")


BoundaryPolicy = (
    HomogeneousBoundaryPolicy
    | StaticPrescribedDirichletPolicy
    | StaticRobinBoundaryPolicy
)


def neumann_q() -> HomogeneousBoundaryPolicy:
    """Return homogeneous Neumann anchoring for the logical Q field."""

    return HomogeneousBoundaryPolicy(
        field_name="Q",
        semantic=BoundarySemantic.PHYSICAL,
        kind="neumann",
    )


def free_slip_velocity() -> HomogeneousBoundaryPolicy:
    """Return no-penetration/free-slip conditions for vector velocity."""

    return HomogeneousBoundaryPolicy(
        field_name="velocity",
        semantic=BoundarySemantic.PHYSICAL,
        kind="free_slip_velocity",
    )


def no_slip_velocity() -> HomogeneousBoundaryPolicy:
    """Return zero velocity for every component on bounded faces."""

    return HomogeneousBoundaryPolicy(
        field_name="velocity",
        semantic=BoundarySemantic.PHYSICAL,
        kind="no_slip_velocity",
    )


def neumann_pressure_compatibility() -> HomogeneousBoundaryPolicy:
    """Return Neumann modal compatibility for the pressure multiplier."""

    return HomogeneousBoundaryPolicy(
        field_name="pressure",
        semantic=BoundarySemantic.ALGEBRAIC_COMPATIBILITY,
        kind="neumann",
    )


__all__ = [
    "BoundaryPolicy",
    "HomogeneousBoundaryPolicy",
    "free_slip_velocity",
    "neumann_pressure_compatibility",
    "neumann_q",
    "no_slip_velocity",
]
