"""Public physical boundary declarations and composition helpers."""

from .homogeneous import (
    HomogeneousBoundaryPolicy,
    assign_boundaries,
    free_slip_velocity,
    no_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)

__all__ = [
    "HomogeneousBoundaryPolicy",
    "assign_boundaries",
    "free_slip_velocity",
    "no_slip_velocity",
    "neumann_pressure_compatibility",
    "neumann_q",
]
