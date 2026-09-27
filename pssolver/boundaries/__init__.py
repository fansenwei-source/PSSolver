"""Public physical boundary declarations and composition helpers."""

from .homogeneous import (
    BoundaryPolicy,
    HomogeneousBoundaryPolicy,
    assign_boundaries,
    free_slip_velocity,
    no_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from .prescribed import (
    PrescribedFaceValue,
    StaticPrescribedDirichletPolicy,
    prescribed_dirichlet,
)
from .robin import (
    RobinFaceLaw,
    StaticRobinBoundaryPolicy,
    robin,
)

__all__ = [
    "BoundaryPolicy",
    "HomogeneousBoundaryPolicy",
    "PrescribedFaceValue",
    "RobinFaceLaw",
    "StaticPrescribedDirichletPolicy",
    "StaticRobinBoundaryPolicy",
    "assign_boundaries",
    "free_slip_velocity",
    "no_slip_velocity",
    "neumann_pressure_compatibility",
    "neumann_q",
    "prescribed_dirichlet",
    "robin",
]
