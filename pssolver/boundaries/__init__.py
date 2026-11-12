"""Public physical boundary declarations and composition helpers."""

from .homogeneous import (
    BoundaryPolicy,
    HomogeneousBoundaryPolicy,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
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
    "free_slip_velocity",
    "no_slip_velocity",
    "neumann_pressure_compatibility",
    "neumann_q",
    "prescribed_dirichlet",
    "robin",
]
