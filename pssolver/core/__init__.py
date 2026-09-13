"""Stable architectural contracts for spectral-first problem assembly.

Stage A intentionally leaves the existing solver runtime untouched.  These
objects describe a problem; later migration stages will compile them into the
already-qualified tensor and transform execution paths.
"""

from .boundary import (
    BoundaryCondition,
    BoundaryKind,
    BoundarySet,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    PeriodicBC,
)
from .domain import DomainSpec, GridPlacement
from .fields import FieldComponentSpec, FieldRole, FieldSpec
from .geometry import AxisTopology, GeometrySpec
from .model import ModelProtocol
from .numerics import (
    DealiasRule,
    NumericsConfig,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from .problem import ProblemSpec

__all__ = [
    "AxisTopology",
    "BoundaryCondition",
    "BoundaryKind",
    "BoundarySet",
    "DealiasRule",
    "DomainSpec",
    "FieldComponentSpec",
    "FieldRole",
    "FieldSpec",
    "GeometrySpec",
    "GridPlacement",
    "HomogeneousDirichletBC",
    "HomogeneousNeumannBC",
    "ModelProtocol",
    "NumericsConfig",
    "PeriodicBC",
    "Precision",
    "ProblemSpec",
    "ProjectedTransformExecution",
    "SpectralStorage",
    "TransformExecutionOrder",
]
