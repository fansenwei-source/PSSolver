"""Execution-facing contracts independent of the legacy solver objects."""

from .algebraic import (
    AlgebraicExecutableModelProtocol,
    AlgebraicSolverContext,
    AlgebraicSolverProtocol,
    AlgebraicSystemSpec,
    AlgebraicUpdatePhase,
)
from .contracts import ExecutableModelProtocol, ModelExecutionContext
from .dispatch import (
    AlgebraicSolverFactory,
    AlgebraicSolverRegistration,
    GeometrySolverRegistry,
)

__all__ = [
    "AlgebraicExecutableModelProtocol",
    "AlgebraicSolverContext",
    "AlgebraicSolverFactory",
    "AlgebraicSolverProtocol",
    "AlgebraicSolverRegistration",
    "AlgebraicSystemSpec",
    "AlgebraicUpdatePhase",
    "ExecutableModelProtocol",
    "GeometrySolverRegistry",
    "ModelExecutionContext",
]
