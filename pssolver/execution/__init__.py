"""Execution-facing contracts independent of the legacy solver objects."""

from .algebraic import (
    AlgebraicDependencyEdge,
    AlgebraicExecutionPlan,
    AlgebraicExecutableModelProtocol,
    AlgebraicRuntimeRestartState,
    AlgebraicSolverContext,
    AlgebraicSolverProtocol,
    AlgebraicSystemRestartState,
    AlgebraicSystemSpec,
    AlgebraicUpdatePhase,
    build_algebraic_execution_plan,
    InspectableAlgebraicSolverProtocol,
)
from .contracts import (
    ExecutableModelProtocol,
    MathematicalOperatorContext,
    ModelExecutionContext,
)
from .dispatch import (
    AlgebraicSolverFactory,
    AlgebraicSolverRegistration,
    GeometrySolverRegistry,
)
from .stokes import (
    INCOMPRESSIBLE_STOKES_CAPABILITY,
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)

__all__ = [
    "AlgebraicExecutableModelProtocol",
    "AlgebraicDependencyEdge",
    "AlgebraicExecutionPlan",
    "AlgebraicRuntimeRestartState",
    "AlgebraicSolverContext",
    "AlgebraicSolverFactory",
    "AlgebraicSolverProtocol",
    "AlgebraicSolverRegistration",
    "AlgebraicSystemRestartState",
    "AlgebraicSystemSpec",
    "AlgebraicUpdatePhase",
    "build_algebraic_execution_plan",
    "ExecutableModelProtocol",
    "GeometrySolverRegistry",
    "INCOMPRESSIBLE_STOKES_CAPABILITY",
    "IncompressibleStokesSystemSpec",
    "InspectableAlgebraicSolverProtocol",
    "ModelExecutionContext",
    "MathematicalOperatorContext",
    "PressureGauge",
    "TangentialZeroModePolicy",
]
