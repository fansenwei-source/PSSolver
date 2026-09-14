"""Execution-facing contracts independent of the legacy solver objects."""

from .algebraic import (
    AlgebraicDependencyEdge,
    AlgebraicExecutionPlan,
    AlgebraicExecutableModelProtocol,
    AlgebraicRuntimeRestartState,
    AlgebraicSolverContext,
    AlgebraicPhysicalDependenciesProtocol,
    AlgebraicSolverProtocol,
    AlgebraicSystemRestartState,
    AlgebraicSystemSpec,
    AlgebraicUpdatePhase,
    build_algebraic_execution_plan,
    InspectableAlgebraicSolverProtocol,
)
from .contracts import (
    ExplicitRHSPhysicalDependenciesProtocol,
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
    "AlgebraicPhysicalDependenciesProtocol",
    "AlgebraicSolverFactory",
    "AlgebraicSolverProtocol",
    "AlgebraicSolverRegistration",
    "AlgebraicSystemRestartState",
    "AlgebraicSystemSpec",
    "AlgebraicUpdatePhase",
    "build_algebraic_execution_plan",
    "ExecutableModelProtocol",
    "ExplicitRHSPhysicalDependenciesProtocol",
    "GeometrySolverRegistry",
    "INCOMPRESSIBLE_STOKES_CAPABILITY",
    "IncompressibleStokesSystemSpec",
    "InspectableAlgebraicSolverProtocol",
    "ModelExecutionContext",
    "MathematicalOperatorContext",
    "PressureGauge",
    "TangentialZeroModePolicy",
]
