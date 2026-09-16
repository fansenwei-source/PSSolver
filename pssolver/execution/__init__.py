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
    ExplicitRHSExecutorProtocol,
    ExplicitRHSPhysicalDependenciesProtocol,
    ExecutableModelProtocol,
    MathematicalOperatorContext,
    ModelExecutionContext,
)
from .dispatch import (
    AlgebraicSolverFactory,
    AlgebraicSolverRegistration,
    ExplicitRHSExecutorFactory,
    ExplicitRHSExecutorRegistration,
    GeometrySolverRegistry,
)
from .policy import (
    AlgebraicExecutionMode,
    AlgebraicExecutionPolicy,
    AlgebraicOutputPublicationMode,
    AlgebraicOutputPublicationPolicy,
    resolve_algebraic_execution_policy,
)
from .stokes import (
    INCOMPRESSIBLE_STOKES_CAPABILITY,
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)

__all__ = [
    "AlgebraicExecutableModelProtocol",
    "AlgebraicExecutionMode",
    "AlgebraicExecutionPolicy",
    "AlgebraicOutputPublicationMode",
    "AlgebraicOutputPublicationPolicy",
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
    "ExplicitRHSExecutorFactory",
    "ExplicitRHSExecutorProtocol",
    "ExplicitRHSExecutorRegistration",
    "ExplicitRHSPhysicalDependenciesProtocol",
    "GeometrySolverRegistry",
    "INCOMPRESSIBLE_STOKES_CAPABILITY",
    "IncompressibleStokesSystemSpec",
    "InspectableAlgebraicSolverProtocol",
    "ModelExecutionContext",
    "MathematicalOperatorContext",
    "PressureGauge",
    "resolve_algebraic_execution_policy",
    "TangentialZeroModePolicy",
]
