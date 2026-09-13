"""Execution-facing contracts independent of the legacy solver objects."""

from .algebraic import (
    AlgebraicExecutableModelProtocol,
    AlgebraicRuntimeRestartState,
    AlgebraicSolverContext,
    AlgebraicSolverProtocol,
    AlgebraicSystemRestartState,
    AlgebraicSystemSpec,
    AlgebraicUpdatePhase,
    InspectableAlgebraicSolverProtocol,
)
from .contracts import ExecutableModelProtocol, ModelExecutionContext
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
    "AlgebraicRuntimeRestartState",
    "AlgebraicSolverContext",
    "AlgebraicSolverFactory",
    "AlgebraicSolverProtocol",
    "AlgebraicSolverRegistration",
    "AlgebraicSystemRestartState",
    "AlgebraicSystemSpec",
    "AlgebraicUpdatePhase",
    "ExecutableModelProtocol",
    "GeometrySolverRegistry",
    "INCOMPRESSIBLE_STOKES_CAPABILITY",
    "IncompressibleStokesSystemSpec",
    "InspectableAlgebraicSolverProtocol",
    "ModelExecutionContext",
    "PressureGauge",
    "TangentialZeroModePolicy",
]
