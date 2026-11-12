"""Frozen v0.4 declaration vocabulary for Functional API 1.1."""

from pssolver.boundaries.homogeneous import (
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.geometries.public import PeriodicBox, PlaneSlab, RectangularChannel

from .active_nematics_public import CompleteStressBerisEdwards
from .api_declarations import (
    GeneratedInitialCondition,
    Output,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from .api_simulation import Simulation
from .boundary_assignment import assign_boundaries
from .simulation import ExecutionSpec, InvocationSpec, SimulationSpec


__all__ = [
    "CompleteStressBerisEdwards",
    "ExecutionSpec",
    "GeneratedInitialCondition",
    "InvocationSpec",
    "Output",
    "PeriodicBox",
    "PlaneSlab",
    "RectangularChannel",
    "Simulation",
    "SimulationSpec",
    "SnapshotInitialCondition",
    "SpectralNumerics",
    "TimeStepping",
    "TorchSpectralExecution",
    "assign_boundaries",
    "free_slip_velocity",
    "neumann_pressure_compatibility",
    "neumann_q",
    "no_slip_velocity",
]
