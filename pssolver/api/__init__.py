"""Supported high-level declaration API for complete simulations."""

from .capabilities import (
    DeclarationCapability,
    PublicCapabilityCatalog,
    QualifiedCombination,
    available_boundary_policies,
    available_combinations,
    available_geometries,
    available_models,
    capability_catalog,
)
from .declarations import (
    GeneratedInitialCondition,
    Output,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from .simulation import Simulation
from .runner import CompiledSimulation, compile_simulation, run_simulation
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)
from pssolver.errors import (
    SimulationExecutionError,
    SimulationInputError,
    SimulationRestartError,
    SimulationStateError,
)
from .results import (
    SimulationDiagnosticProtocol,
    SimulationObservationProtocol,
    SimulationResult,
    SimulationRunStatus,
)

__all__ = [
    "CompiledSimulation",
    "DeclarationCapability",
    "GeneratedInitialCondition",
    "Output",
    "PublicCapabilityCatalog",
    "PublicCompilationRejectionCode",
    "PublicSimulationCompilationError",
    "QualifiedCombination",
    "Simulation",
    "SimulationExecutionError",
    "SimulationInputError",
    "SimulationRestartError",
    "SnapshotInitialCondition",
    "SpectralNumerics",
    "TimeStepping",
    "TorchSpectralExecution",
    "available_boundary_policies",
    "available_combinations",
    "available_geometries",
    "available_models",
    "capability_catalog",
    "compile_simulation",
    "run_simulation",
    "SimulationDiagnosticProtocol",
    "SimulationObservationProtocol",
    "SimulationResult",
    "SimulationRunStatus",
    "SimulationStateError",
]
