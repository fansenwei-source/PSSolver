"""Public declarations and lazy entry points for PSSolver."""

from ._version import __version__


_EXPORTS = {
    "ChannelPressureConvergenceError": "pssolver.linear_solvers.stokes",
    "CompiledSimulation": "pssolver.api.runner",
    "CompleteStressBerisEdwards": "pssolver.api.models",
    "DeclarationCapability": "pssolver.api.capabilities",
    "GeneratedInitialCondition": "pssolver.api.declarations",
    "Output": "pssolver.api.declarations",
    "PeriodicBox": "pssolver.geometries.public",
    "PlaneSlab": "pssolver.geometries.public",
    "PublicCapabilityCatalog": "pssolver.api.capabilities",
    "PublicCompilationRejectionCode": "pssolver.api.errors",
    "PublicSimulationCompilationError": "pssolver.api.errors",
    "QualifiedCombination": "pssolver.api.capabilities",
    "RectangularChannel": "pssolver.geometries.public",
    "SemiImplicitEuler": "pssolver.api.declarations",
    "ShendrukPlaneBerisEdwards": "pssolver.api.models",
    "Simulation": "pssolver.api.simulation",
    "SimulationDiagnosticProtocol": "pssolver.api.results",
    "SimulationExecutionError": "pssolver.errors",
    "SimulationInputError": "pssolver.errors",
    "SimulationInputFileNotFoundError": "pssolver.errors",
    "SimulationInputIOError": "pssolver.errors",
    "SimulationObservationProtocol": "pssolver.api.results",
    "SimulationRestartError": "pssolver.errors",
    "SimulationResult": "pssolver.api.results",
    "SimulationRunStatus": "pssolver.api.results",
    "SimulationStateError": "pssolver.errors",
    "SnapshotInitialCondition": "pssolver.api.declarations",
    "SpectralNumerics": "pssolver.api.declarations",
    "TorchSpectralExecution": "pssolver.api.declarations",
    "available_boundary_policies": "pssolver.api.capabilities",
    "available_combinations": "pssolver.api.capabilities",
    "available_geometries": "pssolver.api.capabilities",
    "available_models": "pssolver.api.capabilities",
    "capability_catalog": "pssolver.api.capabilities",
    "compile_simulation": "pssolver.api.runner",
    "free_slip_velocity": "pssolver.boundaries.homogeneous",
    "neumann_pressure_compatibility": "pssolver.boundaries.homogeneous",
    "neumann_q": "pssolver.boundaries.homogeneous",
    "no_slip_velocity": "pssolver.boundaries.homogeneous",
    "run_simulation": "pssolver.api.runner",
}

__all__ = [
    "ChannelPressureConvergenceError",
    "CompiledSimulation",
    "CompleteStressBerisEdwards",
    "DeclarationCapability",
    "GeneratedInitialCondition",
    "Output",
    "PeriodicBox",
    "PlaneSlab",
    "PublicCapabilityCatalog",
    "PublicCompilationRejectionCode",
    "PublicSimulationCompilationError",
    "QualifiedCombination",
    "RectangularChannel",
    "SemiImplicitEuler",
    "ShendrukPlaneBerisEdwards",
    "Simulation",
    "SimulationDiagnosticProtocol",
    "SimulationExecutionError",
    "SimulationInputError",
    "SimulationInputFileNotFoundError",
    "SimulationInputIOError",
    "SimulationObservationProtocol",
    "SimulationRestartError",
    "SimulationResult",
    "SimulationRunStatus",
    "SimulationStateError",
    "SnapshotInitialCondition",
    "SpectralNumerics",
    "TorchSpectralExecution",
    "__version__",
    "available_boundary_policies",
    "available_combinations",
    "available_geometries",
    "available_models",
    "capability_catalog",
    "compile_simulation",
    "free_slip_velocity",
    "neumann_pressure_compatibility",
    "neumann_q",
    "no_slip_velocity",
    "run_simulation",
]


def __getattr__(name):
    from importlib import import_module

    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return list(__all__)
