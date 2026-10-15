from .solver import SpectralSolver
from ._version import __version__
from .Field import (
    DEFAULT_TRANSFORM_GROUP_INDEXING,
    TRANSFORM_GROUP_INDEXING_MODES,
    Fields,
    Parameters,
)
from .PDEmodel import PDEModel
from .integrator import SemiImplicitEulerIntegrator
from .linear_solvers.stokes import ChannelPressureConvergenceError
from .loop_conventions import (
    CONVENTION_NAME,
    IDEAL_LOOP_MODE_AXES,
    classify_ideal_loop,
    convention_metadata,
    ideal_loop_axis_angles,
)
from .run_metadata import prepare_new_run_directory, write_run_metadata
from .snapshots import (
    SimulationSnapshot,
    REPRESENTATIVE_ORDERED_S_DEFINITION,
    apply_snapshot_to_solver,
    load_snapshot,
    representative_ordered_S,
    require_distinct_output_directory,
)
from .api import (
    CompiledSimulation,
    DeclarationCapability,
    GeneratedInitialCondition,
    Output,
    Simulation,
    SimulationDiagnosticProtocol,
    SimulationExecutionError,
    SimulationRestartError,
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
    SimulationObservationProtocol,
    SimulationResult,
    SimulationRunStatus,
    SimulationStateError,
    PublicCapabilityCatalog,
    QualifiedCombination,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
    available_boundary_policies,
    available_combinations,
    available_geometries,
    available_models,
    capability_catalog,
    run_simulation,
)
from .transforms import (
    BasisAwareSpectralProjector,
    DEALIAS_RULE_FRACTIONS,
    DEFAULT_DEALIAS_RULE,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
    DEFAULT_PERIODIC_TRANSFORM_EXECUTION,
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    FreeSlipModalStokesSolver,
    TensorProductTransformBackend,
    PROJECTED_TRANSFORM_EXECUTION_MODES,
    PERIODIC_TRANSFORM_EXECUTION_MODES,
    SPECTRAL_STORAGE_MODES,
    projected_common_basis_stress_divergence,
    projected_distortion_stress_divergence,
)

__all__ = [
    '__version__',
    'SpectralSolver',
    'Fields',
    'Parameters',
    'DEFAULT_TRANSFORM_GROUP_INDEXING',
    'TRANSFORM_GROUP_INDEXING_MODES',
    'PDEModel',
    'SemiImplicitEulerIntegrator',
    'CONVENTION_NAME',
    'IDEAL_LOOP_MODE_AXES',
    'classify_ideal_loop',
    'convention_metadata',
    'ideal_loop_axis_angles',
    'BasisAwareSpectralProjector',
    'DEALIAS_RULE_FRACTIONS',
    'DEFAULT_DEALIAS_RULE',
    'DEFAULT_PROJECTED_TRANSFORM_EXECUTION',
    'DEFAULT_PERIODIC_TRANSFORM_EXECUTION',
    'DEFAULT_TRANSFORM_EXECUTION_ORDER',
    'FreeSlipModalStokesSolver',
    'TensorProductTransformBackend',
    'PROJECTED_TRANSFORM_EXECUTION_MODES',
    'PERIODIC_TRANSFORM_EXECUTION_MODES',
    'SPECTRAL_STORAGE_MODES',
    'projected_common_basis_stress_divergence',
    'projected_distortion_stress_divergence',
    'prepare_new_run_directory',
    'write_run_metadata',
    'SimulationSnapshot',
    'REPRESENTATIVE_ORDERED_S_DEFINITION',
    'apply_snapshot_to_solver',
    'load_snapshot',
    'representative_ordered_S',
    'require_distinct_output_directory',
    'Simulation',
    'GeneratedInitialCondition',
    'Output',
    'SnapshotInitialCondition',
    'SpectralNumerics',
    'TimeStepping',
    'TorchSpectralExecution',
    'CompiledSimulation',
    'compile_simulation',
    'run_simulation',
    'SimulationDiagnosticProtocol',
    'SimulationObservationProtocol',
    'SimulationResult',
    'SimulationRunStatus',
    'SimulationExecutionError',
    'SimulationRestartError',
    'PublicCompilationRejectionCode',
    'PublicSimulationCompilationError',
    'SimulationStateError',
    'DeclarationCapability',
    'PublicCapabilityCatalog',
    'QualifiedCombination',
    'available_boundary_policies',
    'available_combinations',
    'available_geometries',
    'available_models',
    'capability_catalog',
    'ChannelPressureConvergenceError',
]


def __getattr__(name):
    if name == "DEFAULT_SPECTRAL_STORAGE":
        import warnings

        warnings.warn(
            "pssolver.DEFAULT_SPECTRAL_STORAGE is deprecated because public "
            "simulations require an explicit spectral_storage policy; import "
            "the legacy transform default from pssolver.transforms if needed",
            DeprecationWarning,
            stacklevel=2,
        )
        from .transforms import DEFAULT_SPECTRAL_STORAGE

        return DEFAULT_SPECTRAL_STORAGE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
