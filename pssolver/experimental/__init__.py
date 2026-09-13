"""Explicit opt-in APIs under architectural migration.

Nothing in this package is imported by the production solver entry points.
"""

from .canary_algebraic import create_canary_geometry_solver_registry
from .legacy_assembly import (
    LegacyAssemblySpec,
    LegacyBackendSpec,
    LegacyFieldDeclaration,
    LegacyProjectorSpec,
    create_legacy_projector,
    create_legacy_solver,
    declare_legacy_fields,
    materialize_legacy_assembly,
)
from .model_execution import (
    ExperimentalModelRuntime,
    LegacyAlgebraicFieldsAdapter,
    LegacyAlgebraicSolverContext,
    LegacyExplicitRHSAdapter,
    LegacyModelExecutionContext,
    ResolvedAlgebraicSystem,
    build_experimental_model_runtime,
)
from .stokes import (
    ChannelStokesSolverOptions,
    PlaneStokesSolverOptions,
    create_stokes_geometry_solver_registry,
)

__all__ = [
    "ExperimentalModelRuntime",
    "ChannelStokesSolverOptions",
    "LegacyAlgebraicFieldsAdapter",
    "LegacyAlgebraicSolverContext",
    "LegacyAssemblySpec",
    "LegacyBackendSpec",
    "LegacyExplicitRHSAdapter",
    "LegacyFieldDeclaration",
    "LegacyModelExecutionContext",
    "LegacyProjectorSpec",
    "PlaneStokesSolverOptions",
    "ResolvedAlgebraicSystem",
    "build_experimental_model_runtime",
    "create_canary_geometry_solver_registry",
    "create_legacy_projector",
    "create_legacy_solver",
    "create_stokes_geometry_solver_registry",
    "declare_legacy_fields",
    "materialize_legacy_assembly",
]
