"""Explicit opt-in APIs under architectural migration.

Nothing in this package is imported by the production solver entry points.
"""

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
    LegacyExplicitRHSAdapter,
    LegacyModelExecutionContext,
    build_experimental_model_runtime,
)

__all__ = [
    "LegacyAssemblySpec",
    "LegacyBackendSpec",
    "LegacyExplicitRHSAdapter",
    "LegacyFieldDeclaration",
    "LegacyModelExecutionContext",
    "LegacyProjectorSpec",
    "ExperimentalModelRuntime",
    "build_experimental_model_runtime",
    "create_legacy_projector",
    "create_legacy_solver",
    "declare_legacy_fields",
    "materialize_legacy_assembly",
]
