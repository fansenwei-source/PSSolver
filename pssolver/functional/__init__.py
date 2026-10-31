"""Deprecated compatibility facade for functional execution.

New consumers must import the stable Functional API 1.1 surface from
``pssolver.functional.api``.  The broader facade remains source-compatible
through every v0.4 release.  A deprecation warning is emitted only when a
facade name is accessed; importing ``pssolver.functional.api`` itself stays
silent.
"""

from __future__ import annotations

from importlib import import_module as _import_module

from pssolver.core._compatibility import (
    facade_dir as _facade_dir,
    resolve_deprecated_facade_name as _resolve_deprecated_facade_name,
)


_STABLE_API_EXPORT_NAMES = (
    "CHANNEL_ACTIVITY_RUNTIME_STAGE",
    "CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION",
    "FUNCTIONAL_API_VERSION",
    "FUNCTIONAL_CHECKPOINT_READ_API_VERSIONS",
    "FUNCTIONAL_COMPATIBILITY_POLICY_VERSION",
    "FUNCTIONAL_CONSTRUCTION_API_VERSIONS",
    "FUNCTIONAL_LEGACY_CHECKPOINT_API_VERSIONS",
    "FUNCTIONAL_OBSERVATION_TIME",
    "ChannelPressureSolveDiagnostics",
    "FunctionalAPIVersionSelection",
    "FunctionalCapabilitySet",
    "FunctionalCheckpointBridgeProtocol",
    "FunctionalCheckpointCompatibility",
    "FunctionalCheckpointCompatibilityError",
    "FunctionalCheckpointError",
    "FunctionalCheckpointExistsError",
    "FunctionalCheckpointIntegrityError",
    "FunctionalCheckpointNotFoundError",
    "FunctionalCheckpointState",
    "FunctionalContractError",
    "FunctionalControlFieldSpec",
    "FunctionalControls",
    "FunctionalConvergenceError",
    "FunctionalError",
    "FunctionalExecutionError",
    "FunctionalIdentityError",
    "FunctionalObservationSpec",
    "FunctionalObservations",
    "FunctionalRuntimeConstructionRequest",
    "FunctionalRuntimeDeclaration",
    "FunctionalRuntimeFactoryProtocol",
    "FunctionalRuntimeIdentity",
    "FunctionalRuntimeProtocol",
    "FunctionalState",
    "FunctionalStateSpec",
    "FunctionalTensorSpec",
    "FunctionalTypeError",
    "FunctionalValueError",
    "FunctionalVersionError",
    "functional_protocol_provenance",
    "negotiate_functional_api_version",
)


_COMPATIBILITY_EXPORTS = {
    name: ("pssolver.functional.api", name)
    for name in _STABLE_API_EXPORT_NAMES
}
_COMPATIBILITY_EXPORTS.update(
    {
        "CHANNEL_ACTIVITY_FUNCTIONAL_KIND": (
            "pssolver.functional.channel_activity",
            "CHANNEL_ACTIVITY_FUNCTIONAL_KIND",
        ),
        "CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT": (
            "pssolver.functional.channel_activity",
            "CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT",
        ),
        "CHANNEL_FUNCTIONAL_PRESSURE_WARM_START": (
            "pssolver.functional.channel_activity",
            "CHANNEL_FUNCTIONAL_PRESSURE_WARM_START",
        ),
        "channel_activity_functional_declaration": (
            "pssolver.functional.channel_activity",
            "channel_activity_functional_declaration",
        ),
        "channel_activity_functional_request": (
            "pssolver.functional.channel_activity",
            "channel_activity_functional_request",
        ),
        "CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION": (
            "pssolver.functional.channel_pressure",
            "CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION",
        ),
        "ChannelPressureTransposeOperator": (
            "pssolver.functional.channel_pressure",
            "ChannelPressureTransposeOperator",
        ),
        "ChannelPressureTransposeProtocol": (
            "pssolver.functional.channel_pressure",
            "ChannelPressureTransposeProtocol",
        ),
        "CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION": (
            "pssolver.functional.channel_pressure_adjoint",
            "CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION",
        ),
        "CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES": (
            "pssolver.functional.channel_pressure_adjoint",
            "CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES",
        ),
        "ChannelImplicitPressureAdjoint": (
            "pssolver.functional.channel_pressure_adjoint",
            "ChannelImplicitPressureAdjoint",
        ),
        "ChannelImplicitPressureAdjointProtocol": (
            "pssolver.functional.channel_pressure_adjoint",
            "ChannelImplicitPressureAdjointProtocol",
        ),
        "unrolled_channel_pressure_solve_oracle": (
            "pssolver.functional.channel_pressure_adjoint",
            "unrolled_channel_pressure_solve_oracle",
        ),
        "CHANNEL_ACTIVITY_DETERMINISTIC_REPLAY": (
            "pssolver.functional.channel_activity_runtime",
            "CHANNEL_ACTIVITY_DETERMINISTIC_REPLAY",
        ),
        "ChannelActivityFunctionalRuntime": (
            "pssolver.functional.channel_activity_runtime",
            "ChannelActivityFunctionalRuntime",
        ),
        "ChannelActivityFunctionalRuntimeFactory": (
            "pssolver.functional.channel_activity_runtime",
            "ChannelActivityFunctionalRuntimeFactory",
        ),
        "build_channel_activity_functional_runtime": (
            "pssolver.functional.channel_activity_runtime",
            "build_channel_activity_functional_runtime",
        ),
        "CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION": (
            "pssolver.functional.channel_checkpoint",
            "CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION",
        ),
        "ChannelActivityCheckpointBridge": (
            "pssolver.functional.channel_checkpoint",
            "ChannelActivityCheckpointBridge",
        ),
        "build_channel_functional_checkpoint_identity": (
            "pssolver.functional.channel_checkpoint",
            "build_channel_functional_checkpoint_identity",
        ),
        "upgrade_channel_functional_checkpoint_directory": (
            "pssolver.functional.channel_checkpoint",
            "upgrade_channel_functional_checkpoint_directory",
        ),
        "FunctionalCapabilities": (
            "pssolver.functional.contracts",
            "FunctionalCapabilities",
        ),
        "FunctionalCheckpointDurabilityError": (
            "pssolver.functional.errors",
            "FunctionalCheckpointDurabilityError",
        ),
        "FunctionalCheckpointIOError": (
            "pssolver.functional.errors",
            "FunctionalCheckpointIOError",
        ),
        "PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION": (
            "pssolver.functional.periodic_checkpoint",
            "PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION",
        ),
        "PeriodicActivityCheckpointBridge": (
            "pssolver.functional.periodic_checkpoint",
            "PeriodicActivityCheckpointBridge",
        ),
        "build_periodic_functional_checkpoint_identity": (
            "pssolver.functional.periodic_checkpoint",
            "build_periodic_functional_checkpoint_identity",
        ),
        "upgrade_periodic_functional_checkpoint_directory": (
            "pssolver.functional.periodic_checkpoint",
            "upgrade_periodic_functional_checkpoint_directory",
        ),
        "PeriodicActivityFunctionalRuntime": (
            "pssolver.functional.periodic_activity",
            "PeriodicActivityFunctionalRuntime",
        ),
        "PeriodicActivityFunctionalRuntimeFactory": (
            "pssolver.functional.periodic_activity",
            "PeriodicActivityFunctionalRuntimeFactory",
        ),
        "build_functional_runtime": (
            "pssolver.functional.periodic_activity",
            "build_functional_runtime",
        ),
        "periodic_activity_functional_request": (
            "pssolver.functional.periodic_activity",
            "periodic_activity_functional_request",
        ),
        "DirectionalDerivativeCheck": (
            "pssolver.functional.validation",
            "DirectionalDerivativeCheck",
        ),
        "FunctionalValidationError": (
            "pssolver.functional.validation",
            "FunctionalValidationError",
        ),
        "GradientPathCheck": (
            "pssolver.functional.validation",
            "GradientPathCheck",
        ),
        "PERIODIC_CONSISTENCY_VALIDATION_VERSION": (
            "pssolver.functional.validation",
            "PERIODIC_CONSISTENCY_VALIDATION_VERSION",
        ),
        "PERIODIC_GRADIENT_VALIDATION_VERSION": (
            "pssolver.functional.validation",
            "PERIODIC_GRADIENT_VALIDATION_VERSION",
        ),
        "PeriodicGradientValidationReport": (
            "pssolver.functional.validation",
            "PeriodicGradientValidationReport",
        ),
        "PeriodicProductionConsistencyReport": (
            "pssolver.functional.validation",
            "PeriodicProductionConsistencyReport",
        ),
        "TensorConsistencyCheck": (
            "pssolver.functional.validation",
            "TensorConsistencyCheck",
        ),
        "evaluate_periodic_activity_gradients": (
            "pssolver.functional.validation",
            "evaluate_periodic_activity_gradients",
        ),
        "evaluate_periodic_production_consistency": (
            "pssolver.functional.validation",
            "evaluate_periodic_production_consistency",
        ),
        "validate_periodic_activity_gradients": (
            "pssolver.functional.validation",
            "validate_periodic_activity_gradients",
        ),
        "validate_periodic_production_consistency": (
            "pssolver.functional.validation",
            "validate_periodic_production_consistency",
        ),
    }
)

_IMPLICIT_SUBMODULE_NAMES = frozenset(
    {
        "channel_activity",
        "channel_activity_runtime",
        "channel_checkpoint",
        "channel_pressure",
        "channel_pressure_adjoint",
        "checkpoint_identity",
        "contracts",
        "errors",
        "legacy_identity",
        "periodic_activity",
        "periodic_checkpoint",
        "pressure_metadata",
        "validation",
        "versioning",
    }
)


def __getattr__(name: str) -> object:
    if name in _IMPLICIT_SUBMODULE_NAMES:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    return _resolve_deprecated_facade_name(
        name,
        facade=__name__,
        replacement=(
            "pssolver.functional.api (stable names) or the defining "
            "functional submodule"
        ),
        exports=_COMPATIBILITY_EXPORTS,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    return sorted(
        set(_facade_dir(globals(), _COMPATIBILITY_EXPORTS))
        | _IMPLICIT_SUBMODULE_NAMES
    )


__all__ = [
    "FUNCTIONAL_API_VERSION",
    "FUNCTIONAL_OBSERVATION_TIME",
    "CHANNEL_ACTIVITY_FUNCTIONAL_KIND",
    "CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT",
    "CHANNEL_FUNCTIONAL_PRESSURE_WARM_START",
    "CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION",
    "CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION",
    "CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES",
    "CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION",
    "CHANNEL_ACTIVITY_DETERMINISTIC_REPLAY",
    "CHANNEL_ACTIVITY_RUNTIME_STAGE",
    "CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION",
    "ChannelActivityCheckpointBridge",
    "build_channel_functional_checkpoint_identity",
    "upgrade_channel_functional_checkpoint_directory",
    "ChannelActivityFunctionalRuntime",
    "ChannelActivityFunctionalRuntimeFactory",
    "ChannelImplicitPressureAdjoint",
    "ChannelImplicitPressureAdjointProtocol",
    "ChannelPressureSolveDiagnostics",
    "FunctionalCapabilities",
    "FunctionalCapabilitySet",
    "FunctionalCheckpointBridgeProtocol",
    "FunctionalCheckpointCompatibility",
    "FunctionalCheckpointCompatibilityError",
    "FunctionalCheckpointDurabilityError",
    "FunctionalCheckpointError",
    "FunctionalCheckpointExistsError",
    "FunctionalCheckpointIntegrityError",
    "FunctionalCheckpointIOError",
    "FunctionalCheckpointNotFoundError",
    "FunctionalCheckpointState",
    "FunctionalContractError",
    "FunctionalControlFieldSpec",
    "FunctionalControls",
    "FunctionalConvergenceError",
    "FunctionalError",
    "FunctionalExecutionError",
    "FunctionalIdentityError",
    "FunctionalObservationSpec",
    "FunctionalObservations",
    "FunctionalRuntimeConstructionRequest",
    "FunctionalRuntimeDeclaration",
    "FunctionalRuntimeFactoryProtocol",
    "FunctionalRuntimeIdentity",
    "FunctionalRuntimeProtocol",
    "FunctionalState",
    "FunctionalStateSpec",
    "FunctionalTensorSpec",
    "FunctionalTypeError",
    "FunctionalValueError",
    "FunctionalVersionError",
    "FunctionalAPIVersionSelection",
    "ChannelPressureTransposeOperator",
    "ChannelPressureTransposeProtocol",
    "PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION",
    "PeriodicActivityCheckpointBridge",
    "build_periodic_functional_checkpoint_identity",
    "upgrade_periodic_functional_checkpoint_directory",
    "PeriodicActivityFunctionalRuntime",
    "PeriodicActivityFunctionalRuntimeFactory",
    "DirectionalDerivativeCheck",
    "FunctionalValidationError",
    "GradientPathCheck",
    "PERIODIC_CONSISTENCY_VALIDATION_VERSION",
    "PERIODIC_GRADIENT_VALIDATION_VERSION",
    "PeriodicGradientValidationReport",
    "PeriodicProductionConsistencyReport",
    "TensorConsistencyCheck",
    "build_functional_runtime",
    "build_channel_activity_functional_runtime",
    "channel_activity_functional_declaration",
    "channel_activity_functional_request",
    "evaluate_periodic_activity_gradients",
    "evaluate_periodic_production_consistency",
    "periodic_activity_functional_request",
    "validate_periodic_activity_gradients",
    "validate_periodic_production_consistency",
    "unrolled_channel_pressure_solve_oracle",
    "FUNCTIONAL_CHECKPOINT_READ_API_VERSIONS",
    "FUNCTIONAL_COMPATIBILITY_POLICY_VERSION",
    "FUNCTIONAL_CONSTRUCTION_API_VERSIONS",
    "FUNCTIONAL_LEGACY_CHECKPOINT_API_VERSIONS",
    "functional_protocol_provenance",
    "negotiate_functional_api_version",
]
