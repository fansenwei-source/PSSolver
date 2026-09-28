"""Provisional functional execution declarations and qualified runtimes.

The interface is intentionally imported through ``pssolver.functional`` and
is not part of the stable package-root API.  P9.2--P9.4 implement only the
batch-one periodic complete-stress activity-control runtime, CPU deterministic
replay, its periodic production-checkpoint bridge, and frozen gradient and
production-consistency qualification; unsupported
model--geometry combinations continue to fail before execution.
"""

from .contracts import (
    FUNCTIONAL_API_VERSION,
    FUNCTIONAL_OBSERVATION_TIME,
    FunctionalCapabilities,
    FunctionalCapabilitySet,
    FunctionalCheckpointBridgeProtocol,
    FunctionalCheckpointState,
    FunctionalControlFieldSpec,
    FunctionalControls,
    FunctionalObservationSpec,
    FunctionalObservations,
    FunctionalRuntimeConstructionRequest,
    FunctionalRuntimeFactoryProtocol,
    FunctionalRuntimeIdentity,
    FunctionalRuntimeProtocol,
    FunctionalState,
    FunctionalStateSpec,
    FunctionalTensorSpec,
)
from .periodic_checkpoint import (
    PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION,
    PeriodicActivityCheckpointBridge,
)
from .periodic_activity import (
    PeriodicActivityFunctionalRuntime,
    PeriodicActivityFunctionalRuntimeFactory,
    build_functional_runtime,
    periodic_activity_functional_request,
)
from .validation import (
    DirectionalDerivativeCheck,
    FunctionalValidationError,
    GradientPathCheck,
    PERIODIC_CONSISTENCY_VALIDATION_VERSION,
    PERIODIC_GRADIENT_VALIDATION_VERSION,
    PeriodicGradientValidationReport,
    PeriodicProductionConsistencyReport,
    TensorConsistencyCheck,
    evaluate_periodic_activity_gradients,
    evaluate_periodic_production_consistency,
    validate_periodic_activity_gradients,
    validate_periodic_production_consistency,
)

__all__ = [
    "FUNCTIONAL_API_VERSION",
    "FUNCTIONAL_OBSERVATION_TIME",
    "FunctionalCapabilities",
    "FunctionalCapabilitySet",
    "FunctionalCheckpointBridgeProtocol",
    "FunctionalCheckpointState",
    "FunctionalControlFieldSpec",
    "FunctionalControls",
    "FunctionalObservationSpec",
    "FunctionalObservations",
    "FunctionalRuntimeConstructionRequest",
    "FunctionalRuntimeFactoryProtocol",
    "FunctionalRuntimeIdentity",
    "FunctionalRuntimeProtocol",
    "FunctionalState",
    "FunctionalStateSpec",
    "FunctionalTensorSpec",
    "PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION",
    "PeriodicActivityCheckpointBridge",
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
    "evaluate_periodic_activity_gradients",
    "evaluate_periodic_production_consistency",
    "periodic_activity_functional_request",
    "validate_periodic_activity_gradients",
    "validate_periodic_production_consistency",
]
