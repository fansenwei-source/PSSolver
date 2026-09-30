"""Provisional functional execution declarations and qualified runtimes.

The interface is intentionally imported through ``pssolver.functional`` and
is not part of the stable package-root API.  P9.2--P9.5 implement and qualify
only the batch-one periodic complete-stress activity-control runtime, bitwise
CPU/H100 replay under a fixed execution identity, its periodic
production-checkpoint bridge, and frozen gradient, performance, memory, and
production-consistency gates.  P9.7.1 additionally declares, but does not
execute, the first batch-one complete-stress Channel functional contract;
unsupported model--geometry combinations continue to fail before execution.
"""

from .channel_activity import (
    CHANNEL_ACTIVITY_FUNCTIONAL_KIND,
    CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT,
    CHANNEL_FUNCTIONAL_PRESSURE_WARM_START,
    channel_activity_functional_declaration,
    channel_activity_functional_request,
)
from .channel_pressure import (
    CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION,
    ChannelPressureTransposeOperator,
    ChannelPressureTransposeProtocol,
)
from .channel_pressure_adjoint import (
    CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION,
    CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES,
    ChannelImplicitPressureAdjoint,
    ChannelImplicitPressureAdjointProtocol,
    ChannelPressureSolveDiagnostics,
    unrolled_channel_pressure_solve_oracle,
)
from .channel_activity_runtime import (
    CHANNEL_ACTIVITY_DETERMINISTIC_REPLAY,
    CHANNEL_ACTIVITY_RUNTIME_STAGE,
    ChannelActivityFunctionalRuntime,
    ChannelActivityFunctionalRuntimeFactory,
    build_channel_activity_functional_runtime,
)
from .channel_checkpoint import (
    CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION,
    ChannelActivityCheckpointBridge,
)

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
    FunctionalRuntimeDeclaration,
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
    "CHANNEL_ACTIVITY_FUNCTIONAL_KIND",
    "CHANNEL_FUNCTIONAL_PRESSURE_GRADIENT",
    "CHANNEL_FUNCTIONAL_PRESSURE_WARM_START",
    "CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION",
    "CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION",
    "CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES",
    "CHANNEL_ACTIVITY_DETERMINISTIC_REPLAY",
    "CHANNEL_ACTIVITY_RUNTIME_STAGE",
    "CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION",
    "ChannelActivityCheckpointBridge",
    "ChannelActivityFunctionalRuntime",
    "ChannelActivityFunctionalRuntimeFactory",
    "ChannelImplicitPressureAdjoint",
    "ChannelImplicitPressureAdjointProtocol",
    "ChannelPressureSolveDiagnostics",
    "FunctionalCapabilities",
    "FunctionalCapabilitySet",
    "FunctionalCheckpointBridgeProtocol",
    "FunctionalCheckpointState",
    "FunctionalControlFieldSpec",
    "FunctionalControls",
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
    "ChannelPressureTransposeOperator",
    "ChannelPressureTransposeProtocol",
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
    "build_channel_activity_functional_runtime",
    "channel_activity_functional_declaration",
    "channel_activity_functional_request",
    "evaluate_periodic_activity_gradients",
    "evaluate_periodic_production_consistency",
    "periodic_activity_functional_request",
    "validate_periodic_activity_gradients",
    "validate_periodic_production_consistency",
    "unrolled_channel_pressure_solve_oracle",
]
