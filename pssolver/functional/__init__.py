"""Provisional functional execution declarations and qualified runtimes.

The interface is intentionally imported through ``pssolver.functional`` and
is not part of the stable package-root API.  P9.2--P9.3 implement only the
batch-one periodic complete-stress activity-control runtime, CPU deterministic
replay, and its periodic production-checkpoint bridge; unsupported
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
    "build_functional_runtime",
    "periodic_activity_functional_request",
]
