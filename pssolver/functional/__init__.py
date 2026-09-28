"""Provisional functional execution declarations and qualified runtimes.

The interface is intentionally imported through ``pssolver.functional`` and
is not part of the stable package-root API.  P9.2 implements only the
batch-one periodic complete-stress activity-control runtime; unsupported
model--geometry combinations continue to fail before execution.
"""

from .contracts import (
    FUNCTIONAL_API_VERSION,
    FUNCTIONAL_OBSERVATION_TIME,
    FunctionalCapabilities,
    FunctionalCapabilitySet,
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
    "PeriodicActivityFunctionalRuntime",
    "PeriodicActivityFunctionalRuntimeFactory",
    "build_functional_runtime",
    "periodic_activity_functional_request",
]
