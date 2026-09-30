"""Stable installed-package facade for functional PSSolver execution.

Only names in this module's ``__all__`` are part of the P9.8 stable surface.
The broader :mod:`pssolver.functional` namespace remains a compatibility
facade through the deprecation window recorded by the protocol provenance.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import TypeVar

from . import channel_activity as _channel_declarations
from . import channel_activity_runtime as _channel_runtime
from . import periodic_activity as _periodic_runtime
from .channel_activity_runtime import CHANNEL_ACTIVITY_RUNTIME_STAGE
from .contracts import (
    FUNCTIONAL_OBSERVATION_TIME,
    FunctionalCapabilitySet,
    FunctionalCheckpointBridgeProtocol,
    FunctionalCheckpointCompatibility,
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
from .errors import (
    FunctionalCheckpointCompatibilityError,
    FunctionalCheckpointError,
    FunctionalCheckpointExistsError,
    FunctionalCheckpointIntegrityError,
    FunctionalCheckpointNotFoundError,
    FunctionalContractError,
    FunctionalConvergenceError,
    FunctionalError,
    FunctionalExecutionError,
    FunctionalIdentityError,
    FunctionalTypeError,
    FunctionalValueError,
    FunctionalVersionError,
    translate_functional_exception,
)
from .pressure_metadata import (
    CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
    ChannelPressureSolveDiagnostics,
)
from .versioning import (
    FUNCTIONAL_API_VERSION,
    FUNCTIONAL_CHECKPOINT_READ_API_VERSIONS,
    FUNCTIONAL_COMPATIBILITY_POLICY_VERSION,
    FUNCTIONAL_CONSTRUCTION_API_VERSIONS,
    FUNCTIONAL_LEGACY_CHECKPOINT_API_VERSIONS,
    FunctionalAPIVersionSelection,
    functional_protocol_provenance,
    negotiate_functional_api_version,
)


_T = TypeVar("_T")


def _invoke(operation: str, function: Callable[..., _T], *args, **kwargs) -> _T:
    try:
        return function(*args, **kwargs)
    except Exception as exc:
        translated = translate_functional_exception(
            exc,
            operation=operation,
        )
        if translated is exc:
            raise
        raise translated from exc


class _StableCheckpointBridge:
    """Error-normalizing facade without exposing bridge implementation state."""

    __slots__ = ("_delegate",)

    def __init__(self, delegate: FunctionalCheckpointBridgeProtocol) -> None:
        self._delegate = delegate

    @property
    def format_version(self) -> int:
        return self._delegate.format_version

    def export_checkpoint(
        self,
        directory: str | Path,
        state: FunctionalState,
        *,
        completed_steps: int,
    ) -> Path:
        return _invoke(
            "checkpoint_export",
            self._delegate.export_checkpoint,
            directory,
            state,
            completed_steps=completed_steps,
        )

    def import_checkpoint(
        self,
        directory: str | Path,
    ) -> FunctionalCheckpointState:
        return _invoke(
            "checkpoint_import",
            self._delegate.import_checkpoint,
            directory,
        )


class _StableFunctionalRuntime:
    """Protocol-only facade that prevents reliance on private runtime members."""

    __slots__ = ("_checkpoint_bridge", "_delegate")

    def __init__(self, delegate: FunctionalRuntimeProtocol) -> None:
        self._delegate = delegate
        bridge = delegate.checkpoint_bridge
        self._checkpoint_bridge = (
            None if bridge is None else _StableCheckpointBridge(bridge)
        )

    @property
    def api_version(self) -> str:
        return self._delegate.api_version

    @property
    def state_spec(self) -> FunctionalStateSpec:
        return self._delegate.state_spec

    @property
    def control_specs(self) -> tuple[FunctionalControlFieldSpec, ...]:
        return self._delegate.control_specs

    @property
    def control_field_schema(self) -> tuple[FunctionalControlFieldSpec, ...]:
        return self._delegate.control_field_schema

    @property
    def observation_specs(self) -> tuple[FunctionalObservationSpec, ...]:
        return self._delegate.observation_specs

    @property
    def capabilities(self) -> FunctionalCapabilitySet:
        return self._delegate.capabilities

    @property
    def checkpoint_bridge(self) -> FunctionalCheckpointBridgeProtocol | None:
        return self._checkpoint_bridge

    def identity(self) -> FunctionalRuntimeIdentity:
        return _invoke("runtime_identity", self._delegate.identity)

    def diagnostics(self) -> dict[str, object]:
        """Return an owned JSON snapshot without exposing runtime internals."""

        metadata = _invoke("runtime_diagnostics", self._delegate.diagnostics)
        if not isinstance(metadata, dict):
            raise FunctionalTypeError(
                "runtime diagnostics must be a dict",
                operation="runtime_diagnostics",
            )
        try:
            encoded = json.dumps(
                metadata,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            snapshot = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise FunctionalValueError(
                "runtime diagnostics must be finite JSON data",
                operation="runtime_diagnostics",
            ) from exc
        if not isinstance(snapshot, dict):  # pragma: no cover - guarded above
            raise AssertionError("JSON object diagnostics decoded incorrectly")
        return snapshot

    def initial_state(self) -> FunctionalState:
        return _invoke("initial_state", self._delegate.initial_state)

    def step(
        self,
        state: FunctionalState,
        controls: FunctionalControls,
        step_index: int,
    ) -> FunctionalState:
        return _invoke(
            "step",
            self._delegate.step,
            state,
            controls,
            step_index,
        )

    def observe(
        self,
        state: FunctionalState,
        controls: FunctionalControls | None,
    ) -> FunctionalObservations:
        return _invoke("observe", self._delegate.observe, state, controls)

    def step_and_observe(
        self,
        state: FunctionalState,
        controls: FunctionalControls,
        step_index: int,
    ) -> tuple[FunctionalState, FunctionalObservations]:
        return _invoke(
            "step_and_observe",
            self._delegate.step_and_observe,
            state,
            controls,
            step_index,
        )


def periodic_activity_functional_request(*args, **kwargs):
    return _invoke(
        "periodic_activity_functional_request",
        _periodic_runtime.periodic_activity_functional_request,
        *args,
        **kwargs,
    )


def channel_activity_functional_request(*args, **kwargs):
    return _invoke(
        "channel_activity_functional_request",
        _channel_declarations.channel_activity_functional_request,
        *args,
        **kwargs,
    )


def channel_activity_functional_declaration(*args, **kwargs):
    return _invoke(
        "channel_activity_functional_declaration",
        _channel_declarations.channel_activity_functional_declaration,
        *args,
        **kwargs,
    )


def build_functional_runtime(
    request: FunctionalRuntimeConstructionRequest,
) -> FunctionalRuntimeProtocol:
    runtime = _invoke(
        "build_periodic_functional_runtime",
        _periodic_runtime.build_functional_runtime,
        request,
    )
    return _StableFunctionalRuntime(runtime)


def build_channel_activity_functional_runtime(
    request: FunctionalRuntimeConstructionRequest,
) -> FunctionalRuntimeProtocol:
    runtime = _invoke(
        "build_channel_functional_runtime",
        _channel_runtime.build_channel_activity_functional_runtime,
        request,
    )
    return _StableFunctionalRuntime(runtime)


__all__ = [
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
    "build_channel_activity_functional_runtime",
    "build_functional_runtime",
    "channel_activity_functional_declaration",
    "channel_activity_functional_request",
    "functional_protocol_provenance",
    "negotiate_functional_api_version",
    "periodic_activity_functional_request",
]
