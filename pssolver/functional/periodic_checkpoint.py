"""Durable state bridge for the periodic activity functional runtime."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import torch

from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.workflows.periodic_checkpoint import (
    PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PeriodicWorkflowCheckpoint,
    load_periodic_checkpoint,
    read_periodic_checkpoint_header,
    write_periodic_checkpoint,
)

from .contracts import (
    FUNCTIONAL_API_VERSION,
    FunctionalCheckpointCompatibility,
    FunctionalCheckpointState,
    FunctionalRuntimeIdentity,
    FunctionalState,
    FunctionalStateSpec,
)
from .errors import FunctionalCheckpointCompatibilityError
from .versioning import (
    negotiate_functional_api_version,
    runtime_identity_sha256_for_api_version,
)


PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION = 1
_SOURCE_PRODUCTION = "periodic_production_v1"
_SOURCE_FUNCTIONAL = "periodic_functional_bridge_v1"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


class PeriodicActivityCheckpointBridge:
    """Bidirectional adapter for the production periodic checkpoint schema.

    Durable serialization intentionally detaches tensors.  The independent
    consumer owns checkpoint scheduling and reconstructs each differentiable
    replay segment from the returned owned tensors.
    """

    def __init__(
        self,
        *,
        state_spec: FunctionalStateSpec,
        functional_identity: FunctionalRuntimeIdentity,
        production_runtime_identity_sha256: str,
        backend_restart: Mapping[str, object],
    ) -> None:
        if not isinstance(state_spec, FunctionalStateSpec):
            raise TypeError("state_spec must be a FunctionalStateSpec")
        if not isinstance(functional_identity, FunctionalRuntimeIdentity):
            raise TypeError("functional_identity must be a FunctionalRuntimeIdentity")
        self._state_spec = state_spec
        self._functional_identity_metadata = functional_identity.to_metadata()
        self._functional_identity_sha256 = functional_identity.canonical_sha256()
        self._production_runtime_identity_sha256 = _require_sha256(
            production_runtime_identity_sha256,
            "production_runtime_identity_sha256",
        )
        try:
            self._backend_restart = json.loads(
                json.dumps(
                    dict(backend_restart),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("backend_restart must be JSON-compatible") from exc
        self._state_layout = state_spec.to_metadata()
        self._state_layout_sha256 = _canonical_sha256(self._state_layout)

    @property
    def format_version(self) -> int:
        return PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION

    def _bridge_metadata(
        self,
        *,
        api_version: str = FUNCTIONAL_API_VERSION,
    ) -> dict[str, object]:
        negotiate_functional_api_version(
            api_version,
            purpose="checkpoint_read",
        )
        return {
            "format_version": self.format_version,
            "api_version": api_version,
            "runtime_kind": "periodic_activity_batch_one",
            "functional_runtime_identity_sha256": (
                self._functional_identity_sha256
                if api_version == FUNCTIONAL_API_VERSION
                else runtime_identity_sha256_for_api_version(
                    self._functional_identity_metadata,
                    api_version,
                )
            ),
            "production_runtime_identity_sha256": (
                self._production_runtime_identity_sha256
            ),
            "state_layout": self._state_layout,
            "state_layout_sha256": self._state_layout_sha256,
        }

    def _validate_bridge_metadata(
        self,
        metadata: Mapping[str, object] | None,
    ) -> tuple[str, FunctionalCheckpointCompatibility]:
        if metadata is None:
            return (
                _SOURCE_PRODUCTION,
                FunctionalCheckpointCompatibility(
                    source_api_version=None,
                    target_api_version=FUNCTIONAL_API_VERSION,
                    reader="qualified_periodic_production_v1_reader",
                    exact_current_protocol=False,
                ),
            )
        api_version = metadata.get("api_version")
        selection = negotiate_functional_api_version(
            api_version,
            purpose="checkpoint_read",
        )
        expected = self._bridge_metadata(api_version=selection.requested)
        if dict(metadata) != expected:
            raise FunctionalCheckpointCompatibilityError(
                "functional checkpoint identity or state layout does not match target",
                operation="periodic_checkpoint_import",
                details={"source_api_version": selection.requested},
            )
        return (
            _SOURCE_FUNCTIONAL,
            FunctionalCheckpointCompatibility(
                source_api_version=selection.requested,
                target_api_version=selection.effective,
                reader=(
                    selection.compatibility_reader
                    or "current_periodic_functional_bridge_v1_reader"
                ),
                exact_current_protocol=not selection.legacy,
            ),
        )

    def export_checkpoint(
        self,
        directory: str | Path,
        state: FunctionalState,
        *,
        completed_steps: int,
    ) -> Path:
        """Write an atomic checkpoint accepted by the production runtime."""

        if (
            not isinstance(completed_steps, int)
            or isinstance(completed_steps, bool)
            or completed_steps < 0
        ):
            raise ValueError("completed_steps must be non-negative")
        self._state_spec.validate(state)
        if any(not bool(torch.isfinite(value).all().item()) for value in state):
            raise ValueError("functional state contains NaN or Inf")
        q_physical, q_spectral = state
        checkpoint = PeriodicWorkflowCheckpoint(
            format_version=PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            runtime_identity_sha256=self._production_runtime_identity_sha256,
            completed_steps=completed_steps,
            spectral_refresh_interval=None,
            integrator_step_count=completed_steps,
            integrator_refresh_count=0,
            evolved_spatial={
                name: q_physical[index]
                for index, name in enumerate(Q_COMPONENTS)
            },
            evolved_spectral={
                name: q_spectral[index]
                for index, name in enumerate(Q_COMPONENTS)
            },
            backend_restart=self._backend_restart,
            functional_bridge=self._bridge_metadata(),
        )
        return write_periodic_checkpoint(directory, checkpoint)

    def import_checkpoint(
        self,
        directory: str | Path,
    ) -> FunctionalCheckpointState:
        """Preflight identities, then return fresh tensors on the target device."""

        header = read_periodic_checkpoint_header(directory)
        if (
            header.runtime_identity_sha256
            != self._production_runtime_identity_sha256
        ):
            raise FunctionalCheckpointCompatibilityError(
                "checkpoint scientific/runtime identity does not match",
                operation="periodic_checkpoint_import",
            )
        if dict(header.backend_restart) != self._backend_restart:
            raise FunctionalCheckpointCompatibilityError(
                "checkpoint backend contract does not match target",
                operation="periodic_checkpoint_import",
            )
        if header.spectral_refresh_interval is not None:
            raise FunctionalCheckpointCompatibilityError(
                "functional checkpoint import requires disabled spectral refresh",
                operation="periodic_checkpoint_import",
            )
        source_format, compatibility = self._validate_bridge_metadata(
            header.functional_bridge
        )

        # Tensor payloads are opened only after all small identity and layout
        # metadata gates above have succeeded.
        checkpoint = load_periodic_checkpoint(directory)
        physical = torch.stack(
            tuple(checkpoint.evolved_spatial[name] for name in Q_COMPONENTS),
            dim=0,
        )
        spectral = torch.stack(
            tuple(checkpoint.evolved_spectral[name] for name in Q_COMPONENTS),
            dim=0,
        )
        state = tuple(
            value.to(device=specification.device)
            for value, specification in zip(
                (physical, spectral),
                self._state_spec.components,
                strict=True,
            )
        )
        self._state_spec.validate(state)
        if any(not bool(torch.isfinite(value).all().item()) for value in state):
            raise ValueError("imported functional state contains NaN or Inf")
        return FunctionalCheckpointState(
            state=state,
            completed_steps=checkpoint.completed_steps,
            source_format=source_format,
            compatibility=compatibility,
        )


__all__ = [
    "PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION",
    "PeriodicActivityCheckpointBridge",
]
