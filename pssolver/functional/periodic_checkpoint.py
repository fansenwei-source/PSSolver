"""Durable state bridge for the periodic activity functional runtime."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import torch

from pssolver.configuration.periodic_beris_edwards import (
    PERIODIC_RUNTIME_PATH,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.io.checkpoint_identity import (
    CheckpointCompatibilityIdentity,
    CheckpointFamily,
    LegacyIdentityKey,
    SourceReleaseGeneration,
    resolve_legacy_identity_schema,
)
from pssolver.runtime.periodic_beris_edwards import PERIODIC_BOUNDARIES
from pssolver.workflows.periodic_checkpoint import (
    PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
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
from .checkpoint_identity import (
    build_periodic_functional_checkpoint_identity,
    normalize_functional_checkpoint_identity,
)
from .versioning import (
    negotiate_functional_api_version,
    runtime_identity_sha256_for_api_version,
)


PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION = 3
_LEGACY_PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION = 1
_LEGACY_PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_V2 = 2
_SOURCE_PRODUCTION_V1 = "periodic_production_v1"
_SOURCE_PRODUCTION_V2 = "periodic_production_v2"
_SOURCE_PRODUCTION_V3 = "periodic_production_v3"
_SOURCE_FUNCTIONAL_V1 = "periodic_functional_bridge_v1"
_SOURCE_FUNCTIONAL_V2 = "periodic_functional_bridge_v2"
_SOURCE_FUNCTIONAL_V3 = "periodic_functional_bridge_v3"


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
    """Versioned state bridge for the Periodic functional runtime.

    Durable serialization intentionally detaches tensors.  The independent
    consumer owns checkpoint scheduling and reconstructs each differentiable
    replay segment from the returned owned tensors.  Format 3 carries layered
    forward, derivative, layout, and backend identities.  Provisional v1
    needs explicit non-in-place state migration; qualified post-repair v2
    remains readable and can also be upgraded explicitly.
    """

    def __init__(
        self,
        *,
        state_spec: FunctionalStateSpec,
        functional_identity: FunctionalRuntimeIdentity,
        production_runtime_identity_sha256: str,
        backend_restart: Mapping[str, object],
        projector=None,
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
        self._compatibility_identity = (
            build_periodic_functional_checkpoint_identity(
                functional_identity=functional_identity,
                state_spec=state_spec,
            )
        )
        self._projector = projector

    @property
    def format_version(self) -> int:
        return PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION

    def _bridge_metadata(
        self,
        *,
        api_version: str = FUNCTIONAL_API_VERSION,
        format_version: int | None = None,
        production_runtime_identity_sha256: str | None = None,
    ) -> dict[str, object]:
        negotiate_functional_api_version(
            api_version,
            purpose="checkpoint_read",
        )
        version = (
            self.format_version
            if format_version is None and api_version == FUNCTIONAL_API_VERSION
            else (
                _LEGACY_PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION
                if format_version is None
                else format_version
            )
        )
        functional_sha256 = (
            self._functional_identity_sha256
            if api_version == FUNCTIONAL_API_VERSION
            else runtime_identity_sha256_for_api_version(
                self._functional_identity_metadata,
                api_version,
            )
        )
        production_sha256 = (
            self._production_runtime_identity_sha256
            if production_runtime_identity_sha256 is None
            else _require_sha256(
                production_runtime_identity_sha256,
                "production_runtime_identity_sha256",
            )
        )
        metadata = {
            "format_version": version,
            "api_version": api_version,
            "runtime_kind": "periodic_activity_batch_one",
        }
        if version == self.format_version:
            metadata["checkpoint_compatibility_identity"] = (
                self._compatibility_identity.to_metadata()
            )
            metadata["checkpoint_compatibility_sha256"] = (
                self._compatibility_identity.canonical_sha256()
            )
            metadata["run_provenance"] = {
                "functional_runtime_identity_sha256": functional_sha256,
                "production_runtime_identity_sha256": production_sha256,
            }
        else:
            metadata.update(
                {
                    "functional_runtime_identity_sha256": functional_sha256,
                    "production_runtime_identity_sha256": production_sha256,
                    "state_layout": self._state_layout,
                    "state_layout_sha256": self._state_layout_sha256,
                }
            )
        return metadata

    def _validate_bridge_metadata(
        self,
        metadata: Mapping[str, object] | None,
    ) -> tuple[str, FunctionalCheckpointCompatibility]:
        if metadata is None:
            return (
                _SOURCE_PRODUCTION_V2,
                FunctionalCheckpointCompatibility(
                    source_api_version=None,
                    target_api_version=FUNCTIONAL_API_VERSION,
                    reader="qualified_periodic_production_v2_reader",
                    exact_current_protocol=False,
                ),
            )
        bridge_version = metadata.get("format_version")
        if bridge_version == _LEGACY_PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION:
            raise FunctionalCheckpointCompatibilityError(
                "legacy Periodic functional checkpoint requires explicit state migration",
                operation="periodic_checkpoint_import",
            )
        if bridge_version not in {
            _LEGACY_PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_V2,
            self.format_version,
        }:
            raise FunctionalCheckpointCompatibilityError(
                "unsupported Periodic functional checkpoint version",
                operation="periodic_checkpoint_import",
            )
        api_version = metadata.get("api_version")
        selection = negotiate_functional_api_version(
            api_version,
            purpose="checkpoint_read",
        )
        if selection.legacy:
            raise FunctionalCheckpointCompatibilityError(
                "provisional Periodic checkpoint requires explicit state migration",
                operation="periodic_checkpoint_import",
            )
        expected = self._bridge_metadata(
            api_version=selection.requested,
            format_version=bridge_version,
        )
        observed = dict(metadata)
        if bridge_version == self.format_version:
            provenance = observed.get("run_provenance")
            if not isinstance(provenance, Mapping) or set(provenance) != {
                "functional_runtime_identity_sha256",
                "production_runtime_identity_sha256",
            }:
                raise FunctionalCheckpointCompatibilityError(
                    "functional checkpoint run provenance is invalid",
                    operation="periodic_checkpoint_import",
                )
            for key, value in provenance.items():
                _require_sha256(value, key)
            expected["run_provenance"] = dict(provenance)
            try:
                source_identity = CheckpointCompatibilityIdentity.from_metadata(
                    observed.get("checkpoint_compatibility_identity")
                )
                source_sha256 = _require_sha256(
                    observed.get("checkpoint_compatibility_sha256"),
                    "checkpoint_compatibility_sha256",
                )
                if source_identity.canonical_sha256() != source_sha256:
                    raise ValueError(
                        "checkpoint compatibility identity digest differs"
                    )
                normalized = normalize_functional_checkpoint_identity(
                    source_identity.to_metadata()
                )
            except (TypeError, ValueError) as exc:
                raise FunctionalCheckpointCompatibilityError(
                    "functional checkpoint compatibility identity is invalid",
                    operation="periodic_checkpoint_import",
                ) from exc
            if normalized != self._compatibility_identity:
                raise FunctionalCheckpointCompatibilityError(
                    "functional checkpoint forward compatibility differs",
                    operation="periodic_checkpoint_import",
                )
            observed["checkpoint_compatibility_identity"] = expected[
                "checkpoint_compatibility_identity"
            ]
            observed["checkpoint_compatibility_sha256"] = expected[
                "checkpoint_compatibility_sha256"
            ]
        if observed != expected:
            raise FunctionalCheckpointCompatibilityError(
                "functional checkpoint identity or state layout does not match target",
                operation="periodic_checkpoint_import",
                details={"source_api_version": selection.requested},
            )
        current = bridge_version == self.format_version
        return (
            (
                _SOURCE_FUNCTIONAL_V3
                if current
                else _SOURCE_FUNCTIONAL_V2
            ),
            FunctionalCheckpointCompatibility(
                source_api_version=selection.requested,
                target_api_version=selection.effective,
                reader=(
                    selection.compatibility_reader
                    or (
                        "current_periodic_functional_bridge_v3_reader"
                        if current
                        else "qualified_periodic_functional_bridge_v2_reader"
                    )
                ),
                exact_current_protocol=current,
            ),
        )

    def export_checkpoint(
        self,
        directory: str | Path,
        state: FunctionalState,
        *,
        completed_steps: int,
    ) -> Path:
        """Write an atomic functional checkpoint in the v2 tensor carrier."""

        return self._write_state_checkpoint(
            directory,
            state,
            completed_steps=completed_steps,
            migration_provenance=None,
        )

    def _write_state_checkpoint(
        self,
        directory: str | Path,
        state: FunctionalState,
        *,
        completed_steps: int,
        migration_provenance: Mapping[str, object] | None,
    ) -> Path:
        """Write the current functional carrier with optional migration data."""

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
            # The functional v3 bridge deliberately remains inside the sealed
            # production-v2 tensor carrier.  Production exact restore accepts
            # only bridge v2, so v3 state cannot silently cross families.
            format_version=PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
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
            migration_provenance=migration_provenance,
        )
        return write_periodic_checkpoint(directory, checkpoint)

    def migrate_legacy_checkpoint(
        self,
        source_directory: str | Path,
        target_directory: str | Path,
        *,
        source_release_generation: SourceReleaseGeneration | str,
    ) -> Path:
        """Write a v3 bridge without treating incompatible history as exact."""

        try:
            generation = SourceReleaseGeneration(source_release_generation)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_release_generation is invalid") from exc
        source = Path(source_directory).expanduser().resolve()
        target = Path(target_directory).expanduser().resolve()
        if source == target:
            raise ValueError("legacy migration must write a new directory")
        header = read_periodic_checkpoint_header(source)
        bridge = header.functional_bridge
        if not isinstance(bridge, Mapping):
            raise FunctionalCheckpointCompatibilityError(
                "source is not a Periodic functional checkpoint",
                operation="periodic_checkpoint_migration",
            )
        expected_version = (
            _LEGACY_PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION
            if generation is SourceReleaseGeneration.RC1
            else _LEGACY_PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_V2
        )
        if bridge.get("format_version") != expected_version:
            raise FunctionalCheckpointCompatibilityError(
                "legacy Periodic functional format does not match source release",
                operation="periodic_checkpoint_migration",
            )
        resolve_legacy_identity_schema(
            LegacyIdentityKey(
                checkpoint_family=CheckpointFamily.PERIODIC_FUNCTIONAL,
                format_version=expected_version,
                source_release_generation=generation,
                runtime_path="periodic_activity_batch_one",
                applicability_class="configuration_dependent",
            )
        )
        api_version = (
            "0.1-provisional"
            if generation is SourceReleaseGeneration.RC1
            else FUNCTIONAL_API_VERSION
        )
        expected_bridge = self._bridge_metadata(
            api_version=api_version,
            format_version=expected_version,
            production_runtime_identity_sha256=header.runtime_identity_sha256,
        )
        if dict(bridge) != expected_bridge:
            raise FunctionalCheckpointCompatibilityError(
                "legacy Periodic functional identity does not match frozen schema",
                operation="periodic_checkpoint_migration",
            )
        if dict(header.backend_restart) != self._backend_restart:
            raise FunctionalCheckpointCompatibilityError(
                "legacy Periodic backend contract does not match target",
                operation="periodic_checkpoint_migration",
            )
        if (
            generation is not SourceReleaseGeneration.RC1
            and header.runtime_identity_sha256
            != self._production_runtime_identity_sha256
        ):
            raise FunctionalCheckpointCompatibilityError(
                "legacy Periodic forward identity does not match target",
                operation="periodic_checkpoint_migration",
            )

        # No payload is opened until the complete legacy identity preflight
        # above has succeeded.
        checkpoint = load_periodic_checkpoint(source)
        physical = torch.stack(
            tuple(checkpoint.evolved_spatial[name] for name in Q_COMPONENTS),
            dim=0,
        ).to(device=self._state_spec.components[0].device)
        spectral = torch.stack(
            tuple(checkpoint.evolved_spectral[name] for name in Q_COMPONENTS),
            dim=0,
        ).to(device=self._state_spec.components[1].device)
        state_only = generation is SourceReleaseGeneration.RC1
        if state_only:
            if self._projector is None:
                raise FunctionalCheckpointCompatibilityError(
                    "Periodic state migration requires the current runtime projector",
                    operation="periodic_checkpoint_migration",
                )
            spectral = self._projector.forward_transform(
                physical,
                PERIODIC_BOUNDARIES,
            )
        state = (physical, spectral)
        self._state_spec.validate(state)
        return self._write_state_checkpoint(
            target,
            state,
            completed_steps=checkpoint.completed_steps,
            migration_provenance={
                "migration_kind": (
                    "state_only_migration"
                    if state_only
                    else "exact_legacy_upgrade"
                ),
                "source_api_version": api_version,
                "source_family": CheckpointFamily.PERIODIC_FUNCTIONAL.value,
                "source_format_version": expected_version,
                "source_release_generation": generation.value,
                "spectral_reprojection": (
                    "current_periodic_projector_from_physical_q"
                    if state_only
                    else "not_required"
                ),
                "trajectory_equivalent": not state_only,
            },
        )

    @staticmethod
    def _validate_migration_provenance(
        provenance: object,
    ) -> Mapping[str, object] | None:
        if provenance is None:
            return None
        if not isinstance(provenance, Mapping) or set(provenance) != {
            "migration_kind",
            "source_api_version",
            "source_family",
            "source_format_version",
            "source_release_generation",
            "spectral_reprojection",
            "trajectory_equivalent",
        }:
            raise FunctionalCheckpointCompatibilityError(
                "Periodic migration provenance schema is invalid",
                operation="periodic_checkpoint_import",
            )
        try:
            generation = SourceReleaseGeneration(
                provenance["source_release_generation"]
            )
        except (TypeError, ValueError) as exc:
            raise FunctionalCheckpointCompatibilityError(
                "Periodic migration source release is invalid",
                operation="periodic_checkpoint_import",
            ) from exc
        state_only = generation is SourceReleaseGeneration.RC1
        expected = {
            "migration_kind": (
                "state_only_migration"
                if state_only
                else "exact_legacy_upgrade"
            ),
            "source_api_version": (
                "0.1-provisional"
                if state_only
                else FUNCTIONAL_API_VERSION
            ),
            "source_family": CheckpointFamily.PERIODIC_FUNCTIONAL.value,
            "source_format_version": (
                _LEGACY_PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION
                if state_only
                else _LEGACY_PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_V2
            ),
            "source_release_generation": generation.value,
            "spectral_reprojection": (
                "current_periodic_projector_from_physical_q"
                if state_only
                else "not_required"
            ),
            "trajectory_equivalent": not state_only,
        }
        if dict(provenance) != expected:
            raise FunctionalCheckpointCompatibilityError(
                "Periodic migration provenance is inconsistent",
                operation="periodic_checkpoint_import",
            )
        return provenance

    def import_checkpoint(
        self,
        directory: str | Path,
    ) -> FunctionalCheckpointState:
        """Preflight identities, then return fresh tensors on the target device."""

        directory = Path(directory).expanduser().resolve()
        metadata_path = directory / "checkpoint.json"
        migration_provenance = None
        if metadata_path.is_file():
            try:
                raw_metadata = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                raw_metadata = None
            if isinstance(raw_metadata, Mapping):
                migration_provenance = raw_metadata.get("migration_provenance")
                runtime_path = raw_metadata.get("runtime_path")
                format_kind = raw_metadata.get("format_kind")
                if (
                    runtime_path is not None
                    and runtime_path != PERIODIC_RUNTIME_PATH
                ) or format_kind is not None:
                    raise FunctionalCheckpointCompatibilityError(
                        "checkpoint belongs to a non-periodic runtime schema",
                        operation="periodic_checkpoint_import",
                        details={
                            "runtime_path": runtime_path,
                            "format_kind": format_kind,
                        },
                    )
        header = read_periodic_checkpoint_header(directory)
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
        if source_format == _SOURCE_FUNCTIONAL_V3:
            provenance = header.functional_bridge.get("run_provenance")
            if (
                not isinstance(provenance, Mapping)
                or provenance.get("production_runtime_identity_sha256")
                != header.runtime_identity_sha256
            ):
                raise FunctionalCheckpointCompatibilityError(
                    "functional checkpoint production provenance is inconsistent",
                    operation="periodic_checkpoint_import",
                )
        elif (
            header.runtime_identity_sha256
            != self._production_runtime_identity_sha256
        ):
            raise FunctionalCheckpointCompatibilityError(
                "checkpoint scientific/runtime identity does not match",
                operation="periodic_checkpoint_import",
            )
        migration_provenance = self._validate_migration_provenance(
            migration_provenance
        )
        if (
            source_format == _SOURCE_FUNCTIONAL_V3
            and isinstance(migration_provenance, Mapping)
            and migration_provenance.get("migration_kind")
            == "state_only_migration"
        ):
            compatibility = FunctionalCheckpointCompatibility(
                source_api_version=migration_provenance.get(
                    "source_api_version"
                ),
                target_api_version=FUNCTIONAL_API_VERSION,
                reader="periodic_reprojection_state_migration_v3_reader",
                exact_current_protocol=False,
            )
        if header.functional_bridge is None and header.format_version == 3:
            source_format = _SOURCE_PRODUCTION_V3
            compatibility = FunctionalCheckpointCompatibility(
                source_api_version=None,
                target_api_version=FUNCTIONAL_API_VERSION,
                reader="qualified_periodic_production_v3_reader",
                exact_current_protocol=False,
            )
        if header.functional_bridge is None and header.format_version == 1:
            source_format = _SOURCE_PRODUCTION_V1
            compatibility = FunctionalCheckpointCompatibility(
                source_api_version=None,
                target_api_version=FUNCTIONAL_API_VERSION,
                reader="qualified_periodic_production_v1_reader",
                exact_current_protocol=False,
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


def upgrade_periodic_functional_checkpoint_directory(
    source_directory: str | Path,
    target_directory: str | Path,
    *,
    bridge: PeriodicActivityCheckpointBridge,
    source_release_generation: SourceReleaseGeneration | str,
) -> Path:
    """Public non-in-place entry point for Periodic functional migration."""

    if not isinstance(bridge, PeriodicActivityCheckpointBridge):
        raise TypeError("bridge must be a PeriodicActivityCheckpointBridge")
    return bridge.migrate_legacy_checkpoint(
        source_directory,
        target_directory,
        source_release_generation=source_release_generation,
    )


__all__ = [
    "PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION",
    "PeriodicActivityCheckpointBridge",
    "build_periodic_functional_checkpoint_identity",
    "upgrade_periodic_functional_checkpoint_directory",
]
