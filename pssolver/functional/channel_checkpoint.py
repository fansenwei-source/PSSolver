"""Durable Q-state checkpoints for the pure Channel activity runtime.

The production Channel checkpoint owns algebraic velocity, pressure, and a
history-dependent pressure warm start.  Those values are deliberately absent
from the functional state.  This bridge therefore writes a separate,
functional-only format and can additionally import Q from an existing
production checkpoint after validating its runtime identity.  It never
fabricates production algebraic state on export.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from pssolver.io.checkpoint import (
    seal_checkpoint_metadata,
    verify_checkpoint_metadata,
    verify_functional_checkpoint_provenance,
    write_functional_checkpoint_provenance,
)
from pssolver.io.checkpoint_identity import (
    CheckpointCompatibilityIdentity,
    CheckpointFamily,
    LegacyIdentityKey,
    SourceReleaseGeneration,
    resolve_legacy_identity_schema,
)
from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS
from pssolver.runtime.channel_beris_edwards import (
    CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
)
from pssolver.workflows.channel_beris_edwards import CHECKPOINT_VERSION

from .contracts import (
    FUNCTIONAL_API_VERSION,
    FunctionalCheckpointCompatibility,
    FunctionalCheckpointState,
    FunctionalRuntimeIdentity,
    FunctionalState,
    FunctionalStateSpec,
)
from .errors import (
    FunctionalCheckpointCompatibilityError,
    FunctionalCheckpointExistsError,
    FunctionalCheckpointIntegrityError,
    FunctionalCheckpointNotFoundError,
)
from .checkpoint_identity import (
    build_channel_functional_checkpoint_identity,
    normalize_functional_checkpoint_identity,
)
from .versioning import (
    negotiate_functional_api_version,
    runtime_identity_sha256_for_api_version,
)


CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION = 3
_LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION = 1
_LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_V2 = 2
_FORMAT_KIND = "channel_activity_functional_checkpoint"
_SOURCE_FUNCTIONAL_V1 = "channel_functional_bridge_v1"
_SOURCE_FUNCTIONAL_V2 = "channel_functional_bridge_v2"
_SOURCE_FUNCTIONAL_V3 = "channel_functional_bridge_v3"
_SOURCE_PRODUCTION_V1 = "channel_production_v1"
_SOURCE_PRODUCTION_V2 = "channel_production_v2"
_SOURCE_PRODUCTION_V3 = "channel_production_v3"
_PRODUCTION_STATE_COMPONENTS = (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


class ChannelActivityCheckpointBridge:
    """Checkpoint the explicit functional state without hidden PCG history."""

    def __init__(
        self,
        *,
        state_spec: FunctionalStateSpec,
        functional_identity: FunctionalRuntimeIdentity,
        production_runtime_identity_sha256: str,
        production_backend_restart: Mapping[str, object],
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
            self._production_backend_restart = json.loads(
                json.dumps(
                    dict(production_backend_restart),
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "production_backend_restart must be JSON-compatible"
            ) from exc
        self._state_layout = state_spec.to_metadata()
        self._state_layout_sha256 = _canonical_sha256(self._state_layout)
        self._compatibility_identity = (
            build_channel_functional_checkpoint_identity(
                functional_identity=functional_identity,
                state_spec=state_spec,
            )
        )

    @property
    def format_version(self) -> int:
        return CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION

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
        version = self.format_version if format_version is None else format_version
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
            "runtime_kind": "channel_activity_batch_one",
            "pressure_warm_start": "absent_functional_zero_start",
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

    @staticmethod
    def _write_tensor(path: Path, value: torch.Tensor) -> dict[str, object]:
        array = value.detach().to(device="cpu").contiguous().numpy()
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
        return {
            "file": path.name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": _sha256(path),
        }

    def export_checkpoint(
        self,
        directory: str | Path,
        state: FunctionalState,
        *,
        completed_steps: int,
    ) -> Path:
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
        if (
            not isinstance(completed_steps, int)
            or isinstance(completed_steps, bool)
            or completed_steps < 0
        ):
            raise ValueError("completed_steps must be non-negative")
        self._state_spec.validate(state)
        if any(not bool(value.detach().isfinite().all()) for value in state):
            raise ValueError("functional state contains NaN or Inf")
        target = Path(directory).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FunctionalCheckpointExistsError(
                f"checkpoint directory already exists: {target}",
                operation="channel_checkpoint_export",
            )
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
        )
        try:
            records: dict[str, dict[str, object]] = {}
            for name, value in zip(
                ("q_physical", "q_spectral"), state, strict=True
            ):
                records[name] = self._write_tensor(
                    staging / f"state__{name}.npy",
                    value,
                )
            metadata = {
                "format_kind": _FORMAT_KIND,
                "format_version": self.format_version,
                "completed_steps": completed_steps,
                "functional_bridge": self._bridge_metadata(),
                "tensor_files": records,
            }
            if migration_provenance is not None:
                metadata["migration_provenance"] = json.loads(
                    json.dumps(
                        dict(migration_provenance),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
            metadata = seal_checkpoint_metadata(metadata)
            write_functional_checkpoint_provenance(
                staging,
                metadata,
                metadata["functional_bridge"],
            )
            (staging / "checkpoint.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, target)
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return target

    def migrate_legacy_checkpoint(
        self,
        source_directory: str | Path,
        target_directory: str | Path,
        *,
        source_release_generation: SourceReleaseGeneration | str,
    ) -> Path:
        """Upgrade legacy Channel state without claiming derivative continuity."""

        try:
            generation = SourceReleaseGeneration(source_release_generation)
        except (TypeError, ValueError) as exc:
            raise ValueError("source_release_generation is invalid") from exc
        source = Path(source_directory).expanduser().resolve()
        target = Path(target_directory).expanduser().resolve()
        if source == target:
            raise ValueError("legacy migration must write a new directory")
        metadata = self._metadata(source)
        if metadata.get("format_kind") != _FORMAT_KIND:
            raise FunctionalCheckpointCompatibilityError(
                "source is not a Channel functional checkpoint",
                operation="channel_checkpoint_migration",
            )
        expected_version = (
            _LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION
            if generation is SourceReleaseGeneration.RC1
            else _LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_V2
        )
        if metadata.get("format_version") != expected_version:
            raise FunctionalCheckpointCompatibilityError(
                "legacy Channel functional format does not match source release",
                operation="channel_checkpoint_migration",
            )
        verify_checkpoint_metadata(
            metadata,
            required=expected_version
            == _LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_V2,
        )
        bridge = metadata.get("functional_bridge")
        verify_functional_checkpoint_provenance(
            source,
            metadata,
            bridge,
            required=expected_version
            == _LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_V2,
        )
        if not isinstance(bridge, Mapping):
            raise FunctionalCheckpointCompatibilityError(
                "legacy Channel functional bridge metadata is missing",
                operation="channel_checkpoint_migration",
            )
        resolve_legacy_identity_schema(
            LegacyIdentityKey(
                checkpoint_family=CheckpointFamily.CHANNEL_FUNCTIONAL,
                format_version=expected_version,
                source_release_generation=generation,
                runtime_path="channel_activity_batch_one",
                applicability_class="configuration_dependent",
            )
        )
        api_version = (
            "0.1-provisional"
            if generation is SourceReleaseGeneration.RC1
            else FUNCTIONAL_API_VERSION
        )
        source_production_identity = _require_sha256(
            bridge.get("production_runtime_identity_sha256"),
            "production_runtime_identity_sha256",
        )
        expected_bridge = self._bridge_metadata(
            api_version=api_version,
            format_version=expected_version,
            production_runtime_identity_sha256=source_production_identity,
        )
        if dict(bridge) != expected_bridge:
            raise FunctionalCheckpointCompatibilityError(
                "legacy Channel functional identity does not match frozen schema",
                operation="channel_checkpoint_migration",
            )
        if (
            generation is SourceReleaseGeneration.RC3
            and source_production_identity
            != self._production_runtime_identity_sha256
        ):
            raise FunctionalCheckpointCompatibilityError(
                "legacy Channel forward identity does not match target",
                operation="channel_checkpoint_migration",
            )
        completed = metadata.get("completed_steps")
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or completed < 0
        ):
            raise ValueError("checkpoint completed_steps is invalid")
        records = metadata.get("tensor_files")
        if not isinstance(records, Mapping) or set(records) != {
            "q_physical",
            "q_spectral",
        }:
            raise ValueError("functional checkpoint tensor manifest is incomplete")

        # The derivative applicability decision is made from the target's
        # already-validated state contract, before opening either payload.
        periodic_size = self._state_spec.components[0].shape[-3]
        state_only = (
            generation
            in {SourceReleaseGeneration.RC1, SourceReleaseGeneration.RC2}
            and periodic_size % 2 == 0
        )
        values = tuple(
            self._load_tensor(source, records[name], name)
            for name in ("q_physical", "q_spectral")
        )
        state = tuple(
            value.to(device=spec.device)
            for value, spec in zip(
                values,
                self._state_spec.components,
                strict=True,
            )
        )
        self._state_spec.validate(state)
        return self._write_state_checkpoint(
            target,
            state,
            completed_steps=completed,
            migration_provenance={
                "derivative_dynamics_change": (
                    "pre_rc3_pressure_transpose_nyquist"
                    if state_only
                    else "proven_inapplicable_or_already_repaired"
                ),
                "migration_kind": (
                    "state_only_migration"
                    if state_only
                    else "exact_legacy_upgrade"
                ),
                "periodic_axis_size": periodic_size,
                "source_api_version": api_version,
                "source_family": CheckpointFamily.CHANNEL_FUNCTIONAL.value,
                "source_format_version": expected_version,
                "source_release_generation": generation.value,
                "trajectory_equivalent": not state_only,
            },
        )

    def _validate_migration_provenance(
        self,
        provenance: object,
    ) -> Mapping[str, object] | None:
        if provenance is None:
            return None
        if not isinstance(provenance, Mapping) or set(provenance) != {
            "derivative_dynamics_change",
            "migration_kind",
            "periodic_axis_size",
            "source_api_version",
            "source_family",
            "source_format_version",
            "source_release_generation",
            "trajectory_equivalent",
        }:
            raise FunctionalCheckpointCompatibilityError(
                "Channel migration provenance schema is invalid",
                operation="channel_checkpoint_import",
            )
        try:
            generation = SourceReleaseGeneration(
                provenance["source_release_generation"]
            )
        except (TypeError, ValueError) as exc:
            raise FunctionalCheckpointCompatibilityError(
                "Channel migration source release is invalid",
                operation="channel_checkpoint_import",
            ) from exc
        periodic_size = self._state_spec.components[0].shape[-3]
        state_only = (
            generation
            in {SourceReleaseGeneration.RC1, SourceReleaseGeneration.RC2}
            and periodic_size % 2 == 0
        )
        expected = {
            "derivative_dynamics_change": (
                "pre_rc3_pressure_transpose_nyquist"
                if state_only
                else "proven_inapplicable_or_already_repaired"
            ),
            "migration_kind": (
                "state_only_migration"
                if state_only
                else "exact_legacy_upgrade"
            ),
            "periodic_axis_size": periodic_size,
            "source_api_version": (
                "0.1-provisional"
                if generation is SourceReleaseGeneration.RC1
                else FUNCTIONAL_API_VERSION
            ),
            "source_family": CheckpointFamily.CHANNEL_FUNCTIONAL.value,
            "source_format_version": (
                _LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION
                if generation is SourceReleaseGeneration.RC1
                else _LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_V2
            ),
            "source_release_generation": generation.value,
            "trajectory_equivalent": not state_only,
        }
        if dict(provenance) != expected:
            raise FunctionalCheckpointCompatibilityError(
                "Channel migration provenance is inconsistent",
                operation="channel_checkpoint_import",
            )
        return provenance

    @staticmethod
    def _metadata(directory: Path) -> Mapping[str, object]:
        path = directory / "checkpoint.json"
        if not path.is_file():
            raise FunctionalCheckpointNotFoundError(
                f"checkpoint metadata is missing: {path}",
                operation="channel_checkpoint_import",
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FunctionalCheckpointIntegrityError(
                "checkpoint metadata is not valid JSON",
                operation="channel_checkpoint_import",
            ) from exc
        if not isinstance(value, Mapping):
            raise TypeError("checkpoint metadata must be a JSON object")
        return value

    @staticmethod
    def _load_tensor(
        directory: Path,
        record: object,
        description: str,
    ) -> torch.Tensor:
        if not isinstance(record, Mapping):
            raise TypeError(f"{description} tensor record must be a mapping")
        if set(record) != {"file", "shape", "dtype", "sha256"}:
            raise ValueError(f"{description} tensor record schema differs")
        filename = record.get("file")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".npy")
        ):
            raise ValueError(f"{description} tensor filename is invalid")
        path = directory / filename
        if not path.is_file():
            raise FunctionalCheckpointNotFoundError(
                f"checkpoint tensor is missing: {path}",
                operation="channel_checkpoint_import",
            )
        expected = _require_sha256(record.get("sha256"), "tensor SHA-256")
        if _sha256(path) != expected:
            raise FunctionalCheckpointIntegrityError(
                f"checkpoint tensor checksum mismatch: {path}",
                operation="channel_checkpoint_import",
            )
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != record.get("shape"):
            raise ValueError(f"checkpoint tensor shape metadata differs: {path}")
        if str(array.dtype) != record.get("dtype"):
            raise ValueError(f"checkpoint tensor dtype metadata differs: {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"checkpoint tensor contains NaN or Inf: {path}")
        return torch.from_numpy(np.array(array, copy=True))

    def _import_functional(
        self,
        directory: Path,
        metadata: Mapping[str, object],
    ) -> FunctionalCheckpointState:
        format_version = metadata.get("format_version")
        if format_version in {
            _LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION,
            _LEGACY_CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_V2,
        }:
            raise FunctionalCheckpointCompatibilityError(
                "legacy Channel functional checkpoint requires explicit migration",
                operation="channel_checkpoint_import",
            )
        if format_version != self.format_version:
            raise FunctionalCheckpointCompatibilityError(
                "unsupported Channel functional checkpoint version",
                operation="channel_checkpoint_import",
            )
        bridge = metadata.get("functional_bridge")
        if not isinstance(bridge, Mapping):
            raise FunctionalCheckpointCompatibilityError(
                "functional checkpoint bridge metadata is missing",
                operation="channel_checkpoint_import",
            )
        selection = negotiate_functional_api_version(
            bridge.get("api_version"),
            purpose="checkpoint_read",
        )
        bridge_version = bridge.get("format_version")
        if bridge_version != self.format_version:
            raise FunctionalCheckpointCompatibilityError(
                "unsupported Channel functional bridge version",
                operation="channel_checkpoint_import",
            )
        if bridge_version != format_version:
            raise FunctionalCheckpointCompatibilityError(
                "functional checkpoint schema versions do not match",
                operation="channel_checkpoint_import",
            )
        expected = self._bridge_metadata(
            api_version=selection.requested,
            format_version=bridge_version,
        )
        observed = dict(bridge)
        provenance = observed.get("run_provenance")
        if not isinstance(provenance, Mapping) or set(provenance) != {
            "functional_runtime_identity_sha256",
            "production_runtime_identity_sha256",
        }:
            raise FunctionalCheckpointCompatibilityError(
                "functional checkpoint run provenance is invalid",
                operation="channel_checkpoint_import",
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
                operation="channel_checkpoint_import",
            ) from exc
        if normalized != self._compatibility_identity:
            raise FunctionalCheckpointCompatibilityError(
                "functional checkpoint forward compatibility differs",
                operation="channel_checkpoint_import",
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
                operation="channel_checkpoint_import",
                details={"source_api_version": selection.requested},
            )
        if selection.legacy:
            raise FunctionalCheckpointCompatibilityError(
                "legacy Channel functional API requires explicit migration",
                operation="channel_checkpoint_import",
            )
        source = _SOURCE_FUNCTIONAL_V3
        completed = metadata.get("completed_steps")
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or completed < 0
        ):
            raise ValueError("checkpoint completed_steps is invalid")
        records = metadata.get("tensor_files")
        if not isinstance(records, Mapping) or set(records) != {
            "q_physical",
            "q_spectral",
        }:
            raise ValueError("functional checkpoint tensor manifest is incomplete")
        migration = self._validate_migration_provenance(
            metadata.get("migration_provenance")
        )
        values = tuple(
            self._load_tensor(directory, records[name], name)
            for name in ("q_physical", "q_spectral")
        )
        state = tuple(
            value.to(device=spec.device)
            for value, spec in zip(
                values, self._state_spec.components, strict=True
            )
        )
        return self._validated_state(
            state,
            completed,
            source,
            FunctionalCheckpointCompatibility(
                source_api_version=(
                    migration.get("source_api_version")
                    if migration is not None
                    and migration.get("migration_kind") == "state_only_migration"
                    else selection.requested
                ),
                target_api_version=selection.effective,
                reader=(
                    "channel_state_only_migration_v3_reader"
                    if migration is not None
                    and migration.get("migration_kind") == "state_only_migration"
                    else "current_channel_functional_bridge_v3_reader"
                ),
                exact_current_protocol=(
                    migration is None
                    or migration.get("migration_kind") != "state_only_migration"
                ),
            ),
        )

    @staticmethod
    def _validated_production_progress(
        metadata: Mapping[str, object],
    ) -> int:
        completed = metadata.get("completed_steps")
        if (
            not isinstance(completed, int)
            or isinstance(completed, bool)
            or completed < 0
        ):
            raise ValueError("checkpoint completed_steps is invalid")
        integrator = metadata.get("integrator")
        if not isinstance(integrator, Mapping) or set(integrator) != {
            "spectral_refresh_interval",
            "step_count",
            "refresh_count",
        }:
            raise ValueError("checkpoint integrator metadata differs")
        interval = integrator["spectral_refresh_interval"]
        if interval is not None and (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval <= 0
        ):
            raise ValueError("checkpoint spectral refresh interval is invalid")
        for name in ("step_count", "refresh_count"):
            value = integrator[name]
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"checkpoint {name} is invalid")
        if interval is None:
            expected_refresh, expected_step = 0, completed
        else:
            expected_refresh, expected_step = divmod(completed, interval)
        if (
            integrator["step_count"] != expected_step
            or integrator["refresh_count"] != expected_refresh
        ):
            raise ValueError("checkpoint progress counters differ")
        return completed

    def _import_production(
        self,
        directory: Path,
        metadata: Mapping[str, object],
    ) -> FunctionalCheckpointState:
        format_version = metadata.get("format_version")
        if (
            not isinstance(format_version, int)
            or isinstance(format_version, bool)
            or format_version not in {1, 2, CHECKPOINT_VERSION}
        ):
            raise FunctionalCheckpointCompatibilityError(
                "unsupported complete-stress Channel checkpoint version",
                operation="channel_checkpoint_import",
            )
        verify_checkpoint_metadata(
            metadata,
            required=format_version in {2, CHECKPOINT_VERSION},
        )
        if metadata.get("runtime_path") != CHANNEL_COMPLETE_STRESS_RUNTIME_PATH:
            raise FunctionalCheckpointCompatibilityError(
                "checkpoint runtime identity does not match target",
                operation="channel_checkpoint_import",
            )
        runtime_identity = metadata.get("runtime_identity_sha256")
        if format_version == CHECKPOINT_VERSION:
            provenance = metadata.get("run_provenance")
            if not isinstance(provenance, Mapping):
                raise FunctionalCheckpointCompatibilityError(
                    "checkpoint run provenance is missing",
                    operation="channel_checkpoint_import",
                )
            runtime_identity = provenance.get(
                "legacy_runtime_identity_sha256"
            )
        if runtime_identity != self._production_runtime_identity_sha256:
            raise FunctionalCheckpointCompatibilityError(
                "checkpoint scientific/runtime identity does not match",
                operation="channel_checkpoint_import",
            )
        if metadata.get("backend_restart") != self._production_backend_restart:
            raise FunctionalCheckpointCompatibilityError(
                "checkpoint backend contract does not match target",
                operation="channel_checkpoint_import",
            )
        completed = self._validated_production_progress(metadata)
        manifest = metadata.get("tensor_files")
        if not isinstance(manifest, Mapping) or set(manifest) != {
            "spatial",
            "spectral",
        }:
            raise ValueError("production checkpoint tensor manifest is missing")
        spatial = manifest.get("spatial")
        spectral = manifest.get("spectral")
        if not isinstance(spatial, Mapping) or not isinstance(
            spectral, Mapping
        ) or set(spatial) != set(_PRODUCTION_STATE_COMPONENTS) or set(
            spectral
        ) != set(_PRODUCTION_STATE_COMPONENTS):
            raise ValueError("production checkpoint Q manifest is incomplete")
        backend_files = metadata.get("backend_files")
        if not isinstance(backend_files, Mapping) or set(backend_files) != {
            "pressure_guess"
        }:
            raise ValueError("production checkpoint backend manifest is incomplete")
        loaded = {
            kind: {
                name: self._load_tensor(
                    directory,
                    records[name],
                    f"{kind} {name}",
                )
                for name in _PRODUCTION_STATE_COMPONENTS
            }
            for kind, records in (("spatial", spatial), ("spectral", spectral))
        }
        self._load_tensor(
            directory,
            backend_files["pressure_guess"],
            "pressure guess",
        )
        values = (
            torch.stack(
                tuple(loaded["spatial"][name] for name in Q_COMPONENTS)
            ),
            torch.stack(
                tuple(loaded["spectral"][name] for name in Q_COMPONENTS)
            ),
        )
        state = tuple(
            value.to(device=spec.device)
            for value, spec in zip(
                values, self._state_spec.components, strict=True
            )
        )
        return self._validated_state(
            state,
            completed,
            (
                _SOURCE_PRODUCTION_V3
                if format_version == CHECKPOINT_VERSION
                else (
                    _SOURCE_PRODUCTION_V2
                    if format_version == 2
                    else _SOURCE_PRODUCTION_V1
                )
            ),
            FunctionalCheckpointCompatibility(
                source_api_version=None,
                target_api_version=FUNCTIONAL_API_VERSION,
                reader=(
                    "qualified_channel_production_v3_reader"
                    if format_version == CHECKPOINT_VERSION
                    else (
                        "qualified_channel_production_v2_reader"
                        if format_version == 2
                        else "qualified_channel_production_v1_reader"
                    )
                ),
                exact_current_protocol=False,
            ),
        )

    def _validated_state(
        self,
        state: FunctionalState,
        completed_steps: int,
        source_format: str,
        compatibility: FunctionalCheckpointCompatibility,
    ) -> FunctionalCheckpointState:
        self._state_spec.validate(state)
        if any(not bool(value.isfinite().all()) for value in state):
            raise ValueError("imported functional state contains NaN or Inf")
        return FunctionalCheckpointState(
            state=state,
            completed_steps=completed_steps,
            source_format=source_format,
            compatibility=compatibility,
        )

    def import_checkpoint(
        self,
        directory: str | Path,
    ) -> FunctionalCheckpointState:
        directory = Path(directory).expanduser().resolve()
        metadata = self._metadata(directory)
        format_version = metadata.get("format_version")
        functional = metadata.get("format_kind") == _FORMAT_KIND
        verify_checkpoint_metadata(
            metadata,
            required=functional and format_version == self.format_version,
        )
        verify_functional_checkpoint_provenance(
            directory,
            metadata,
            metadata.get("functional_bridge"),
            required=functional and format_version == self.format_version,
        )
        if functional:
            return self._import_functional(directory, metadata)
        return self._import_production(directory, metadata)


def upgrade_channel_functional_checkpoint_directory(
    source_directory: str | Path,
    target_directory: str | Path,
    *,
    bridge: ChannelActivityCheckpointBridge,
    source_release_generation: SourceReleaseGeneration | str,
) -> Path:
    """Public non-in-place entry point for Channel functional migration."""

    if not isinstance(bridge, ChannelActivityCheckpointBridge):
        raise TypeError("bridge must be a ChannelActivityCheckpointBridge")
    return bridge.migrate_legacy_checkpoint(
        source_directory,
        target_directory,
        source_release_generation=source_release_generation,
    )


__all__ = [
    "CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION",
    "ChannelActivityCheckpointBridge",
    "build_channel_functional_checkpoint_identity",
    "upgrade_channel_functional_checkpoint_directory",
]
