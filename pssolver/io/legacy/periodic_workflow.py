"""Versioned exact-restart checkpoints for the periodic Beris--Edwards runtime."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from pssolver.core._canonical import canonical_json
from pssolver.io._checkpoint_artifacts import write_standard_tensor_record
from pssolver.io.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    checkpoint_directory_transaction,
    checkpoint_file_sha256,
    load_checkpoint_numpy,
    read_checkpoint_json,
    require_exact_checkpoint_keys,
    resolve_checkpoint_payload,
    seal_checkpoint_metadata,
    verify_checkpoint_metadata,
    verify_functional_checkpoint_provenance,
    write_functional_checkpoint_provenance,
)
from pssolver.io.checkpoint_identity import (
    CheckpointCompatibilityIdentity,
    CheckpointFamily,
    SourceReleaseGeneration,
)


PERIODIC_RUNTIME_PATH = "periodic_spectral"
Q_COMPONENTS = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")

PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1 = 1
PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 2
PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 3
_SUPPORTED_PERIODIC_WORKFLOW_CHECKPOINT_FORMATS = frozenset(
    {
        PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1,
        PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    }
)
_PERIODIC_COMMON_REQUIRED_KEYS = frozenset(
    {
        "format_version",
        "runtime_path",
        "completed_steps",
        "integrator",
        "backend_restart",
        "tensor_files",
        "metadata_sha256",
    }
)


def _sha256(path: Path) -> str:
    return checkpoint_file_sha256(path, operation="periodic_checkpoint_io")


def _require_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _json_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    try:
        encoded = canonical_json(dict(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{description} must be JSON-compatible") from exc
    return MappingProxyType(json.loads(encoded))


def _clone_tensors(
    values: Mapping[str, torch.Tensor],
    description: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(values, Mapping) or tuple(values) != Q_COMPONENTS:
        raise ValueError(f"{description} must contain ordered Q components")
    cloned = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{description} values must be tensors")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{description} tensors must be finite")
        cloned[name] = value.detach().clone()
    return MappingProxyType(cloned)


def _validate_progress(
    completed_steps: int,
    spectral_refresh_interval: int | None,
    integrator_step_count: int,
    integrator_refresh_count: int,
) -> None:
    for value, description in (
        (completed_steps, "completed_steps"),
        (integrator_step_count, "integrator_step_count"),
        (integrator_refresh_count, "integrator_refresh_count"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{description} must be non-negative")
    interval = spectral_refresh_interval
    if interval is not None and (
        not isinstance(interval, int)
        or isinstance(interval, bool)
        or interval <= 0
    ):
        raise ValueError("spectral refresh interval must be positive or None")
    if interval is None:
        valid = integrator_refresh_count == 0 and integrator_step_count == completed_steps
    else:
        valid = (
            integrator_step_count < interval
            and integrator_refresh_count * interval + integrator_step_count
            == completed_steps
        )
    if not valid:
        raise ValueError("checkpoint refresh counters are inconsistent")


@dataclass(frozen=True, slots=True)
class PeriodicCheckpointHeader:
    """Small identity record read before any checkpoint array is loaded."""

    format_version: int
    runtime_identity_sha256: str
    completed_steps: int
    spectral_refresh_interval: int | None
    integrator_step_count: int
    integrator_refresh_count: int
    backend_restart: Mapping[str, object]
    functional_bridge: Mapping[str, object] | None = None
    compatibility_identity: CheckpointCompatibilityIdentity | None = None

    @property
    def compatibility_identity_sha256(self) -> str | None:
        identity = self.compatibility_identity
        return None if identity is None else identity.canonical_sha256()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.format_version, int)
            or isinstance(self.format_version, bool)
            or self.format_version
            not in _SUPPORTED_PERIODIC_WORKFLOW_CHECKPOINT_FORMATS
        ):
            raise ValueError("unsupported periodic workflow checkpoint version")
        _require_sha256(self.runtime_identity_sha256, "runtime_identity_sha256")
        _validate_progress(
            self.completed_steps,
            self.spectral_refresh_interval,
            self.integrator_step_count,
            self.integrator_refresh_count,
        )
        object.__setattr__(
            self,
            "backend_restart",
            _json_mapping(self.backend_restart, "backend_restart"),
        )
        if self.functional_bridge is not None:
            object.__setattr__(
                self,
                "functional_bridge",
                _json_mapping(self.functional_bridge, "functional_bridge"),
            )
        identity = self.compatibility_identity
        if self.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            if not isinstance(identity, CheckpointCompatibilityIdentity):
                raise TypeError("periodic compatibility identity is missing")
            if identity.family is not CheckpointFamily.PERIODIC_WORKFLOW:
                raise ValueError("periodic compatibility identity family differs")
            if identity.runtime_path != PERIODIC_RUNTIME_PATH:
                raise ValueError("periodic compatibility runtime path differs")
        elif identity is not None:
            raise ValueError("legacy periodic checkpoints cannot carry v3 identity")


@dataclass(frozen=True, slots=True)
class PeriodicWorkflowCheckpoint:
    """Complete evolved Q state and integrator phase for periodic execution."""

    format_version: int
    runtime_identity_sha256: str
    completed_steps: int
    spectral_refresh_interval: int | None
    integrator_step_count: int
    integrator_refresh_count: int
    evolved_spatial: Mapping[str, torch.Tensor]
    evolved_spectral: Mapping[str, torch.Tensor]
    backend_restart: Mapping[str, object]
    functional_bridge: Mapping[str, object] | None = None
    compatibility_identity: CheckpointCompatibilityIdentity | None = None
    migration_provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.format_version, int)
            or isinstance(self.format_version, bool)
            or self.format_version
            not in _SUPPORTED_PERIODIC_WORKFLOW_CHECKPOINT_FORMATS
        ):
            raise ValueError("unsupported periodic workflow checkpoint version")
        _require_sha256(self.runtime_identity_sha256, "runtime_identity_sha256")
        _validate_progress(
            self.completed_steps,
            self.spectral_refresh_interval,
            self.integrator_step_count,
            self.integrator_refresh_count,
        )
        object.__setattr__(
            self,
            "evolved_spatial",
            _clone_tensors(self.evolved_spatial, "evolved_spatial"),
        )
        object.__setattr__(
            self,
            "evolved_spectral",
            _clone_tensors(self.evolved_spectral, "evolved_spectral"),
        )
        object.__setattr__(
            self,
            "backend_restart",
            _json_mapping(self.backend_restart, "backend_restart"),
        )
        if self.functional_bridge is not None:
            object.__setattr__(
                self,
                "functional_bridge",
                _json_mapping(self.functional_bridge, "functional_bridge"),
            )
        identity = self.compatibility_identity
        if self.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            if not isinstance(identity, CheckpointCompatibilityIdentity):
                raise TypeError("periodic compatibility identity is missing")
            if identity.family is not CheckpointFamily.PERIODIC_WORKFLOW:
                raise ValueError("periodic compatibility identity family differs")
            if identity.runtime_path != PERIODIC_RUNTIME_PATH:
                raise ValueError("periodic compatibility runtime path differs")
        elif identity is not None:
            raise ValueError("legacy periodic checkpoints cannot carry v3 identity")
        if self.migration_provenance is not None:
            object.__setattr__(
                self,
                "migration_provenance",
                _json_mapping(self.migration_provenance, "migration_provenance"),
            )

    def to_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "format_version": self.format_version,
            "runtime_path": PERIODIC_RUNTIME_PATH,
            "completed_steps": self.completed_steps,
            "integrator": {
                "spectral_refresh_interval": self.spectral_refresh_interval,
                "step_count": self.integrator_step_count,
                "refresh_count": self.integrator_refresh_count,
            },
            "backend_restart": dict(self.backend_restart),
        }
        if self.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            identity = self.compatibility_identity
            if identity is None:
                raise RuntimeError("periodic compatibility identity is missing")
            metadata["run_provenance"] = {
                "legacy_runtime_identity_sha256": self.runtime_identity_sha256,
            }
            metadata["compatibility_identity"] = identity.to_metadata()
            metadata["compatibility_identity_sha256"] = (
                identity.canonical_sha256()
            )
        else:
            metadata["runtime_identity_sha256"] = self.runtime_identity_sha256
        if self.functional_bridge is not None:
            metadata["functional_bridge"] = dict(self.functional_bridge)
        if self.migration_provenance is not None:
            metadata["migration_provenance"] = dict(self.migration_provenance)
        return metadata


def _write_tensor(path: Path, value: torch.Tensor) -> dict[str, object]:
    return write_standard_tensor_record(
        path,
        value,
        hash_file=_sha256,
    )


def write_periodic_checkpoint(
    directory: str | Path,
    checkpoint: PeriodicWorkflowCheckpoint,
) -> Path:
    """Atomically write a production-compatible checkpoint without pickle."""

    if not isinstance(checkpoint, PeriodicWorkflowCheckpoint):
        raise TypeError("checkpoint must be PeriodicWorkflowCheckpoint")
    with checkpoint_directory_transaction(
        directory,
        operation="periodic_checkpoint_export",
    ) as (staging, target):
        files = {"spatial": {}, "spectral": {}}
        for kind, tensors in (
            ("spatial", checkpoint.evolved_spatial),
            ("spectral", checkpoint.evolved_spectral),
        ):
            for name, value in tensors.items():
                files[kind][name] = _write_tensor(
                    staging / f"{kind}__{name}.npy",
                    value,
                )
        metadata = checkpoint.to_metadata()
        metadata["tensor_files"] = files
        if checkpoint.format_version in {
            PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        }:
            metadata = seal_checkpoint_metadata(metadata)
        if checkpoint.functional_bridge is not None:
            write_functional_checkpoint_provenance(
                staging,
                metadata,
                checkpoint.functional_bridge,
            )
        (staging / "checkpoint.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return target


def _read_metadata(directory: Path) -> Mapping[str, object]:
    metadata = read_checkpoint_json(
        directory / "checkpoint.json",
        operation="periodic_checkpoint_import",
    )
    version = metadata.get("format_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in _SUPPORTED_PERIODIC_WORKFLOW_CHECKPOINT_FORMATS
    ):
        raise CheckpointCompatibilityError(
            "unsupported periodic workflow checkpoint version",
            operation="periodic_checkpoint_import",
        )
    verify_checkpoint_metadata(
        metadata,
        required=version
        in {
            PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        },
        operation="periodic_checkpoint_import",
    )
    bridge = metadata.get("functional_bridge")
    bridge_version = (
        bridge.get("format_version") if isinstance(bridge, Mapping) else None
    )
    if version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        require_exact_checkpoint_keys(
            metadata,
            required=_PERIODIC_COMMON_REQUIRED_KEYS
            | {
                "run_provenance",
                "compatibility_identity",
                "compatibility_identity_sha256",
            },
            optional={"migration_provenance"},
            description="Periodic checkpoint metadata",
            operation="periodic_checkpoint_import",
        )
        require_exact_checkpoint_keys(
            metadata.get("run_provenance"),
            required={"legacy_runtime_identity_sha256"},
            description="Periodic checkpoint run provenance",
            operation="periodic_checkpoint_import",
        )
    elif (
        version == PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
        and bridge_version == 3
    ):
        require_exact_checkpoint_keys(
            metadata,
            required=_PERIODIC_COMMON_REQUIRED_KEYS
            | {"runtime_identity_sha256", "functional_bridge"},
            optional={"migration_provenance"},
            description="Periodic functional checkpoint metadata",
            operation="periodic_checkpoint_import",
        )
    if version in {
        PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    } and (version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION or bridge_version == 3):
        require_exact_checkpoint_keys(
            metadata.get("integrator"),
            required={"spectral_refresh_interval", "step_count", "refresh_count"},
            description="Periodic checkpoint integrator",
            operation="periodic_checkpoint_import",
        )
    verify_functional_checkpoint_provenance(
        directory,
        metadata,
        bridge,
        required=bridge_version in {2, 3},
    )
    return metadata


def read_periodic_checkpoint_header(
    directory: str | Path,
) -> PeriodicCheckpointHeader:
    """Read all small restart identities without opening tensor payloads."""

    metadata = _read_metadata(Path(directory).expanduser().resolve())
    if metadata.get("runtime_path") != PERIODIC_RUNTIME_PATH:
        raise CheckpointCompatibilityError(
            "checkpoint runtime identity does not match target",
            operation="periodic_checkpoint_import",
        )
    format_version = metadata.get("format_version")
    compatibility_identity = None
    if format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        run_provenance = metadata.get("run_provenance")
        if not isinstance(run_provenance, Mapping):
            raise CheckpointIntegrityError(
                "periodic checkpoint run provenance is missing",
                operation="periodic_checkpoint_import",
            )
        try:
            identity = _require_sha256(
                run_provenance.get("legacy_runtime_identity_sha256"),
                "legacy_runtime_identity_sha256",
            )
            compatibility_identity = (
                CheckpointCompatibilityIdentity.from_metadata(
                    metadata.get("compatibility_identity")
                )
            )
            expected = _require_sha256(
                metadata.get("compatibility_identity_sha256"),
                "compatibility_identity_sha256",
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointIntegrityError(
                "periodic checkpoint compatibility identity is invalid",
                operation="periodic_checkpoint_import",
                legacy_exception_type=type(exc).__name__,
            ) from exc
        if compatibility_identity.canonical_sha256() != expected:
            raise CheckpointIntegrityError(
                "periodic checkpoint compatibility identity checksum mismatch",
                operation="periodic_checkpoint_import",
            )
    else:
        try:
            identity = _require_sha256(
                metadata.get("runtime_identity_sha256"),
                "runtime_identity_sha256",
            )
        except ValueError as exc:
            raise CheckpointIntegrityError(
                "periodic checkpoint runtime identity is invalid",
                operation="periodic_checkpoint_import",
                legacy_exception_type=type(exc).__name__,
            ) from exc
    completed = metadata.get("completed_steps")
    integrator = metadata.get("integrator")
    if not isinstance(integrator, Mapping):
        raise CheckpointIntegrityError(
            "checkpoint integrator metadata is missing",
            operation="periodic_checkpoint_import",
        )
    try:
        return PeriodicCheckpointHeader(
            format_version=format_version,
            runtime_identity_sha256=identity,
            completed_steps=completed,
            spectral_refresh_interval=integrator.get("spectral_refresh_interval"),
            integrator_step_count=integrator.get("step_count"),
            integrator_refresh_count=integrator.get("refresh_count"),
            backend_restart=metadata.get("backend_restart"),
            functional_bridge=metadata.get("functional_bridge"),
            compatibility_identity=compatibility_identity,
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointIntegrityError(
            "periodic checkpoint header is invalid",
            operation="periodic_checkpoint_import",
            legacy_exception_type=type(exc).__name__,
        ) from exc


def _load_tensor(
    directory: Path,
    record: object,
    description: str,
    expected_filename: str,
) -> torch.Tensor:
    record = require_exact_checkpoint_keys(
        record,
        required={"file", "shape", "dtype", "sha256"},
        description=f"{description} tensor record",
        operation="periodic_checkpoint_import",
    )
    filename = record.get("file")
    path = resolve_checkpoint_payload(
        directory,
        filename,
        operation="periodic_checkpoint_import",
        description=f"{description} tensor",
        expected_filename=expected_filename,
    )
    try:
        expected = _require_sha256(record.get("sha256"), "tensor SHA-256")
    except ValueError as exc:
        raise CheckpointIntegrityError(
            f"{description} tensor SHA-256 is invalid",
            operation="periodic_checkpoint_import",
            legacy_exception_type=type(exc).__name__,
        ) from exc
    if _sha256(path) != expected:
        raise CheckpointIntegrityError(
            f"checkpoint tensor checksum mismatch: {path}",
            operation="periodic_checkpoint_import",
        )
    try:
        values = load_checkpoint_numpy(
            path,
            operation="periodic_checkpoint_import",
        )
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(
            f"checkpoint tensor is invalid: {path}",
            operation="periodic_checkpoint_import",
            legacy_exception_type=type(exc).__name__,
        ) from exc
    if list(values.shape) != record.get("shape"):
        raise CheckpointIntegrityError(
            f"checkpoint tensor shape metadata differs: {path}",
            operation="periodic_checkpoint_import",
        )
    if str(values.dtype) != record.get("dtype"):
        raise CheckpointIntegrityError(
            f"checkpoint tensor dtype metadata differs: {path}",
            operation="periodic_checkpoint_import",
        )
    if not np.isfinite(values).all():
        raise CheckpointIntegrityError(
            f"checkpoint tensor contains NaN or Inf: {path}",
            operation="periodic_checkpoint_import",
        )
    return torch.from_numpy(np.array(values, copy=True))


def load_periodic_checkpoint(
    directory: str | Path,
) -> PeriodicWorkflowCheckpoint:
    """Load and checksum-validate a complete periodic workflow checkpoint."""

    directory = Path(directory).expanduser().resolve()
    metadata = _read_metadata(directory)
    header = read_periodic_checkpoint_header(directory)
    files = metadata.get("tensor_files")
    files = require_exact_checkpoint_keys(
        files,
        required={"spatial", "spectral"},
        description="Periodic checkpoint tensor-file manifest",
        operation="periodic_checkpoint_import",
    )
    loaded = {}
    for kind in ("spatial", "spectral"):
        records = files.get(kind)
        if not isinstance(records, Mapping) or tuple(records) != Q_COMPONENTS:
            raise CheckpointIntegrityError(
                f"checkpoint {kind} manifest is incomplete",
                operation="periodic_checkpoint_import",
            )
        loaded[kind] = {
            name: _load_tensor(
                directory,
                records[name],
                f"{kind}.{name}",
                f"{kind}__{name}.npy",
            )
            for name in Q_COMPONENTS
        }
    return PeriodicWorkflowCheckpoint(
        format_version=metadata.get("format_version"),
        runtime_identity_sha256=header.runtime_identity_sha256,
        completed_steps=header.completed_steps,
        spectral_refresh_interval=header.spectral_refresh_interval,
        integrator_step_count=header.integrator_step_count,
        integrator_refresh_count=header.integrator_refresh_count,
        evolved_spatial=loaded["spatial"],
        evolved_spectral=loaded["spectral"],
        backend_restart=header.backend_restart,
        functional_bridge=header.functional_bridge,
        compatibility_identity=header.compatibility_identity,
        migration_provenance=metadata.get("migration_provenance"),
    )


def _legacy_periodic_format_for_generation(
    generation: SourceReleaseGeneration,
) -> int:
    if not isinstance(generation, SourceReleaseGeneration):
        raise TypeError("source_release_generation must be a SourceReleaseGeneration")
    return (
        PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1
        if generation is SourceReleaseGeneration.RC1
        else PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    )
