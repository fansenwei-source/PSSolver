"""Versioned exact-restart checkpoints for the periodic Beris--Edwards runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType

import numpy as np
import torch

from pssolver.configuration.periodic_beris_edwards import PERIODIC_RUNTIME_PATH
from pssolver.io.checkpoint import (
    seal_checkpoint_metadata,
    verify_checkpoint_metadata,
    verify_functional_checkpoint_provenance,
    write_functional_checkpoint_provenance,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.periodic_beris_edwards import PeriodicRuntimeAdapterProtocol


PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 2
_LEGACY_PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 1


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


def _json_mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    try:
        encoded = json.dumps(
            dict(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
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

    def __post_init__(self) -> None:
        if (
            not isinstance(self.format_version, int)
            or isinstance(self.format_version, bool)
            or self.format_version not in {
                _LEGACY_PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
                PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            }
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

    def __post_init__(self) -> None:
        if (
            not isinstance(self.format_version, int)
            or isinstance(self.format_version, bool)
            or self.format_version not in {
                _LEGACY_PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
                PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            }
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

    def to_metadata(self) -> dict[str, object]:
        metadata = {
            "format_version": self.format_version,
            "runtime_path": PERIODIC_RUNTIME_PATH,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "completed_steps": self.completed_steps,
            "integrator": {
                "spectral_refresh_interval": self.spectral_refresh_interval,
                "step_count": self.integrator_step_count,
                "refresh_count": self.integrator_refresh_count,
            },
            "backend_restart": dict(self.backend_restart),
        }
        if self.functional_bridge is not None:
            metadata["functional_bridge"] = dict(self.functional_bridge)
        return metadata


def capture_periodic_checkpoint(
    adapter: PeriodicRuntimeAdapterProtocol,
    *,
    runtime_identity_sha256: str,
) -> PeriodicWorkflowCheckpoint:
    """Synchronize algebraic fields and capture the exact production state."""

    if not isinstance(adapter, PeriodicRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PeriodicRuntimeAdapterProtocol")
    _require_sha256(runtime_identity_sha256, "runtime_identity_sha256")
    adapter.synchronize_for_observation()
    fields = adapter.fields
    integrator = adapter.solver.integrator
    return PeriodicWorkflowCheckpoint(
        format_version=PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_identity_sha256=runtime_identity_sha256,
        completed_steps=adapter.completed_steps,
        spectral_refresh_interval=integrator.spectral_refresh_interval,
        integrator_step_count=int(integrator.step_count),
        integrator_refresh_count=int(integrator.refresh_count),
        evolved_spatial={name: fields[name] for name in Q_COMPONENTS},
        evolved_spectral={name: fields[f"{name}.hat"] for name in Q_COMPONENTS},
        backend_restart=adapter.backend_restart_metadata(),
    )


def restore_periodic_checkpoint(
    adapter: PeriodicRuntimeAdapterProtocol,
    checkpoint: PeriodicWorkflowCheckpoint,
    *,
    runtime_identity_sha256: str,
) -> int:
    """Restore only after every identity, shape, and dtype gate has passed."""

    if not isinstance(adapter, PeriodicRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PeriodicRuntimeAdapterProtocol")
    if not isinstance(checkpoint, PeriodicWorkflowCheckpoint):
        raise TypeError("checkpoint must be PeriodicWorkflowCheckpoint")
    if checkpoint.runtime_identity_sha256 != runtime_identity_sha256:
        raise ValueError("checkpoint scientific/runtime identity does not match")
    if dict(checkpoint.backend_restart) != adapter.backend_restart_metadata():
        raise ValueError("checkpoint backend contract does not match target")
    fields = adapter.fields
    targets: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    for name in Q_COMPONENTS:
        for suffix, source in (
            ("", checkpoint.evolved_spatial[name]),
            (".hat", checkpoint.evolved_spectral[name]),
        ):
            target = fields[f"{name}{suffix}"]
            if target.shape != source.shape:
                raise ValueError(f"checkpoint tensor shape differs for {name}{suffix}")
            if target.dtype != source.dtype:
                raise ValueError(f"checkpoint tensor dtype differs for {name}{suffix}")
            targets.append((target, source, f"{name}{suffix}"))

    # No target is mutated until the complete preflight above succeeds.
    for target, source, _description in targets:
        target.copy_(source.to(device=target.device))
    adapter.synchronize_for_observation()
    adapter.restore_progress(
        completed_steps=checkpoint.completed_steps,
        spectral_refresh_interval=checkpoint.spectral_refresh_interval,
        integrator_step_count=checkpoint.integrator_step_count,
        integrator_refresh_count=checkpoint.integrator_refresh_count,
    )
    if adapter.completed_steps != checkpoint.completed_steps:
        raise RuntimeError("restored completed-step count is inconsistent")
    return checkpoint.completed_steps


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


def write_periodic_checkpoint(
    directory: str | Path,
    checkpoint: PeriodicWorkflowCheckpoint,
) -> Path:
    """Atomically write a production-compatible checkpoint without pickle."""

    if not isinstance(checkpoint, PeriodicWorkflowCheckpoint):
        raise TypeError("checkpoint must be PeriodicWorkflowCheckpoint")
    target = Path(directory).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint directory already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    # mkdtemp deliberately creates mode 0700.  The atomic rename would make
    # that private staging mode the durable checkpoint mode, preventing
    # read-only service accounts from consuming an otherwise shared run.
    staging.chmod(0o755)
    try:
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
        if checkpoint.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
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
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def _read_metadata(directory: Path) -> Mapping[str, object]:
    path = directory / "checkpoint.json"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint metadata is missing: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint metadata is not valid JSON") from exc
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a JSON object")
    version = metadata.get("format_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version
        not in {
            _LEGACY_PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        }
    ):
        raise ValueError("unsupported periodic workflow checkpoint version")
    verify_checkpoint_metadata(
        metadata,
        required=version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    )
    bridge = metadata.get("functional_bridge")
    bridge_version = (
        bridge.get("format_version") if isinstance(bridge, Mapping) else None
    )
    verify_functional_checkpoint_provenance(
        directory,
        metadata,
        bridge,
        required=bridge_version == 2,
    )
    return metadata


def read_periodic_checkpoint_header(
    directory: str | Path,
) -> PeriodicCheckpointHeader:
    """Read all small restart identities without opening tensor payloads."""

    metadata = _read_metadata(Path(directory).expanduser().resolve())
    if metadata.get("runtime_path") != PERIODIC_RUNTIME_PATH:
        raise ValueError("checkpoint runtime identity does not match target")
    identity = _require_sha256(
        metadata.get("runtime_identity_sha256"),
        "runtime_identity_sha256",
    )
    completed = metadata.get("completed_steps")
    integrator = metadata.get("integrator")
    if not isinstance(integrator, Mapping):
        raise ValueError("checkpoint integrator metadata is missing")
    return PeriodicCheckpointHeader(
        format_version=metadata.get("format_version"),
        runtime_identity_sha256=identity,
        completed_steps=completed,
        spectral_refresh_interval=integrator.get("spectral_refresh_interval"),
        integrator_step_count=integrator.get("step_count"),
        integrator_refresh_count=integrator.get("refresh_count"),
        backend_restart=metadata.get("backend_restart"),
        functional_bridge=metadata.get("functional_bridge"),
    )


def _load_tensor(
    directory: Path,
    record: object,
    description: str,
) -> torch.Tensor:
    if not isinstance(record, Mapping):
        raise TypeError(f"{description} tensor record must be a mapping")
    filename = record.get("file")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".npy")
    ):
        raise ValueError(f"{description} tensor filename is invalid")
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint tensor is missing: {path}")
    expected = _require_sha256(record.get("sha256"), "tensor SHA-256")
    if _sha256(path) != expected:
        raise ValueError(f"checkpoint tensor checksum mismatch: {path}")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != record.get("shape"):
        raise ValueError(f"checkpoint tensor shape metadata differs: {path}")
    if str(values.dtype) != record.get("dtype"):
        raise ValueError(f"checkpoint tensor dtype metadata differs: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"checkpoint tensor contains NaN or Inf: {path}")
    return torch.from_numpy(np.array(values, copy=True))


def load_periodic_checkpoint(
    directory: str | Path,
) -> PeriodicWorkflowCheckpoint:
    """Load and checksum-validate a complete periodic workflow checkpoint."""

    directory = Path(directory).expanduser().resolve()
    metadata = _read_metadata(directory)
    header = read_periodic_checkpoint_header(directory)
    files = metadata.get("tensor_files")
    if not isinstance(files, Mapping):
        raise ValueError("checkpoint tensor-file manifest is missing")
    loaded = {}
    for kind in ("spatial", "spectral"):
        records = files.get(kind)
        if not isinstance(records, Mapping) or tuple(records) != Q_COMPONENTS:
            raise ValueError(f"checkpoint {kind} manifest is incomplete")
        loaded[kind] = {
            name: _load_tensor(directory, records[name], f"{kind}.{name}")
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
    )


__all__ = [
    "PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PeriodicCheckpointHeader",
    "PeriodicWorkflowCheckpoint",
    "capture_periodic_checkpoint",
    "load_periodic_checkpoint",
    "read_periodic_checkpoint_header",
    "restore_periodic_checkpoint",
    "write_periodic_checkpoint",
]
