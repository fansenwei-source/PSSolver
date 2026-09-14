"""Versioned exact-restart checkpoints for the dual-path Plane workflow."""

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

from pssolver.configuration import PlaneRuntimePath
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime import PlaneRuntimeAdapterProtocol


PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 1


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


@dataclass(frozen=True, slots=True)
class PlaneCheckpointHeader:
    """Small header that can reject cross-backend restart before arrays load."""

    runtime_path: PlaneRuntimePath
    runtime_identity_sha256: str
    completed_steps: int


@dataclass(frozen=True, slots=True)
class PlaneWorkflowCheckpoint:
    """Complete evolved state and refresh phase for one Plane backend."""

    format_version: int
    runtime_path: PlaneRuntimePath
    runtime_identity_sha256: str
    completed_steps: int
    spectral_refresh_interval: int | None
    integrator_step_count: int
    integrator_refresh_count: int
    evolved_spatial: Mapping[str, torch.Tensor]
    evolved_spectral: Mapping[str, torch.Tensor]
    backend_restart: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.format_version != PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported Plane workflow checkpoint version")
        if not isinstance(self.runtime_path, PlaneRuntimePath):
            raise TypeError("checkpoint runtime_path must be PlaneRuntimePath")
        _require_sha256(
            self.runtime_identity_sha256,
            "runtime_identity_sha256",
        )
        for value, description in (
            (self.completed_steps, "completed_steps"),
            (self.integrator_step_count, "integrator_step_count"),
            (self.integrator_refresh_count, "integrator_refresh_count"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{description} must be non-negative")
        interval = self.spectral_refresh_interval
        if interval is not None and (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval <= 0
        ):
            raise ValueError("spectral refresh interval must be positive or None")
        if interval is None:
            counters_match = (
                self.integrator_refresh_count == 0
                and self.integrator_step_count == self.completed_steps
            )
        else:
            counters_match = (
                self.integrator_step_count < interval
                and self.integrator_refresh_count * interval
                + self.integrator_step_count
                == self.completed_steps
            )
        if not counters_match:
            raise ValueError("checkpoint refresh counters are inconsistent")
        spatial = _clone_tensors(self.evolved_spatial, "evolved_spatial")
        spectral = _clone_tensors(self.evolved_spectral, "evolved_spectral")
        if not isinstance(self.backend_restart, Mapping):
            raise TypeError("backend_restart must be a mapping")
        backend = dict(self.backend_restart)
        try:
            json.dumps(backend, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("backend restart metadata is not JSON-compatible") from exc
        object.__setattr__(self, "evolved_spatial", spatial)
        object.__setattr__(self, "evolved_spectral", spectral)
        object.__setattr__(self, "backend_restart", MappingProxyType(backend))

    def to_metadata(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "runtime_path": self.runtime_path.value,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "completed_steps": self.completed_steps,
            "integrator": {
                "spectral_refresh_interval": self.spectral_refresh_interval,
                "step_count": self.integrator_step_count,
                "refresh_count": self.integrator_refresh_count,
            },
            "evolved_components": list(Q_COMPONENTS),
            "backend_restart": dict(self.backend_restart),
        }


def capture_plane_checkpoint(
    adapter: PlaneRuntimeAdapterProtocol,
    *,
    runtime_identity_sha256: str,
) -> PlaneWorkflowCheckpoint:
    """Synchronize algebraic fields and capture exact evolved state."""

    if not isinstance(adapter, PlaneRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PlaneRuntimeAdapterProtocol")
    _require_sha256(runtime_identity_sha256, "runtime_identity_sha256")
    adapter.synchronize_for_observation()
    fields = adapter.fields
    integrator = adapter.solver.integrator
    return PlaneWorkflowCheckpoint(
        format_version=PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_path=adapter.runtime_path,
        runtime_identity_sha256=runtime_identity_sha256,
        completed_steps=adapter.completed_steps,
        spectral_refresh_interval=integrator.spectral_refresh_interval,
        integrator_step_count=int(integrator.step_count),
        integrator_refresh_count=int(integrator.refresh_count),
        evolved_spatial={name: fields[name] for name in Q_COMPONENTS},
        evolved_spectral={
            name: fields[f"{name}.hat"] for name in Q_COMPONENTS
        },
        backend_restart=adapter.backend_restart_metadata(),
    )


def restore_plane_checkpoint(
    adapter: PlaneRuntimeAdapterProtocol,
    checkpoint: PlaneWorkflowCheckpoint,
    *,
    runtime_identity_sha256: str,
) -> int:
    """Restore a checksum-validated checkpoint on its exact runtime path."""

    if not isinstance(adapter, PlaneRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PlaneRuntimeAdapterProtocol")
    if not isinstance(checkpoint, PlaneWorkflowCheckpoint):
        raise TypeError("checkpoint must be PlaneWorkflowCheckpoint")
    if checkpoint.runtime_path is not adapter.runtime_path:
        raise ValueError("cross-runtime Plane checkpoint restart is unsupported")
    if checkpoint.runtime_identity_sha256 != runtime_identity_sha256:
        raise ValueError("checkpoint runtime identity does not match target")
    if dict(checkpoint.backend_restart) != adapter.backend_restart_metadata():
        raise ValueError("checkpoint backend restart contract does not match target")

    fields = adapter.fields
    for name in Q_COMPONENTS:
        for suffix, source in (
            ("", checkpoint.evolved_spatial[name]),
            (".hat", checkpoint.evolved_spectral[name]),
        ):
            target = fields[f"{name}{suffix}"]
            if target.shape != source.shape:
                raise ValueError(f"checkpoint shape for {name}{suffix} differs")
            if target.dtype != source.dtype:
                raise ValueError(f"checkpoint dtype for {name}{suffix} differs")
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


def write_plane_checkpoint(
    directory: str | Path,
    checkpoint: PlaneWorkflowCheckpoint,
) -> Path:
    """Atomically write a new checkpoint directory without pickle."""

    if not isinstance(checkpoint, PlaneWorkflowCheckpoint):
        raise TypeError("checkpoint must be PlaneWorkflowCheckpoint")
    target = Path(directory).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint directory already exists: {target}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        files = {"evolved_spatial": {}, "evolved_spectral": {}}
        for kind, tensors in (
            ("evolved_spatial", checkpoint.evolved_spatial),
            ("evolved_spectral", checkpoint.evolved_spectral),
        ):
            for name, value in tensors.items():
                files[kind][name] = _write_tensor(
                    staging / f"{kind}__{name}.npy",
                    value,
                )
        metadata = checkpoint.to_metadata()
        metadata["tensor_files"] = files
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
    return metadata


def read_plane_checkpoint_header(directory: str | Path) -> PlaneCheckpointHeader:
    """Read only the versioned identity fields needed for an early gate."""

    metadata = _read_metadata(Path(directory).expanduser().resolve())
    if metadata.get("format_version") != PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported Plane workflow checkpoint version")
    try:
        runtime_path = PlaneRuntimePath(metadata.get("runtime_path"))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint runtime path is invalid") from exc
    identity = _require_sha256(
        metadata.get("runtime_identity_sha256"),
        "runtime_identity_sha256",
    )
    completed = metadata.get("completed_steps")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
        raise ValueError("checkpoint completed_steps is invalid")
    return PlaneCheckpointHeader(runtime_path, identity, completed)


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


def load_plane_checkpoint(directory: str | Path) -> PlaneWorkflowCheckpoint:
    """Load and checksum-validate a complete Plane workflow checkpoint."""

    directory = Path(directory).expanduser().resolve()
    metadata = _read_metadata(directory)
    header = read_plane_checkpoint_header(directory)
    files = metadata.get("tensor_files")
    if not isinstance(files, Mapping):
        raise ValueError("checkpoint tensor-file manifest is missing")
    loaded = {}
    for kind in ("evolved_spatial", "evolved_spectral"):
        records = files.get(kind)
        if not isinstance(records, Mapping) or tuple(records) != Q_COMPONENTS:
            raise ValueError(f"checkpoint {kind} manifest is incomplete")
        loaded[kind] = {
            name: _load_tensor(directory, records[name], f"{kind}.{name}")
            for name in Q_COMPONENTS
        }
    integrator = metadata.get("integrator")
    if not isinstance(integrator, Mapping):
        raise ValueError("checkpoint integrator metadata is missing")
    backend = metadata.get("backend_restart")
    if not isinstance(backend, Mapping):
        raise ValueError("checkpoint backend restart metadata is missing")
    return PlaneWorkflowCheckpoint(
        format_version=metadata.get("format_version"),
        runtime_path=header.runtime_path,
        runtime_identity_sha256=header.runtime_identity_sha256,
        completed_steps=header.completed_steps,
        spectral_refresh_interval=integrator.get("spectral_refresh_interval"),
        integrator_step_count=integrator.get("step_count"),
        integrator_refresh_count=integrator.get("refresh_count"),
        evolved_spatial=loaded["evolved_spatial"],
        evolved_spectral=loaded["evolved_spectral"],
        backend_restart=backend,
    )


__all__ = [
    "PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "PlaneCheckpointHeader",
    "PlaneWorkflowCheckpoint",
    "capture_plane_checkpoint",
    "load_plane_checkpoint",
    "read_plane_checkpoint_header",
    "restore_plane_checkpoint",
    "write_plane_checkpoint",
]
