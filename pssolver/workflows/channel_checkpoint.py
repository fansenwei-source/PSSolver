"""Versioned exact-restart checkpoints for the P7.4 Channel facade."""

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

from pssolver.configuration.channel_active_nematics_declarations import (
    ChannelRuntimePath,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.channel_active_nematics import (
    ChannelRuntimeAdapterProtocol,
)


CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION = 1
CHANNEL_ALGEBRAIC_COMPONENTS = ("ux", "uy", "uz", "p")


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


def _clone_component_mapping(
    values: Mapping[str, torch.Tensor],
    description: str,
    components: tuple[str, ...],
) -> Mapping[str, torch.Tensor]:
    if not isinstance(values, Mapping) or tuple(values) != components:
        raise ValueError(
            f"{description} must contain ordered components {components!r}"
        )
    cloned = {}
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{description} values must be tensors")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{description} tensors must be finite")
        cloned[name] = value.detach().clone()
    return MappingProxyType(cloned)


def _clone_pressure(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("pressure_guess must be a tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("pressure_guess must be finite")
    return value.detach().clone()


@dataclass(frozen=True, slots=True)
class ChannelCheckpointHeader:
    """Identity-only header used to reject incompatible restarts early."""

    runtime_path: ChannelRuntimePath
    runtime_identity_sha256: str
    completed_steps: int


@dataclass(frozen=True, slots=True)
class ChannelWorkflowCheckpoint:
    """Complete evolved state, refresh phase, and PCG warm-start state."""

    format_version: int
    runtime_path: ChannelRuntimePath
    runtime_identity_sha256: str
    completed_steps: int
    spectral_refresh_interval: int | None
    integrator_step_count: int
    integrator_refresh_count: int
    evolved_spatial: Mapping[str, torch.Tensor]
    evolved_spectral: Mapping[str, torch.Tensor]
    algebraic_spatial: Mapping[str, torch.Tensor]
    algebraic_spectral: Mapping[str, torch.Tensor]
    pressure_guess: torch.Tensor
    backend_restart: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.format_version != CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported Channel checkpoint version")
        if not isinstance(self.runtime_path, ChannelRuntimePath):
            raise TypeError("runtime_path must be a ChannelRuntimePath")
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
                self.integrator_step_count == self.completed_steps
                and self.integrator_refresh_count == 0
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
        spatial = _clone_component_mapping(
            self.evolved_spatial,
            "evolved_spatial",
            Q_COMPONENTS,
        )
        spectral = _clone_component_mapping(
            self.evolved_spectral,
            "evolved_spectral",
            Q_COMPONENTS,
        )
        algebraic_spatial = _clone_component_mapping(
            self.algebraic_spatial,
            "algebraic_spatial",
            CHANNEL_ALGEBRAIC_COMPONENTS,
        )
        algebraic_spectral = _clone_component_mapping(
            self.algebraic_spectral,
            "algebraic_spectral",
            CHANNEL_ALGEBRAIC_COMPONENTS,
        )
        pressure = _clone_pressure(self.pressure_guess)
        if not isinstance(self.backend_restart, Mapping):
            raise TypeError("backend_restart must be a mapping")
        backend = dict(self.backend_restart)
        try:
            json.dumps(backend, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "backend restart metadata is not JSON-compatible"
            ) from exc
        object.__setattr__(self, "evolved_spatial", spatial)
        object.__setattr__(self, "evolved_spectral", spectral)
        object.__setattr__(self, "algebraic_spatial", algebraic_spatial)
        object.__setattr__(self, "algebraic_spectral", algebraic_spectral)
        object.__setattr__(self, "pressure_guess", pressure)
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
            "algebraic_components": list(CHANNEL_ALGEBRAIC_COMPONENTS),
            "persistent_algebraic": ["pressure_guess"],
            "backend_restart": dict(self.backend_restart),
        }


def capture_channel_checkpoint(
    adapter: ChannelRuntimeAdapterProtocol,
    *,
    runtime_identity_sha256: str,
) -> ChannelWorkflowCheckpoint:
    """Synchronize u/p, then capture exact Q and PCG warm-start state."""

    if not isinstance(adapter, ChannelRuntimeAdapterProtocol):
        raise TypeError("adapter must implement ChannelRuntimeAdapterProtocol")
    _require_sha256(runtime_identity_sha256, "runtime_identity_sha256")
    adapter.synchronize_for_observation()
    fields = adapter.fields
    integrator = adapter.solver.integrator
    return ChannelWorkflowCheckpoint(
        format_version=CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
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
        algebraic_spatial={
            name: fields[name] for name in CHANNEL_ALGEBRAIC_COMPONENTS
        },
        algebraic_spectral={
            name: fields[f"{name}.hat"]
            for name in CHANNEL_ALGEBRAIC_COMPONENTS
        },
        pressure_guess=adapter.capture_pressure_guess(),
        backend_restart=adapter.backend_restart_metadata(),
    )


def _validated_copy_plan(
    adapter: ChannelRuntimeAdapterProtocol,
    checkpoint: ChannelWorkflowCheckpoint,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    copies = []
    for name in Q_COMPONENTS:
        for suffix, source in (
            ("", checkpoint.evolved_spatial[name]),
            (".hat", checkpoint.evolved_spectral[name]),
        ):
            target = adapter.fields[f"{name}{suffix}"]
            if target.shape != source.shape:
                raise ValueError(f"checkpoint shape for {name}{suffix} differs")
            if target.dtype != source.dtype:
                raise ValueError(f"checkpoint dtype for {name}{suffix} differs")
            if not bool(torch.isfinite(source).all().item()):
                raise ValueError(f"checkpoint tensor for {name}{suffix} is non-finite")
            copies.append((target, source))
    for name in CHANNEL_ALGEBRAIC_COMPONENTS:
        for suffix, source in (
            ("", checkpoint.algebraic_spatial[name]),
            (".hat", checkpoint.algebraic_spectral[name]),
        ):
            target = adapter.fields[f"{name}{suffix}"]
            if target.shape != source.shape:
                raise ValueError(f"checkpoint shape for {name}{suffix} differs")
            if target.dtype != source.dtype:
                raise ValueError(f"checkpoint dtype for {name}{suffix} differs")
            if not bool(torch.isfinite(source).all().item()):
                raise ValueError(f"checkpoint tensor for {name}{suffix} is non-finite")
            copies.append((target, source))
    target_pressure = adapter.fields["p.hat"]
    source_pressure = checkpoint.pressure_guess
    if target_pressure.shape != source_pressure.shape:
        raise ValueError("checkpoint pressure_guess shape differs")
    if target_pressure.dtype != source_pressure.dtype:
        raise ValueError("checkpoint pressure_guess dtype differs")
    if not bool(torch.isfinite(source_pressure).all().item()):
        raise ValueError("checkpoint pressure_guess is non-finite")
    return tuple(copies)


def restore_channel_checkpoint(
    adapter: ChannelRuntimeAdapterProtocol,
    checkpoint: ChannelWorkflowCheckpoint,
    *,
    runtime_identity_sha256: str,
) -> int:
    """Restore a fully validated checkpoint on its exact runtime path."""

    if not isinstance(adapter, ChannelRuntimeAdapterProtocol):
        raise TypeError("adapter must implement ChannelRuntimeAdapterProtocol")
    if not isinstance(checkpoint, ChannelWorkflowCheckpoint):
        raise TypeError("checkpoint must be ChannelWorkflowCheckpoint")
    if checkpoint.runtime_path is not adapter.runtime_path:
        raise ValueError("cross-runtime Channel checkpoint restart is unsupported")
    if checkpoint.runtime_identity_sha256 != runtime_identity_sha256:
        raise ValueError("checkpoint runtime identity does not match target")
    if dict(checkpoint.backend_restart) != adapter.backend_restart_metadata():
        raise ValueError("checkpoint backend restart contract does not match target")
    copies = _validated_copy_plan(adapter, checkpoint)

    for target, source in copies:
        target.copy_(source.to(device=target.device))
    current_pressure = adapter.fields["p.hat"]
    adapter.restore_pressure_guess(
        checkpoint.pressure_guess.to(device=current_pressure.device)
    )
    adapter.restore_progress(
        completed_steps=checkpoint.completed_steps,
        spectral_refresh_interval=checkpoint.spectral_refresh_interval,
        integrator_step_count=checkpoint.integrator_step_count,
        integrator_refresh_count=checkpoint.integrator_refresh_count,
    )
    if adapter.completed_steps != checkpoint.completed_steps:
        raise RuntimeError("restored Channel completed-step count is inconsistent")
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


def write_channel_checkpoint(
    directory: str | Path,
    checkpoint: ChannelWorkflowCheckpoint,
) -> Path:
    """Atomically write a new Channel checkpoint without pickle."""

    if not isinstance(checkpoint, ChannelWorkflowCheckpoint):
        raise TypeError("checkpoint must be ChannelWorkflowCheckpoint")
    target = Path(directory).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint directory already exists: {target}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    )
    try:
        files = {
            "evolved_spatial": {},
            "evolved_spectral": {},
            "algebraic_spatial": {},
            "algebraic_spectral": {},
        }
        for kind, tensors in (
            ("evolved_spatial", checkpoint.evolved_spatial),
            ("evolved_spectral", checkpoint.evolved_spectral),
            ("algebraic_spatial", checkpoint.algebraic_spatial),
            ("algebraic_spectral", checkpoint.algebraic_spectral),
        ):
            for name, value in tensors.items():
                files[kind][name] = _write_tensor(
                    staging / f"{kind}__{name}.npy",
                    value,
                )
        files["pressure_guess"] = _write_tensor(
            staging / "persistent__pressure_guess.npy",
            checkpoint.pressure_guess,
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


def read_channel_checkpoint_header(
    directory: str | Path,
) -> ChannelCheckpointHeader:
    """Read only the identity needed for early runtime rejection."""

    metadata = _read_metadata(Path(directory).expanduser().resolve())
    if metadata.get("format_version") != CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported Channel checkpoint version")
    try:
        runtime_path = ChannelRuntimePath(metadata.get("runtime_path"))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint runtime path is invalid") from exc
    identity = _require_sha256(
        metadata.get("runtime_identity_sha256"),
        "runtime_identity_sha256",
    )
    completed = metadata.get("completed_steps")
    if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
        raise ValueError("checkpoint completed_steps is invalid")
    return ChannelCheckpointHeader(runtime_path, identity, completed)


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


def load_channel_checkpoint(directory: str | Path) -> ChannelWorkflowCheckpoint:
    """Load and checksum-validate a complete Channel checkpoint."""

    directory = Path(directory).expanduser().resolve()
    metadata = _read_metadata(directory)
    header = read_channel_checkpoint_header(directory)
    files = metadata.get("tensor_files")
    if not isinstance(files, Mapping):
        raise ValueError("checkpoint tensor-file manifest is missing")
    loaded = {}
    for kind, components in (
        ("evolved_spatial", Q_COMPONENTS),
        ("evolved_spectral", Q_COMPONENTS),
        ("algebraic_spatial", CHANNEL_ALGEBRAIC_COMPONENTS),
        ("algebraic_spectral", CHANNEL_ALGEBRAIC_COMPONENTS),
    ):
        records = files.get(kind)
        if (
            not isinstance(records, Mapping)
            or len(records) != len(components)
            or set(records) != set(components)
        ):
            raise ValueError(f"checkpoint {kind} manifest is incomplete")
        loaded[kind] = {
            name: _load_tensor(directory, records[name], f"{kind}.{name}")
            for name in components
        }
    integrator = metadata.get("integrator")
    if not isinstance(integrator, Mapping):
        raise ValueError("checkpoint integrator metadata is missing")
    backend = metadata.get("backend_restart")
    if not isinstance(backend, Mapping):
        raise ValueError("checkpoint backend restart metadata is missing")
    return ChannelWorkflowCheckpoint(
        format_version=metadata.get("format_version"),
        runtime_path=header.runtime_path,
        runtime_identity_sha256=header.runtime_identity_sha256,
        completed_steps=header.completed_steps,
        spectral_refresh_interval=integrator.get("spectral_refresh_interval"),
        integrator_step_count=integrator.get("step_count"),
        integrator_refresh_count=integrator.get("refresh_count"),
        evolved_spatial=loaded["evolved_spatial"],
        evolved_spectral=loaded["evolved_spectral"],
        algebraic_spatial=loaded["algebraic_spatial"],
        algebraic_spectral=loaded["algebraic_spectral"],
        pressure_guess=_load_tensor(
            directory,
            files.get("pressure_guess"),
            "pressure_guess",
        ),
        backend_restart=backend,
    )


__all__ = [
    "CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
    "CHANNEL_ALGEBRAIC_COMPONENTS",
    "ChannelCheckpointHeader",
    "ChannelWorkflowCheckpoint",
    "capture_channel_checkpoint",
    "load_channel_checkpoint",
    "read_channel_checkpoint_header",
    "restore_channel_checkpoint",
    "write_channel_checkpoint",
]
