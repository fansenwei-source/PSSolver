"""Complete, checksum-validated checkpoints for experimental runtimes."""

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

from pssolver.execution import (
    AlgebraicRuntimeRestartState,
    AlgebraicSystemRestartState,
)

from ._shadow_support import (
    SHADOW_CHECKPOINT_FORMAT_VERSION,
    completed_steps,
    evolved_names,
    ordered_tensor_sha256,
    require_nonnegative_integer,
    require_sha256,
    shadow_runtime_identity_sha256,
)
from .model_execution import ExperimentalModelRuntime


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_tensor_mapping(
    values: Mapping[str, torch.Tensor],
    description: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{description} must be a mapping")
    cloned: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"{description} keys must be Python identifiers")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{description} values must be tensors")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{description} tensors must be finite")
        cloned[name] = value.detach().clone()
    return MappingProxyType(cloned)


def _tensor_metadata(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device_at_capture": str(value.device),
        "sha256": ordered_tensor_sha256({"value": value}),
    }


@dataclass(frozen=True, slots=True)
class ShadowRunCheckpoint:
    """Complete evolved and solver state for an exact shadow-run restart."""

    format_version: int
    runtime_identity_sha256: str
    completed_steps: int
    spectral_refresh_interval: int | None
    integrator_step_count: int
    integrator_refresh_count: int
    evolved_spatial: Mapping[str, torch.Tensor]
    evolved_spectral: Mapping[str, torch.Tensor]
    algebraic_restart: AlgebraicRuntimeRestartState

    def __post_init__(self) -> None:
        if self.format_version != SHADOW_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported shadow checkpoint format version")
        require_sha256(
            self.runtime_identity_sha256,
            "runtime_identity_sha256",
        )
        completed = require_nonnegative_integer(
            self.completed_steps,
            "completed_steps",
        )
        step_count = require_nonnegative_integer(
            self.integrator_step_count,
            "integrator_step_count",
        )
        refresh_count = require_nonnegative_integer(
            self.integrator_refresh_count,
            "integrator_refresh_count",
        )
        interval = self.spectral_refresh_interval
        if interval is not None and (
            not isinstance(interval, int)
            or isinstance(interval, bool)
            or interval <= 0
        ):
            raise ValueError(
                "spectral_refresh_interval must be positive or None"
            )
        if interval is None:
            if refresh_count != 0 or step_count != completed:
                raise ValueError(
                    "disabled-refresh counters are inconsistent with steps"
                )
        elif (
            step_count >= interval
            or refresh_count * interval + step_count != completed
        ):
            raise ValueError(
                "spectral-refresh counters are inconsistent with steps"
            )
        spatial = _clone_tensor_mapping(
            self.evolved_spatial,
            "evolved_spatial",
        )
        spectral = _clone_tensor_mapping(
            self.evolved_spectral,
            "evolved_spectral",
        )
        if not spatial or tuple(spatial) != tuple(spectral):
            raise ValueError(
                "spatial and spectral checkpoint components must match"
            )
        if not isinstance(
            self.algebraic_restart,
            AlgebraicRuntimeRestartState,
        ):
            raise TypeError(
                "algebraic_restart must be an AlgebraicRuntimeRestartState"
            )
        object.__setattr__(self, "evolved_spatial", spatial)
        object.__setattr__(self, "evolved_spectral", spectral)

    def to_metadata(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "completed_steps": self.completed_steps,
            "integrator": {
                "spectral_refresh_interval": self.spectral_refresh_interval,
                "step_count": self.integrator_step_count,
                "refresh_count": self.integrator_refresh_count,
            },
            "evolved_components": list(self.evolved_spatial),
            "evolved_spatial": {
                name: _tensor_metadata(value)
                for name, value in self.evolved_spatial.items()
            },
            "evolved_spectral": {
                name: _tensor_metadata(value)
                for name, value in self.evolved_spectral.items()
            },
            "algebraic_restart": self.algebraic_restart.to_metadata(),
        }


def capture_shadow_checkpoint(
    runtime: ExperimentalModelRuntime,
) -> ShadowRunCheckpoint:
    """Synchronize observations and capture a complete restart state."""

    if not isinstance(runtime, ExperimentalModelRuntime):
        raise TypeError("runtime must be an ExperimentalModelRuntime")
    runtime.synchronize_algebraic_for_observation()
    names = evolved_names(runtime)
    fields = runtime.solver.fields
    integrator = runtime.solver.integrator
    return ShadowRunCheckpoint(
        format_version=SHADOW_CHECKPOINT_FORMAT_VERSION,
        runtime_identity_sha256=shadow_runtime_identity_sha256(runtime),
        completed_steps=completed_steps(runtime),
        spectral_refresh_interval=integrator.spectral_refresh_interval,
        integrator_step_count=int(integrator.step_count),
        integrator_refresh_count=int(integrator.refresh_count),
        evolved_spatial={name: fields[name] for name in names},
        evolved_spectral={name: fields[f"{name}.hat"] for name in names},
        algebraic_restart=runtime.capture_algebraic_restart_state(),
    )


def _restart_on_runtime_device(
    runtime: ExperimentalModelRuntime,
    restart: AlgebraicRuntimeRestartState,
) -> AlgebraicRuntimeRestartState:
    return AlgebraicRuntimeRestartState(
        restart.format_version,
        tuple(
            AlgebraicSystemRestartState(
                system_name=system.system_name,
                capability=system.capability,
                implementation_name=system.implementation_name,
                provenance_sha256=system.provenance_sha256,
                tensors={
                    name: value.to(
                        device=runtime.context.device,
                        dtype=(
                            runtime.solver.transform_backend.spectral_dtype
                            if value.is_complex()
                            else runtime.context.real_dtype
                        ),
                    )
                    for name, value in system.tensors.items()
                },
            )
            for system in restart.systems
        ),
    )


def restore_shadow_checkpoint(
    runtime: ExperimentalModelRuntime,
    checkpoint: ShadowRunCheckpoint,
) -> int:
    """Restore a compatible checkpoint without rebuilding its spectral state."""

    if not isinstance(runtime, ExperimentalModelRuntime):
        raise TypeError("runtime must be an ExperimentalModelRuntime")
    if not isinstance(checkpoint, ShadowRunCheckpoint):
        raise TypeError("checkpoint must be a ShadowRunCheckpoint")
    identity = shadow_runtime_identity_sha256(runtime)
    if checkpoint.runtime_identity_sha256 != identity:
        raise ValueError("checkpoint runtime identity does not match target")
    expected_names = evolved_names(runtime)
    if tuple(checkpoint.evolved_spatial) != expected_names:
        raise ValueError("checkpoint evolved components do not match target")

    fields = runtime.solver.fields
    for name in expected_names:
        field_index = fields.name_to_idx[name]
        spatial = checkpoint.evolved_spatial[name]
        spectral = checkpoint.evolved_spectral[name]
        expected_spatial = fields.spatial[field_index]
        expected_spectral = fields.spectral[field_index]
        if spatial.shape != expected_spatial.shape:
            raise ValueError(f"checkpoint spatial shape for {name!r} differs")
        if spectral.shape != expected_spectral.shape:
            raise ValueError(f"checkpoint spectral shape for {name!r} differs")
        if spatial.dtype != expected_spatial.dtype:
            raise ValueError(f"checkpoint spatial dtype for {name!r} differs")
        if spectral.dtype != expected_spectral.dtype:
            raise ValueError(f"checkpoint spectral dtype for {name!r} differs")
        expected_spatial.copy_(spatial.to(device=expected_spatial.device))
        expected_spectral.copy_(spectral.to(device=expected_spectral.device))

    if runtime.algebraic_fields_adapter is not None:
        runtime.algebraic_fields_adapter.clear_cached_outputs()
    runtime.restore_algebraic_restart_state(
        _restart_on_runtime_device(runtime, checkpoint.algebraic_restart)
    )
    runtime.synchronize_algebraic_for_observation()
    integrator = runtime.solver.integrator
    integrator.set_spectral_refresh_interval(
        checkpoint.spectral_refresh_interval
    )
    integrator.restore_progress(
        checkpoint.completed_steps,
        static_fields_are_current=(
            runtime.algebraic_fields_adapter is not None
        ),
    )
    if (
        integrator.step_count != checkpoint.integrator_step_count
        or integrator.refresh_count != checkpoint.integrator_refresh_count
    ):
        raise RuntimeError("restored integrator counters are inconsistent")
    return checkpoint.completed_steps


def _write_npy(path: Path, value: torch.Tensor) -> dict[str, object]:
    array = value.detach().to(device="cpu").contiguous().numpy()
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": _file_sha256(path),
    }


def write_shadow_checkpoint(
    directory: str | Path,
    checkpoint: ShadowRunCheckpoint,
) -> Path:
    """Atomically write a complete checkpoint into a new directory."""

    if not isinstance(checkpoint, ShadowRunCheckpoint):
        raise TypeError("checkpoint must be a ShadowRunCheckpoint")
    target = Path(directory).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"checkpoint directory already exists: {target}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
    )
    try:
        tensor_files: dict[str, dict[str, dict[str, object]]] = {
            "evolved_spatial": {},
            "evolved_spectral": {},
            "algebraic_restart": {},
        }
        for kind, values in (
            ("evolved_spatial", checkpoint.evolved_spatial),
            ("evolved_spectral", checkpoint.evolved_spectral),
        ):
            for name, value in values.items():
                filename = f"{kind}__{name}.npy"
                tensor_files[kind][name] = _write_npy(
                    staging / filename,
                    value,
                )
        for system in checkpoint.algebraic_restart.systems:
            for name, value in system.tensors.items():
                key = f"{system.system_name}.{name}"
                filename = (
                    f"algebraic__{system.system_name}__{name}.npy"
                )
                tensor_files["algebraic_restart"][key] = _write_npy(
                    staging / filename,
                    value,
                )
        metadata = checkpoint.to_metadata()
        metadata["tensor_files"] = tensor_files
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


def _load_tensor_record(
    directory: Path,
    record: Mapping[str, object],
) -> torch.Tensor:
    if not isinstance(record, Mapping):
        raise TypeError("checkpoint tensor record must be a mapping")
    filename = record.get("file")
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".npy")
    ):
        raise ValueError("checkpoint tensor filename is invalid")
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint tensor is missing: {path}")
    expected_sha = require_sha256(record.get("sha256"), "tensor sha256")
    if _file_sha256(path) != expected_sha:
        raise ValueError(f"checkpoint tensor checksum mismatch: {path}")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != record.get("shape"):
        raise ValueError(f"checkpoint tensor shape metadata differs: {path}")
    if str(values.dtype) != record.get("dtype"):
        raise ValueError(f"checkpoint tensor dtype metadata differs: {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"checkpoint tensor contains NaN or Inf: {path}")
    return torch.from_numpy(np.array(values, copy=True))


def load_shadow_checkpoint(directory: str | Path) -> ShadowRunCheckpoint:
    """Load and checksum-validate a shadow checkpoint without pickle."""

    directory = Path(directory).expanduser().resolve()
    metadata_path = directory / "checkpoint.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint metadata is not valid JSON") from exc
    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    files = metadata.get("tensor_files")
    if not isinstance(files, Mapping):
        raise ValueError("checkpoint tensor-file manifest is missing")

    component_names = metadata.get("evolved_components")
    if not isinstance(component_names, list) or any(
        not isinstance(name, str) or not name.isidentifier()
        for name in component_names
    ):
        raise ValueError("checkpoint evolved component metadata is invalid")
    loaded: dict[str, dict[str, torch.Tensor]] = {}
    for kind in ("evolved_spatial", "evolved_spectral"):
        records = files.get(kind)
        if not isinstance(records, Mapping) or set(records) != set(component_names):
            raise ValueError(f"checkpoint {kind} manifest is incomplete")
        loaded[kind] = {
            name: _load_tensor_record(directory, records[name])
            for name in component_names
        }

    restart_metadata = metadata.get("algebraic_restart")
    if not isinstance(restart_metadata, Mapping):
        raise ValueError("checkpoint algebraic restart metadata is missing")
    systems_metadata = restart_metadata.get("systems")
    if not isinstance(systems_metadata, list):
        raise ValueError("checkpoint algebraic systems metadata is invalid")
    restart_records = files.get("algebraic_restart")
    if not isinstance(restart_records, Mapping):
        raise ValueError("checkpoint algebraic tensor manifest is invalid")
    systems = []
    used_restart_keys: set[str] = set()
    for system in systems_metadata:
        if not isinstance(system, Mapping):
            raise TypeError("checkpoint algebraic system must be a mapping")
        tensor_metadata = system.get("tensors")
        if not isinstance(tensor_metadata, Mapping):
            raise ValueError("checkpoint algebraic tensor metadata is invalid")
        system_name = system.get("system_name")
        tensors = {}
        for name in tensor_metadata:
            key = f"{system_name}.{name}"
            if key not in restart_records:
                raise ValueError("checkpoint algebraic tensor file is missing")
            tensors[name] = _load_tensor_record(
                directory,
                restart_records[key],
            )
            used_restart_keys.add(key)
        systems.append(
            AlgebraicSystemRestartState(
                system_name=system_name,
                capability=system.get("capability"),
                implementation_name=system.get("implementation_name"),
                provenance_sha256=system.get("provenance_sha256"),
                tensors=tensors,
            )
        )
    if used_restart_keys != set(restart_records):
        raise ValueError("checkpoint has unreferenced algebraic tensor files")

    integrator = metadata.get("integrator")
    if not isinstance(integrator, Mapping):
        raise ValueError("checkpoint integrator metadata is missing")
    return ShadowRunCheckpoint(
        format_version=metadata.get("format_version"),
        runtime_identity_sha256=metadata.get("runtime_identity_sha256"),
        completed_steps=metadata.get("completed_steps"),
        spectral_refresh_interval=integrator.get(
            "spectral_refresh_interval"
        ),
        integrator_step_count=integrator.get("step_count"),
        integrator_refresh_count=integrator.get("refresh_count"),
        evolved_spatial=loaded["evolved_spatial"],
        evolved_spectral=loaded["evolved_spectral"],
        algebraic_restart=AlgebraicRuntimeRestartState(
            restart_metadata.get("format_version"),
            tuple(systems),
        ),
    )


__all__ = [
    "ShadowRunCheckpoint",
    "capture_shadow_checkpoint",
    "load_shadow_checkpoint",
    "restore_shadow_checkpoint",
    "write_shadow_checkpoint",
]
