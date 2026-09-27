"""File-backed workflow and exact restart for the P8.5 finite-Q pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.runtime.finite_q_anchoring import (
    FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION,
    PlaneFiniteQAnchoringCheckpoint,
    PlaneFiniteQAnchoringRuntime,
)
from pssolver.runtime.robin_scalar import PlaneRobinScalarCheckpoint


FINITE_Q_ANCHORING_WORKFLOW_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, object]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value in {path}: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _tensor_record(path: Path, value: torch.Tensor) -> dict[str, object]:
    materialized = value.detach().cpu().contiguous().numpy()
    with path.open("wb") as handle:
        np.save(handle, materialized, allow_pickle=False)
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "shape": list(materialized.shape),
        "dtype": str(materialized.dtype),
    }


def write_finite_q_anchoring_checkpoint(
    directory: str | Path,
    checkpoint: PlaneFiniteQAnchoringCheckpoint,
) -> None:
    """Atomically create a complete, pickle-free checkpoint directory."""

    if not isinstance(checkpoint, PlaneFiniteQAnchoringCheckpoint):
        raise TypeError(
            "checkpoint must be a PlaneFiniteQAnchoringCheckpoint"
        )
    target = Path(directory)
    if target.exists():
        raise FileExistsError(f"checkpoint directory already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
    )
    try:
        metadata = checkpoint.to_metadata()
        for component, scalar in checkpoint.components:
            tensors = metadata["components"][component]["tensors"]
            for tensor_name, tensor in (
                ("remainder", scalar.remainder),
                ("bounded_modal", scalar.bounded_modal),
            ):
                filename = f"{component}.{tensor_name}.npy"
                tensors[tensor_name]["file"] = _tensor_record(
                    staging / filename,
                    tensor,
                )
        _atomic_json(staging / "checkpoint.json", metadata)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_tensor(
    directory: Path,
    record: object,
    *,
    expected_shape: object,
    expected_dtype: object,
) -> torch.Tensor:
    if not isinstance(record, dict):
        raise ValueError("checkpoint tensor file record is missing")
    if set(record) != {"path", "size_bytes", "sha256", "shape", "dtype"}:
        raise ValueError("checkpoint tensor file record schema differs")
    filename = record["path"]
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("checkpoint tensor path is invalid")
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint tensor is missing: {path}")
    if path.stat().st_size != record["size_bytes"]:
        raise ValueError("checkpoint tensor file size differs")
    if _sha256_file(path) != record["sha256"]:
        raise ValueError("checkpoint tensor file checksum differs")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("checkpoint tensor file is invalid") from exc
    if list(array.shape) != record["shape"] or list(array.shape) != expected_shape:
        raise ValueError("checkpoint tensor shape metadata differs")
    normalized_expected_dtype = str(expected_dtype).removeprefix("torch.")
    if (
        str(array.dtype) != record["dtype"]
        or str(array.dtype) != normalized_expected_dtype
    ):
        raise ValueError("checkpoint tensor dtype metadata differs")
    if not np.isfinite(array).all():
        raise ValueError("checkpoint tensor contains NaN or Inf")
    return torch.from_numpy(array.copy()).contiguous()


def load_finite_q_anchoring_checkpoint(
    directory: str | Path,
) -> PlaneFiniteQAnchoringCheckpoint:
    """Load and fully validate one finite-Q checkpoint file set."""

    source = Path(directory)
    metadata = _strict_json(source / "checkpoint.json")
    if metadata.get("schema_version") != (
        FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError("unsupported finite-Q checkpoint version")
    if metadata.get("component_order") != list(Q_COMPONENTS):
        raise ValueError("checkpoint component order differs")
    identity = metadata.get("identity")
    identity_json = json.dumps(
        identity,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    if hashlib.sha256(identity_json.encode("utf-8")).hexdigest() != (
        metadata.get("identity_sha256")
    ):
        raise ValueError("checkpoint aggregate identity checksum differs")
    components = metadata.get("components")
    if not isinstance(components, dict) or tuple(components) != Q_COMPONENTS:
        raise ValueError("checkpoint component manifest is incomplete")
    snapshots = []
    for component in Q_COMPONENTS:
        scalar = components[component]
        if not isinstance(scalar, dict):
            raise ValueError("checkpoint scalar metadata is invalid")
        tensors = scalar.get("tensors")
        if not isinstance(tensors, dict) or set(tensors) != {
            "remainder",
            "bounded_modal",
        }:
            raise ValueError("checkpoint tensor manifest is incomplete")
        remainder_metadata = tensors["remainder"]
        modal_metadata = tensors["bounded_modal"]
        remainder = _load_tensor(
            source,
            remainder_metadata.get("file"),
            expected_shape=remainder_metadata.get("shape"),
            expected_dtype=remainder_metadata.get("dtype"),
        )
        modal = _load_tensor(
            source,
            modal_metadata.get("file"),
            expected_shape=modal_metadata.get("shape"),
            expected_dtype=modal_metadata.get("dtype"),
        )
        snapshots.append(
            (
                component,
                PlaneRobinScalarCheckpoint(
                    identity_json=json.dumps(
                        scalar.get("identity"),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    remainder=remainder,
                    bounded_modal=modal,
                    progress_json=json.dumps(
                        scalar.get("progress"),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    representations_json=json.dumps(
                        scalar.get("representations"),
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    remainder_sha256=remainder_metadata.get("sha256"),
                    bounded_modal_sha256=modal_metadata.get("sha256"),
                ),
            )
        )
    checkpoint = PlaneFiniteQAnchoringCheckpoint(
        identity_json=identity_json,
        components=tuple(snapshots),
    )
    if checkpoint.completed_steps != metadata.get("completed_steps"):
        raise ValueError("checkpoint aggregate clock metadata differs")
    return checkpoint


@dataclass(frozen=True, slots=True)
class PlaneFiniteQAnchoringWorkflowResult:
    start_step: int
    final_step: int
    saved_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    final_q: torch.Tensor


class PlaneFiniteQAnchoringWorkflow:
    """Small CPU-only workflow around the finite-Q relaxation oracle."""

    def __init__(
        self,
        runtime: PlaneFiniteQAnchoringRuntime,
        output_directory: str | Path,
        *,
        save_interval: int = 1,
        checkpoint_interval: int | None = None,
    ) -> None:
        if not isinstance(runtime, PlaneFiniteQAnchoringRuntime):
            raise TypeError("runtime must be a PlaneFiniteQAnchoringRuntime")
        for value, description in (
            (save_interval, "save_interval"),
            (checkpoint_interval, "checkpoint_interval"),
        ):
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{description} must be positive or None")
        self.runtime = runtime
        self.output_directory = Path(output_directory)
        self.save_interval = save_interval
        self.checkpoint_interval = checkpoint_interval

    def run(
        self,
        final_step: int,
        *,
        restart_from: str | Path | None = None,
    ) -> PlaneFiniteQAnchoringWorkflowResult:
        if (
            not isinstance(final_step, int)
            or isinstance(final_step, bool)
            or final_step < 0
        ):
            raise ValueError("final_step must be a non-negative integer")
        if self.output_directory.exists() and any(
            self.output_directory.iterdir()
        ):
            raise FileExistsError("workflow output directory is not empty")
        if restart_from is not None:
            self.runtime.restore_checkpoint(
                load_finite_q_anchoring_checkpoint(restart_from)
            )
        start_step = self.runtime.completed_steps
        if final_step < start_step:
            raise ValueError("final_step precedes restored progress")
        self.output_directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": FINITE_Q_ANCHORING_WORKFLOW_SCHEMA_VERSION,
            "workflow_kind": "plane_finite_q_anchoring_relaxation_pilot",
            "runtime": self.runtime.to_metadata(),
            "start_step": start_step,
            "requested_final_step": final_step,
            "save_interval": self.save_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "complete_q_timestep_connected": False,
            "public_runner_connected": False,
        }
        _atomic_json(self.output_directory / "metadata.json", metadata)
        saved_steps = []
        checkpoint_steps = []
        while self.runtime.completed_steps < final_step:
            self.runtime.advance()
            step = self.runtime.completed_steps
            if step % self.save_interval == 0:
                _atomic_npy(
                    self.output_directory / f"Q_{step}.npy",
                    self.runtime.physical_q().detach().cpu().numpy(),
                )
                saved_steps.append(step)
            if (
                self.checkpoint_interval is not None
                and step % self.checkpoint_interval == 0
            ):
                write_finite_q_anchoring_checkpoint(
                    self.output_directory / f"checkpoint_{step}",
                    self.runtime.capture_checkpoint(),
                )
                checkpoint_steps.append(step)
        final_q = self.runtime.physical_q().detach().clone().contiguous()
        _atomic_json(
            self.output_directory / "COMPLETE",
            {
                "schema_version": 1,
                "completed_steps": final_step,
                "saved_steps": saved_steps,
                "checkpoint_steps": checkpoint_steps,
                "finite": bool(torch.isfinite(final_q).all().item()),
            },
        )
        return PlaneFiniteQAnchoringWorkflowResult(
            start_step=start_step,
            final_step=final_step,
            saved_steps=tuple(saved_steps),
            checkpoint_steps=tuple(checkpoint_steps),
            final_q=final_q,
        )


__all__ = [
    "FINITE_Q_ANCHORING_WORKFLOW_SCHEMA_VERSION",
    "PlaneFiniteQAnchoringWorkflow",
    "PlaneFiniteQAnchoringWorkflowResult",
    "load_finite_q_anchoring_checkpoint",
    "write_finite_q_anchoring_checkpoint",
]
