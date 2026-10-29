"""File-backed workflow and exact restart for the P8.5 finite-Q pilot."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from pssolver.core._canonical import canonical_json
from pssolver.io.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    checkpoint_directory_tree_sha256,
    checkpoint_file_sha256,
    checkpoint_directory_transaction,
    load_checkpoint_numpy,
    read_checkpoint_json,
    require_exact_checkpoint_keys,
    require_sealed_upgrade_provenance,
    resolve_checkpoint_payload,
    seal_checkpoint_metadata,
    verify_checkpoint_metadata,
)

from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.runtime.finite_q_anchoring import (
    FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION,
    FINITE_Q_ANCHORING_LEGACY_CHECKPOINT_FORMAT_VERSION,
    PlaneFiniteQAnchoringCheckpoint,
    PlaneFiniteQAnchoringRuntime,
)
from pssolver.runtime.robin_scalar import PlaneRobinScalarCheckpoint


FINITE_Q_ANCHORING_WORKFLOW_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    return checkpoint_file_sha256(path, operation="finite_q_checkpoint_io")


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
) -> Path:
    """Atomically create a complete, pickle-free checkpoint directory."""

    if not isinstance(checkpoint, PlaneFiniteQAnchoringCheckpoint):
        raise TypeError(
            "checkpoint must be a PlaneFiniteQAnchoringCheckpoint"
        )
    if checkpoint.format_version != FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            "v0.3 can only write finite-Q checkpoint format version 2; "
            "legacy format 1 is read-only",
            operation="finite_q_checkpoint_export",
        )
    if checkpoint.migration_provenance is not None:
        require_sealed_upgrade_provenance(
            checkpoint.migration_provenance,
            operation="finite_q_checkpoint_export",
        )
    with checkpoint_directory_transaction(
        directory,
        operation="finite_q_checkpoint_export",
    ) as (staging, target):
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
        _atomic_json(
            staging / "checkpoint.json",
            seal_checkpoint_metadata(metadata),
        )
    return target


def _load_tensor(
    directory: Path,
    record: object,
    *,
    expected_shape: object,
    expected_dtype: object,
    expected_filename: str,
) -> torch.Tensor:
    record = require_exact_checkpoint_keys(
        record,
        required={"path", "size_bytes", "sha256", "shape", "dtype"},
        description="finite-Q checkpoint tensor file record",
        operation="finite_q_checkpoint_import",
    )
    filename = record["path"]
    path = resolve_checkpoint_payload(
        directory,
        filename,
        operation="finite_q_checkpoint_import",
        description="finite-Q checkpoint tensor",
        expected_filename=expected_filename,
    )
    if path.stat().st_size != record["size_bytes"]:
        raise CheckpointIntegrityError(
            "checkpoint tensor file size differs",
            operation="finite_q_checkpoint_import",
        )
    if _sha256_file(path) != record["sha256"]:
        raise CheckpointIntegrityError(
            "checkpoint tensor file checksum differs",
            operation="finite_q_checkpoint_import",
        )
    array = load_checkpoint_numpy(path, operation="finite_q_checkpoint_import")
    if list(array.shape) != record["shape"] or list(array.shape) != expected_shape:
        raise CheckpointIntegrityError(
            "checkpoint tensor shape metadata differs",
            operation="finite_q_checkpoint_import",
        )
    normalized_expected_dtype = str(expected_dtype).removeprefix("torch.")
    if (
        str(array.dtype) != record["dtype"]
        or str(array.dtype) != normalized_expected_dtype
    ):
        raise CheckpointIntegrityError(
            "checkpoint tensor dtype metadata differs",
            operation="finite_q_checkpoint_import",
        )
    if not np.isfinite(array).all():
        raise CheckpointIntegrityError(
            "checkpoint tensor contains NaN or Inf",
            operation="finite_q_checkpoint_import",
        )
    return torch.from_numpy(array.copy()).contiguous()


def load_finite_q_anchoring_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> PlaneFiniteQAnchoringCheckpoint:
    """Load and fully validate one finite-Q checkpoint file set."""

    source = Path(directory)
    metadata = read_checkpoint_json(
        source / "checkpoint.json",
        operation="finite_q_checkpoint_import",
    )
    format_version = metadata.get("schema_version")
    if format_version not in {
        FINITE_Q_ANCHORING_LEGACY_CHECKPOINT_FORMAT_VERSION,
        FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION,
    }:
        raise CheckpointCompatibilityError(
            "unsupported finite-Q checkpoint version",
            operation="finite_q_checkpoint_import",
        )
    verify_checkpoint_metadata(
        metadata,
        required=(format_version == FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION),
        operation="finite_q_checkpoint_import",
    )
    require_exact_checkpoint_keys(
        metadata,
        required={
            "schema_version",
            "identity",
            "identity_sha256",
            "component_order",
            "completed_steps",
            "components",
            *(
                {"metadata_sha256"}
                if format_version == FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION
                else set()
            ),
        },
        optional=(
            {"migration_provenance"}
            if format_version == FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION
            else ()
        ),
        description="finite-Q checkpoint metadata",
        operation="finite_q_checkpoint_import",
    )
    if metadata.get("component_order") != list(Q_COMPONENTS):
        raise ValueError("checkpoint component order differs")
    identity = metadata.get("identity")
    identity_json = canonical_json(identity)
    if hashlib.sha256(identity_json.encode("utf-8")).hexdigest() != (
        metadata.get("identity_sha256")
    ):
        raise ValueError("checkpoint aggregate identity checksum differs")
    components = metadata.get("components")
    if not isinstance(components, dict) or tuple(components) != Q_COMPONENTS:
        raise ValueError("checkpoint component manifest is incomplete")
    normalized_device = torch.device(device)
    if normalized_device.type == "cuda" and normalized_device.index is None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        normalized_device = torch.device("cuda", torch.cuda.current_device())
    snapshots = []
    for component in Q_COMPONENTS:
        scalar = components[component]
        scalar = require_exact_checkpoint_keys(
            scalar,
            required={
                "schema_version",
                "identity",
                "progress",
                "representations",
                "tensors",
            },
            description=f"finite-Q checkpoint component {component}",
            operation="finite_q_checkpoint_import",
        )
        tensors = scalar.get("tensors")
        tensors = require_exact_checkpoint_keys(
            tensors,
            required={"remainder", "bounded_modal"},
            description=f"finite-Q checkpoint component {component} tensors",
            operation="finite_q_checkpoint_import",
        )
        remainder_metadata = tensors["remainder"]
        modal_metadata = tensors["bounded_modal"]
        for tensor_name, tensor_metadata in (
            ("remainder", remainder_metadata),
            ("bounded_modal", modal_metadata),
        ):
            require_exact_checkpoint_keys(
                tensor_metadata,
                required={"shape", "dtype", "device", "sha256", "file"},
                description=(
                    f"finite-Q checkpoint {component}.{tensor_name} metadata"
                ),
                operation="finite_q_checkpoint_import",
            )
        remainder = _load_tensor(
            source,
            remainder_metadata.get("file"),
            expected_shape=remainder_metadata.get("shape"),
            expected_dtype=remainder_metadata.get("dtype"),
            expected_filename=f"{component}.remainder.npy",
        )
        modal = _load_tensor(
            source,
            modal_metadata.get("file"),
            expected_shape=modal_metadata.get("shape"),
            expected_dtype=modal_metadata.get("dtype"),
            expected_filename=f"{component}.bounded_modal.npy",
        )
        remainder = remainder.to(normalized_device)
        modal = modal.to(normalized_device)
        snapshots.append(
            (
                component,
                PlaneRobinScalarCheckpoint(
                    identity_json=canonical_json(scalar.get("identity")),
                    remainder=remainder,
                    bounded_modal=modal,
                    progress_json=canonical_json(scalar.get("progress")),
                    representations_json=canonical_json(scalar.get("representations")),
                    remainder_sha256=remainder_metadata.get("sha256"),
                    bounded_modal_sha256=modal_metadata.get("sha256"),
                ),
            )
        )
    checkpoint = PlaneFiniteQAnchoringCheckpoint(
        identity_json=identity_json,
        components=tuple(snapshots),
        format_version=format_version,
        migration_provenance=metadata.get("migration_provenance"),
    )
    if checkpoint.completed_steps != metadata.get("completed_steps"):
        raise ValueError("checkpoint aggregate clock metadata differs")
    return checkpoint


def upgrade_finite_q_anchoring_checkpoint_directory(
    source_directory: str | Path,
    target_directory: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Path:
    """Seal a validated format-1 checkpoint into a new format-2 directory."""

    source = Path(source_directory).expanduser().resolve()
    target = Path(target_directory).expanduser().resolve()
    if target == source or source in target.parents:
        raise ValueError("checkpoint upgrade target must be outside the source")
    source_tree_sha256 = checkpoint_directory_tree_sha256(
        source,
        operation="finite_q_checkpoint_upgrade",
    )
    checkpoint = load_finite_q_anchoring_checkpoint(source, device=device)
    if checkpoint.format_version != FINITE_Q_ANCHORING_LEGACY_CHECKPOINT_FORMAT_VERSION:
        raise CheckpointCompatibilityError(
            "only finite-Q checkpoint format version 1 can be upgraded",
            operation="finite_q_checkpoint_upgrade",
        )
    upgraded = replace(
        checkpoint,
        format_version=FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION,
        migration_provenance={
            "kind": "finite_q_checkpoint_metadata_seal_v1_to_v2",
            "source_format_version": 1,
            "source_integrity_status": "unsealed_legacy",
            "source_tree_sha256": source_tree_sha256,
            "source_historical_integrity_authenticated": False,
            "source_checkpoint_mutated": False,
        },
    )
    written = write_finite_q_anchoring_checkpoint(target, upgraded)
    if checkpoint_directory_tree_sha256(
        source,
        operation="finite_q_checkpoint_upgrade",
    ) != source_tree_sha256:
        raise RuntimeError("legacy source checkpoint changed during upgrade")
    return written


@dataclass(frozen=True, slots=True)
class PlaneFiniteQAnchoringWorkflowResult:
    start_step: int
    final_step: int
    saved_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    final_q: torch.Tensor


class PlaneFiniteQAnchoringWorkflow:
    """Small file-backed workflow around the finite-Q relaxation oracle."""

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
                load_finite_q_anchoring_checkpoint(
                    restart_from,
                    device=self.runtime.device,
                )
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
    "upgrade_finite_q_anchoring_checkpoint_directory",
    "write_finite_q_anchoring_checkpoint",
]
