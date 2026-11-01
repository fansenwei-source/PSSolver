"""Provisional checkpoint for the Phase 4.5 SBDF2/block reference canary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from pssolver.io import checkpoint_directory_transaction
from pssolver.io._checkpoint_artifacts import (
    canonical_indented_json_bytes as _canonical_json_bytes,
    indented_identity_sha256 as _identity_sha256,
    raw_file_sha256 as _sha256_file,
    read_reference_tensor_record,
    sha256_bytes as _sha256_bytes,
    write_reference_tensor_record as _write_tensor,
)

from pssolver.integrators.history import SBDF2History
from pssolver.experimental.modal_block_reference import (
    TwoComponentReferenceState,
    TwoComponentSBDF2ReferenceStepper,
)


FORMAT_IDENTITY = "provisional_generic_multistep_v1"
WORKFLOW_IDENTITY = "two_component_sbdf2_modal_block_canary"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPLETE_NAME = "COMPLETE"

_TENSOR_FILES = {
    "physical": "physical.npy",
    "native_spectrum": "native_spectrum.npy",
    "previous_evolved_native_spectrum": (
        "previous_evolved_native_spectrum.npy"
    ),
    "previous_explicit_native_spectral_rhs": (
        "previous_explicit_native_spectral_rhs.npy"
    ),
}


def _read_tensor(
    root: Path,
    name: str,
    record: object,
) -> torch.Tensor:
    filename = _TENSOR_FILES[name]
    return read_reference_tensor_record(
        root,
        name,
        record,
        filename=filename,
    )


def _phase(state: TwoComponentReferenceState) -> str:
    if state.completed_steps == 0:
        return "initial"
    if state.completed_steps == 1:
        return "startup_complete"
    return "multistep"


def save_two_component_reference_checkpoint(
    path: str | Path,
    *,
    stepper: TwoComponentSBDF2ReferenceStepper,
    state: TwoComponentReferenceState,
) -> dict[str, object]:
    """Atomically save a complete CPU canary state and its owned history."""

    if not isinstance(stepper, TwoComponentSBDF2ReferenceStepper):
        raise TypeError("stepper must be TwoComponentSBDF2ReferenceStepper")
    stepper.validate_state(state)
    if state.history is None:
        raise ValueError("P4.5 checkpoints require complete SBDF2 history")
    with checkpoint_directory_transaction(
        path,
        operation="two_component_reference_checkpoint_export",
    ) as (temporary, _target):
        values = {
            "physical": state.physical,
            "native_spectrum": state.native_spectrum,
            "previous_evolved_native_spectrum": (
                state.history.previous_evolved_native_spectrum
            ),
            "previous_explicit_native_spectral_rhs": (
                state.history.previous_explicit_native_spectral_rhs
            ),
        }
        tensors = {
            name: _write_tensor(temporary / _TENSOR_FILES[name], value)
            for name, value in values.items()
        }
        stepper_identity = stepper.to_metadata()
        manifest: dict[str, Any] = {
            "format": FORMAT_IDENTITY,
            "workflow": WORKFLOW_IDENTITY,
            "schema_version": SCHEMA_VERSION,
            "stepper": stepper_identity,
            "stepper_identity_sha256": _identity_sha256(stepper_identity),
            "runtime": {
                "completed_steps": state.completed_steps,
                "time": state.time(stepper.spec.dt),
                "phase": _phase(state),
                "representation_validity": {
                    "physical": "current",
                    "native_spectrum": "current",
                },
                "spectral_refresh": {
                    "interval": state.refresh_interval,
                    "step_count": state.refresh_step_count,
                    "refresh_count": state.refresh_count,
                },
            },
            "history": {
                "present": True,
                "depth": 1,
                "source_completed_steps": (
                    state.history.source_completed_steps
                ),
                "dt": state.history.dt,
            },
            "tensors": tensors,
            "connection": {
                "reference_only": True,
                "plane_checkpoint_v1": False,
            },
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        (temporary / MANIFEST_NAME).write_bytes(manifest_bytes)
        (temporary / COMPLETE_NAME).write_text(
            f"manifest_sha256={_sha256_bytes(manifest_bytes)}\n",
            encoding="utf-8",
        )
    return manifest


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError("checkpoint manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise ValueError("checkpoint manifest must be an object")
    if manifest.get("format") != FORMAT_IDENTITY:
        raise ValueError("checkpoint format identity is unsupported")
    if manifest.get("workflow") != WORKFLOW_IDENTITY:
        raise ValueError("checkpoint workflow identity is unsupported")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is unsupported")
    complete = root / COMPLETE_NAME
    if not complete.is_file():
        raise ValueError("checkpoint COMPLETE marker is missing")
    expected = f"manifest_sha256={_sha256_file(manifest_path)}\n"
    if complete.read_text(encoding="utf-8") != expected:
        raise ValueError("checkpoint manifest identity does not match COMPLETE")
    return manifest


def load_two_component_reference_checkpoint(
    path: str | Path,
    *,
    stepper: TwoComponentSBDF2ReferenceStepper,
) -> TwoComponentReferenceState:
    """Load and validate one checkpoint against an exact canary plan."""

    if not isinstance(stepper, TwoComponentSBDF2ReferenceStepper):
        raise TypeError("stepper must be TwoComponentSBDF2ReferenceStepper")
    root = Path(path)
    if not root.is_dir():
        raise ValueError("checkpoint path must be a directory")
    manifest = _read_manifest(root)
    identity = stepper.to_metadata()
    if manifest.get("stepper") != identity or manifest.get(
        "stepper_identity_sha256"
    ) != _identity_sha256(identity):
        raise ValueError("checkpoint stepper identity mismatch")

    runtime = manifest.get("runtime")
    history_record = manifest.get("history")
    records = manifest.get("tensors")
    if not isinstance(runtime, dict) or not isinstance(history_record, dict):
        raise ValueError("checkpoint runtime/history metadata is invalid")
    if not isinstance(records, dict) or set(records) != set(_TENSOR_FILES):
        raise ValueError("checkpoint tensor record set is incomplete")
    if history_record.get("present") is not True or history_record.get("depth") != 1:
        raise ValueError("checkpoint history metadata is invalid")
    tensors = {
        name: _read_tensor(root, name, records[name])
        for name in sorted(_TENSOR_FILES)
    }
    history = SBDF2History(
        previous_evolved_native_spectrum=tensors[
            "previous_evolved_native_spectrum"
        ],
        previous_explicit_native_spectral_rhs=tensors[
            "previous_explicit_native_spectral_rhs"
        ],
        source_completed_steps=history_record.get("source_completed_steps"),
        dt=history_record.get("dt"),
    )
    refresh = runtime.get("spectral_refresh")
    if not isinstance(refresh, dict):
        raise ValueError("checkpoint refresh metadata is invalid")
    if runtime.get("representation_validity") != {
        "physical": "current",
        "native_spectrum": "current",
    }:
        raise ValueError("checkpoint representation validity is invalid")
    state = TwoComponentReferenceState(
        physical=tensors["physical"],
        native_spectrum=tensors["native_spectrum"],
        completed_steps=runtime.get("completed_steps"),
        refresh_interval=refresh.get("interval"),
        refresh_step_count=refresh.get("step_count"),
        refresh_count=refresh.get("refresh_count"),
        history=history,
    )
    if runtime.get("time") != state.time(stepper.spec.dt):
        raise ValueError("checkpoint time is inconsistent")
    if runtime.get("phase") != _phase(state):
        raise ValueError("checkpoint startup/multistep phase is inconsistent")
    stepper.validate_state(state)
    return state


__all__ = [
    "COMPLETE_NAME",
    "FORMAT_IDENTITY",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "WORKFLOW_IDENTITY",
    "load_two_component_reference_checkpoint",
    "save_two_component_reference_checkpoint",
]
