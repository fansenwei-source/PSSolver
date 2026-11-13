"""Provisional checkpoint schema for the disconnected Phase 4 reference.

The Plane workflow checkpoint implementation is intentionally not imported.
This directory-based format exists only to qualify multistep history and
restart identity before a generic production checkpoint is designed.
"""

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

from pssolver.core.integrators import IntegratorScheme
from pssolver.integrators.history import SBDF2History
from pssolver.integrators.scalar_reference import (
    ScalarPeriodicReferenceStepper,
    ScalarReferenceState,
)


FORMAT_IDENTITY = "provisional_generic_multistep_v1"
SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPLETE_NAME = "COMPLETE"

_CURRENT_TENSORS = {
    "physical": "physical.npy",
    "native_spectrum": "native_spectrum.npy",
}
_HISTORY_TENSORS = {
    "previous_evolved_native_spectrum": ("previous_evolved_native_spectrum.npy"),
    "previous_explicit_native_spectral_rhs": ("previous_explicit_native_spectral_rhs.npy"),
}


def _phase(stepper: ScalarPeriodicReferenceStepper, state: ScalarReferenceState) -> str:
    if stepper.spec.scheme is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER:
        return "one_step"
    if state.completed_steps == 0:
        return "initial"
    if state.completed_steps == 1:
        return "startup_complete"
    return "multistep"


def _read_tensor(
    root: Path,
    name: str,
    record: object,
) -> torch.Tensor:
    expected_filename = (_CURRENT_TENSORS | _HISTORY_TENSORS)[name]
    return read_reference_tensor_record(
        root,
        name,
        record,
        filename=expected_filename,
    )


def save_scalar_reference_checkpoint(
    path: str | Path,
    *,
    stepper: ScalarPeriodicReferenceStepper,
    state: ScalarReferenceState,
) -> dict[str, object]:
    """Atomically write one complete provisional reference checkpoint."""

    if not isinstance(stepper, ScalarPeriodicReferenceStepper):
        raise TypeError("stepper must be ScalarPeriodicReferenceStepper")
    stepper.validate_state(state)
    with checkpoint_directory_transaction(
        path,
        operation="scalar_reference_checkpoint_export",
    ) as (temporary, _target):
        tensors: dict[str, dict[str, object]] = {}
        for name, filename in _CURRENT_TENSORS.items():
            tensors[name] = _write_tensor(
                temporary / filename,
                getattr(state, name),
            )

        history = state.history
        if history is not None:
            for name, filename in _HISTORY_TENSORS.items():
                tensors[name] = _write_tensor(
                    temporary / filename,
                    getattr(history, name),
                )

        integrator_identity = stepper.spec.to_metadata()
        model_identity = stepper.model.to_metadata()
        manifest: dict[str, Any] = {
            "format": FORMAT_IDENTITY,
            "schema_version": SCHEMA_VERSION,
            "integrator": integrator_identity,
            "integrator_identity_sha256": _identity_sha256(integrator_identity),
            "model_discretization": model_identity,
            "model_discretization_identity_sha256": _identity_sha256(model_identity),
            "runtime": {
                "completed_steps": state.completed_steps,
                "time": state.time(stepper.spec.dt),
                "phase": _phase(stepper, state),
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
                "present": history is not None,
                "depth": 0 if history is None else 1,
                "source_completed_steps": (
                    None if history is None else history.source_completed_steps
                ),
                "dt": None if history is None else history.dt,
            },
            "tensors": tensors,
            "connection": {
                "reference_only": True,
                "plane_checkpoint_v1": False,
            },
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        (temporary / MANIFEST_NAME).write_bytes(manifest_bytes)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        (temporary / COMPLETE_NAME).write_text(
            f"manifest_sha256={manifest_sha256}\n",
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
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema version is unsupported")
    complete_path = root / COMPLETE_NAME
    if not complete_path.is_file():
        raise ValueError("checkpoint COMPLETE marker is missing")
    expected_complete = f"manifest_sha256={_sha256_file(manifest_path)}\n"
    if complete_path.read_text(encoding="utf-8") != expected_complete:
        raise ValueError("checkpoint manifest identity does not match COMPLETE")
    return manifest


def load_scalar_reference_checkpoint(
    path: str | Path,
    *,
    stepper: ScalarPeriodicReferenceStepper,
) -> ScalarReferenceState:
    """Load and validate a checkpoint against one exact reference plan."""

    if not isinstance(stepper, ScalarPeriodicReferenceStepper):
        raise TypeError("stepper must be ScalarPeriodicReferenceStepper")
    root = Path(path)
    if not root.is_dir():
        raise ValueError("checkpoint path must be a directory")
    manifest = _read_manifest(root)

    integrator_identity = stepper.spec.to_metadata()
    model_identity = stepper.model.to_metadata()
    if manifest.get("integrator") != integrator_identity or manifest.get(
        "integrator_identity_sha256"
    ) != _identity_sha256(integrator_identity):
        raise ValueError("checkpoint integrator identity mismatch")
    if manifest.get("model_discretization") != model_identity or manifest.get(
        "model_discretization_identity_sha256"
    ) != _identity_sha256(model_identity):
        raise ValueError("checkpoint model/discretization identity mismatch")

    runtime = manifest.get("runtime")
    history_record = manifest.get("history")
    tensor_records = manifest.get("tensors")
    if not isinstance(runtime, dict) or not isinstance(history_record, dict):
        raise ValueError("checkpoint runtime/history metadata is invalid")
    if not isinstance(tensor_records, dict):
        raise ValueError("checkpoint tensor records are invalid")
    history_present = history_record.get("present")
    if not isinstance(history_present, bool):
        raise ValueError("checkpoint history presence is invalid")
    required_names = set(_CURRENT_TENSORS)
    if history_present:
        required_names.update(_HISTORY_TENSORS)
    if set(tensor_records) != required_names:
        raise ValueError("checkpoint tensor record set is incomplete")

    tensors = {
        name: _read_tensor(root, name, tensor_records[name]) for name in sorted(required_names)
    }
    history = None
    if history_present:
        if history_record.get("depth") != 1:
            raise ValueError("checkpoint history depth is invalid")
        history = SBDF2History(
            previous_evolved_native_spectrum=tensors["previous_evolved_native_spectrum"],
            previous_explicit_native_spectral_rhs=tensors["previous_explicit_native_spectral_rhs"],
            source_completed_steps=history_record.get("source_completed_steps"),
            dt=history_record.get("dt"),
        )
    elif history_record != {
        "present": False,
        "depth": 0,
        "source_completed_steps": None,
        "dt": None,
    }:
        raise ValueError("checkpoint absent-history metadata is invalid")

    refresh = runtime.get("spectral_refresh")
    representations = runtime.get("representation_validity")
    if not isinstance(refresh, dict):
        raise ValueError("checkpoint refresh metadata is invalid")
    if representations != {
        "physical": "current",
        "native_spectrum": "current",
    }:
        raise ValueError("checkpoint representation validity is invalid")
    state = ScalarReferenceState(
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
    if runtime.get("phase") != _phase(stepper, state):
        raise ValueError("checkpoint startup/multistep phase is inconsistent")
    stepper.validate_state(state)
    return state


__all__ = [
    "COMPLETE_NAME",
    "FORMAT_IDENTITY",
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "load_scalar_reference_checkpoint",
    "save_scalar_reference_checkpoint",
]
