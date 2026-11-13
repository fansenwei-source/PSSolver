"""Sealed engine checkpoints; plan interpretation stays with the caller."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pssolver.core._canonical import canonical_sha256
from pssolver.engine.state import State

from ._checkpoint_artifacts import write_standard_tensor_record
from .checkpoint import (
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
)


_OP = "engine_state_import"
_PRESSURE_EXPONENT = "pressure_guess_action_exponent"
_KEYS = frozenset(
    {
        "format_kind",
        "format_version",
        "plan_identity_sha256",
        "plan",
        "dtype",
        "device_kind",
        "batch",
        "completed_steps",
        "refresh_interval",
        "refresh_phase",
        "constrained_current",
        "history_length",
        "solver_keys",
        "tensor_files",
        "run_provenance",
        "metadata_sha256",
    }
)


def _hash_file(path):
    return checkpoint_file_sha256(path, operation="engine_state_io")


def _compatibility_error(message):
    return CheckpointCompatibilityError(message, operation=_OP)


def _plan_digest(metadata):
    return canonical_sha256(
        {key: value for key, value in metadata.items() if key != "identity_sha256"}
    )


def _reject_test_integrator(metadata, *, validate_seal=False):
    if validate_seal:
        if (
            metadata.get("format_kind") != "engine_state"
            or type(metadata.get("format_version")) is not int
            or metadata["format_version"] != 1
        ):
            return
        try:
            verify_checkpoint_metadata(metadata, required=True, operation=_OP)
            require_exact_checkpoint_keys(
                metadata, required=_KEYS, optional={"solver_scalars"},
                description="engine state", operation=_OP,
            )
        except CheckpointIntegrityError:
            # Dry runs historically ignore other invalid or absent restart headers.
            return
    plan = metadata.get("plan")
    integrator = plan.get("integrator") if isinstance(plan, dict) else None
    scheme = integrator.get("scheme") if isinstance(integrator, dict) else None
    length = metadata.get("history_length")
    if (type(length) is int and length >= 1) or (
        isinstance(scheme, str) and scheme != "semi_implicit_euler"
    ):
        raise _compatibility_error(
            "engine checkpoint requires the public Euler integrator with empty history"
        )


def read_header(directory, *, plan_identity_sha256, dtype, batch, _reference=None):
    """Validate all header contracts before inspecting or loading any tensor."""
    metadata = read_checkpoint_json(Path(directory) / "checkpoint.json", operation=_OP)
    if (
        metadata.get("format_kind") != "engine_state"
        or type(metadata.get("format_version")) is not int
        or metadata["format_version"] != 1
    ):
        raise _compatibility_error("unsupported engine state checkpoint format")
    verify_checkpoint_metadata(metadata, required=True, operation=_OP)
    require_exact_checkpoint_keys(
        metadata,
        required=_KEYS,
        optional={"solver_scalars"},
        description="engine state",
        operation=_OP,
    )
    if _reference is None:
        _reject_test_integrator(metadata)
    else:
        scheme, dt = _reference
        plan = metadata["plan"]
        integrator = plan.get("integrator") if isinstance(plan, dict) else None
        if (
            not isinstance(integrator, dict)
            or integrator.get("scheme") != scheme
            or scheme != "sbdf2_test"
        ):
            raise _compatibility_error("engine checkpoint reference scheme does not match target")
        if integrator.get("dt") != dt:
            raise _compatibility_error("SBDF2 history requires unchanged dt")
    if (
        metadata["plan_identity_sha256"] != plan_identity_sha256
        or _plan_digest(metadata["plan"]) != plan_identity_sha256
    ):
        raise _compatibility_error("engine checkpoint plan identity does not match target")
    if metadata["dtype"] != str(dtype).removeprefix("torch."):
        raise _compatibility_error("engine checkpoint dtype does not match target")
    if type(metadata["batch"]) is not int or metadata["batch"] != batch:
        raise _compatibility_error("engine checkpoint batch does not match target")
    for key in ("completed_steps", "history_length"):
        if type(metadata[key]) is not int or metadata[key] < 0:
            raise _compatibility_error(f"engine checkpoint {key} must be non-negative")
    if _reference is not None and metadata["history_length"] != int(metadata["completed_steps"] > 0):
        raise _compatibility_error("engine checkpoint reference history length is invalid")
    phase = metadata["refresh_phase"]
    if (
        not isinstance(phase, list)
        or len(phase) != 2
        or any(type(value) is not int or value < 0 for value in phase)
    ):
        raise _compatibility_error("engine checkpoint refresh phase is invalid")
    interval = metadata["refresh_interval"]
    if interval is not None and (type(interval) is not int or interval <= 0):
        raise _compatibility_error("engine checkpoint refresh interval is invalid")
    expected_phase = (
        [metadata["completed_steps"], 0]
        if interval is None
        else [metadata["completed_steps"] % interval, metadata["completed_steps"] // interval]
    )
    if phase != expected_phase:
        raise _compatibility_error("engine checkpoint progress is inconsistent")
    if type(metadata["constrained_current"]) is not bool:
        raise _compatibility_error("engine checkpoint constrained_current must be boolean")
    solver_keys = metadata["solver_keys"]
    if (
        not isinstance(solver_keys, list)
        or not all(isinstance(key, str) for key in solver_keys)
        or len(set(solver_keys)) != len(solver_keys)
    ):
        raise _compatibility_error("engine checkpoint solver keys are invalid")
    expected = (
        {"spectral", "spatial"}
        | {f"history_{index}" for index in range(metadata["history_length"])}
        | {f"solver.{key}" for key in solver_keys}
    )
    files = metadata["tensor_files"]
    if not isinstance(files, dict) or set(files) != expected:
        raise _compatibility_error("engine checkpoint tensor keys are invalid")
    if "solver_scalars" in metadata:
        scalars = metadata["solver_scalars"]
        if not solver_keys or not isinstance(scalars, dict) or not scalars.keys() <= {
            _PRESSURE_EXPONENT
        }:
            raise _compatibility_error("engine checkpoint solver scalars are invalid")
        if _PRESSURE_EXPONENT in scalars:
            exponent = scalars[_PRESSURE_EXPONENT]
            if batch == 1:
                valid = exponent is None or type(exponent) is int
                has_action = exponent is not None
            else:
                valid = (
                    isinstance(exponent, list)
                    and len(exponent) == batch
                    and all(value is None or type(value) is int for value in exponent)
                )
                has_action = valid and any(value is not None for value in exponent)
            if (
                not valid
                or not {"pressure_guess", "pressure_guess_action"} <= set(solver_keys)
                or has_action != (files["solver.pressure_guess_action"] is not None)
                or (has_action and files["solver.pressure_guess"] is None)
            ):
                raise _compatibility_error("engine checkpoint pressure cache exponent is invalid")
    for name, record in files.items():
        if record is None and name.startswith("solver."):
            continue
        require_exact_checkpoint_keys(
            record,
            required={"file", "sha256", "shape", "dtype"},
            description=f"engine tensor {name}",
            operation=_OP,
        )
    if _reference is not None and metadata["history_length"]:
        spectrum, history = files["spectral"], files["history_0"]
        if (
            not isinstance(spectrum["shape"], list)
            or history["shape"] != [2, *spectrum["shape"]]
            or history["dtype"] != spectrum["dtype"]
        ):
            raise _compatibility_error("engine checkpoint reference history shape or dtype is invalid")
    return metadata


def write(
    directory, state, *, plan_identity_sha256, plan_metadata, refresh_interval, run_provenance
):
    """Publish one state with the existing durable no-overwrite transaction."""
    if _plan_digest(plan_metadata) != plan_identity_sha256:
        raise ValueError("engine plan metadata and identity differ")
    tensors = {"spectral": state.spectral, "spatial": state.spatial}
    tensors.update((f"history_{index}", value) for index, value in enumerate(state.history))
    solver_tensors = {key: value for key, value in state.solver.items() if key != _PRESSURE_EXPONENT}
    solver_scalars = (
        {_PRESSURE_EXPONENT: state.solver[_PRESSURE_EXPONENT]}
        if _PRESSURE_EXPONENT in state.solver
        else {}
    )
    tensors.update((f"solver.{key}", value) for key, value in solver_tensors.items())
    with checkpoint_directory_transaction(directory, operation="engine_state_export") as (
        tmp,
        target,
    ):
        files = {}
        for index, (name, value) in enumerate(tensors.items()):
            if value is None:
                files[name] = None
            else:
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"engine state tensor {name} must be finite")
                files[name] = write_standard_tensor_record(
                    tmp / f"tensor_{index}.npy",
                    value,
                    hash_file=_hash_file,
                )
        metadata = seal_checkpoint_metadata(
            {
                "format_kind": "engine_state",
                "format_version": 1,
                "plan_identity_sha256": plan_identity_sha256,
                "plan": plan_metadata,
                "dtype": str(state.spatial.dtype).removeprefix("torch."),
                "device_kind": state.spatial.device.type,
                "batch": state.spatial.shape[1],
                "completed_steps": state.completed_steps,
                "refresh_interval": refresh_interval,
                "refresh_phase": list(state.refresh_phase),
                "constrained_current": state.constrained_current,
                "history_length": len(state.history),
                "solver_keys": list(solver_tensors),
                "tensor_files": files,
                "run_provenance": run_provenance,
                **({"solver_scalars": solver_scalars} if state.solver else {}),
            }
        )
        (tmp / "checkpoint.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return target


def read(directory, *, plan_identity_sha256, dtype, batch, device, _reference=None):
    """Read a state only after header validation; map tensors to the caller's device."""
    root = Path(directory)
    metadata = read_header(
        root, plan_identity_sha256=plan_identity_sha256, dtype=dtype, batch=batch,
        _reference=_reference,
    )
    if _reference is not None and metadata["device_kind"] != torch.device(device).type:
        raise _compatibility_error("engine checkpoint reference device does not match target")
    tensors = {}
    for name, record in metadata["tensor_files"].items():
        if record is None:
            tensors[name] = None
            continue
        path = resolve_checkpoint_payload(
            root, record["file"], operation=_OP, description=f"engine tensor {name}"
        )
        if _hash_file(path) != record["sha256"]:
            raise CheckpointIntegrityError(f"engine tensor {name} checksum mismatch", operation=_OP)
        array = load_checkpoint_numpy(path, operation=_OP)
        if (
            list(array.shape) != record["shape"]
            or str(array.dtype) != record["dtype"]
            or not np.isfinite(array).all()
        ):
            raise CheckpointIntegrityError(f"engine tensor {name} record mismatch", operation=_OP)
        tensors[name] = torch.from_numpy(np.array(array, copy=True)).to(device=device)
    solver = {key: tensors[f"solver.{key}"] for key in metadata["solver_keys"]}
    if "pressure_guess_action" in solver:
        exponent = metadata.get("solver_scalars", {}).get(_PRESSURE_EXPONENT)
        if exponent is None:
            solver["pressure_guess_action"] = None
        solver[_PRESSURE_EXPONENT] = (
            tuple(exponent) if isinstance(exponent, list) else
            ((None,) * batch if batch > 1 else exponent)
        )
    return State(
        tensors["spectral"],
        tensors["spatial"],
        tuple(tensors[f"history_{index}"] for index in range(metadata["history_length"])),
        solver,
        metadata["completed_steps"],
        tuple(metadata["refresh_phase"]),
        metadata["constrained_current"],
    )
