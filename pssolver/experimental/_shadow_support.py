"""Private invariants shared by experimental shadow-run modules."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import BerisEdwardsPlaneCoupledModel

from .model_execution import ExperimentalModelRuntime


SHADOW_CHECKPOINT_FORMAT_VERSION = 1
SHADOW_RUN_SCHEMA_VERSION = 1
SHADOW_SCRIPT_ID = "experimental_plane_beris_edwards_shadow"


def canonical_json_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of one regular file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_tensor_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in values.values():
        array = tensor.detach().to(device="cpu").contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def require_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def require_nonnegative_integer(value: object, description: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{description} must be a non-negative integer")
    return value


def evolved_names(runtime: ExperimentalModelRuntime) -> tuple[str, ...]:
    return tuple(field.name for field in runtime.assembly.evolved_fields)


def completed_steps(runtime: ExperimentalModelRuntime) -> int:
    integrator = runtime.solver.integrator
    interval = integrator.spectral_refresh_interval
    if interval is None:
        if integrator.refresh_count != 0:
            raise RuntimeError(
                "disabled spectral refresh cannot have a nonzero refresh count"
            )
        return int(integrator.step_count)
    return int(integrator.refresh_count * interval + integrator.step_count)


def require_plane_beris_edwards_runtime(
    runtime: ExperimentalModelRuntime,
) -> BerisEdwardsPlaneCoupledModel:
    if not isinstance(runtime, ExperimentalModelRuntime):
        raise TypeError("runtime must be an ExperimentalModelRuntime")
    if not isinstance(runtime.problem.geometry, PlaneSlab):
        raise TypeError("shadow runner requires PlaneSlab geometry")
    model = runtime.problem.model
    if not isinstance(model, BerisEdwardsPlaneCoupledModel):
        raise TypeError("shadow runner requires BerisEdwardsPlaneCoupledModel")
    return model


def shadow_runtime_identity_sha256(
    runtime: ExperimentalModelRuntime,
) -> str:
    """Return the trajectory-defining identity of one experimental runtime."""

    if not isinstance(runtime, ExperimentalModelRuntime):
        raise TypeError("runtime must be an ExperimentalModelRuntime")
    return canonical_json_sha256(
        {
            "runtime": runtime.to_metadata(),
            "dt": float(runtime.solver.dt),
            "spectral_refresh_interval": (
                runtime.solver.integrator.spectral_refresh_interval
            ),
        }
    )


__all__ = [
    "SHADOW_CHECKPOINT_FORMAT_VERSION",
    "SHADOW_RUN_SCHEMA_VERSION",
    "SHADOW_SCRIPT_ID",
    "completed_steps",
    "evolved_names",
    "file_sha256",
    "ordered_tensor_sha256",
    "require_nonnegative_integer",
    "require_plane_beris_edwards_runtime",
    "require_sha256",
    "shadow_runtime_identity_sha256",
]
