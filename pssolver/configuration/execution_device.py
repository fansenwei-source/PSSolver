"""Validation and resolution of public execution-device requests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from collections.abc import Iterator

import torch

from pssolver.configuration.device_tokens import validate_execution_device
_TF32_SCOPE_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class TF32ExecutionState:
    """Exact caller-owned Torch precision state around one execution scope."""

    torch_float32_matmul_precision: str
    cuda_matmul_allow_tf32: bool
    cudnn_allow_tf32: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "torch_float32_matmul_precision": (
                self.torch_float32_matmul_precision
            ),
            "cuda_matmul_allow_tf32": self.cuda_matmul_allow_tf32,
            "cudnn_allow_tf32": self.cudnn_allow_tf32,
        }


def snapshot_tf32_execution_state() -> TF32ExecutionState:
    """Capture every process-global Torch flag owned by the caller."""

    return TF32ExecutionState(
        torch_float32_matmul_precision=(
            torch.get_float32_matmul_precision()
        ),
        cuda_matmul_allow_tf32=bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
    )


def _resolve_tf32_execution(
    requested: object,
    *,
    real_dtype: torch.dtype,
    device: torch.device | str,
) -> tuple[bool, torch.device]:
    if requested not in {"off", "on"}:
        raise ValueError("TF32 policy must be 'off' or 'on'")
    if real_dtype not in {torch.float32, torch.float64}:
        raise ValueError("TF32 configuration requires float32 or float64")
    resolved_device = torch.device(device)
    effective = bool(
        requested == "on"
        and real_dtype == torch.float32
        and resolved_device.type == "cuda"
    )
    return effective, resolved_device


def _tf32_metadata(requested: object, effective: bool) -> dict[str, object]:
    precision = "high" if effective else "highest"
    return {
        "tf32_requested": requested,
        "tf32_effective": effective,
        "float32_matmul_precision": precision,
        "cuda_matmul_allow_tf32": effective,
        "cudnn_allow_tf32": effective,
    }


def _apply_tf32_execution(effective: bool) -> None:
    torch.set_float32_matmul_precision("high" if effective else "highest")
    torch.backends.cuda.matmul.allow_tf32 = effective
    torch.backends.cudnn.allow_tf32 = effective


def _restore_tf32_execution(state: TF32ExecutionState) -> None:
    # ``allow_tf32`` and the public matmul-precision policy are coupled by
    # Torch.  In particular, assigning ``allow_tf32 = True`` rewrites an
    # existing ``"medium"`` policy to ``"high"``.  Restore the low-level
    # flags first and the caller-owned public policy last so all reachable
    # caller states, including ``"medium"``, round-trip exactly.
    torch.backends.cuda.matmul.allow_tf32 = (
        state.cuda_matmul_allow_tf32
    )
    torch.backends.cudnn.allow_tf32 = state.cudnn_allow_tf32
    torch.set_float32_matmul_precision(
        state.torch_float32_matmul_precision
    )


def configure_tf32_execution(
    requested: object,
    *,
    real_dtype: torch.dtype,
    device: torch.device | str,
) -> dict[str, object]:
    """Validate and describe a TF32 policy without changing global state.

    The historical name is retained for internal compatibility. Scientific
    execution must use :func:`scoped_tf32_execution`, which applies this
    policy only while PSSolver owns execution and restores the caller's exact
    state on every exit.
    """

    effective, _ = _resolve_tf32_execution(
        requested,
        real_dtype=real_dtype,
        device=device,
    )
    return _tf32_metadata(requested, effective)


@contextmanager
def scoped_tf32_execution(
    requested: object,
    *,
    real_dtype: torch.dtype,
    device: torch.device | str,
) -> Iterator[dict[str, object]]:
    """Apply TF32 policy for one owned region and restore caller state.

    Validation happens before acquiring or mutating process-global state.
    The re-entrant lock serializes competing scopes and gives same-thread
    nested scopes strict LIFO restoration semantics.
    """

    effective, _ = _resolve_tf32_execution(
        requested,
        real_dtype=real_dtype,
        device=device,
    )
    metadata = _tf32_metadata(requested, effective)
    with _TF32_SCOPE_LOCK:
        before = snapshot_tf32_execution_state()
        try:
            _apply_tf32_execution(effective)
            yield metadata
        finally:
            _restore_tf32_execution(before)


def resolve_execution_device(requested: object) -> torch.device:
    """Resolve ``auto`` consistently and return a canonical Torch device."""

    validated = validate_execution_device(requested)
    if validated == "auto":
        validated = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(validated)


def execution_device_normalization(requested: object) -> dict[str, str]:
    """Return the public request and its host-resolved execution identity."""

    validated = validate_execution_device(requested)
    return {
        "requested_execution_device": validated,
        "resolved_execution_device": str(resolve_execution_device(validated)),
    }


def resolve_allocated_execution_device(requested: object) -> torch.device:
    """Resolve a public request to one concrete device available now.

    Unlike :func:`resolve_execution_device`, this helper is intended for
    tensor-explicit declarations.  It binds an unindexed CUDA request to the
    current allocated device and rejects unavailable devices before a tensor
    identity is constructed.
    """

    device = resolve_execution_device(requested)
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("functional CUDA execution requires CUDA")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("CUDA execution device index is unavailable")
    return torch.device("cuda", index)


__all__ = [
    "TF32ExecutionState",
    "configure_tf32_execution",
    "execution_device_normalization",
    "resolve_allocated_execution_device",
    "resolve_execution_device",
    "scoped_tf32_execution",
    "snapshot_tf32_execution_state",
    "validate_execution_device",
]
