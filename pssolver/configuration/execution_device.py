"""Validation and resolution of public execution-device requests."""

from __future__ import annotations

import torch


def configure_tf32_execution(
    requested: object,
    *,
    real_dtype: torch.dtype,
    device: torch.device | str,
) -> dict[str, object]:
    """Apply the requested TF32 policy and report the effective backend state.

    Torch's TF32 switches are process-global.  Every production application
    must therefore set them explicitly instead of assuming that a previous
    caller left the process in the requested state.
    """

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
    torch.set_float32_matmul_precision("high" if effective else "highest")
    torch.backends.cuda.matmul.allow_tf32 = effective
    torch.backends.cudnn.allow_tf32 = effective
    return {
        "tf32_requested": requested,
        "tf32_effective": effective,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def validate_execution_device(requested: object) -> str:
    """Return a validated request without requiring a device allocation."""

    if not isinstance(requested, str) or not requested:
        raise ValueError("execution device must be a non-empty string")
    if requested == "auto":
        return requested
    try:
        torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid execution device: {requested!r}") from exc
    return requested


def resolve_execution_device(requested: object) -> torch.device:
    """Resolve ``auto`` consistently and return a canonical Torch device."""

    validated = validate_execution_device(requested)
    if validated == "auto":
        validated = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(validated)


__all__ = [
    "configure_tf32_execution",
    "resolve_execution_device",
    "validate_execution_device",
]
