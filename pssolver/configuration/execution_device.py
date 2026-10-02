"""Validation and resolution of public execution-device requests."""

from __future__ import annotations

import torch


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


__all__ = ["resolve_execution_device", "validate_execution_device"]
