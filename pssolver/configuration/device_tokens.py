"""Tensor-free grammar for public execution-device request tokens."""

from __future__ import annotations

import re


_PUBLIC_DEVICE_PATTERN = re.compile(
    r"(?:auto|cpu|cuda|cuda:(?:0|[1-9][0-9]*))\Z"
)


def validate_execution_device(requested: object) -> str:
    """Return a canonical public request without importing an array runtime."""

    if not isinstance(requested, str) or _PUBLIC_DEVICE_PATTERN.fullmatch(
        requested
    ) is None:
        raise ValueError(
            "invalid execution device; expected exactly 'auto', 'cpu', "
            "'cuda', or 'cuda:N' with canonical non-negative N"
        )
    return requested


__all__ = ["validate_execution_device"]
