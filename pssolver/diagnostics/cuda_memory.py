"""Phase-aware CUDA allocator measurements without solver-side state."""

from __future__ import annotations

from collections.abc import Mapping
import numbers

import torch


_CURRENT_KEYS = ("allocated_bytes", "reserved_bytes")
_PEAK_KEYS = ("peak_allocated_bytes", "peak_reserved_bytes")


def cuda_memory_snapshot(device: object) -> dict[str, int | None]:
    """Return current and peak allocator counters for one resolved device."""

    resolved = torch.device(device)
    if resolved.type != "cuda":
        return {
            "allocated_bytes": None,
            "reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(resolved)),
        "reserved_bytes": int(torch.cuda.memory_reserved(resolved)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(resolved)),
    }


def _counter(snapshot: Mapping[str, object], key: str) -> int | None:
    value = snapshot.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"memory counter {key!r} must be an integer or None")
    result = int(value)
    if result < 0:
        raise ValueError(f"memory counter {key!r} must be nonnegative")
    return result


def cuda_memory_window(
    start: Mapping[str, object],
    end: Mapping[str, object],
    peak: Mapping[str, object],
) -> dict[str, int | None]:
    """Normalize one allocator window and report retained-byte growth."""

    start_allocated = _counter(start, _CURRENT_KEYS[0])
    start_reserved = _counter(start, _CURRENT_KEYS[1])
    end_allocated = _counter(end, _CURRENT_KEYS[0])
    end_reserved = _counter(end, _CURRENT_KEYS[1])
    peak_allocated = _counter(peak, _PEAK_KEYS[0])
    peak_reserved = _counter(peak, _PEAK_KEYS[1])
    values = (
        start_allocated,
        start_reserved,
        end_allocated,
        end_reserved,
        peak_allocated,
        peak_reserved,
    )
    if all(value is None for value in values):
        return {
            "start_allocated_bytes": None,
            "start_reserved_bytes": None,
            "end_allocated_bytes": None,
            "end_reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "retained_allocated_growth_bytes": None,
            "retained_reserved_growth_bytes": None,
        }
    if any(value is None for value in values):
        raise ValueError("CUDA memory snapshots must be uniformly available")
    assert start_allocated is not None
    assert start_reserved is not None
    assert end_allocated is not None
    assert end_reserved is not None
    assert peak_allocated is not None
    assert peak_reserved is not None
    if peak_allocated < max(start_allocated, end_allocated):
        raise ValueError("peak allocated memory is below a window endpoint")
    if peak_reserved < max(start_reserved, end_reserved):
        raise ValueError("peak reserved memory is below a window endpoint")
    return {
        "start_allocated_bytes": start_allocated,
        "start_reserved_bytes": start_reserved,
        "end_allocated_bytes": end_allocated,
        "end_reserved_bytes": end_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "retained_allocated_growth_bytes": max(
            0,
            end_allocated - start_allocated,
        ),
        "retained_reserved_growth_bytes": max(
            0,
            end_reserved - start_reserved,
        ),
    }


__all__ = ["cuda_memory_snapshot", "cuda_memory_window"]
