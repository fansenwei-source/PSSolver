"""Read-only diagnostics shared by benchmark and qualification tooling."""

from .cuda_memory import (
    cuda_memory_snapshot,
    cuda_memory_window,
)

__all__ = ["cuda_memory_snapshot", "cuda_memory_window"]
