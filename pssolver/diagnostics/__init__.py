"""Read-only diagnostics shared by benchmark and qualification tooling."""

from .cuda_memory import (
    cuda_memory_snapshot,
    cuda_memory_window,
)
from .tensor_inventory import build_tensor_inventory

__all__ = [
    "build_tensor_inventory",
    "cuda_memory_snapshot",
    "cuda_memory_window",
]
