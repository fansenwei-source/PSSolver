"""Execution plans for one bounded DCT/DST tensor-product axis.

The tensor-product backend owns transform ordering and axis movement.  This
module owns only the device-bound kernel used after the active axis has been
moved to the final tensor dimension.  Keeping that boundary narrow lets later
FFT or pruned implementations be evaluated without changing the mathematical
transform contract or the tensor-product scheduler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Protocol

import torch


BOUNDED_TRANSFORM_KINDS = ("dct", "dst")
BOUNDED_VALUE_TYPES = ("real", "complex")


@dataclass(frozen=True, slots=True)
class BoundedAxisPlanKey:
    """Static facts that identify one reusable bounded-axis executor."""

    kind: str
    physical_size: int
    retained_count: int
    device: torch.device
    real_dtype: torch.dtype
    value_type: str

    def __post_init__(self) -> None:
        if self.kind not in BOUNDED_TRANSFORM_KINDS:
            raise ValueError(
                f"kind must be one of {BOUNDED_TRANSFORM_KINDS}, "
                f"got {self.kind!r}"
            )
        if (
            not isinstance(self.physical_size, int)
            or isinstance(self.physical_size, bool)
            or self.physical_size <= 0
        ):
            raise ValueError("physical_size must be a positive integer")
        if (
            not isinstance(self.retained_count, int)
            or isinstance(self.retained_count, bool)
            or not 0 <= self.retained_count <= self.physical_size
        ):
            raise ValueError(
                "retained_count must be an integer in "
                "[0, physical_size]"
            )
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be a torch.device")
        if self.real_dtype not in (torch.float32, torch.float64):
            raise ValueError("real_dtype must be torch.float32 or torch.float64")
        if self.value_type not in BOUNDED_VALUE_TYPES:
            raise ValueError(
                f"value_type must be one of {BOUNDED_VALUE_TYPES}, "
                f"got {self.value_type!r}"
            )


class BoundedAxisExecutionPlan(Protocol):
    """Narrow interface for one preselected bounded-axis algorithm."""

    algorithm: str
    key: BoundedAxisPlanKey

    def forward_last_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform a physical final axis into retained coefficients."""

        ...

    def inverse_last_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transform retained final-axis coefficients back to physical values."""

        ...


@dataclass(frozen=True, slots=True)
class DenseBoundedAxisExecutionPlan:
    """Dense orthonormal DCT-II/DST-II execution for one static signature."""

    key: BoundedAxisPlanKey
    matrix: torch.Tensor
    algorithm: ClassVar[str] = "dense"

    def __post_init__(self) -> None:
        expected_shape = (self.key.retained_count, self.key.physical_size)
        if tuple(self.matrix.shape) != expected_shape:
            raise ValueError(
                f"matrix shape must be {expected_shape}, "
                f"got {tuple(self.matrix.shape)}"
            )
        if self.matrix.device != self.key.device:
            raise ValueError("matrix device does not match the plan key")
        if self.matrix.dtype != self.key.real_dtype:
            raise ValueError("matrix dtype does not match the plan key")

    def _matrix_for(self, tensor: torch.Tensor) -> torch.Tensor:
        # This intentionally preserves the v0.1.1 conversion point.  PyTorch
        # returns the original tensor when device and dtype already match.
        return self.matrix.to(device=tensor.device, dtype=tensor.dtype)

    def forward_last_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        matrix = self._matrix_for(tensor)
        return tensor @ matrix.transpose(-1, -2)

    def inverse_last_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        matrix = self._matrix_for(tensor)
        return tensor @ matrix


def build_dense_orthonormal_matrix(
    kind: str,
    size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the frozen cell-centered orthonormal DCT-II/DST-II matrix."""

    n = torch.arange(size, device=device, dtype=dtype)
    k = n.unsqueeze(1)
    phase = math.pi * (n + 0.5) / size

    if kind == "dct":
        matrix = torch.cos(k * phase)
        matrix[0] *= math.sqrt(1.0 / size)
        if size > 1:
            matrix[1:] *= math.sqrt(2.0 / size)
    elif kind == "dst":
        matrix = math.sqrt(2.0 / size) * torch.sin((k + 1.0) * phase)
        if size > 0:
            matrix[-1] *= math.sqrt(0.5)
    else:
        raise ValueError(f"Unsupported transform kind '{kind}'.")
    return matrix
