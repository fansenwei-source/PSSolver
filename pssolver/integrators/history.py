"""Disconnected persistent-history contracts for Phase 4.

The history object stores caller-owned tensors by identity.  It does not
clone, convert, transform, allocate, or connect itself to ``RuntimeState``.
That connection is a separately gated Phase 4 slice.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _storage_identity(value: torch.Tensor) -> tuple[str, int]:
    storage = value.untyped_storage()
    return str(value.device), int(storage.data_ptr())


@dataclass(frozen=True, slots=True)
class SBDF2History:
    """One complete SBDF2 history level in native spectral storage."""

    previous_evolved_native_spectrum: torch.Tensor
    previous_explicit_native_spectral_rhs: torch.Tensor
    source_completed_steps: int
    dt: float

    def __post_init__(self) -> None:
        spectrum = self.previous_evolved_native_spectrum
        rhs = self.previous_explicit_native_spectral_rhs
        if not isinstance(spectrum, torch.Tensor):
            raise TypeError(
                "previous_evolved_native_spectrum must be a tensor"
            )
        if not isinstance(rhs, torch.Tensor):
            raise TypeError(
                "previous_explicit_native_spectral_rhs must be a tensor"
            )
        if spectrum.ndim < 1 or spectrum.numel() == 0:
            raise ValueError("SBDF2 history tensors must be nonempty")
        if (
            spectrum.shape != rhs.shape
            or spectrum.dtype != rhs.dtype
            or spectrum.device != rhs.device
        ):
            raise ValueError(
                "SBDF2 state and RHS history storage must match"
            )
        if spectrum is rhs or _storage_identity(spectrum) == _storage_identity(rhs):
            raise ValueError("SBDF2 state and RHS history must not share storage")
        if (
            not isinstance(self.source_completed_steps, int)
            or isinstance(self.source_completed_steps, bool)
            or self.source_completed_steps < 0
        ):
            raise ValueError(
                "source_completed_steps must be a non-negative integer"
            )
        if (
            not isinstance(self.dt, (int, float))
            or isinstance(self.dt, bool)
            or not math.isfinite(float(self.dt))
            or float(self.dt) <= 0.0
        ):
            raise ValueError("dt must be positive and finite")
        object.__setattr__(self, "dt", float(self.dt))

    def validate_for_current(
        self,
        current_native_spectrum: torch.Tensor,
        *,
        completed_steps: int,
        dt: float,
    ) -> None:
        """Reject history that cannot precede the supplied current state."""

        if not isinstance(current_native_spectrum, torch.Tensor):
            raise TypeError("current_native_spectrum must be a tensor")
        if (
            current_native_spectrum.shape
            != self.previous_evolved_native_spectrum.shape
            or current_native_spectrum.dtype
            != self.previous_evolved_native_spectrum.dtype
            or current_native_spectrum.device
            != self.previous_evolved_native_spectrum.device
        ):
            raise ValueError("current spectrum is incompatible with history")
        if (
            not isinstance(completed_steps, int)
            or isinstance(completed_steps, bool)
            or completed_steps < 0
        ):
            raise ValueError("completed_steps must be a non-negative integer")
        if completed_steps != self.source_completed_steps + 1:
            raise ValueError("SBDF2 history does not immediately precede state")
        if (
            not isinstance(dt, (int, float))
            or isinstance(dt, bool)
            or not math.isfinite(float(dt))
            or float(dt) <= 0.0
        ):
            raise ValueError("dt must be positive and finite")
        if float(dt) != self.dt:
            raise ValueError("changing dt with SBDF2 history is unsupported")

    def to_metadata(self) -> dict[str, object]:
        """Describe history identity without serializing tensor values."""

        spectrum = self.previous_evolved_native_spectrum
        rhs = self.previous_explicit_native_spectral_rhs
        return {
            "schema_version": 1,
            "scheme": "sbdf2",
            "history_depth": 1,
            "source_completed_steps": self.source_completed_steps,
            "dt": self.dt,
            "stores_tensors_by_identity": True,
            "previous_evolved_native_spectrum": {
                "shape": list(spectrum.shape),
                "dtype": str(spectrum.dtype),
                "device": str(spectrum.device),
            },
            "previous_explicit_native_spectral_rhs": {
                "shape": list(rhs.shape),
                "dtype": str(rhs.dtype),
                "device": str(rhs.device),
            },
        }


__all__ = ["SBDF2History"]
