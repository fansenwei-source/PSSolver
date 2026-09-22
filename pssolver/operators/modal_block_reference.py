"""Analytic two-component reaction--diffusion modal-block reference."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from pssolver.core.modal_blocks import TwoComponentModalOperatorSpec
from pssolver.operators.modal_block import (
    BoundTwoComponentModalOperator,
    TwoComponentModalSolveWorkspace,
    bind_two_component_modal_operator,
)


def _finite(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{description} must be finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class TwoComponentPeriodicReactionDiffusionReference:
    """One coupled cosine mode with an exact matrix-exponential solution."""

    operator_spec: TwoComponentModalOperatorSpec
    point_count: int = 32
    length: float = 2.0 * math.pi
    initial_mode: int = 2
    initial_amplitudes: tuple[float, float] = (0.8, -0.35)

    def __post_init__(self) -> None:
        if not isinstance(self.operator_spec, TwoComponentModalOperatorSpec):
            raise TypeError("operator_spec must be TwoComponentModalOperatorSpec")
        if (
            not isinstance(self.point_count, int)
            or isinstance(self.point_count, bool)
            or self.point_count < 8
            or self.point_count % 2
        ):
            raise ValueError("point_count must be an even integer of at least 8")
        length = _finite(self.length, "length")
        if length <= 0.0:
            raise ValueError("length must be positive")
        object.__setattr__(self, "length", length)
        if (
            not isinstance(self.initial_mode, int)
            or isinstance(self.initial_mode, bool)
            or self.initial_mode < 0
            or self.initial_mode >= self.point_count // 2
        ):
            raise ValueError("initial_mode must be a retained non-Nyquist mode")
        if (
            not isinstance(self.initial_amplitudes, tuple)
            or len(self.initial_amplitudes) != 2
        ):
            raise ValueError("initial_amplitudes must contain exactly two values")
        object.__setattr__(
            self,
            "initial_amplitudes",
            tuple(
                _finite(value, "initial amplitude")
                for value in self.initial_amplitudes
            ),
        )

    @property
    def component_order(self) -> tuple[str, str]:
        return self.operator_spec.component_order

    def coordinates(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return (
            torch.arange(self.point_count, dtype=dtype, device=device)
            * (self.length / self.point_count)
        )

    def laplacian_eigenvalues(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        frequencies = torch.fft.rfftfreq(
            self.point_count,
            d=self.length / self.point_count,
            dtype=dtype,
            device=device,
        )
        return -(2.0 * math.pi * frequencies).square()

    def initial_physical(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        coordinate = self.coordinates(dtype=dtype, device=device)
        wave_number = 2.0 * math.pi * self.initial_mode / self.length
        profile = torch.cos(wave_number * coordinate)
        amplitudes = torch.tensor(
            self.initial_amplitudes,
            dtype=dtype,
            device=device,
        )
        return profile.unsqueeze(-1) * amplitudes

    def initial_native_spectrum(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.fft.rfft(
            self.initial_physical(dtype=dtype, device=device),
            dim=0,
        )

    def selected_mode_linear_matrix(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        wave_number = 2.0 * math.pi * self.initial_mode / self.length
        matrix = self.operator_spec.coefficient_matrix(-(wave_number**2))
        return torch.tensor(matrix, dtype=dtype, device=device)

    def exact_physical(
        self,
        time: float,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        normalized_time = _finite(time, "time")
        if normalized_time < 0.0:
            raise ValueError("time must be non-negative")
        matrix = self.selected_mode_linear_matrix(dtype=dtype, device=device)
        initial = torch.tensor(
            self.initial_amplitudes,
            dtype=dtype,
            device=device,
        )
        amplitudes = torch.matrix_exp(normalized_time * matrix) @ initial
        coordinate = self.coordinates(dtype=dtype, device=device)
        wave_number = 2.0 * math.pi * self.initial_mode / self.length
        return torch.cos(wave_number * coordinate).unsqueeze(-1) * amplitudes

    def bind_operator(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        implementation: str,
    ) -> BoundTwoComponentModalOperator:
        normalized_device = torch.device(device)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("reference real dtype must be float32 or float64")
        spectral_dtype = (
            torch.complex64 if dtype is torch.float32 else torch.complex128
        )
        return bind_two_component_modal_operator(
            self.operator_spec,
            self.laplacian_eigenvalues(
                dtype=dtype,
                device=normalized_device,
            ),
            component_order=self.component_order,
            geometry_identity="periodic_1d",
            basis_signature=("periodic",),
            spectral_dtype=spectral_dtype,
            implementation=implementation,
        )

    def implicit_euler_step(
        self,
        native_spectrum: torch.Tensor,
        *,
        dt: float,
        operator: BoundTwoComponentModalOperator,
        workspace: TwoComponentModalSolveWorkspace,
    ) -> torch.Tensor:
        normalized_dt = _finite(dt, "dt")
        if normalized_dt <= 0.0:
            raise ValueError("dt must be positive")
        if operator.spec != self.operator_spec:
            raise ValueError("bound operator does not match reference model")
        rhs = native_spectrum / normalized_dt
        return operator.solve_into(
            rhs,
            alpha=1.0 / normalized_dt,
            workspace=workspace,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "identity": "two_component_periodic_reaction_diffusion_reference",
            "operator": self.operator_spec.to_metadata(),
            "point_count": self.point_count,
            "length": self.length,
            "initial_mode": self.initial_mode,
            "initial_amplitudes": list(self.initial_amplitudes),
            "analytic_solution": "matrix_exponential_single_cosine_mode",
            "component_packing": "trailing_component_axis_size_2",
        }


__all__ = ["TwoComponentPeriodicReactionDiffusionReference"]
