"""No-slip rectangular-Channel modal Stokes/Brinkman solver.

The solver consumes force coefficients in periodic/Dirichlet/Dirichlet
velocity bases and returns velocity coefficients in the same bases plus a
zero-mean pressure in periodic/Neumann/Neumann space.  It deliberately keeps
the legacy Channel operation order while separating the linear solve from the
active-nematic force model.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch


CHANNEL_VELOCITY_BOUNDARY_CONDITIONS = (
    "periodic",
    "dirichlet",
    "dirichlet",
)
CHANNEL_PRESSURE_BOUNDARY_CONDITIONS = (
    "periodic",
    "neumann",
    "neumann",
)


def _finite_coefficient(
    value: object,
    description: str,
    *,
    positive: bool,
) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
        or (not positive and float(value) < 0.0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{description} must be finite and {qualifier}")
    return float(value)


def _positive_integer(value: object, description: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or int(value) <= 0
    ):
        raise ValueError(f"{description} must be a positive integer")
    return int(value)


class ChannelNoSlipModalStokesSolver(torch.nn.Module):
    """Matrix-free mixed FFT/DST/DCT Channel Stokes/Brinkman solve."""

    def __init__(
        self,
        transform_backend,
        *,
        friction: float = 0.0,
        viscosity: float = 1.0,
        pressure_relative_tolerance: float = 1.0e-6,
        pressure_max_iterations: int = 80,
        pressure_fixed_iterations: int | None = None,
    ) -> None:
        super().__init__()
        if not hasattr(transform_backend, "get_metadata"):
            raise TypeError("transform_backend must provide transform metadata")
        if len(tuple(transform_backend.shape)) != 3:
            raise ValueError("Channel Stokes solver requires three dimensions")
        friction = _finite_coefficient(
            friction,
            "friction",
            positive=False,
        )
        viscosity = _finite_coefficient(
            viscosity,
            "viscosity",
            positive=True,
        )
        pressure_relative_tolerance = _finite_coefficient(
            pressure_relative_tolerance,
            "pressure_relative_tolerance",
            positive=True,
        )
        pressure_max_iterations = _positive_integer(
            pressure_max_iterations,
            "pressure_max_iterations",
        )
        if pressure_fixed_iterations is not None:
            pressure_fixed_iterations = _positive_integer(
                pressure_fixed_iterations,
                "pressure_fixed_iterations",
            )

        backend = transform_backend
        spectral_dtype = backend.spectral_dtype
        real_dtype = backend.real_dtype
        device = backend.device
        velocity_metadata = backend.get_metadata(
            CHANNEL_VELOCITY_BOUNDARY_CONDITIONS
        )
        pressure_metadata = backend.get_metadata(
            CHANNEL_PRESSURE_BOUNDARY_CONDITIONS
        )
        qx = velocity_metadata.axis_modes[0]
        ky_d = velocity_metadata.axis_modes[1]
        kz_d = velocity_metadata.axis_modes[2]
        ky_n = pressure_metadata.axis_modes[1]
        kz_n = pressure_metadata.axis_modes[2]
        nx, ny, nz = qx.numel(), ky_d.numel(), kz_d.numel()

        wall_ops = {}
        for axis, size, k_dirichlet in (
            (1, ny, ky_d),
            (2, nz, kz_d),
        ):
            dst = backend._get_matrix("dst", size).to(
                device=device,
                dtype=real_dtype,
            )
            dct = backend._get_matrix("dct", size).to(
                device=device,
                dtype=real_dtype,
            )
            d_d_to_n = torch.zeros(
                (size, size),
                device=device,
                dtype=real_dtype,
            )
            if size > 1:
                index = torch.arange(size - 1, device=device)
                d_d_to_n[index, index + 1] = k_dirichlet[:-1]
            wall_ops[axis] = {
                "n_to_d": (dct @ dst.transpose(0, 1)).to(spectral_dtype),
                "d_to_n": (dst @ dct.transpose(0, 1)).to(spectral_dtype),
                "dn_to_dd": (-d_d_to_n.transpose(0, 1)).to(spectral_dtype),
                "dd_to_dn": d_d_to_n.to(spectral_dtype),
            }

        a_diag = friction + viscosity * (
            qx.square().view(nx, 1, 1)
            + ky_d.square().view(1, ny, 1)
            + kz_d.square().view(1, 1, nz)
        )
        pressure_q2 = (
            qx.square().view(nx, 1, 1)
            + ky_n.square().view(1, ny, 1)
            + kz_n.square().view(1, 1, nz)
        )
        schur_diag = pressure_q2 / (
            friction
            + viscosity
            * pressure_q2.clamp_min(torch.finfo(real_dtype).eps)
        )
        null_mask = torch.zeros(
            (1, nx, ny, nz),
            device=device,
            dtype=torch.bool,
        )
        null_mask[:, 0, 0, 0] = True
        schur_safe = schur_diag.unsqueeze(0).clone()
        schur_safe[null_mask] = 1.0

        self.register_buffer(
            "ikx",
            (1j * qx).view(1, nx, 1, 1).to(spectral_dtype),
        )
        self.register_buffer("a_inv", (1.0 / a_diag).unsqueeze(0))
        for axis_name, axis in (("y", 1), ("z", 2)):
            for operator_name, operator in wall_ops[axis].items():
                self.register_buffer(f"{operator_name}_{axis_name}", operator)
        self.register_buffer("schur_diag_safe", schur_safe)
        self.register_buffer("pressure_null_mask", null_mask)
        self.pressure_rel_tol = pressure_relative_tolerance
        self.pressure_max_iter = pressure_max_iterations
        self.pressure_fixed_iterations = pressure_fixed_iterations
        self.pressure_guess = None
        self.last_pressure_iterations = 0
        self.last_pressure_residual = 0.0
        self.last_pressure_relative_residual = 0.0

    def _apply_axis_matrix(self, tensor, matrix, axis):
        spectral_axis = tensor.ndim - 3 + axis
        moved = tensor.movedim(spectral_axis, -1)
        return torch.matmul(moved, matrix).movedim(-1, spectral_axis)

    def _project_pressure_gauge(self, pressure_hat):
        return pressure_hat.masked_fill(self.pressure_null_mask, 0)

    def _helmholtz_inverse(self, rhs_hat):
        return rhs_hat * self.a_inv

    def _pressure_to_velocity(self, pressure_hat):
        out = self._apply_axis_matrix(pressure_hat, self.n_to_d_y, axis=1)
        return self._apply_axis_matrix(out, self.n_to_d_z, axis=2)

    def _velocity_to_pressure(self, velocity_hat):
        out = self._apply_axis_matrix(velocity_hat, self.d_to_n_y, axis=1)
        return self._apply_axis_matrix(out, self.d_to_n_z, axis=2)

    def _pressure_gradient(self, pressure_hat, axis):
        if axis == 0:
            return self.ikx * self._pressure_to_velocity(pressure_hat)
        if axis == 1:
            out = self._apply_axis_matrix(
                pressure_hat,
                self.dn_to_dd_y,
                axis=1,
            )
            return self._apply_axis_matrix(out, self.n_to_d_z, axis=2)
        if axis == 2:
            out = self._apply_axis_matrix(
                pressure_hat,
                self.n_to_d_y,
                axis=1,
            )
            return self._apply_axis_matrix(out, self.dn_to_dd_z, axis=2)
        raise IndexError(f"Unsupported pressure-gradient axis {axis}.")

    def _velocity_divergence_component(self, velocity_hat, axis):
        if axis == 0:
            return self._velocity_to_pressure(self.ikx * velocity_hat)
        if axis == 1:
            out = self._apply_axis_matrix(
                velocity_hat,
                self.dd_to_dn_y,
                axis=1,
            )
            return self._apply_axis_matrix(out, self.d_to_n_z, axis=2)
        if axis == 2:
            out = self._apply_axis_matrix(
                velocity_hat,
                self.d_to_n_y,
                axis=1,
            )
            return self._apply_axis_matrix(out, self.dd_to_dn_z, axis=2)
        raise IndexError(f"Unsupported velocity-divergence axis {axis}.")

    def pressure_gradient_hats(self, pressure_hat):
        """Return pressure-gradient coefficients in velocity bases."""

        return tuple(
            self._pressure_gradient(pressure_hat, axis)
            for axis in range(3)
        )

    def divergence_hat(self, ux_hat, uy_hat, uz_hat):
        """Return velocity divergence in the native pressure basis."""

        return sum(
            self._velocity_divergence_component(velocity_hat, axis)
            for axis, velocity_hat in enumerate((ux_hat, uy_hat, uz_hat))
        )

    def _pressure_operator(self, pressure_hat):
        pressure_hat = self._project_pressure_gauge(pressure_hat)
        velocity_hats = [
            self._helmholtz_inverse(
                self._pressure_gradient(pressure_hat, axis)
            )
            for axis in range(3)
        ]
        divergence_hat = self.divergence_hat(*velocity_hats)
        return self._project_pressure_gauge(-divergence_hat)

    def _solve_pressure(self, rhs_hat):
        rhs_hat = self._project_pressure_gauge(rhs_hat)
        rhs_norm = torch.linalg.vector_norm(rhs_hat.reshape(-1)).item()
        if rhs_norm == 0.0:
            self.last_pressure_iterations = 0
            self.last_pressure_residual = 0.0
            self.last_pressure_relative_residual = 0.0
            return torch.zeros_like(rhs_hat)
        if self.pressure_guess is None or self.pressure_guess.shape != rhs_hat.shape:
            pressure_hat = torch.zeros_like(rhs_hat)
        else:
            pressure_hat = self._project_pressure_gauge(
                self.pressure_guess.to(
                    device=rhs_hat.device,
                    dtype=rhs_hat.dtype,
                )
            )
        residual = rhs_hat - self._pressure_operator(pressure_hat)
        preconditioned = residual / self.schur_diag_safe
        direction = preconditioned.clone()
        rz_old = torch.sum(torch.conj(residual) * preconditioned).real
        tolerance = self.pressure_rel_tol * rhs_norm
        residual_norm = torch.linalg.vector_norm(residual.reshape(-1)).item()
        iterations = 0
        iteration_limit = (
            self.pressure_fixed_iterations
            if self.pressure_fixed_iterations is not None
            else self.pressure_max_iter
        )
        while iterations < iteration_limit and (
            self.pressure_fixed_iterations is not None
            or residual_norm > tolerance
        ):
            operator_direction = self._pressure_operator(direction)
            denominator = torch.sum(
                torch.conj(direction) * operator_direction
            ).real
            if denominator.abs().item() < 1e-30:
                break
            step = rz_old / denominator
            pressure_hat = self._project_pressure_gauge(
                pressure_hat + step * direction
            )
            residual = self._project_pressure_gauge(
                residual - step * operator_direction
            )
            residual_norm = torch.linalg.vector_norm(
                residual.reshape(-1)
            ).item()
            iterations += 1
            if residual_norm <= tolerance:
                break
            preconditioned = residual / self.schur_diag_safe
            rz_new = torch.sum(torch.conj(residual) * preconditioned).real
            if rz_old.abs().item() < 1e-30:
                break
            direction = preconditioned + (rz_new / rz_old) * direction
            rz_old = rz_new
        self.pressure_guess = pressure_hat.detach()
        self.last_pressure_iterations = iterations
        self.last_pressure_residual = residual_norm
        self.last_pressure_relative_residual = residual_norm / rhs_norm
        return pressure_hat

    def solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        """Solve the Channel saddle system for native force spectra."""

        force_hats = (fx_hat, fy_hat, fz_hat)
        free_velocity = [
            self._helmholtz_inverse(force_hat)
            for force_hat in force_hats
        ]
        provisional_divergence = self.divergence_hat(*free_velocity)
        pressure_hat = self._solve_pressure(
            self._project_pressure_gauge(-provisional_divergence)
        )
        velocity_hat = [
            free_velocity[axis]
            - self._helmholtz_inverse(
                self._pressure_gradient(pressure_hat, axis)
            )
            for axis in range(3)
        ]
        return (*velocity_hat, pressure_hat)


__all__ = [
    "CHANNEL_PRESSURE_BOUNDARY_CONDITIONS",
    "CHANNEL_VELOCITY_BOUNDARY_CONDITIONS",
    "ChannelNoSlipModalStokesSolver",
]
