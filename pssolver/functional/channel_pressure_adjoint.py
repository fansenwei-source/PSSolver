"""Custom implicit Channel pressure adjoint and small-grid CPU oracle.

P9.7.3 differentiates only the pressure right-hand side.  The pressure
operator coefficients are fixed runtime identity, and both the primal and
transpose PCG solves begin from zero.  The custom autograd node retains no PCG
iteration tensors.  A separately exposed fixed-iteration implementation is
restricted to small CPU problems and exists only as a qualification oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Protocol, runtime_checkable

import torch
from torch.autograd.function import once_differentiable

from pssolver.linear_solvers.stokes.channel_no_slip import (
    ChannelNoSlipModalStokesSolver,
)

from .channel_pressure import ChannelPressureTransposeOperator


CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION = "1"
CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES = 512


@dataclass(frozen=True, slots=True)
class ChannelPressureSolveDiagnostics:
    """Scalar diagnostics from one graph-free pressure solve."""

    iterations: int
    residual: float
    relative_residual: float


@runtime_checkable
class ChannelImplicitPressureAdjointProtocol(Protocol):
    """Batch-one implicit pressure solve with an explicit custom VJP."""

    def solve(self, rhs_hat: torch.Tensor) -> torch.Tensor:
        """Return the zero-gauge pressure with a custom implicit VJP."""

        ...

    def solve_force_hats(
        self,
        fx_hat: torch.Tensor,
        fy_hat: torch.Tensor,
        fz_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return velocity and pressure hats without pressure warm state."""

        ...

    def implicit_adjoint_metadata(self) -> dict[str, object]:
        """Return the JSON-safe differentiation identity."""

        ...


class _ImplicitPressureSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs_hat, operator):
        ctx.operator = operator
        return operator._solve_primal_no_grad(rhs_hat)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_pressure_hat):
        grad_rhs = ctx.operator._solve_transpose_no_grad(grad_pressure_hat)
        return grad_rhs, None


class ChannelImplicitPressureAdjoint:
    """Zero-warm-start Channel pressure solve with an implicit custom VJP."""

    def __init__(self, solver: ChannelNoSlipModalStokesSolver) -> None:
        if not isinstance(solver, ChannelNoSlipModalStokesSolver):
            raise TypeError(
                "solver must be a ChannelNoSlipModalStokesSolver"
            )
        self._solver = solver
        self._operators = ChannelPressureTransposeOperator(solver)
        self.last_primal_diagnostics = ChannelPressureSolveDiagnostics(
            iterations=0,
            residual=0.0,
            relative_residual=0.0,
        )
        self.last_transpose_diagnostics = ChannelPressureSolveDiagnostics(
            iterations=0,
            residual=0.0,
            relative_residual=0.0,
        )

    def _validate_rhs(self, rhs_hat: torch.Tensor) -> None:
        if not isinstance(rhs_hat, torch.Tensor):
            raise TypeError("pressure right-hand side must be a torch.Tensor")
        expected_shape = tuple(self._solver.pressure_null_mask.shape)
        if tuple(rhs_hat.shape) != expected_shape:
            raise ValueError(
                "P9.7.3 pressure right-hand side shape must be "
                f"{expected_shape}, got {tuple(rhs_hat.shape)}"
            )
        if rhs_hat.dtype is not self._solver.ikx.dtype:
            raise TypeError(
                "pressure right-hand side dtype must match the solver "
                f"spectral dtype {self._solver.ikx.dtype}"
            )
        if rhs_hat.device != self._solver.ikx.device:
            raise ValueError(
                "pressure right-hand side device must match the solver device"
            )
        if not bool(torch.isfinite(rhs_hat.detach()).all().item()):
            raise ValueError("pressure right-hand side must be finite")

    @torch.no_grad()
    def _solve_no_grad(self, rhs_hat, action):
        rhs_hat = self._operators._project_gauge(rhs_hat)
        rhs_norm = torch.linalg.vector_norm(rhs_hat.reshape(-1)).item()
        if rhs_norm == 0.0:
            return torch.zeros_like(rhs_hat), ChannelPressureSolveDiagnostics(
                iterations=0,
                residual=0.0,
                relative_residual=0.0,
            )
        pressure_hat = torch.zeros_like(rhs_hat)
        residual = rhs_hat.clone()
        preconditioned = residual / self._solver.schur_diag_safe
        direction = preconditioned.clone()
        rz_old = torch.sum(torch.conj(residual) * preconditioned).real
        tolerance = self._solver.pressure_rel_tol * rhs_norm
        residual_norm = rhs_norm
        iterations = 0
        iteration_limit = (
            self._solver.pressure_fixed_iterations
            if self._solver.pressure_fixed_iterations is not None
            else self._solver.pressure_max_iter
        )
        while iterations < iteration_limit and (
            self._solver.pressure_fixed_iterations is not None
            or residual_norm > tolerance
        ):
            operator_direction = action(direction)
            denominator = torch.sum(
                torch.conj(direction) * operator_direction
            ).real
            if denominator.abs().item() < 1e-30:
                break
            step = rz_old / denominator
            pressure_hat = self._operators._project_gauge(
                pressure_hat + step * direction
            )
            residual = self._operators._project_gauge(
                residual - step * operator_direction
            )
            residual_norm = torch.linalg.vector_norm(
                residual.reshape(-1)
            ).item()
            iterations += 1
            if residual_norm <= tolerance:
                break
            preconditioned = residual / self._solver.schur_diag_safe
            rz_new = torch.sum(torch.conj(residual) * preconditioned).real
            if rz_old.abs().item() < 1e-30:
                break
            direction = preconditioned + (rz_new / rz_old) * direction
            rz_old = rz_new
        diagnostics = ChannelPressureSolveDiagnostics(
            iterations=iterations,
            residual=residual_norm,
            relative_residual=residual_norm / rhs_norm,
        )
        return pressure_hat, diagnostics

    def _solve_primal_no_grad(self, rhs_hat):
        value, diagnostics = self._solve_no_grad(
            rhs_hat,
            self._operators.apply_pressure_operator,
        )
        self._validate_solve_diagnostics(diagnostics, "primal")
        self.last_primal_diagnostics = diagnostics
        return value

    def _solve_transpose_no_grad(self, rhs_hat):
        value, diagnostics = self._solve_no_grad(
            rhs_hat,
            self._operators.apply_pressure_operator_transpose,
        )
        self._validate_solve_diagnostics(diagnostics, "transpose")
        self.last_transpose_diagnostics = diagnostics
        return value

    def _validate_solve_diagnostics(
        self,
        diagnostics: ChannelPressureSolveDiagnostics,
        description: str,
    ) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                diagnostics.residual,
                diagnostics.relative_residual,
            )
        ):
            raise RuntimeError(
                f"functional pressure {description} solve is non-finite"
            )
        fixed = self._solver.pressure_fixed_iterations
        if fixed is None:
            if diagnostics.relative_residual > self._solver.pressure_rel_tol:
                raise RuntimeError(
                    f"functional pressure {description} solve did not meet "
                    "its relative-residual tolerance"
                )
            return
        if diagnostics.iterations > fixed:
            raise RuntimeError(
                f"functional pressure {description} solve exceeded its "
                "fixed-iteration contract"
            )
        if (
            diagnostics.iterations < fixed
            and diagnostics.relative_residual > self._solver.pressure_rel_tol
        ):
            raise RuntimeError(
                f"functional pressure {description} solve stopped before "
                "its fixed-iteration or residual contract"
            )

    def solve(self, rhs_hat: torch.Tensor) -> torch.Tensor:
        """Solve pressure and attach the graph-free implicit VJP."""

        self._validate_rhs(rhs_hat)
        return _ImplicitPressureSolve.apply(rhs_hat, self)

    def solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        """Solve the surrounding Stokes map with functional pressure state."""

        force_hats = (fx_hat, fy_hat, fz_hat)
        if any(
            not isinstance(value, torch.Tensor) for value in force_hats
        ):
            raise TypeError("Channel force coefficients must be tensors")
        expected = tuple(self._solver.a_inv.shape)
        for value in force_hats:
            if tuple(value.shape) != expected:
                raise ValueError(
                    "Channel force coefficient shape must be "
                    f"{expected}, got {tuple(value.shape)}"
                )
            if value.dtype is not self._solver.ikx.dtype:
                raise TypeError(
                    "Channel force coefficient dtype must match the solver"
                )
            if value.device != self._solver.ikx.device:
                raise ValueError(
                    "Channel force coefficient device must match the solver"
                )
            if not bool(torch.isfinite(value.detach()).all().item()):
                raise ValueError("Channel force coefficients must be finite")
        free_velocity = tuple(
            self._solver._helmholtz_inverse(value) for value in force_hats
        )
        pressure_rhs = self._operators._project_gauge(
            -self._solver.divergence_hat(*free_velocity)
        )
        pressure_hat = self.solve(pressure_rhs)
        pressure_gradient = self._solver.pressure_gradient_hats(pressure_hat)
        velocity_hat = tuple(
            free_velocity[axis]
            - self._solver._helmholtz_inverse(pressure_gradient[axis])
            for axis in range(3)
        )
        return (*velocity_hat, pressure_hat)

    def implicit_adjoint_metadata(self) -> dict[str, object]:
        """Describe the qualified scope without claiming a Channel runtime."""

        return {
            "version": CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION,
            "batch_size": 1,
            "differentiable_input": "pressure_rhs_hat",
            "operator_coefficients_differentiable": False,
            "primal_solve": "zero_initial_guess_pcg",
            "backward_solve": "explicit_transpose_zero_initial_guess_pcg",
            "production_pressure_warm_start_read": False,
            "production_pressure_warm_start_written": False,
            "forward_iteration_tensors_saved": 0,
            "backward_iteration_tensors_saved": 0,
            "higher_order_derivatives": False,
            "unrolled_oracle": "small_grid_cpu_fixed_iteration_only",
            "functional_runtime_executable": False,
        }


def unrolled_channel_pressure_solve_oracle(
    rhs_hat: torch.Tensor,
    solver: ChannelNoSlipModalStokesSolver,
    *,
    iterations: int,
) -> torch.Tensor:
    """Differentiate a fixed-iteration primal PCG only on a small CPU grid."""

    if not isinstance(solver, ChannelNoSlipModalStokesSolver):
        raise TypeError("solver must be a ChannelNoSlipModalStokesSolver")
    if (
        not isinstance(iterations, Integral)
        or isinstance(iterations, bool)
        or int(iterations) <= 0
    ):
        raise ValueError("iterations must be a positive integer")
    operator = ChannelImplicitPressureAdjoint(solver)
    operator._validate_rhs(rhs_hat)
    if rhs_hat.device.type != "cpu":
        raise ValueError("the unrolled pressure oracle is CPU-only")
    if rhs_hat.numel() > CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES:
        raise ValueError(
            "the unrolled pressure oracle is restricted to small grids"
        )

    actions = ChannelPressureTransposeOperator(solver)
    rhs_hat = actions._project_gauge(rhs_hat)
    pressure_hat = torch.zeros_like(rhs_hat)
    residual = rhs_hat
    preconditioned = residual / solver.schur_diag_safe
    direction = preconditioned
    rz_old = torch.sum(torch.conj(residual) * preconditioned).real
    for index in range(int(iterations)):
        operator_direction = actions.apply_pressure_operator(direction)
        denominator = torch.sum(
            torch.conj(direction) * operator_direction
        ).real
        step = rz_old / denominator
        pressure_hat = actions._project_gauge(
            pressure_hat + step * direction
        )
        residual = actions._project_gauge(
            residual - step * operator_direction
        )
        preconditioned = residual / solver.schur_diag_safe
        rz_new = torch.sum(torch.conj(residual) * preconditioned).real
        if index + 1 < int(iterations):
            direction = preconditioned + (rz_new / rz_old) * direction
        rz_old = rz_new
    return pressure_hat


__all__ = [
    "CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION",
    "CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES",
    "ChannelImplicitPressureAdjoint",
    "ChannelImplicitPressureAdjointProtocol",
    "ChannelPressureSolveDiagnostics",
    "unrolled_channel_pressure_solve_oracle",
]
