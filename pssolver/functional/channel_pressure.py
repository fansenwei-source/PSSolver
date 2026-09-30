"""Explicit Channel pressure-transpose operators for functional execution.

P9.7.2 deliberately adapts the frozen production Channel Stokes solver rather
than modifying it.  The adapter exposes the primal pressure Schur action and
an independently assembled conjugate-transpose action in the native complex
coefficient-space inner product.  It does not define an autograd rule; that
remains the separately qualified P9.7.3 slice.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from pssolver.linear_solvers.stokes.channel_no_slip import (
    ChannelNoSlipModalStokesSolver,
)

from .pressure_metadata import (
    channel_pressure_solve_diagnostics,
)


CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION = "1"


@runtime_checkable
class ChannelPressureTransposeProtocol(Protocol):
    """Primal, explicit-transpose, and transpose-solve pressure contract."""

    def apply_pressure_operator(
        self,
        pressure_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the zero-gauge Channel pressure Schur operator."""

        ...

    def apply_pressure_operator_transpose(
        self,
        pressure_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the explicit conjugate-transpose Schur action."""

        ...

    def solve_pressure_transpose(
        self,
        rhs_hat: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the transpose system from a zero initial guess."""

        ...

    def pressure_transpose_metadata(self) -> dict[str, object]:
        """Return the JSON-safe operator and solve identity."""

        ...

    def pressure_solve_metadata(self) -> dict[str, object]:
        """Return the most recent stable transpose-solve outcome."""

        ...


class ChannelPressureTransposeOperator:
    """Functional adapter over a frozen production Channel pressure solver.

    The primal action delegates to the canonical solver.  The transpose action
    reverses the actual ``gauge -> gradient -> Helmholtz -> divergence ->
    gauge`` dataflow and applies every stored factor's conjugate transpose.
    Consequently symmetry is a measured property, not an implementation
    assumption.
    """

    def __init__(self, solver: ChannelNoSlipModalStokesSolver) -> None:
        if not isinstance(solver, ChannelNoSlipModalStokesSolver):
            raise TypeError(
                "solver must be a ChannelNoSlipModalStokesSolver"
            )
        self._solver = solver
        self.last_transpose_iterations = 0
        self.last_transpose_residual = 0.0
        self.last_transpose_relative_residual = 0.0
        self.last_transpose_diagnostics = channel_pressure_solve_diagnostics(
            solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            achieved_iteration_count=0,
            achieved_absolute_residual=0.0,
            achieved_relative_residual=0.0,
            termination_reason="not_run",
        )

    def _apply_axis_matrix(self, tensor, matrix, axis):
        spectral_axis = tensor.ndim - 3 + axis
        moved = tensor.movedim(spectral_axis, -1)
        return torch.matmul(moved, matrix).movedim(-1, spectral_axis)

    def _apply_axis_matrix_adjoint(self, tensor, matrix, axis):
        return self._apply_axis_matrix(
            tensor,
            torch.conj(matrix).transpose(-2, -1),
            axis,
        )

    def _project_gauge(self, pressure_hat):
        return pressure_hat.masked_fill(self._solver.pressure_null_mask, 0)

    def _pressure_to_velocity_adjoint(self, velocity_hat):
        out = self._apply_axis_matrix_adjoint(
            velocity_hat,
            self._solver.n_to_d_z,
            axis=2,
        )
        return self._apply_axis_matrix_adjoint(
            out,
            self._solver.n_to_d_y,
            axis=1,
        )

    def _velocity_to_pressure_adjoint(self, pressure_hat):
        out = self._apply_axis_matrix_adjoint(
            pressure_hat,
            self._solver.d_to_n_z,
            axis=2,
        )
        return self._apply_axis_matrix_adjoint(
            out,
            self._solver.d_to_n_y,
            axis=1,
        )

    def _pressure_gradient_adjoint(self, velocity_hat, axis):
        if axis == 0:
            return self._pressure_to_velocity_adjoint(
                torch.conj(self._solver.ikx) * velocity_hat
            )
        if axis == 1:
            out = self._apply_axis_matrix_adjoint(
                velocity_hat,
                self._solver.n_to_d_z,
                axis=2,
            )
            return self._apply_axis_matrix_adjoint(
                out,
                self._solver.dn_to_dd_y,
                axis=1,
            )
        if axis == 2:
            out = self._apply_axis_matrix_adjoint(
                velocity_hat,
                self._solver.dn_to_dd_z,
                axis=2,
            )
            return self._apply_axis_matrix_adjoint(
                out,
                self._solver.n_to_d_y,
                axis=1,
            )
        raise IndexError(f"Unsupported pressure-gradient adjoint axis {axis}.")

    def _velocity_divergence_component_adjoint(self, pressure_hat, axis):
        if axis == 0:
            return torch.conj(
                self._solver.ikx
            ) * self._velocity_to_pressure_adjoint(pressure_hat)
        if axis == 1:
            out = self._apply_axis_matrix_adjoint(
                pressure_hat,
                self._solver.d_to_n_z,
                axis=2,
            )
            return self._apply_axis_matrix_adjoint(
                out,
                self._solver.dd_to_dn_y,
                axis=1,
            )
        if axis == 2:
            out = self._apply_axis_matrix_adjoint(
                pressure_hat,
                self._solver.dd_to_dn_z,
                axis=2,
            )
            return self._apply_axis_matrix_adjoint(
                out,
                self._solver.d_to_n_y,
                axis=1,
            )
        raise IndexError(
            f"Unsupported velocity-divergence adjoint axis {axis}."
        )

    def apply_pressure_operator(self, pressure_hat):
        """Apply the unchanged production pressure Schur action."""

        return self._solver._pressure_operator(pressure_hat)

    def apply_pressure_operator_transpose(self, pressure_hat):
        """Apply the explicit conjugate transpose of the Schur operator."""

        pressure_hat = self._project_gauge(pressure_hat)
        divergence_adjoint_hats = [
            self._velocity_divergence_component_adjoint(
                -pressure_hat,
                axis,
            )
            for axis in range(3)
        ]
        helmholtz_adjoint_hats = [
            torch.conj(self._solver.a_inv) * value
            for value in divergence_adjoint_hats
        ]
        result = sum(
            self._pressure_gradient_adjoint(value, axis)
            for axis, value in enumerate(helmholtz_adjoint_hats)
        )
        return self._project_gauge(result)

    @torch.no_grad()
    def solve_pressure_transpose(self, rhs_hat):
        """Solve the transpose Schur system without production warm state.

        This operator-level PCG exists for P9.7.2 manufactured validation.  It
        always starts at zero and neither reads nor writes ``solver``'s
        pressure warm start.  P9.7.3 owns the custom implicit autograd rule.
        """

        rhs_hat = self._project_gauge(rhs_hat)
        rhs_norm = torch.linalg.vector_norm(rhs_hat.reshape(-1)).item()
        if rhs_norm == 0.0:
            self.last_transpose_iterations = 0
            self.last_transpose_residual = 0.0
            self.last_transpose_relative_residual = 0.0
            self.last_transpose_diagnostics = (
                channel_pressure_solve_diagnostics(
                    self._solver,
                    operator_identity=(
                        "channel_pressure_schur_conjugate_transpose"
                    ),
                    achieved_iteration_count=0,
                    achieved_absolute_residual=0.0,
                    achieved_relative_residual=0.0,
                    termination_reason="zero_rhs",
                )
            )
            return torch.zeros_like(rhs_hat)
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
        termination_reason = "iteration_limit_reached"
        if (
            self._solver.pressure_fixed_iterations is None
            and residual_norm <= tolerance
        ):
            termination_reason = "relative_tolerance_met"
        while iterations < iteration_limit and (
            self._solver.pressure_fixed_iterations is not None
            or residual_norm > tolerance
        ):
            operator_direction = self.apply_pressure_operator_transpose(
                direction
            )
            denominator = torch.sum(
                torch.conj(direction) * operator_direction
            ).real
            if denominator.abs().item() < 1e-30:
                termination_reason = "operator_breakdown"
                break
            step = rz_old / denominator
            pressure_hat = self._project_gauge(pressure_hat + step * direction)
            residual = self._project_gauge(
                residual - step * operator_direction
            )
            residual_norm = torch.linalg.vector_norm(
                residual.reshape(-1)
            ).item()
            iterations += 1
            if residual_norm <= tolerance:
                termination_reason = "relative_tolerance_met"
                break
            preconditioned = residual / self._solver.schur_diag_safe
            rz_new = torch.sum(torch.conj(residual) * preconditioned).real
            if rz_old.abs().item() < 1e-30:
                termination_reason = "preconditioned_residual_breakdown"
                break
            direction = preconditioned + (rz_new / rz_old) * direction
            rz_old = rz_new
        self.last_transpose_iterations = iterations
        self.last_transpose_residual = residual_norm
        self.last_transpose_relative_residual = residual_norm / rhs_norm
        self.last_transpose_diagnostics = channel_pressure_solve_diagnostics(
            self._solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            achieved_iteration_count=iterations,
            achieved_absolute_residual=residual_norm,
            achieved_relative_residual=residual_norm / rhs_norm,
            termination_reason=termination_reason,
        )
        return pressure_hat

    def pressure_transpose_metadata(self) -> dict[str, object]:
        """Describe the exact operator-level contract without runtime claims."""

        return {
            "protocol_version": CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION,
            "inner_product": "native_complex_euclidean_coefficient_space",
            "gauge": "zero_mean_pressure_mode",
            "primal_action": "frozen_production_schur_operator",
            "transpose_action": "explicit_reverse_dataflow_conjugate_transpose",
            "transpose_solve": "zero_initial_guess_pcg",
            "production_pressure_warm_start_read": False,
            "production_pressure_warm_start_written": False,
            "pcg_iteration_graph_retained": False,
            "custom_autograd_rule": False,
            "functional_runtime_executable": False,
        }

    def pressure_solve_metadata(self) -> dict[str, object]:
        """Return the latest transpose solve without changing PCG behavior."""

        return self.last_transpose_diagnostics.to_metadata()


__all__ = [
    "CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION",
    "ChannelPressureTransposeOperator",
    "ChannelPressureTransposeProtocol",
]
