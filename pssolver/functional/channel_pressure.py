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
    channel_pressure_active_rms,
    channel_pressure_magnitude_ratio,
    channel_pressure_relative_breakdown,
    channel_pressure_scale_power_of_two,
    channel_pressure_working_vector,
)

from .errors import FunctionalConvergenceError
from .pressure_metadata import (
    channel_pressure_solve_diagnostics,
)


CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION = "2"


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
        self._transpose_solve_sequence_id = 0
        self.last_transpose_diagnostics = channel_pressure_solve_diagnostics(
            solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="not_run",
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
        return pressure_hat.masked_fill(
            self._solver.pressure_null_mask | self._solver.nyquist_mask,
            0,
        )

    def _helmholtz_inverse_adjoint(self, rhs_hat):
        """Apply the transpose Helmholtz inverse on the production subspace."""

        return (torch.conj(self._solver.a_inv) * rhs_hat).masked_fill(
            self._solver.nyquist_mask,
            0,
        )

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
            self._helmholtz_inverse_adjoint(value)
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
        self._transpose_solve_sequence_id += 1
        sequence = self._transpose_solve_sequence_id
        if not bool(torch.isfinite(rhs_hat).all().item()):
            diagnostics = channel_pressure_solve_diagnostics(
                self._solver,
                operator_identity="channel_pressure_schur_conjugate_transpose",
                solve_phase="standalone_transpose_qualification",
                solve_sequence_id=sequence,
                achieved_normalized_residual=None,
                termination_reason="nonfinite_input",
            )
            self._record_transpose_diagnostics(diagnostics)
            raise FunctionalConvergenceError(
                "standalone Channel pressure transpose input is non-finite",
                operation="channel_pressure_transpose_solve",
                details=diagnostics.to_metadata(),
            )
        working_rhs, working_exponent, rhs_magnitude = (
            channel_pressure_working_vector(
                rhs_hat,
                self._solver.pressure_inactive_mask,
            )
        )
        if rhs_magnitude.is_zero:
            diagnostics = channel_pressure_solve_diagnostics(
                self._solver,
                operator_identity="channel_pressure_schur_conjugate_transpose",
                solve_phase="standalone_transpose_qualification",
                solve_sequence_id=sequence,
                rhs_norm=rhs_magnitude,
                initial_absolute_residual=rhs_magnitude,
                achieved_absolute_residual=rhs_magnitude,
                achieved_normalized_residual=0.0,
                working_scale_binary_exponent=None,
                achieved_iteration_count=0,
                termination_reason="zero_rhs",
            )
            self._record_transpose_diagnostics(diagnostics)
            return torch.zeros_like(rhs_hat)
        pressure_hat = torch.zeros_like(working_rhs)
        residual = working_rhs.clone()
        initial_residual = rhs_magnitude
        rhs_work_magnitude = channel_pressure_active_rms(
            working_rhs,
            self._solver.pressure_inactive_mask,
        )
        preconditioned = residual / self._solver.schur_diag_safe
        if channel_pressure_relative_breakdown(
            residual,
            preconditioned,
            self._solver.pressure_inactive_mask,
        ):
            return self._fail_transpose_solve(
                sequence=sequence,
                rhs_norm=rhs_magnitude,
                initial_residual=initial_residual,
                achieved_residual=initial_residual,
                relative_residual=1.0,
                working_exponent=working_exponent,
                iterations=0,
                termination_reason="preconditioned_residual_breakdown",
            )
        direction = preconditioned.clone()
        rz_old = torch.sum(torch.conj(residual) * preconditioned).real
        residual_work_magnitude = rhs_work_magnitude
        residual_magnitude = initial_residual
        relative_residual = 1.0
        iterations = 0
        fixed_work = self._solver.pressure_fixed_iterations is not None
        iteration_limit = self._solver.pressure_iteration_limit
        termination_reason = "iteration_limit_reached_without_postcondition"
        acceptable = False
        while iterations < iteration_limit:
            operator_direction = self.apply_pressure_operator_transpose(
                direction
            )
            if not bool(torch.isfinite(operator_direction).all().item()):
                termination_reason = "nonfinite_intermediate"
                break
            if channel_pressure_relative_breakdown(
                direction,
                operator_direction,
                self._solver.pressure_inactive_mask,
            ):
                termination_reason = "operator_breakdown"
                break
            denominator = torch.sum(
                torch.conj(direction) * operator_direction
            ).real
            step = rz_old / denominator
            pressure_hat = self._project_gauge(pressure_hat + step * direction)
            residual = self._project_gauge(
                residual - step * operator_direction
            )
            iterations += 1
            if not bool(
                torch.isfinite(pressure_hat).all().item()
                and torch.isfinite(residual).all().item()
            ):
                termination_reason = "nonfinite_intermediate"
                break
            residual_work_magnitude = channel_pressure_active_rms(
                residual,
                self._solver.pressure_inactive_mask,
            )
            residual_magnitude = residual_work_magnitude.shifted(
                int(working_exponent)
            )
            relative_residual = channel_pressure_magnitude_ratio(
                residual_work_magnitude,
                rhs_work_magnitude,
            )
            if residual_work_magnitude.is_zero:
                termination_reason = "exact_zero_residual"
                acceptable = True
                break
            if not fixed_work and (
                relative_residual <= self._solver.pressure_rel_tol
            ):
                termination_reason = "relative_tolerance_met"
                acceptable = True
                break
            if iterations == iteration_limit:
                if fixed_work:
                    termination_reason = "fixed_work_completed"
                    acceptable = True
                break
            preconditioned = residual / self._solver.schur_diag_safe
            if not bool(torch.isfinite(preconditioned).all().item()):
                termination_reason = "nonfinite_intermediate"
                break
            if channel_pressure_relative_breakdown(
                residual,
                preconditioned,
                self._solver.pressure_inactive_mask,
            ):
                termination_reason = "preconditioned_residual_breakdown"
                break
            rz_new = torch.sum(torch.conj(residual) * preconditioned).real
            direction = preconditioned + (rz_new / rz_old) * direction
            rz_old = rz_new
        if not acceptable:
            return self._fail_transpose_solve(
                sequence=sequence,
                rhs_norm=rhs_magnitude,
                initial_residual=initial_residual,
                achieved_residual=residual_magnitude,
                relative_residual=relative_residual,
                working_exponent=working_exponent,
                iterations=iterations,
                termination_reason=termination_reason,
            )
        diagnostics = channel_pressure_solve_diagnostics(
            self._solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="standalone_transpose_qualification",
            solve_sequence_id=sequence,
            rhs_norm=rhs_magnitude,
            initial_absolute_residual=initial_residual,
            achieved_absolute_residual=residual_magnitude,
            achieved_normalized_residual=relative_residual,
            working_scale_binary_exponent=working_exponent,
            achieved_iteration_count=iterations,
            termination_reason=termination_reason,
        )
        self._record_transpose_diagnostics(diagnostics)
        result = channel_pressure_scale_power_of_two(
            pressure_hat,
            int(working_exponent),
        )
        if not bool(torch.isfinite(result).all().item()):
            return self._fail_transpose_solve(
                sequence=sequence,
                rhs_norm=rhs_magnitude,
                initial_residual=initial_residual,
                achieved_residual=residual_magnitude,
                relative_residual=relative_residual,
                working_exponent=working_exponent,
                iterations=iterations,
                termination_reason="nonfinite_intermediate",
            )
        return result

    def _record_transpose_diagnostics(self, diagnostics) -> None:
        self.last_transpose_diagnostics = diagnostics
        self.last_transpose_iterations = diagnostics.iterations
        self.last_transpose_residual = diagnostics.residual
        self.last_transpose_relative_residual = diagnostics.relative_residual

    def _fail_transpose_solve(
        self,
        *,
        sequence,
        rhs_norm,
        initial_residual,
        achieved_residual,
        relative_residual,
        working_exponent,
        iterations,
        termination_reason,
    ):
        diagnostics = channel_pressure_solve_diagnostics(
            self._solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="standalone_transpose_qualification",
            solve_sequence_id=sequence,
            rhs_norm=rhs_norm,
            initial_absolute_residual=initial_residual,
            achieved_absolute_residual=achieved_residual,
            achieved_normalized_residual=relative_residual,
            working_scale_binary_exponent=working_exponent,
            achieved_iteration_count=iterations,
            termination_reason=termination_reason,
        )
        self._record_transpose_diagnostics(diagnostics)
        raise FunctionalConvergenceError(
            "standalone Channel pressure transpose solve is unacceptable",
            operation="channel_pressure_transpose_solve",
            details=diagnostics.to_metadata(),
        )

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
            "component_role": "pressure_transpose_operator",
            "component_is_runtime": False,
            "component_connected_to_executable_runtime": True,
            "connected_runtime_kind": "channel_activity_batch_one",
            "custom_autograd_rule": False,
            "public_explicit_vjp_method": False,
        }

    def pressure_solve_metadata(self) -> dict[str, object]:
        """Return the latest transpose solve without changing PCG behavior."""

        return self.last_transpose_diagnostics.to_metadata()


__all__ = [
    "CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION",
    "ChannelPressureTransposeOperator",
    "ChannelPressureTransposeProtocol",
]
