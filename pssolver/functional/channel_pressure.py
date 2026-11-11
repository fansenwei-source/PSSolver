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
    PCGRole,
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
            raise TypeError("solver must be a ChannelNoSlipModalStokesSolver")
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
        return self._solver._apply_axis_matrix_adjoint(tensor, matrix, axis)

    def _project_gauge(self, pressure_hat):
        return pressure_hat.masked_fill(
            self._solver.pressure_null_mask | self._solver.nyquist_mask,
            0,
        )

    def _helmholtz_inverse_adjoint(self, rhs_hat):
        return self._solver._helmholtz_inverse_adjoint(rhs_hat)

    def _pressure_to_velocity_adjoint(self, velocity_hat):
        return self._solver._pressure_to_velocity_adjoint(velocity_hat)

    def _velocity_to_pressure_adjoint(self, pressure_hat):
        return self._solver._velocity_to_pressure_adjoint(pressure_hat)

    def _pressure_gradient_adjoint(self, velocity_hat, axis):
        return self._solver._pressure_gradient_adjoint(velocity_hat, axis)

    def _velocity_divergence_component_adjoint(self, pressure_hat, axis):
        return self._solver._velocity_divergence_component_adjoint(pressure_hat, axis)

    def apply_pressure_operator(self, pressure_hat):
        """Apply the unchanged production pressure Schur action."""

        return self._solver._pressure_operator(pressure_hat)

    def apply_pressure_operator_transpose(self, pressure_hat):
        return self._solver.apply_pressure_operator_transpose(pressure_hat)

    @torch.no_grad()
    def solve_pressure_transpose(self, rhs_hat):
        return self._solver.solve_transpose(
            rhs_hat,
            {
                "context": self,
                "action": self.apply_pressure_operator_transpose,
                "diagnostics": channel_pressure_solve_diagnostics,
                "error": FunctionalConvergenceError,
                "standalone": True,
            },
            role=PCGRole.FUNCTIONAL_TRANSPOSE,
        )

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
