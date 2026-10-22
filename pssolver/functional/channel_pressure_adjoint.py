"""Custom implicit Channel pressure adjoint and small-grid CPU oracle.

P9.7.3 differentiates only the pressure right-hand side.  The pressure
operator coefficients are fixed runtime identity, and both the primal and
transpose PCG solves begin from zero.  The custom autograd node retains no PCG
iteration tensors.  A separately exposed fixed-iteration implementation is
restricted to small CPU problems and exists only as a qualification oracle.
"""

from __future__ import annotations

from numbers import Integral
from typing import Protocol, runtime_checkable

import torch
from torch.autograd.function import once_differentiable

from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION,
    ChannelNoSlipModalStokesSolver,
    channel_pressure_active_rms,
    channel_pressure_fixed_work_roundoff_saturated,
    channel_pressure_magnitude_ratio,
    channel_pressure_minimum_relative_tolerance,
    channel_pressure_relative_breakdown,
    channel_pressure_scale_power_of_two,
    channel_pressure_working_vector,
)

from .channel_pressure import ChannelPressureTransposeOperator
from .errors import FunctionalConvergenceError
from .pressure_metadata import (
    CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
    ChannelPressureSolveDiagnostics,
    channel_pressure_solve_diagnostics,
)


CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION = "2"
CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES = 512


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

    def pressure_solve_metadata(self) -> dict[str, object]:
        """Return stable primal and transpose convergence outcomes."""

        ...


class _ImplicitPressureSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs_hat, operator):
        ctx.operator = operator
        value = operator._solve_primal_no_grad(rhs_hat)
        ctx.solve_sequence_id = operator._solve_sequence_id
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_pressure_hat):
        grad_rhs = ctx.operator._solve_transpose_no_grad(
            grad_pressure_hat,
            source_primal_sequence_id=ctx.solve_sequence_id,
        )
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
        self._solve_sequence_id = 0
        self.last_primal_diagnostics = channel_pressure_solve_diagnostics(
            solver,
            operator_identity="channel_pressure_schur_primal",
            solve_phase="not_run",
        )
        self.last_transpose_diagnostics = channel_pressure_solve_diagnostics(
            solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="not_run",
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
    @torch.no_grad()
    def _solve_no_grad(
        self,
        rhs_hat,
        action,
        operator_identity,
        *,
        solve_phase,
        solve_sequence_id,
        source_primal_sequence_id,
    ):
        rhs_hat = self._operators._project_gauge(rhs_hat)
        if not bool(torch.isfinite(rhs_hat).all().item()):
            diagnostics = channel_pressure_solve_diagnostics(
                self._solver,
                operator_identity=operator_identity,
                solve_phase=solve_phase,
                solve_sequence_id=solve_sequence_id,
                source_primal_sequence_id=source_primal_sequence_id,
                achieved_iteration_count=0,
                achieved_normalized_residual=None,
                termination_reason="nonfinite_input",
            )
            self._persist_solve_diagnostics(diagnostics)
            raise FunctionalConvergenceError(
                f"functional pressure {solve_phase} input is non-finite",
                operation=f"channel_pressure_{solve_phase}_solve",
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
                operator_identity=operator_identity,
                solve_phase=solve_phase,
                solve_sequence_id=solve_sequence_id,
                source_primal_sequence_id=source_primal_sequence_id,
                rhs_norm=rhs_magnitude,
                initial_absolute_residual=rhs_magnitude,
                achieved_absolute_residual=rhs_magnitude,
                achieved_normalized_residual=0.0,
                working_scale_binary_exponent=None,
                achieved_iteration_count=0,
                termination_reason="zero_rhs",
            )
            return torch.zeros_like(rhs_hat), diagnostics
        pressure_hat = torch.zeros_like(working_rhs)
        residual = working_rhs.clone()
        rhs_work_magnitude = channel_pressure_active_rms(
            working_rhs,
            self._solver.pressure_inactive_mask,
        )
        initial_residual = rhs_magnitude
        preconditioned = residual / self._solver.schur_diag_safe
        if channel_pressure_relative_breakdown(
            residual,
            preconditioned,
            self._solver.pressure_inactive_mask,
        ):
            diagnostics = channel_pressure_solve_diagnostics(
                self._solver,
                operator_identity=operator_identity,
                solve_phase=solve_phase,
                solve_sequence_id=solve_sequence_id,
                source_primal_sequence_id=source_primal_sequence_id,
                rhs_norm=rhs_magnitude,
                initial_absolute_residual=initial_residual,
                achieved_absolute_residual=initial_residual,
                achieved_normalized_residual=1.0,
                working_scale_binary_exponent=working_exponent,
                achieved_iteration_count=0,
                termination_reason="preconditioned_residual_breakdown",
            )
            return pressure_hat, diagnostics
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
        fixed_work_floor_pressure = None
        fixed_work_floor_iterations = None
        fixed_work_floor_relative_residual = None
        saturation_failure_reason = (
            "iteration_limit_reached_without_postcondition"
        )
        if fixed_work and channel_pressure_fixed_work_roundoff_saturated(
            relative_residual,
            residual.dtype,
        ):
            fixed_work_floor_pressure = pressure_hat.clone()
            fixed_work_floor_iterations = 0
            fixed_work_floor_relative_residual = float(relative_residual)
        while iterations < iteration_limit:
            operator_direction = action(direction)
            if not bool(torch.isfinite(operator_direction).all().item()):
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_hat = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = "nonfinite_intermediate"
                else:
                    termination_reason = "nonfinite_intermediate"
                break
            if channel_pressure_relative_breakdown(
                direction,
                operator_direction,
                self._solver.pressure_inactive_mask,
            ):
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_hat = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = "operator_breakdown"
                else:
                    termination_reason = "operator_breakdown"
                break
            denominator = torch.sum(
                torch.conj(direction) * operator_direction
            ).real
            step = rz_old / denominator
            pressure_hat = self._operators._project_gauge(
                pressure_hat + step * direction
            )
            residual = self._operators._project_gauge(
                residual - step * operator_direction
            )
            iterations += 1
            if not bool(
                torch.isfinite(pressure_hat).all().item()
                and torch.isfinite(residual).all().item()
            ):
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_hat = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = "nonfinite_intermediate"
                else:
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
            if fixed_work:
                if channel_pressure_fixed_work_roundoff_saturated(
                    relative_residual,
                    residual.dtype,
                ):
                    candidate_relative_residual = float(relative_residual)
                    if (
                        fixed_work_floor_relative_residual is None
                        or candidate_relative_residual
                        < fixed_work_floor_relative_residual
                    ):
                        fixed_work_floor_pressure = pressure_hat.clone()
                        fixed_work_floor_iterations = iterations
                        fixed_work_floor_relative_residual = (
                            candidate_relative_residual
                        )
                elif fixed_work_floor_pressure is not None:
                    pressure_hat = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = (
                        "iteration_limit_reached_without_postcondition"
                    )
                    break
            if not fixed_work and (
                relative_residual <= self._solver.pressure_rel_tol
            ):
                residual = self._operators._project_gauge(
                    working_rhs - action(pressure_hat)
                )
                if not bool(torch.isfinite(residual).all().item()):
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
                if relative_residual <= self._solver.pressure_rel_tol:
                    termination_reason = "relative_tolerance_met"
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
                rz_old = torch.sum(
                    torch.conj(residual) * preconditioned
                ).real
                direction = preconditioned.clone()
                continue
            if iterations == iteration_limit:
                if fixed_work:
                    termination_reason = "fixed_work_completed"
                    acceptable = True
                break
            preconditioned = residual / self._solver.schur_diag_safe
            if not bool(torch.isfinite(preconditioned).all().item()):
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_hat = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = "nonfinite_intermediate"
                else:
                    termination_reason = "nonfinite_intermediate"
                break
            if channel_pressure_relative_breakdown(
                residual,
                preconditioned,
                self._solver.pressure_inactive_mask,
            ):
                if fixed_work and fixed_work_floor_pressure is not None:
                    pressure_hat = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    termination_reason = (
                        CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    )
                    acceptable = True
                    saturation_failure_reason = (
                        "preconditioned_residual_breakdown"
                    )
                else:
                    termination_reason = "preconditioned_residual_breakdown"
                break
            rz_new = torch.sum(torch.conj(residual) * preconditioned).real
            direction = preconditioned + (rz_new / rz_old) * direction
            rz_old = rz_new
        if (
            acceptable
            and fixed_work_floor_pressure is not None
            and termination_reason
            in {
                "fixed_work_completed",
                CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION,
            }
        ):
            completed_pressure = (
                pressure_hat.clone()
                if termination_reason == "fixed_work_completed"
                else None
            )
            try:
                residual = self._operators._project_gauge(
                    working_rhs - action(pressure_hat)
                )
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
            except ValueError:
                acceptable = False
                termination_reason = "nonfinite_intermediate"
            else:
                represented_at_floor = (
                    channel_pressure_fixed_work_roundoff_saturated(
                        relative_residual,
                        residual.dtype,
                    )
                )
                completed_residual_magnitude = (
                    residual_magnitude
                    if completed_pressure is not None
                    else None
                )
                completed_relative_residual = (
                    relative_residual
                    if completed_pressure is not None
                    else None
                )
                if (
                    not represented_at_floor
                    and completed_pressure is not None
                    and not torch.equal(
                        completed_pressure,
                        fixed_work_floor_pressure,
                    )
                ):
                    pressure_hat = fixed_work_floor_pressure
                    iterations = fixed_work_floor_iterations
                    try:
                        residual = self._operators._project_gauge(
                            working_rhs - action(pressure_hat)
                        )
                        residual_work_magnitude = (
                            channel_pressure_active_rms(
                                residual,
                                self._solver.pressure_inactive_mask,
                            )
                        )
                        residual_magnitude = residual_work_magnitude.shifted(
                            int(working_exponent)
                        )
                        relative_residual = channel_pressure_magnitude_ratio(
                            residual_work_magnitude,
                            rhs_work_magnitude,
                        )
                    except ValueError:
                        acceptable = False
                        termination_reason = "nonfinite_intermediate"
                    else:
                        represented_at_floor = (
                            channel_pressure_fixed_work_roundoff_saturated(
                                relative_residual,
                                residual.dtype,
                            )
                        )
                        if represented_at_floor:
                            termination_reason = (
                                CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                            )
                if (
                    not represented_at_floor
                    and completed_pressure is not None
                    and completed_relative_residual is not None
                    and float(completed_relative_residual)
                    <= 2.0
                    * channel_pressure_minimum_relative_tolerance(
                        residual.dtype
                    )
                ):
                    pressure_hat = completed_pressure
                    iterations = iteration_limit
                    residual_magnitude = completed_residual_magnitude
                    relative_residual = completed_relative_residual
                    termination_reason = "fixed_work_completed"
                    represented_at_floor = True
                if not represented_at_floor:
                    acceptable = False
                    termination_reason = saturation_failure_reason
                elif (
                    termination_reason
                    == CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                    and not channel_pressure_fixed_work_roundoff_saturated(
                    relative_residual,
                    residual.dtype,
                    )
                ):
                    acceptable = False
                    termination_reason = saturation_failure_reason
        diagnostics = channel_pressure_solve_diagnostics(
            self._solver,
            operator_identity=operator_identity,
            solve_phase=solve_phase,
            solve_sequence_id=solve_sequence_id,
            source_primal_sequence_id=source_primal_sequence_id,
            rhs_norm=rhs_magnitude,
            initial_absolute_residual=initial_residual,
            achieved_absolute_residual=residual_magnitude,
            achieved_normalized_residual=relative_residual,
            working_scale_binary_exponent=working_exponent,
            achieved_iteration_count=iterations,
            termination_reason=termination_reason,
        )
        if acceptable != diagnostics.acceptable:
            raise RuntimeError("pressure-loop and diagnostics postconditions disagree")
        if diagnostics.acceptable:
            pressure_hat = channel_pressure_scale_power_of_two(
                pressure_hat,
                int(working_exponent),
            )
            if not bool(torch.isfinite(pressure_hat).all().item()):
                diagnostics = channel_pressure_solve_diagnostics(
                    self._solver,
                    operator_identity=operator_identity,
                    solve_phase=solve_phase,
                    solve_sequence_id=solve_sequence_id,
                    source_primal_sequence_id=source_primal_sequence_id,
                    rhs_norm=rhs_magnitude,
                    initial_absolute_residual=initial_residual,
                    achieved_absolute_residual=residual_magnitude,
                    achieved_normalized_residual=relative_residual,
                    working_scale_binary_exponent=working_exponent,
                    achieved_iteration_count=iterations,
                    termination_reason="nonfinite_intermediate",
                )
        return pressure_hat, diagnostics

    def _solve_primal_no_grad(self, rhs_hat):
        self._solve_sequence_id += 1
        sequence = self._solve_sequence_id
        self.last_transpose_diagnostics = channel_pressure_solve_diagnostics(
            self._solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="not_run",
            solve_sequence_id=sequence,
            source_primal_sequence_id=sequence,
        )
        value, diagnostics = self._solve_no_grad(
            rhs_hat,
            self._operators.apply_pressure_operator,
            "channel_pressure_schur_primal",
            solve_phase="functional_primal",
            solve_sequence_id=sequence,
            source_primal_sequence_id=None,
        )
        self.last_primal_diagnostics = diagnostics
        self._validate_solve_diagnostics(diagnostics, "primal")
        return value

    def _solve_transpose_no_grad(
        self,
        rhs_hat,
        *,
        source_primal_sequence_id=None,
    ):
        if source_primal_sequence_id is None:
            source_primal_sequence_id = self._solve_sequence_id
        value, diagnostics = self._solve_no_grad(
            rhs_hat,
            self._operators.apply_pressure_operator_transpose,
            "channel_pressure_schur_conjugate_transpose",
            solve_phase="functional_transpose",
            solve_sequence_id=source_primal_sequence_id,
            source_primal_sequence_id=source_primal_sequence_id,
        )
        if source_primal_sequence_id == self._solve_sequence_id:
            self.last_transpose_diagnostics = diagnostics
        self._validate_solve_diagnostics(diagnostics, "transpose")
        return value

    def _persist_solve_diagnostics(
        self,
        diagnostics: ChannelPressureSolveDiagnostics,
    ) -> None:
        if diagnostics.solve_phase == "functional_primal":
            self.last_primal_diagnostics = diagnostics
        elif (
            diagnostics.solve_phase == "functional_transpose"
            and diagnostics.source_primal_sequence_id
            == self._solve_sequence_id
        ):
            self.last_transpose_diagnostics = diagnostics

    def _validate_solve_diagnostics(
        self,
        diagnostics: ChannelPressureSolveDiagnostics,
        description: str,
    ) -> None:
        if not diagnostics.acceptable:
            raise FunctionalConvergenceError(
                f"functional pressure {description} solve is unacceptable",
                operation=f"channel_pressure_{description}_solve",
                details=diagnostics.to_metadata(),
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
            "component_role": "implicit_pressure_adjoint",
            "component_is_runtime": False,
            "component_connected_to_executable_runtime": True,
            "connected_runtime_kind": "channel_activity_batch_one",
            "custom_autograd_rule": True,
            "public_explicit_vjp_method": False,
        }

    def pressure_solve_metadata(self) -> dict[str, object]:
        """Return stable outcomes for the most recent primal/transpose solves."""

        return {
            "schema_version": CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
            "solve_sequence_id": self._solve_sequence_id,
            "primal": self.last_primal_diagnostics.to_metadata(),
            "transpose": self.last_transpose_diagnostics.to_metadata(),
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
    if not bool(torch.isfinite(rhs_hat).all().item()):
        raise FunctionalConvergenceError(
            "unrolled pressure oracle input is non-finite",
            operation="channel_pressure_unrolled_oracle",
            details={"termination_reason": "nonfinite_input"},
        )
    working_rhs, working_exponent, rhs_magnitude = (
        channel_pressure_working_vector(
            rhs_hat,
            solver.pressure_inactive_mask,
        )
    )
    if rhs_magnitude.is_zero:
        return torch.zeros_like(rhs_hat)
    pressure_hat = torch.zeros_like(working_rhs)
    residual = working_rhs
    rhs_work_magnitude = channel_pressure_active_rms(
        working_rhs,
        solver.pressure_inactive_mask,
    )
    relative_residual = 1.0
    preconditioned = residual / solver.schur_diag_safe
    if channel_pressure_relative_breakdown(
        residual,
        preconditioned,
        solver.pressure_inactive_mask,
    ):
        raise FunctionalConvergenceError(
            "unrolled pressure oracle preconditioner broke down",
            operation="channel_pressure_unrolled_oracle",
            details={
                "termination_reason": "preconditioned_residual_breakdown"
            },
        )
    direction = preconditioned
    rz_old = torch.sum(torch.conj(residual) * preconditioned).real
    fixed_work_floor_pressure = None
    fixed_work_floor_relative_residual = None
    fixed_work_saturated = False
    for index in range(int(iterations)):
        operator_direction = actions.apply_pressure_operator(direction)
        if not bool(torch.isfinite(operator_direction).all().item()):
            if fixed_work_floor_pressure is not None:
                pressure_hat = fixed_work_floor_pressure
                fixed_work_saturated = True
                break
            raise FunctionalConvergenceError(
                "unrolled pressure oracle intermediate is non-finite",
                operation="channel_pressure_unrolled_oracle",
                details={"termination_reason": "nonfinite_intermediate"},
            )
        if channel_pressure_relative_breakdown(
            direction,
            operator_direction,
            solver.pressure_inactive_mask,
        ):
            if fixed_work_floor_pressure is not None:
                pressure_hat = fixed_work_floor_pressure
                fixed_work_saturated = True
                break
            raise FunctionalConvergenceError(
                "unrolled pressure oracle operator broke down",
                operation="channel_pressure_unrolled_oracle",
                details={"termination_reason": "operator_breakdown"},
            )
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
        if not bool(
            torch.isfinite(pressure_hat).all().item()
            and torch.isfinite(residual).all().item()
        ):
            if fixed_work_floor_pressure is not None:
                pressure_hat = fixed_work_floor_pressure
                fixed_work_saturated = True
                break
            raise FunctionalConvergenceError(
                "unrolled pressure oracle intermediate is non-finite",
                operation="channel_pressure_unrolled_oracle",
                details={"termination_reason": "nonfinite_intermediate"},
            )
        residual_work_magnitude = channel_pressure_active_rms(
            residual,
            solver.pressure_inactive_mask,
        )
        if residual_work_magnitude.is_zero:
            break
        relative_residual = channel_pressure_magnitude_ratio(
            residual_work_magnitude,
            rhs_work_magnitude,
        )
        if channel_pressure_fixed_work_roundoff_saturated(
            relative_residual,
            residual.dtype,
        ):
            candidate_relative_residual = float(relative_residual)
            if (
                fixed_work_floor_relative_residual is None
                or candidate_relative_residual
                < fixed_work_floor_relative_residual
            ):
                fixed_work_floor_pressure = pressure_hat.clone()
                fixed_work_floor_relative_residual = candidate_relative_residual
        elif fixed_work_floor_pressure is not None:
            pressure_hat = fixed_work_floor_pressure
            fixed_work_saturated = True
            break
        if index + 1 == int(iterations):
            continue
        preconditioned = residual / solver.schur_diag_safe
        if not bool(torch.isfinite(preconditioned).all().item()):
            if fixed_work_floor_pressure is not None:
                pressure_hat = fixed_work_floor_pressure
                fixed_work_saturated = True
                break
            raise FunctionalConvergenceError(
                "unrolled pressure oracle intermediate is non-finite",
                operation="channel_pressure_unrolled_oracle",
                details={"termination_reason": "nonfinite_intermediate"},
            )
        if channel_pressure_relative_breakdown(
            residual,
            preconditioned,
            solver.pressure_inactive_mask,
        ):
            if fixed_work_floor_pressure is not None:
                pressure_hat = fixed_work_floor_pressure
                fixed_work_saturated = True
                break
            raise FunctionalConvergenceError(
                "unrolled pressure oracle preconditioner broke down",
                operation="channel_pressure_unrolled_oracle",
                details={
                    "termination_reason": (
                        "preconditioned_residual_breakdown"
                    )
                },
            )
        rz_new = torch.sum(torch.conj(residual) * preconditioned).real
        direction = preconditioned + (rz_new / rz_old) * direction
        rz_old = rz_new
    if fixed_work_saturated or fixed_work_floor_pressure is not None:
        completed_pressure = (
            None if fixed_work_saturated else pressure_hat.clone()
        )
        try:
            residual = actions._project_gauge(
                working_rhs
                - actions.apply_pressure_operator(pressure_hat)
            )
            residual_work_magnitude = channel_pressure_active_rms(
                residual,
                solver.pressure_inactive_mask,
            )
            relative_residual = channel_pressure_magnitude_ratio(
                residual_work_magnitude,
                rhs_work_magnitude,
            )
        except ValueError as exc:
            raise FunctionalConvergenceError(
                "unrolled pressure oracle saturation state is non-finite",
                operation="channel_pressure_unrolled_oracle",
                details={"termination_reason": "nonfinite_intermediate"},
            ) from exc
        represented_at_floor = channel_pressure_fixed_work_roundoff_saturated(
            relative_residual,
            residual.dtype,
        )
        completed_relative_residual = (
            relative_residual if completed_pressure is not None else None
        )
        if (
            not represented_at_floor
            and completed_pressure is not None
            and not torch.equal(
                completed_pressure,
                fixed_work_floor_pressure,
            )
        ):
            pressure_hat = fixed_work_floor_pressure
            try:
                residual = actions._project_gauge(
                    working_rhs
                    - actions.apply_pressure_operator(pressure_hat)
                )
                residual_work_magnitude = channel_pressure_active_rms(
                    residual,
                    solver.pressure_inactive_mask,
                )
                relative_residual = channel_pressure_magnitude_ratio(
                    residual_work_magnitude,
                    rhs_work_magnitude,
                )
            except ValueError as exc:
                raise FunctionalConvergenceError(
                    "unrolled pressure oracle saturation state is non-finite",
                    operation="channel_pressure_unrolled_oracle",
                    details={"termination_reason": "nonfinite_intermediate"},
                ) from exc
            represented_at_floor = (
                channel_pressure_fixed_work_roundoff_saturated(
                    relative_residual,
                    residual.dtype,
                )
            )
        if (
            not represented_at_floor
            and completed_pressure is not None
            and completed_relative_residual is not None
            and float(completed_relative_residual)
            <= 2.0
            * channel_pressure_minimum_relative_tolerance(residual.dtype)
        ):
            pressure_hat = completed_pressure
            relative_residual = completed_relative_residual
            represented_at_floor = True
        if not represented_at_floor:
            raise FunctionalConvergenceError(
                "unrolled pressure oracle saturation postcondition failed",
                operation="channel_pressure_unrolled_oracle",
                details={
                    "termination_reason": (
                        "iteration_limit_reached_without_postcondition"
                    ),
                    "achieved_normalized_residual": relative_residual,
                },
            )
    return channel_pressure_scale_power_of_two(
        pressure_hat,
        int(working_exponent),
    )


__all__ = [
    "CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION",
    "CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES",
    "ChannelImplicitPressureAdjoint",
    "ChannelImplicitPressureAdjointProtocol",
    "ChannelPressureSolveDiagnostics",
    "unrolled_channel_pressure_solve_oracle",
]
