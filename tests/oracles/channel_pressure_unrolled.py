"""Independent unrolled pressure oracle for small CPU grids."""

from __future__ import annotations

from numbers import Integral

import torch

from pssolver.functional.errors import FunctionalConvergenceError
from pssolver.functional.pressure_metadata import channel_pressure_solve_diagnostics
from pssolver.linear_solvers.stokes._channel_pcg_control import ChannelPCGControl
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


CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES = 512


def _initialize_oracle_diagnostics(solver, identities):
    """Retain the old constructors' observable schema validation in the oracle."""
    for identity in identities:
        channel_pressure_solve_diagnostics(solver, operator_identity=identity, solve_phase="not_run")


def _validate_rhs(solver: ChannelNoSlipModalStokesSolver, rhs_hat: torch.Tensor) -> None:
    if not isinstance(rhs_hat, torch.Tensor):
        raise TypeError("pressure right-hand side must be a torch.Tensor")
    expected_shape = tuple(solver.pressure_null_mask.shape)
    if tuple(rhs_hat.shape) != expected_shape:
        raise ValueError(
            "P9.7.3 pressure right-hand side shape must be "
            f"{expected_shape}, got {tuple(rhs_hat.shape)}"
        )
    if rhs_hat.dtype is not solver.ikx.dtype:
        raise TypeError(
            "pressure right-hand side dtype must match the solver "
            f"spectral dtype {solver.ikx.dtype}"
        )
    if rhs_hat.device != solver.ikx.device:
        raise ValueError(
            "pressure right-hand side device must match the solver device"
        )


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
    transpose = "channel_pressure_schur_conjugate_transpose"
    _initialize_oracle_diagnostics(solver, (transpose, "channel_pressure_schur_primal", transpose))
    _validate_rhs(solver, rhs_hat)
    if rhs_hat.device.type != "cpu":
        raise ValueError("the unrolled pressure oracle is CPU-only")
    if rhs_hat.numel() > CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES:
        raise ValueError("the unrolled pressure oracle is restricted to small grids")

    _initialize_oracle_diagnostics(solver, (transpose,))
    rhs_hat = solver._project_pressure_gauge(rhs_hat)
    if not bool(torch.isfinite(rhs_hat).all().item()):
        raise FunctionalConvergenceError(
            "unrolled pressure oracle input is non-finite",
            operation="channel_pressure_unrolled_oracle",
            details={"termination_reason": "nonfinite_input"},
        )
    working_rhs, working_exponent, rhs_magnitude = channel_pressure_working_vector(
        rhs_hat,
        solver.pressure_inactive_mask,
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
            details={"termination_reason": "preconditioned_residual_breakdown"},
        )
    direction = preconditioned
    rz_old = torch.sum(torch.conj(residual) * preconditioned).real
    pcg_control = ChannelPCGControl(fixed_work=True)
    fixed_work_saturated = False
    for index in range(int(iterations)):
        operator_direction = solver._pressure_operator(direction)
        if not bool(torch.isfinite(operator_direction).all().item()):
            stop = pcg_control.stop_for_failure(
                pressure_hat,
                iterations=index,
                failure_reason="nonfinite_intermediate",
            )
            if stop.rolled_back_to_floor:
                pressure_hat = stop.value
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
            stop = pcg_control.stop_for_failure(
                pressure_hat,
                iterations=index,
                failure_reason="operator_breakdown",
            )
            if stop.rolled_back_to_floor:
                pressure_hat = stop.value
                fixed_work_saturated = True
                break
            raise FunctionalConvergenceError(
                "unrolled pressure oracle operator broke down",
                operation="channel_pressure_unrolled_oracle",
                details={"termination_reason": "operator_breakdown"},
            )
        denominator = torch.sum(torch.conj(direction) * operator_direction).real
        step = rz_old / denominator
        pressure_hat = solver._project_pressure_gauge(pressure_hat + step * direction)
        residual = solver._project_pressure_gauge(residual - step * operator_direction)
        if not bool(
            torch.isfinite(pressure_hat).all().item()
            and torch.isfinite(residual).all().item()
        ):
            stop = pcg_control.stop_for_failure(
                pressure_hat,
                iterations=index + 1,
                failure_reason="nonfinite_intermediate",
            )
            if stop.rolled_back_to_floor:
                pressure_hat = stop.value
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
        represented_at_floor = channel_pressure_fixed_work_roundoff_saturated(
            relative_residual,
            residual.dtype,
        )
        if pcg_control.should_capture_floor(
            represented_at_floor=represented_at_floor,
            relative_residual=relative_residual,
        ):
            pcg_control.capture_floor(
                pressure_hat.clone(),
                iterations=index + 1,
                relative_residual=relative_residual,
            )
        elif pcg_control.should_rollback_after_floor_exit(
            represented_at_floor=represented_at_floor,
        ):
            stop = pcg_control.stop_after_floor_exit(
                pressure_hat,
                iterations=index + 1,
            )
            pressure_hat = stop.value
            fixed_work_saturated = True
            break
        if index + 1 == int(iterations):
            continue
        preconditioned = residual / solver.schur_diag_safe
        if not bool(torch.isfinite(preconditioned).all().item()):
            stop = pcg_control.stop_for_failure(
                pressure_hat,
                iterations=index + 1,
                failure_reason="nonfinite_intermediate",
            )
            if stop.rolled_back_to_floor:
                pressure_hat = stop.value
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
            stop = pcg_control.stop_for_failure(
                pressure_hat,
                iterations=index + 1,
                failure_reason="preconditioned_residual_breakdown",
            )
            if stop.rolled_back_to_floor:
                pressure_hat = stop.value
                fixed_work_saturated = True
                break
            raise FunctionalConvergenceError(
                "unrolled pressure oracle preconditioner broke down",
                operation="channel_pressure_unrolled_oracle",
                details={"termination_reason": ("preconditioned_residual_breakdown")},
            )
        rz_new = torch.sum(torch.conj(residual) * preconditioned).real
        direction = preconditioned + (rz_new / rz_old) * direction
        rz_old = rz_new
    if fixed_work_saturated or pcg_control.has_floor:
        completed_pressure = None if fixed_work_saturated else pressure_hat.clone()
        try:
            residual = solver._project_pressure_gauge(
                working_rhs - solver._pressure_operator(pressure_hat)
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
        if pcg_control.should_retry_floor_postcondition(
            represented_at_floor=represented_at_floor,
            completed_value_available=completed_pressure is not None,
            completed_equals_floor=(
                torch.equal(
                    completed_pressure,
                    pcg_control.floor_value,
                )
                if not represented_at_floor and completed_pressure is not None
                else True
            ),
        ):
            pressure_hat = pcg_control.floor_value
            try:
                residual = solver._project_pressure_gauge(
                    working_rhs - solver._pressure_operator(pressure_hat)
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
        if pcg_control.may_keep_completed_with_roundoff_margin(
            represented_at_floor=represented_at_floor,
            completed_value_available=completed_pressure is not None,
            completed_relative_residual_available=(
                completed_relative_residual is not None
            ),
            completed_within_margin=(
                float(completed_relative_residual)
                <= 2.0 * channel_pressure_minimum_relative_tolerance(residual.dtype)
                if not represented_at_floor
                and completed_pressure is not None
                and completed_relative_residual is not None
                else False
            ),
        ):
            pressure_hat = completed_pressure
            relative_residual = completed_relative_residual
            represented_at_floor = True
        postcondition_acceptable, _ = pcg_control.finish_fixed_work_postcondition(
            acceptable=True,
            represented_at_floor=represented_at_floor,
            termination_reason=(
                CHANNEL_PRESSURE_FIXED_WORK_ROUNDOFF_SATURATION
                if fixed_work_saturated
                else "fixed_work_completed"
            ),
            saturation_reason_still_at_floor=True,
        )
        if not postcondition_acceptable:
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
