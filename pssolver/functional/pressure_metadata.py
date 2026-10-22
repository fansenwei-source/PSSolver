"""Stable convergence metadata for Channel pressure and transpose solves."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from pssolver.core.pressure import (
    channel_pressure_minimum_relative_tolerance_for_precision,
)

from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_BREAKDOWN_DEFINITION,
    CHANNEL_PRESSURE_NORMALIZED_RESIDUAL_DEFINITION,
    CHANNEL_PRESSURE_RHS_NORM_DEFINITION,
    CHANNEL_PRESSURE_WORKING_VECTOR_SCALING,
    ChannelNoSlipModalStokesSolver,
    ChannelPressureMagnitude,
)


CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION = 2

_SOLVE_PHASES = {
    "not_run",
    "production_primal",
    "functional_primal",
    "functional_transpose",
    "standalone_transpose_qualification",
    "unrolled_oracle_qualification",
}
_INITIAL_GUESS_POLICIES = {
    "persistent_production_warm_start",
    "zero_every_call",
}
_TERMINATION_REASONS = {
    "not_run",
    "zero_rhs",
    "exact_zero_residual",
    "fixed_work_completed",
    "fixed_work_roundoff_saturation",
    "relative_tolerance_met",
    "iteration_limit_reached_without_postcondition",
    "operator_breakdown",
    "preconditioned_residual_breakdown",
    "nonfinite_input",
    "nonfinite_intermediate",
}


def _validate_magnitude(
    value: ChannelPressureMagnitude,
    description: str,
) -> None:
    if not isinstance(value, ChannelPressureMagnitude):
        raise TypeError(f"{description} must be a ChannelPressureMagnitude")
    # Constructing a JSON-safe copy exercises the complete representation.
    value.to_metadata()


@dataclass(frozen=True, slots=True)
class ChannelPressureSolveDiagnostics:
    """Complete, JSON-safe outcome of one graph-free pressure solve."""

    solve_sequence_id: int
    source_primal_sequence_id: int | None
    solver_identity: str
    operator_identity: str
    solve_phase: str
    initial_guess_policy: str
    convergence_mode: str
    requested_relative_tolerance: float
    requested_fixed_iterations: int | None
    requested_max_iterations: int
    authoritative_iteration_limit: int
    authoritative_iteration_limit_source: str
    max_iterations_role: str
    relative_tolerance_role: str
    active_pressure_degrees_of_freedom: int
    rhs_norm_definition: str
    rhs_norm: ChannelPressureMagnitude
    initial_absolute_residual: ChannelPressureMagnitude
    achieved_absolute_residual: ChannelPressureMagnitude
    normalized_residual_definition: str
    achieved_normalized_residual: float | None
    working_vector_scaling: str
    working_scale_binary_exponent: int | None
    breakdown_definition: str
    achieved_iteration_count: int
    termination_reason: str
    required_postcondition: str
    postcondition_satisfied: bool
    acceptable: bool

    def __post_init__(self) -> None:
        for value, description in (
            (self.solver_identity, "solver_identity"),
            (self.operator_identity, "operator_identity"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{description} must be a non-empty string")
        if self.solve_phase not in _SOLVE_PHASES:
            raise ValueError("unsupported pressure solve phase")
        if self.initial_guess_policy not in _INITIAL_GUESS_POLICIES:
            raise ValueError("unsupported pressure initial-guess policy")
        if self.convergence_mode not in {"fixed_work", "relative_tolerance"}:
            raise ValueError("unsupported pressure convergence mode")
        for value, description in (
            (self.solve_sequence_id, "solve_sequence_id"),
            (self.requested_max_iterations, "requested_max_iterations"),
            (
                self.authoritative_iteration_limit,
                "authoritative_iteration_limit",
            ),
            (
                self.active_pressure_degrees_of_freedom,
                "active_pressure_degrees_of_freedom",
            ),
            (self.achieved_iteration_count, "achieved_iteration_count"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{description} must be a non-negative integer")
        if self.requested_max_iterations == 0:
            raise ValueError("requested_max_iterations must be positive")
        if self.authoritative_iteration_limit == 0:
            raise ValueError("authoritative_iteration_limit must be positive")
        if self.active_pressure_degrees_of_freedom == 0:
            raise ValueError("active pressure degree count must be positive")
        if self.achieved_iteration_count > self.authoritative_iteration_limit:
            raise ValueError("achieved iterations exceed the authoritative limit")
        if self.source_primal_sequence_id is not None and (
            not isinstance(self.source_primal_sequence_id, int)
            or isinstance(self.source_primal_sequence_id, bool)
            or self.source_primal_sequence_id < 0
        ):
            raise ValueError("source_primal_sequence_id must be non-negative")
        if self.requested_fixed_iterations is not None and (
            not isinstance(self.requested_fixed_iterations, int)
            or isinstance(self.requested_fixed_iterations, bool)
            or self.requested_fixed_iterations <= 0
        ):
            raise ValueError("requested_fixed_iterations must be positive or null")
        if (
            not isinstance(self.requested_relative_tolerance, float)
            or not math.isfinite(self.requested_relative_tolerance)
            or self.requested_relative_tolerance <= 0.0
        ):
            raise ValueError("requested_relative_tolerance must be finite and positive")
        if self.achieved_normalized_residual is not None and (
            not isinstance(self.achieved_normalized_residual, float)
            or not math.isfinite(self.achieved_normalized_residual)
            or self.achieved_normalized_residual < 0.0
        ):
            raise ValueError(
                "achieved_normalized_residual must be finite, non-negative, or null"
            )
        for value, expected, description in (
            (
                self.rhs_norm_definition,
                CHANNEL_PRESSURE_RHS_NORM_DEFINITION,
                "rhs_norm_definition",
            ),
            (
                self.normalized_residual_definition,
                CHANNEL_PRESSURE_NORMALIZED_RESIDUAL_DEFINITION,
                "normalized_residual_definition",
            ),
            (
                self.working_vector_scaling,
                CHANNEL_PRESSURE_WORKING_VECTOR_SCALING,
                "working_vector_scaling",
            ),
            (
                self.breakdown_definition,
                CHANNEL_PRESSURE_BREAKDOWN_DEFINITION,
                "breakdown_definition",
            ),
        ):
            if value != expected:
                raise ValueError(f"{description} does not match the frozen contract")
        for value, description in (
            (self.rhs_norm, "rhs_norm"),
            (self.initial_absolute_residual, "initial_absolute_residual"),
            (self.achieved_absolute_residual, "achieved_absolute_residual"),
        ):
            _validate_magnitude(value, description)
        if self.working_scale_binary_exponent is not None and (
            not isinstance(self.working_scale_binary_exponent, int)
            or isinstance(self.working_scale_binary_exponent, bool)
        ):
            raise TypeError("working scale binary exponent must be integer or null")
        if self.termination_reason not in _TERMINATION_REASONS:
            raise ValueError("unsupported pressure-solve termination reason")
        if not isinstance(self.postcondition_satisfied, bool):
            raise TypeError("postcondition_satisfied must be bool")
        if not isinstance(self.acceptable, bool):
            raise TypeError("pressure-solve acceptable flag must be bool")
        if self.acceptable != self.postcondition_satisfied:
            raise ValueError("acceptable must equal the enforced postcondition")

    @property
    def iterations(self) -> int:
        """Compatibility spelling retained through at least package 0.3."""

        return self.achieved_iteration_count

    @property
    def residual(self) -> float | None:
        """Compatibility projection of the scale-safe absolute residual."""

        return self.achieved_absolute_residual.float64_value

    @property
    def relative_residual(self) -> float | None:
        """Compatibility spelling retained through at least package 0.3."""

        return self.achieved_normalized_residual

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
            "solve_sequence_id": self.solve_sequence_id,
            "source_primal_sequence_id": self.source_primal_sequence_id,
            "solver_identity": self.solver_identity,
            "operator_identity": self.operator_identity,
            "solve_phase": self.solve_phase,
            "initial_guess_policy": self.initial_guess_policy,
            "convergence_mode": self.convergence_mode,
            "requested_relative_tolerance": self.requested_relative_tolerance,
            "requested_fixed_iterations": self.requested_fixed_iterations,
            "requested_max_iterations": self.requested_max_iterations,
            "authoritative_iteration_limit": self.authoritative_iteration_limit,
            "authoritative_iteration_limit_source": (
                self.authoritative_iteration_limit_source
            ),
            "max_iterations_role": self.max_iterations_role,
            "relative_tolerance_role": self.relative_tolerance_role,
            "active_pressure_degrees_of_freedom": (
                self.active_pressure_degrees_of_freedom
            ),
            "rhs_norm_definition": self.rhs_norm_definition,
            "rhs_norm": self.rhs_norm.to_metadata(),
            "initial_absolute_residual": (
                self.initial_absolute_residual.to_metadata()
            ),
            "achieved_absolute_residual": (
                self.achieved_absolute_residual.to_metadata()
            ),
            "normalized_residual_definition": (
                self.normalized_residual_definition
            ),
            "achieved_normalized_residual": self.achieved_normalized_residual,
            "working_vector_scaling": self.working_vector_scaling,
            "working_scale_binary_exponent": (
                self.working_scale_binary_exponent
            ),
            "breakdown_definition": self.breakdown_definition,
            "achieved_iteration_count": self.achieved_iteration_count,
            "termination_reason": self.termination_reason,
            "required_postcondition": self.required_postcondition,
            "postcondition_satisfied": self.postcondition_satisfied,
            "acceptable": self.acceptable,
        }


def _pressure_postcondition(
    solver: ChannelNoSlipModalStokesSolver,
    *,
    achieved_iteration_count: int,
    achieved_normalized_residual: float | None,
    termination_reason: str,
) -> tuple[str, bool]:
    if solver.pressure_fixed_iterations is not None:
        required = (
            "zero_rhs_or_exact_zero_residual_or_exact_requested_fixed_work_or_"
            "fixed_work_roundoff_saturation"
        )
        satisfied = termination_reason in {"zero_rhs", "exact_zero_residual"}
        if termination_reason == "fixed_work_completed":
            satisfied = achieved_iteration_count == solver.pressure_fixed_iterations
        if termination_reason == "fixed_work_roundoff_saturation":
            satisfied = bool(
                achieved_normalized_residual is not None
                and achieved_normalized_residual
                <= channel_pressure_minimum_relative_tolerance_for_precision(
                    "float32"
                    if solver.ikx.dtype == torch.complex64
                    else "float64"
                )
            )
        return required, bool(satisfied)
    required = (
        "zero_rhs_or_exact_zero_residual_or_pcg_recurrence_normalized_"
        "residual_not_above_requested_relative_tolerance"
    )
    satisfied = termination_reason in {"zero_rhs", "exact_zero_residual"}
    if termination_reason == "relative_tolerance_met":
        satisfied = bool(
            achieved_normalized_residual is not None
            and achieved_normalized_residual <= solver.pressure_rel_tol
        )
    return required, bool(satisfied)


def channel_pressure_solve_diagnostics(
    solver: ChannelNoSlipModalStokesSolver,
    *,
    operator_identity: str,
    solve_phase: str = "not_run",
    initial_guess_policy: str = "zero_every_call",
    solve_sequence_id: int = 0,
    source_primal_sequence_id: int | None = None,
    rhs_norm: ChannelPressureMagnitude | None = None,
    initial_absolute_residual: ChannelPressureMagnitude | None = None,
    achieved_absolute_residual: ChannelPressureMagnitude | None = None,
    achieved_normalized_residual: float | None = None,
    working_scale_binary_exponent: int | None = None,
    achieved_iteration_count: int = 0,
    termination_reason: str = "not_run",
) -> ChannelPressureSolveDiagnostics:
    """Build an immutable schema-v2 solve snapshot."""

    if not isinstance(solver, ChannelNoSlipModalStokesSolver):
        raise TypeError("solver must be a ChannelNoSlipModalStokesSolver")
    policy = solver.pressure_convergence_policy()
    if rhs_norm is None:
        rhs_norm = ChannelPressureMagnitude(True, 0.0, False, None, None, None)
    if initial_absolute_residual is None:
        initial_absolute_residual = rhs_norm
    if achieved_absolute_residual is None:
        achieved_absolute_residual = initial_absolute_residual
    required, satisfied = _pressure_postcondition(
        solver,
        achieved_iteration_count=achieved_iteration_count,
        achieved_normalized_residual=achieved_normalized_residual,
        termination_reason=termination_reason,
    )
    active = int((~solver.pressure_inactive_mask).sum().item())
    return ChannelPressureSolveDiagnostics(
        solve_sequence_id=int(solve_sequence_id),
        source_primal_sequence_id=source_primal_sequence_id,
        solver_identity="zero_initial_guess_preconditioned_conjugate_gradient",
        operator_identity=operator_identity,
        solve_phase=solve_phase,
        initial_guess_policy=initial_guess_policy,
        convergence_mode=str(policy["convergence_mode"]),
        requested_relative_tolerance=float(solver.pressure_rel_tol),
        requested_fixed_iterations=solver.pressure_fixed_iterations,
        requested_max_iterations=int(solver.pressure_max_iter),
        authoritative_iteration_limit=int(
            policy["authoritative_iteration_limit"]
        ),
        authoritative_iteration_limit_source=str(
            policy["authoritative_iteration_limit_source"]
        ),
        max_iterations_role=str(policy["max_iterations_role"]),
        relative_tolerance_role=str(policy["relative_tolerance_role"]),
        active_pressure_degrees_of_freedom=active,
        rhs_norm_definition=CHANNEL_PRESSURE_RHS_NORM_DEFINITION,
        rhs_norm=rhs_norm,
        initial_absolute_residual=initial_absolute_residual,
        achieved_absolute_residual=achieved_absolute_residual,
        normalized_residual_definition=(
            CHANNEL_PRESSURE_NORMALIZED_RESIDUAL_DEFINITION
        ),
        achieved_normalized_residual=achieved_normalized_residual,
        working_vector_scaling=CHANNEL_PRESSURE_WORKING_VECTOR_SCALING,
        working_scale_binary_exponent=working_scale_binary_exponent,
        breakdown_definition=CHANNEL_PRESSURE_BREAKDOWN_DEFINITION,
        achieved_iteration_count=int(achieved_iteration_count),
        termination_reason=termination_reason,
        required_postcondition=required,
        postcondition_satisfied=satisfied,
        acceptable=satisfied,
    )


__all__ = [
    "CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION",
    "ChannelPressureSolveDiagnostics",
    "channel_pressure_solve_diagnostics",
]
