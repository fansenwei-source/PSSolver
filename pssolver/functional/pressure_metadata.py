"""Stable convergence metadata for Channel pressure and transpose solves."""

from __future__ import annotations

from dataclasses import dataclass
import math

from pssolver.linear_solvers.stokes.channel_no_slip import (
    ChannelNoSlipModalStokesSolver,
)


CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION = 1

_TERMINATION_REASONS = {
    "not_run",
    "zero_rhs",
    "relative_tolerance_met",
    "iteration_limit_reached",
    "operator_breakdown",
    "preconditioned_residual_breakdown",
}


@dataclass(frozen=True, slots=True)
class ChannelPressureSolveDiagnostics:
    """Complete, JSON-safe outcome of one graph-free pressure solve."""

    solver_identity: str
    operator_identity: str
    convergence_mode: str
    requested_iteration_limit: int
    achieved_iteration_count: int
    achieved_absolute_residual: float
    achieved_relative_residual: float
    requested_relative_tolerance: float
    tolerance_semantics: str
    termination_reason: str
    acceptable: bool

    def __post_init__(self) -> None:
        for value, description in (
            (self.solver_identity, "solver_identity"),
            (self.operator_identity, "operator_identity"),
            (self.convergence_mode, "convergence_mode"),
            (self.tolerance_semantics, "tolerance_semantics"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{description} must be a non-empty string")
        for value, description in (
            (self.requested_iteration_limit, "requested_iteration_limit"),
            (self.achieved_iteration_count, "achieved_iteration_count"),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{description} must be non-negative")
        if self.achieved_iteration_count > self.requested_iteration_limit:
            raise ValueError("achieved iterations exceed the requested limit")
        for value, description in (
            (self.achieved_absolute_residual, "achieved_absolute_residual"),
            (self.achieved_relative_residual, "achieved_relative_residual"),
            (self.requested_relative_tolerance, "requested_relative_tolerance"),
        ):
            if not isinstance(value, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{description} must be a finite non-negative float")
        if self.termination_reason not in _TERMINATION_REASONS:
            raise ValueError("unsupported pressure-solve termination reason")
        if not isinstance(self.acceptable, bool):
            raise TypeError("pressure-solve acceptable flag must be bool")

    @property
    def iterations(self) -> int:
        """Compatibility spelling retained through at least package 0.3."""

        return self.achieved_iteration_count

    @property
    def residual(self) -> float:
        """Compatibility spelling retained through at least package 0.3."""

        return self.achieved_absolute_residual

    @property
    def relative_residual(self) -> float:
        """Compatibility spelling retained through at least package 0.3."""

        return self.achieved_relative_residual

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
            "solver_identity": self.solver_identity,
            "operator_identity": self.operator_identity,
            "convergence_mode": self.convergence_mode,
            "requested_iteration_limit": self.requested_iteration_limit,
            "achieved_iteration_count": self.achieved_iteration_count,
            "achieved_absolute_residual": self.achieved_absolute_residual,
            "achieved_relative_residual": self.achieved_relative_residual,
            "requested_relative_tolerance": self.requested_relative_tolerance,
            "tolerance_semantics": self.tolerance_semantics,
            "termination_reason": self.termination_reason,
            "acceptable": self.acceptable,
        }


def channel_pressure_solve_diagnostics(
    solver: ChannelNoSlipModalStokesSolver,
    *,
    operator_identity: str,
    achieved_iteration_count: int,
    achieved_absolute_residual: float,
    achieved_relative_residual: float,
    termination_reason: str,
) -> ChannelPressureSolveDiagnostics:
    """Build diagnostics using the unchanged production PCG configuration."""

    if not isinstance(solver, ChannelNoSlipModalStokesSolver):
        raise TypeError("solver must be a ChannelNoSlipModalStokesSolver")
    fixed = solver.pressure_fixed_iterations
    if fixed is None:
        limit = int(solver.pressure_max_iter)
        mode = "relative_tolerance_required_with_maximum_iteration_cap"
        semantics = "required_postcondition"
        acceptable = termination_reason in {
            "zero_rhs",
            "relative_tolerance_met",
        }
    else:
        limit = int(fixed)
        mode = "deterministic_iteration_cap_with_early_convergence"
        semantics = "early_convergence_criterion_not_required_at_cap"
        acceptable = termination_reason in {
            "zero_rhs",
            "relative_tolerance_met",
            "iteration_limit_reached",
        } or achieved_iteration_count == limit
    if termination_reason == "not_run":
        acceptable = False
    finite = math.isfinite(float(achieved_absolute_residual)) and math.isfinite(
        float(achieved_relative_residual)
    )
    acceptable = bool(
        acceptable
        and finite
        and achieved_iteration_count <= limit
    )
    return ChannelPressureSolveDiagnostics(
        solver_identity="zero_initial_guess_preconditioned_conjugate_gradient",
        operator_identity=operator_identity,
        convergence_mode=mode,
        requested_iteration_limit=limit,
        achieved_iteration_count=int(achieved_iteration_count),
        achieved_absolute_residual=float(achieved_absolute_residual),
        achieved_relative_residual=float(achieved_relative_residual),
        requested_relative_tolerance=float(solver.pressure_rel_tol),
        tolerance_semantics=semantics,
        termination_reason=termination_reason,
        acceptable=acceptable,
    )


__all__ = [
    "CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION",
    "ChannelPressureSolveDiagnostics",
    "channel_pressure_solve_diagnostics",
]
