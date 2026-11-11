"""Caller-owned fixtures for Channel equation and diagnostics tests."""

from pssolver.engine.constraints import _FunctionalPressureRun
from pssolver.functional.errors import FunctionalConvergenceError
from pssolver.functional.pressure_metadata import (
    CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
    channel_pressure_solve_diagnostics,
)
from qualification.s28a_pcg_equivalence import StandaloneTransposeDiagnostics


def functional_pressure(solver):
    return _FunctionalPressureRun(
        solver,
        make_diagnostics=channel_pressure_solve_diagnostics,
        error=FunctionalConvergenceError,
        schema_version=CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
    )


def empty_pressure_state(guess=None):
    return {
        "pressure_guess": guess,
        "pressure_guess_action": None,
        "pressure_guess_action_exponent": None,
    }


def production_pressure(solver, rhs, *, guess=None):
    return solver.solve(rhs, empty_pressure_state(guess), role="production")


def transpose_pressure(solver, rhs):
    run = StandaloneTransposeDiagnostics(solver)
    result = run.solve(rhs)
    return result, run.last_transpose_diagnostics
