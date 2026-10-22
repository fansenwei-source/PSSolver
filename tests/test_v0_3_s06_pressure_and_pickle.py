"""S-06 pressure precision, fixed-work saturation, and pickle regressions."""

from __future__ import annotations

import pickle

import pytest
import torch

from pssolver import ChannelPressureConvergenceError, SpectralSolver
from pssolver.functional import (
    ChannelImplicitPressureAdjoint,
    ChannelPressureTransposeOperator,
    FunctionalConvergenceError,
    unrolled_channel_pressure_solve_oracle,
)
from pssolver.linear_solvers.stokes import (
    ChannelNoSlipModalStokesSolver,
    channel_pressure_minimum_relative_tolerance,
)


def _solver(shape, *, fixed_iterations):
    spectral = SpectralSolver(
        shape,
        L=tuple(1.0 + 0.37 * value for value in shape),
        device="cpu",
        dtype=torch.float32,
    )
    solver = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=1.0e-6,
        pressure_max_iterations=fixed_iterations,
        pressure_fixed_iterations=fixed_iterations,
    )
    return solver


def _rhs(solver, shape, seed):
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn((1, *shape), generator=generator, dtype=torch.float64)
    imag = torch.randn((1, *shape), generator=generator, dtype=torch.float64)
    operator = ChannelPressureTransposeOperator(solver)
    return operator._project_gauge(
        torch.complex(real, imag).to(torch.complex64)
    )


@pytest.mark.parametrize(
    ("shape", "seed", "fixed_iterations"),
    (((8, 6, 4), 9, 50), ((16, 12, 10), 10, 50)),
)
def test_float32_fixed_work_saturates_before_roundoff_breakdown(
    shape,
    seed,
    fixed_iterations,
):
    solver = _solver(shape, fixed_iterations=fixed_iterations)
    rhs = _rhs(solver, shape, seed)

    pressure = solver._solve_pressure(rhs)

    assert bool(torch.isfinite(pressure).all().item())
    assert solver.last_pressure_termination_reason == (
        "fixed_work_roundoff_saturation"
    )
    assert solver.last_pressure_acceptable is True
    assert solver.last_pressure_iterations < fixed_iterations
    assert solver.last_pressure_relative_residual <= (
        channel_pressure_minimum_relative_tolerance(torch.float32)
    )


def test_fixed_work_roundoff_saturation_is_shared_by_functional_solves():
    shape = (8, 6, 4)
    solver = _solver(shape, fixed_iterations=50)
    rhs = _rhs(solver, shape, 9)

    standalone = ChannelPressureTransposeOperator(solver)
    transpose = standalone.solve_pressure_transpose(rhs)
    implicit = ChannelImplicitPressureAdjoint(solver)
    primal = implicit.solve(rhs)
    oracle = unrolled_channel_pressure_solve_oracle(
        rhs,
        solver,
        iterations=50,
    )

    assert bool(torch.isfinite(transpose).all().item())
    assert bool(torch.isfinite(primal).all().item())
    assert torch.equal(primal, oracle)
    assert standalone.last_transpose_diagnostics.termination_reason == (
        "fixed_work_roundoff_saturation"
    )
    assert implicit.last_primal_diagnostics.termination_reason == (
        "fixed_work_roundoff_saturation"
    )
    assert standalone.last_transpose_diagnostics.acceptable is True
    assert implicit.last_primal_diagnostics.acceptable is True


def test_fixed_work_breakdown_above_precision_floor_still_fails_closed(
    monkeypatch,
):
    shape = (8, 6, 4)
    solver = _solver(shape, fixed_iterations=50)
    rhs = _rhs(solver, shape, 9)
    monkeypatch.setattr(
        solver,
        "_pressure_operator",
        lambda value: torch.zeros_like(value),
    )

    with pytest.raises(ChannelPressureConvergenceError) as captured:
        solver._solve_pressure(rhs)

    assert captured.value.details["termination_reason"] == "operator_breakdown"
    assert captured.value.details["achieved_normalized_residual"] == 1.0
    assert captured.value.details["acceptable"] is False


@pytest.mark.parametrize("protocol", (4, pickle.HIGHEST_PROTOCOL))
@pytest.mark.parametrize(
    "error",
    (
        ChannelPressureConvergenceError(
            "pressure failed",
            details={"acceptable": False, "iteration": 7},
        ),
        FunctionalConvergenceError(
            "functional pressure failed",
            operation="step",
            details={"acceptable": False, "iteration": 8},
        ),
    ),
)
def test_channel_pressure_errors_round_trip_through_pickle(error, protocol):
    restored = pickle.loads(pickle.dumps(error, protocol=protocol))

    assert type(restored) is type(error)
    assert str(restored) == str(error)
    assert restored.args == error.args
    assert restored.to_metadata() == error.to_metadata()
    assert restored.details == error.details
    assert restored.diagnostics == error.diagnostics
    if isinstance(error, FunctionalConvergenceError):
        assert restored.operation == error.operation
