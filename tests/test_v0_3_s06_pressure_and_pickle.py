"""S-06 pressure precision, fixed-work saturation, and pickle regressions."""

from __future__ import annotations

import pickle

import pytest
import torch

from pssolver import ChannelPressureConvergenceError, SpectralSolver
from pssolver.functional import (
    FunctionalConvergenceError,
)
from pssolver.linear_solvers.stokes import (
    ChannelNoSlipModalStokesSolver,
    channel_pressure_minimum_relative_tolerance,
)
from pssolver.linear_solvers.stokes.channel_no_slip import (
    channel_pressure_active_rms,
)
from tests._s28a_pressure import functional_pressure, production_pressure, transpose_pressure
from tests.oracles import channel_pressure_unrolled as pressure_adjoint_module
from tests.oracles.channel_pressure_unrolled import unrolled_channel_pressure_solve_oracle


def _solver(
    shape,
    *,
    fixed_iterations,
    friction=0.2,
    viscosity=0.73,
    length_scale=1.0,
    dtype=torch.float32,
):
    spectral = SpectralSolver(
        shape,
        L=tuple(
            length_scale * (1.0 + 0.37 * value) for value in shape
        ),
        device="cpu",
        dtype=dtype,
    )
    solver = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=friction,
        viscosity=viscosity,
        pressure_relative_tolerance=1.0e-6,
        pressure_max_iterations=fixed_iterations,
        pressure_fixed_iterations=fixed_iterations,
    )
    return solver


def _rhs(solver, shape, seed, *, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn((1, *shape), generator=generator, dtype=torch.float64)
    imag = torch.randn((1, *shape), generator=generator, dtype=torch.float64)
    complex_dtype = (
        torch.complex128 if dtype == torch.float64 else torch.complex64
    )
    return solver._project_pressure_gauge(torch.complex(real, imag).to(complex_dtype))


def _represented_relative_residual(solver, rhs, pressure, action):
    residual = solver._project_pressure_gauge(rhs - action(pressure))
    residual_rms = channel_pressure_active_rms(
        residual,
        solver.pressure_inactive_mask,
    )
    rhs_rms = channel_pressure_active_rms(
        rhs,
        solver.pressure_inactive_mask,
    )
    return residual_rms.float64_value / rhs_rms.float64_value


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

    pressure, _, report = production_pressure(solver, rhs)

    assert bool(torch.isfinite(pressure).all().item())
    assert report["termination_reason"] == (
        "fixed_work_roundoff_saturation"
    )
    assert report["acceptable"] is True
    assert report["achieved_iteration_count"] < fixed_iterations
    assert report["achieved_normalized_residual"] <= (
        channel_pressure_minimum_relative_tolerance(torch.float32)
    )


def test_fixed_work_roundoff_saturation_is_shared_by_functional_solves():
    shape = (8, 6, 4)
    solver = _solver(shape, fixed_iterations=50)
    rhs = _rhs(solver, shape, 9)

    transpose, transpose_report = transpose_pressure(solver, rhs)
    implicit = functional_pressure(solver)
    primal = implicit.solve(rhs)
    oracle = unrolled_channel_pressure_solve_oracle(
        rhs,
        solver,
        iterations=50,
    )

    assert bool(torch.isfinite(transpose).all().item())
    assert bool(torch.isfinite(primal).all().item())
    assert torch.equal(primal, oracle)
    assert transpose_report.termination_reason == (
        "fixed_work_roundoff_saturation"
    )
    assert implicit.last_primal_diagnostics.termination_reason == (
        "fixed_work_roundoff_saturation"
    )
    assert transpose_report.acceptable is True
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
        production_pressure(solver, rhs)

    assert captured.value.details["termination_reason"] == "operator_breakdown"
    assert captured.value.details["achieved_normalized_residual"] == 1.0
    assert captured.value.details["acceptable"] is False


@pytest.mark.parametrize(
    ("shape", "friction", "viscosity", "seed"),
    (
        ((8, 6, 4), 0.2, 20.0, 0),
        ((12, 8, 32), 1.0e-3, 0.73, 2),
    ),
)
@pytest.mark.parametrize("fixed_iterations", (200, 250, 300, 400))
@pytest.mark.parametrize(
    "path",
    ("production", "functional_transpose", "implicit", "oracle"),
)
def test_excess_float32_fixed_work_never_accepts_a_diverged_return_state(
    monkeypatch,
    shape,
    friction,
    viscosity,
    seed,
    fixed_iterations,
    path,
):
    solver = _solver(
        shape,
        fixed_iterations=fixed_iterations,
        friction=friction,
        viscosity=viscosity,
        length_scale=8.0,
    )
    rhs = _rhs(solver, shape, seed)

    try:
        if path == "production":
            pressure, _, report = production_pressure(solver, rhs.clone())
            action = solver._pressure_operator
            acceptable = report["acceptable"]
        elif path == "functional_transpose":
            pressure, transpose_report = transpose_pressure(solver, rhs.clone())
            action = solver.apply_pressure_operator_transpose
            acceptable = transpose_report.acceptable
        elif path == "implicit":
            implicit = functional_pressure(solver)
            pressure = implicit.solve(rhs.clone()).detach()
            action = solver._pressure_operator
            acceptable = implicit.last_primal_diagnostics.acceptable
        else:
            # The oracle remains publicly restricted to 512 modes.  Widen
            # only this regression's qualification guard so the exact
            # 12x8x32 R-0014 counterexample exercises the unchanged oracle
            # algorithm.
            monkeypatch.setattr(
                pressure_adjoint_module,
                "CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES",
                max(512, rhs.numel()),
            )
            pressure = unrolled_channel_pressure_solve_oracle(
                rhs.clone(),
                solver,
                iterations=fixed_iterations,
            ).detach()
            action = solver._pressure_operator
            acceptable = True
    except (ChannelPressureConvergenceError, FunctionalConvergenceError) as error:
        details = error.details
        assert details.get("acceptable", False) is False
        assert details["termination_reason"] in {
            "iteration_limit_reached_without_postcondition",
            "nonfinite_intermediate",
            "operator_breakdown",
            "preconditioned_residual_breakdown",
        }
        return

    assert bool(torch.isfinite(pressure).all().item())
    represented = _represented_relative_residual(
        solver,
        rhs,
        pressure,
        action,
    )
    floor = channel_pressure_minimum_relative_tolerance(torch.float32)
    assert not acceptable or represented <= floor


def test_fixed_work_completion_keeps_the_v29_roundoff_margin():
    shape = (12, 7, 6)
    solver = _solver(
        shape,
        fixed_iterations=25,
        friction=0.2,
        viscosity=0.73,
        dtype=torch.float64,
    )
    rhs = _rhs(solver, shape, 9, dtype=torch.float64)

    pressure, _, report = production_pressure(solver, rhs)

    represented = _represented_relative_residual(
        solver,
        rhs,
        pressure,
        solver._pressure_operator,
    )
    floor = channel_pressure_minimum_relative_tolerance(torch.float64)
    assert report["acceptable"] is True
    assert report["termination_reason"] == "fixed_work_completed"
    assert represented <= 2.0 * floor


def test_transpose_fixed_work_uses_the_minimum_floor_state():
    shape = (12, 8, 32)
    solver = _solver(
        shape,
        fixed_iterations=200,
        friction=1.0e-3,
        viscosity=0.73,
        length_scale=8.0,
    )
    rhs = _rhs(solver, shape, 0)

    pressure, transpose_report = transpose_pressure(solver, rhs)

    represented = _represented_relative_residual(
        solver,
        rhs,
        pressure,
        solver.apply_pressure_operator_transpose,
    )
    floor = channel_pressure_minimum_relative_tolerance(torch.float32)
    assert transpose_report.acceptable is True
    assert represented <= floor


def test_production_fixed_work_uses_the_minimum_floor_state():
    shape = (24, 20, 16)
    solver = _solver(
        shape,
        fixed_iterations=400,
        friction=1.0e-3,
        viscosity=20.0,
        length_scale=1.0,
    )
    rhs = _rhs(solver, shape, 1)

    pressure, _, report = production_pressure(solver, rhs)

    represented = _represented_relative_residual(
        solver,
        rhs,
        pressure,
        solver._pressure_operator,
    )
    floor = channel_pressure_minimum_relative_tolerance(torch.float32)
    assert report["acceptable"] is True
    assert represented <= floor


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
