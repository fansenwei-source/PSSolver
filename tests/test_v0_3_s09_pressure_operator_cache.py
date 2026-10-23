"""S-09 warm-start operator-cache correctness and invalidation gates."""

from __future__ import annotations

import pytest
import torch

from pssolver import SpectralSolver
from pssolver.functional import ChannelPressureTransposeOperator
from pssolver.linear_solvers.stokes import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
)
from pssolver.linear_solvers.stokes.channel_no_slip import (
    channel_pressure_scale_power_of_two,
)


SHAPE = (12, 8, 6)


def _solver() -> tuple[SpectralSolver, ChannelNoSlipModalStokesSolver]:
    spectral = SpectralSolver(
        SHAPE,
        L=(12.0, 8.0, 6.0),
        device="cpu",
        dtype=torch.float64,
    )
    pressure = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=1.0e-5,
        pressure_max_iterations=100,
        pressure_fixed_iterations=None,
    )
    return spectral, pressure


def _rhs(
    spectral: SpectralSolver,
    pressure: ChannelNoSlipModalStokesSolver,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    physical = torch.randn(
        (1, *SHAPE),
        generator=generator,
        dtype=torch.float64,
    )
    exact = spectral.transform_tensor(
        physical,
        CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    )
    exact = pressure._project_pressure_gauge(exact)
    return ChannelPressureTransposeOperator(pressure).apply_pressure_operator(
        exact
    )


@pytest.mark.parametrize("next_exponent", (-100, 0, 100))
def test_cached_warm_operator_is_byte_identical_and_saves_one_action(
    monkeypatch,
    next_exponent,
):
    cached_spectral, cached = _solver()
    direct_spectral, direct = _solver()
    first_cached = _rhs(cached_spectral, cached, seed=9101)
    first_direct = _rhs(direct_spectral, direct, seed=9101)
    assert torch.equal(first_cached, first_direct)
    cached._solve_pressure(first_cached)
    direct._solve_pressure(first_direct)

    # Replacing the tensor with equal bytes is an external warm-start update
    # and must invalidate the direct solver's cache.
    direct.pressure_guess = direct.pressure_guess.detach().clone()
    counts = {id(cached): 0, id(direct): 0}
    original = ChannelNoSlipModalStokesSolver._pressure_operator

    def counted(instance, value):
        counts[id(instance)] += 1
        return original(instance, value)

    monkeypatch.setattr(
        ChannelNoSlipModalStokesSolver,
        "_pressure_operator",
        counted,
    )
    next_cached = channel_pressure_scale_power_of_two(
        _rhs(cached_spectral, cached, seed=9102),
        next_exponent,
    )
    next_direct = channel_pressure_scale_power_of_two(
        _rhs(direct_spectral, direct, seed=9102),
        next_exponent,
    )
    counts[id(cached)] = 0
    counts[id(direct)] = 0
    cached_result = cached._solve_pressure(next_cached)
    direct_result = direct._solve_pressure(next_direct)

    assert torch.equal(cached_result, direct_result)
    assert cached.last_pressure_diagnostics == direct.last_pressure_diagnostics
    assert counts[id(cached)] + 1 == counts[id(direct)]


def test_in_place_warm_start_update_invalidates_operator_cache(monkeypatch):
    spectral, pressure = _solver()
    pressure._solve_pressure(_rhs(spectral, pressure, seed=9201))
    cached_action = pressure._pressure_guess_operator_action
    assert isinstance(cached_action, torch.Tensor)

    pressure.pressure_guess.add_(0.0)
    original = ChannelNoSlipModalStokesSolver._pressure_operator
    calls = 0

    def counted(instance, value):
        nonlocal calls
        calls += 1
        return original(instance, value)

    monkeypatch.setattr(
        ChannelNoSlipModalStokesSolver,
        "_pressure_operator",
        counted,
    )
    pressure._solve_pressure(_rhs(spectral, pressure, seed=9202))

    # One initial action plus one acceptance action proves that the stale
    # cached initial action was not reused.
    assert calls >= 2
    assert pressure.last_pressure_acceptable is True


def test_external_none_warm_start_clears_operator_cache():
    spectral, pressure = _solver()
    pressure._solve_pressure(_rhs(spectral, pressure, seed=9301))
    assert isinstance(pressure._pressure_guess_operator_action, torch.Tensor)

    pressure.pressure_guess = None
    pressure._solve_pressure(_rhs(spectral, pressure, seed=9302))

    assert pressure.last_pressure_acceptable is True


def test_cached_and_direct_warm_start_sequences_are_byte_identical():
    cached_spectral, cached = _solver()
    direct_spectral, direct = _solver()

    for step in range(20):
        exponent = (-20, 0, 20)[step % 3]
        cached_rhs = channel_pressure_scale_power_of_two(
            _rhs(cached_spectral, cached, seed=9400 + step),
            exponent,
        )
        direct_rhs = channel_pressure_scale_power_of_two(
            _rhs(direct_spectral, direct, seed=9400 + step),
            exponent,
        )
        if direct.pressure_guess is not None:
            direct.pressure_guess = direct.pressure_guess.detach().clone()

        cached_result = cached._solve_pressure(cached_rhs)
        direct_result = direct._solve_pressure(direct_rhs)

        assert torch.equal(cached_result, direct_result)
        assert torch.equal(cached.pressure_guess, direct.pressure_guess)
        assert (
            cached.last_pressure_diagnostics
            == direct.last_pressure_diagnostics
        )
