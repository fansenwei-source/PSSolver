"""Tests for optional host-side pressure residual diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from pssolver.transforms import (
    FreeSlipModalStokesSolver,
    TensorProductTransformBackend,
)


TANGENTIAL_BCS = ("periodic", "periodic", "neumann")
NORMAL_BCS = ("periodic", "periodic", "dirichlet")
PRESSURE_BCS = ("periodic", "periodic", "neumann")


def _make_solver(*, pressure_diagnostics):
    backend = TensorProductTransformBackend(
        shape=(8, 7, 6),
        lengths=(4.0, 3.0, 2.0),
        device="cpu",
        dtype=torch.float64,
    )
    solver = FreeSlipModalStokesSolver(
        backend,
        tangential_boundary_conditions=TANGENTIAL_BCS,
        normal_boundary_conditions=NORMAL_BCS,
        pressure_boundary_conditions=PRESSURE_BCS,
        friction=0.0,
        viscosity=2.0 / 3.0,
        zero_mode_policy="zero_mean",
        pressure_diagnostics=pressure_diagnostics,
    )
    return backend, solver


def _force_hats(backend):
    torch.manual_seed(20260908)
    force = torch.randn(3, 1, *backend.shape, dtype=torch.float64)
    tangential_hat = backend.forward(force[:2], TANGENTIAL_BCS)
    normal_hat = backend.forward(force[2], NORMAL_BCS)
    return tangential_hat[0], tangential_hat[1], normal_hat


def test_disabling_pressure_diagnostics_preserves_stokes_solution():
    tracked_backend, tracked = _make_solver(pressure_diagnostics=True)
    fast_backend, fast = _make_solver(pressure_diagnostics=False)

    tracked_hats = _force_hats(tracked_backend)
    fast_hats = _force_hats(fast_backend)
    tracked_solution = tracked.solve_force_hats(*tracked_hats)
    fast_solution = fast.solve_force_hats(*fast_hats)

    for actual, expected in zip(fast_solution, tracked_solution, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    assert tracked.last_pressure_iterations == 1
    assert math.isfinite(tracked.last_pressure_residual)
    assert math.isfinite(tracked.last_pressure_relative_residual)
    assert fast.last_pressure_iterations == 1
    assert math.isnan(fast.last_pressure_residual)
    assert math.isnan(fast.last_pressure_relative_residual)


def test_disabled_pressure_diagnostics_do_not_evaluate_vector_norm(monkeypatch):
    backend, solver = _make_solver(pressure_diagnostics=False)
    force_hats = _force_hats(backend)

    def unexpected_vector_norm(*args, **kwargs):
        raise AssertionError("production pressure solve must not evaluate vector_norm")

    monkeypatch.setattr(torch.linalg, "vector_norm", unexpected_vector_norm)
    solver.solve_force_hats(*force_hats)


def test_enabled_pressure_diagnostics_keep_zero_rhs_shortcut():
    backend, solver = _make_solver(pressure_diagnostics=True)
    tangential_shape = (1, *backend.shape)
    zero_tangential = torch.zeros(
        tangential_shape,
        dtype=backend.spectral_dtype,
    )
    zero_normal = torch.zeros_like(zero_tangential)

    solution = solver.solve_force_hats(
        zero_tangential,
        zero_tangential,
        zero_normal,
    )

    assert all(torch.count_nonzero(field).item() == 0 for field in solution)
    assert solver.last_pressure_iterations == 0
    assert solver.last_pressure_residual == 0.0
    assert solver.last_pressure_relative_residual == 0.0


@pytest.mark.parametrize("value", (None, 0, 1, "false"))
def test_pressure_diagnostics_requires_bool(value):
    backend = TensorProductTransformBackend(
        shape=(8, 7, 6),
        lengths=(4.0, 3.0, 2.0),
        device="cpu",
        dtype=torch.float64,
    )
    with pytest.raises(TypeError, match="pressure_diagnostics must be a bool"):
        FreeSlipModalStokesSolver(
            backend,
            tangential_boundary_conditions=TANGENTIAL_BCS,
            normal_boundary_conditions=NORMAL_BCS,
            pressure_boundary_conditions=PRESSURE_BCS,
            zero_mode_policy="zero_mean",
            pressure_diagnostics=value,
        )
