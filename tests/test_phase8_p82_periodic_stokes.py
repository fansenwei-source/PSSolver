"""P8.2 manufactured tests for the fully periodic Stokes lowering."""

from __future__ import annotations

import math

import pytest
import torch

from pssolver.linear_solvers.stokes import PeriodicModalStokesSolver
from pssolver.transforms import TensorProductTransformBackend


PERIODIC = ("periodic", "periodic", "periodic")


def _backend(*, storage="full_complex"):
    return TensorProductTransformBackend(
        (12, 10, 8),
        (2.0 * math.pi, 2.5 * math.pi, 3.0 * math.pi),
        device="cpu",
        dtype=torch.float64,
        execution_order="real_first",
        spectral_storage=storage,
        hermitian_axis=1,
    )


@pytest.mark.parametrize("storage", ("full_complex", "hermitian_half"))
def test_periodic_stokes_manufactured_solution_and_pressure_gauge(storage):
    backend = _backend(storage=storage)
    viscosity = 0.71
    friction = 0.23
    stokes = PeriodicModalStokesSolver(
        backend,
        friction=friction,
        viscosity=viscosity,
        zero_mode_policy="friction",
    )
    x, y, z = backend.spatial_grids
    kx = 2.0
    ky = 2.0 * math.pi / backend.lengths[1]
    kz = 2.0 * math.pi / backend.lengths[2]
    phase = kx * x + ky * y + kz * z
    ux = ky * torch.cos(phase)
    uy = -kx * torch.cos(phase)
    uz = torch.zeros_like(ux)
    pressure = 0.37 * torch.sin(phase)
    helmholtz = friction + viscosity * (kx * kx + ky * ky + kz * kz)
    fx = helmholtz * ux + 0.37 * kx * torch.cos(phase)
    fy = helmholtz * uy + 0.37 * ky * torch.cos(phase)
    fz = helmholtz * uz + 0.37 * kz * torch.cos(phase)

    force_hat = backend.forward(torch.stack((fx, fy, fz)), PERIODIC)
    ux_hat, uy_hat, uz_hat, p_hat = stokes.solve_force_hats(*force_hat)
    observed_u = backend.inverse(
        torch.stack((ux_hat, uy_hat, uz_hat)), PERIODIC
    )
    observed_p = backend.inverse(p_hat, PERIODIC)

    torch.testing.assert_close(
        observed_u,
        torch.stack((ux, uy, uz)),
        rtol=4.0e-12,
        atol=4.0e-12,
    )
    torch.testing.assert_close(
        observed_p,
        pressure,
        rtol=4.0e-12,
        atol=4.0e-12,
    )
    divergence = stokes.divergence_hat(ux_hat, uy_hat, uz_hat)
    torch.testing.assert_close(
        divergence,
        torch.zeros_like(divergence),
        rtol=0,
        atol=2.0e-12,
    )
    assert abs(float(observed_p.mean())) < 2.0e-14


def test_periodic_zero_mean_policy_removes_all_uniform_velocity_components():
    backend = _backend()
    stokes = PeriodicModalStokesSolver(
        backend,
        viscosity=0.71,
        friction=0.0,
        zero_mode_policy="zero_mean",
    )
    force = torch.stack(
        tuple(
            backend.forward(
                torch.full(backend.shape, value, dtype=torch.float64),
                PERIODIC,
            )
            for value in (1.0, -2.0, 3.0)
        )
    )
    values = stokes.solve_force_hats(*force)

    for observed in values:
        torch.testing.assert_close(
            observed,
            torch.zeros_like(observed),
            rtol=0,
            atol=0,
        )


def test_periodic_friction_policy_retains_uniform_velocity_but_not_pressure():
    backend = _backend()
    friction = 0.4
    stokes = PeriodicModalStokesSolver(
        backend,
        viscosity=0.71,
        friction=friction,
        zero_mode_policy="friction",
    )
    physical_force = torch.stack(
        tuple(
            torch.full(backend.shape, value, dtype=torch.float64)
            for value in (1.0, -2.0, 3.0)
        )
    )
    force = backend.forward(physical_force, PERIODIC)
    *velocity_hat, pressure_hat = stokes.solve_force_hats(*force)
    velocity = backend.inverse(torch.stack(velocity_hat), PERIODIC)
    pressure = backend.inverse(pressure_hat, PERIODIC)

    torch.testing.assert_close(
        velocity,
        physical_force / friction,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        pressure,
        torch.zeros_like(pressure),
        rtol=0,
        atol=0,
    )


def test_periodic_stokes_rejects_implicit_or_inconsistent_nullspace_choices():
    backend = _backend()
    with pytest.raises(ValueError, match="zero_mean mode requires"):
        PeriodicModalStokesSolver(
            backend,
            friction=0.1,
            zero_mode_policy="zero_mean",
        )
    with pytest.raises(ValueError, match="friction mode requires"):
        PeriodicModalStokesSolver(
            backend,
            friction=0.0,
            zero_mode_policy="friction",
        )
    with pytest.raises(ValueError, match="periodic boundaries"):
        PeriodicModalStokesSolver(
            backend,
            boundary_conditions=("periodic", "periodic", "neumann"),
        )
