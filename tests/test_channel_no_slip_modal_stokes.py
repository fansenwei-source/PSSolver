"""Equation-level contracts for the canonical no-slip Channel Stokes solve."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pssolver import SpectralSolver
from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
)


SHAPE = (7, 6, 5)


LENGTHS = (4.0, 3.0, 2.5)


FRICTION = 0.2


VISCOSITY = 0.73


ROOT = Path(__file__).resolve().parents[1]


NOTES = ROOT / "notes" / "architecture_v0_2"


def _spectral_solver(shape=SHAPE):
    lengths = LENGTHS if shape == SHAPE else tuple(float(n) for n in shape)
    return SpectralSolver(
        shape,
        L=lengths,
        device="cpu",
        dtype=torch.float64,
    )


def _canonical(solver, **overrides):
    options = {
        "friction": FRICTION,
        "viscosity": VISCOSITY,
        "pressure_relative_tolerance": 1.0e-12,
        "pressure_max_iterations": 300,
    }
    options.update(overrides)
    return ChannelNoSlipModalStokesSolver(
        solver.transform_backend,
        **options,
    )


def _force_hats(solver, *, batch_size=2):
    generator = torch.Generator().manual_seed(472)
    forces = tuple(
        torch.randn(
            (batch_size, *SHAPE),
            generator=generator,
            dtype=torch.float64,
        )
        for _ in range(3)
    )
    return tuple(
        solver.transform_tensor(
            force,
            CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
        )
        for force in forces
    )


def _relative_norm(value, reference):
    return float(
        torch.linalg.vector_norm(value.reshape(-1))
        / torch.linalg.vector_norm(reference.reshape(-1))
    )


def test_channel_boundary_spaces_and_registered_operator_state_are_frozen():
    solver = _spectral_solver()
    stokes = _canonical(solver)
    velocity_metadata = solver.transform_backend.get_metadata(CHANNEL_VELOCITY_BOUNDARY_CONDITIONS)
    pressure_metadata = solver.transform_backend.get_metadata(CHANNEL_PRESSURE_BOUNDARY_CONDITIONS)

    assert CHANNEL_VELOCITY_BOUNDARY_CONDITIONS == (
        "periodic",
        "dirichlet",
        "dirichlet",
    )
    assert CHANNEL_PRESSURE_BOUNDARY_CONDITIONS == (
        "periodic",
        "neumann",
        "neumann",
    )
    assert velocity_metadata.transform_kinds == ("fft", "dst", "dst")
    assert pressure_metadata.transform_kinds == ("fft", "dct", "dct")
    assert tuple(stokes.state_dict()) == (
        "ikx",
        "a_inv",
        "n_to_d_y",
        "d_to_n_y",
        "dn_to_dd_y",
        "dd_to_dn_y",
        "n_to_d_z",
        "d_to_n_z",
        "dn_to_dd_z",
        "dd_to_dn_z",
        "schur_diag_safe",
        "pressure_null_mask",
    )


def test_force_solve_satisfies_momentum_incompressibility_and_pressure_gauge():
    solver = _spectral_solver()
    stokes = _canonical(solver)
    force_hats = _force_hats(solver)

    ux_hat, uy_hat, uz_hat, pressure_hat = stokes.solve_force_hats(*force_hats)
    velocity_hats = (ux_hat, uy_hat, uz_hat)
    pressure_gradient_hats = stokes.pressure_gradient_hats(pressure_hat)

    for force_hat, velocity_hat, gradient_hat in zip(
        force_hats,
        velocity_hats,
        pressure_gradient_hats,
        strict=True,
    ):
        momentum_residual = force_hat - gradient_hat - velocity_hat / stokes.a_inv
        assert _relative_norm(momentum_residual, force_hat) < 2.0e-15

    divergence_hat = stokes.divergence_hat(*velocity_hats)
    velocity_norm = sum(torch.linalg.vector_norm(value.reshape(-1)) for value in velocity_hats)
    assert float(torch.linalg.vector_norm(divergence_hat.reshape(-1)) / velocity_norm) < 2.0e-11
    assert bool((pressure_hat[:, 0, 0, 0] == 0).all().item())
    assert stokes.last_pressure_relative_residual < 2.0e-12


def test_manufactured_schur_problem_recovers_the_zero_gauge_pressure():
    solver = _spectral_solver()
    stokes = _canonical(solver)
    generator = torch.Generator().manual_seed(813)
    physical_pressure = torch.randn(
        (2, *SHAPE),
        generator=generator,
        dtype=torch.float64,
    )
    exact_pressure_hat = solver.transform_tensor(
        physical_pressure,
        CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    )
    exact_pressure_hat = stokes._project_pressure_gauge(exact_pressure_hat)
    rhs_hat = stokes._pressure_operator(exact_pressure_hat)

    actual_pressure_hat = stokes._solve_pressure(rhs_hat)

    assert (
        _relative_norm(
            stokes._pressure_operator(actual_pressure_hat) - rhs_hat,
            rhs_hat,
        )
        < 2.0e-12
    )
    assert (
        _relative_norm(
            actual_pressure_hat - exact_pressure_hat,
            exact_pressure_hat,
        )
        < 3.0e-12
    )
    assert bool((actual_pressure_hat[:, 0, 0, 0] == 0).all().item())


def test_zero_force_has_exact_zero_solution_and_diagnostics():
    solver = _spectral_solver()
    stokes = _canonical(solver)
    zero = torch.zeros(
        (2, *solver.spectral_shape),
        dtype=torch.complex128,
    )

    solution = stokes.solve_force_hats(zero, zero, zero)

    assert all(bool((value == 0).all().item()) for value in solution)
    assert stokes.last_pressure_iterations == 0
    assert stokes.last_pressure_residual == 0.0
    assert stokes.last_pressure_relative_residual == 0.0


def test_even_channel_removes_unresolved_periodic_nyquist_plane():
    solver = _spectral_solver(shape=(8, 6, 5))
    stokes = _canonical(solver)
    zero = torch.zeros(
        (2, *solver.spectral_shape),
        dtype=torch.complex128,
    )
    tangential_force = zero.clone()
    tangential_force[:, solver.shape[0] // 2, 0, 0] = 1.0

    solution = stokes.solve_force_hats(zero, tangential_force, zero)

    assert all(bool((value == 0).all().item()) for value in solution)
    assert stokes.last_pressure_iterations == 0
    assert stokes.last_pressure_residual == 0.0
    assert stokes.last_pressure_relative_residual == 0.0


def test_fixed_iteration_pressure_mode_preserves_the_requested_work_count():
    solver = _spectral_solver()
    stokes = _canonical(
        solver,
        pressure_relative_tolerance=1.0e-30,
        pressure_fixed_iterations=4,
    )

    stokes.solve_force_hats(*_force_hats(solver))

    assert stokes.last_pressure_iterations == 4
    assert stokes.pressure_fixed_iterations == 4
    assert stokes.pressure_convergence_policy() == {
        "schema_version": 1,
        "convergence_mode": "fixed_work",
        "authoritative_iteration_limit": 4,
        "authoritative_iteration_limit_source": "fixed_iterations",
        "max_iterations_role": "inactive_compatibility_field",
        "relative_tolerance_role": "diagnostic_only_not_early_exit",
        "initial_guess_policy": "persistent_production_warm_start",
        "rhs_norm_definition": "active_mode_root_mean_square.v1",
        "working_vector_scaling": ("largest_power_of_two_not_greater_than_active_component_amax"),
        "breakdown_definition": (
            "abs(mean_active(conj(left)*right))<=eps*active_rms(left)*active_rms(right)"
        ),
    }


def test_tolerance_pressure_mode_reports_maximum_iteration_precedence():
    stokes = _canonical(
        _spectral_solver(),
        pressure_relative_tolerance=1.0e-9,
        pressure_max_iterations=37,
    )

    assert stokes.pressure_convergence_policy() == {
        "schema_version": 1,
        "convergence_mode": "relative_tolerance",
        "authoritative_iteration_limit": 37,
        "authoritative_iteration_limit_source": "max_iterations",
        "max_iterations_role": "authoritative_iteration_cap",
        "relative_tolerance_role": "required_postcondition",
        "initial_guess_policy": "persistent_production_warm_start",
        "rhs_norm_definition": "active_mode_root_mean_square.v1",
        "working_vector_scaling": ("largest_power_of_two_not_greater_than_active_component_amax"),
        "breakdown_definition": (
            "abs(mean_active(conj(left)*right))<=eps*active_rms(left)*active_rms(right)"
        ),
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"friction": -0.1}, "friction"),
        ({"viscosity": 0.0}, "viscosity"),
        ({"pressure_relative_tolerance": 0.0}, "relative_tolerance"),
        ({"pressure_max_iterations": 0}, "max_iterations"),
        ({"pressure_fixed_iterations": 0}, "fixed_iterations"),
    ),
)
def test_constructor_rejects_invalid_solver_coefficients(overrides, message):
    with pytest.raises(ValueError, match=message):
        _canonical(_spectral_solver(), **overrides)


def test_constructor_rejects_non_three_dimensional_transform_backend():
    solver = _spectral_solver(shape=(7, 6))

    with pytest.raises(ValueError, match="three dimensions"):
        _canonical(solver)
