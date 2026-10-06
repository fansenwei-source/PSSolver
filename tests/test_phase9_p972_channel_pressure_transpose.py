"""P9.7.2 equation-level Channel pressure-transpose qualification."""

from __future__ import annotations

import inspect

import pytest
import torch

from pssolver import SpectralSolver
from pssolver.functional import (
    ChannelPressureTransposeOperator,
    ChannelPressureTransposeProtocol,
)
from pssolver.linear_solvers.stokes import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
)


SHAPE = (7, 6, 5)
LENGTHS = (4.0, 3.0, 2.5)


def _solver() -> tuple[
    SpectralSolver,
    ChannelNoSlipModalStokesSolver,
    ChannelPressureTransposeOperator,
]:
    spectral = SpectralSolver(
        SHAPE,
        L=LENGTHS,
        device="cpu",
        dtype=torch.float64,
    )
    pressure = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=1.0e-12,
        pressure_max_iterations=300,
    )
    return spectral, pressure, ChannelPressureTransposeOperator(pressure)


def _random_complex(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn((2, *SHAPE), generator=generator, dtype=torch.float64)
    imaginary = torch.randn(
        (2, *SHAPE),
        generator=generator,
        dtype=torch.float64,
    )
    return torch.complex(real, imaginary)


def _random_pressure_hat(
    spectral: SpectralSolver,
    pressure: ChannelNoSlipModalStokesSolver,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    physical = torch.randn(
        (2, *SHAPE),
        generator=generator,
        dtype=torch.float64,
    )
    value = spectral.transform_tensor(
        physical,
        CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    )
    return pressure._project_pressure_gauge(value)


def _relative_norm(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(value.reshape(-1))
        / torch.linalg.vector_norm(reference.reshape(-1))
    )


def _assert_complex_close(left: torch.Tensor, right: torch.Tensor) -> None:
    scale = max(float(torch.abs(left)), float(torch.abs(right)), 1.0)
    assert float(torch.abs(left - right)) <= 3.0e-14 * scale


def test_solver_satisfies_the_explicit_pressure_transpose_protocol():
    _, _, pressure = _solver()

    assert isinstance(pressure, ChannelPressureTransposeProtocol)
    source = inspect.getsource(pressure.apply_pressure_operator_transpose)
    assert "_pressure_gradient_adjoint" in source
    assert "_velocity_divergence_component_adjoint" in source
    assert "apply_pressure_operator(" not in source
    assert pressure.pressure_transpose_metadata() == {
        "protocol_version": "1",
        "inner_product": "native_complex_euclidean_coefficient_space",
        "gauge": "zero_mean_pressure_mode",
        "primal_action": "frozen_production_schur_operator",
        "transpose_action": "explicit_reverse_dataflow_conjugate_transpose",
        "transpose_solve": "zero_initial_guess_pcg",
        "production_pressure_warm_start_read": False,
        "production_pressure_warm_start_written": False,
        "pcg_iteration_graph_retained": False,
        "custom_autograd_rule": False,
        "functional_runtime_executable": False,
    }


def test_primal_and_transpose_actions_are_complex_linear_and_zero_gauge():
    _, _, pressure = _solver()
    first = _random_complex(17)
    second = _random_complex(19)
    alpha = 0.3 - 0.7j
    beta = -1.2 + 0.4j

    for action in (
        pressure.apply_pressure_operator,
        pressure.apply_pressure_operator_transpose,
    ):
        combined = action(alpha * first + beta * second)
        separate = alpha * action(first) + beta * action(second)
        torch.testing.assert_close(combined, separate, rtol=2.0e-15, atol=2.0e-14)
        assert bool((combined[:, 0, 0, 0] == 0).all().item())

        shifted = first.clone()
        shifted[:, 0, 0, 0] += 11.0 - 4.0j
        torch.testing.assert_close(
            action(shifted),
            action(first),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("axis", range(3))
def test_gradient_and_divergence_building_blocks_have_explicit_adjoint(axis):
    _, production, pressure = _solver()
    pressure_hat = production._project_pressure_gauge(
        _random_complex(31 + axis)
    )
    velocity_hat = _random_complex(41 + axis)

    gradient_pairing = torch.sum(
        torch.conj(production._pressure_gradient(pressure_hat, axis))
        * velocity_hat
    )
    gradient_adjoint_pairing = torch.sum(
        torch.conj(pressure_hat)
        * pressure._pressure_gradient_adjoint(velocity_hat, axis)
    )
    _assert_complex_close(gradient_pairing, gradient_adjoint_pairing)

    divergence_pairing = torch.sum(
        torch.conj(
            production._velocity_divergence_component(velocity_hat, axis)
        )
        * pressure_hat
    )
    divergence_adjoint_pairing = torch.sum(
        torch.conj(velocity_hat)
        * pressure._velocity_divergence_component_adjoint(pressure_hat, axis)
    )
    _assert_complex_close(divergence_pairing, divergence_adjoint_pairing)


def test_pressure_operator_satisfies_the_complex_dot_product_identity():
    _, production, pressure = _solver()
    primal_input = production._project_pressure_gauge(_random_complex(53))
    transpose_input = production._project_pressure_gauge(_random_complex(59))

    left = torch.sum(
        torch.conj(pressure.apply_pressure_operator(primal_input))
        * transpose_input
    )
    right = torch.sum(
        torch.conj(primal_input)
        * pressure.apply_pressure_operator_transpose(transpose_input)
    )

    _assert_complex_close(left, right)


@pytest.mark.parametrize("shape", ((7, 6, 5), (8, 6, 4)))
def test_pressure_transpose_uses_the_production_subspace_on_all_grid_parities(
    shape,
):
    spectral = SpectralSolver(
        shape,
        L=tuple(float(length) for length in shape),
        device="cpu",
        dtype=torch.float64,
    )
    production = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=1.0e-12,
        pressure_max_iterations=300,
    )
    pressure = ChannelPressureTransposeOperator(production)
    generator = torch.Generator().manual_seed(20261006 + shape[0])

    def random_complex():
        real = torch.randn(
            (2, *shape), generator=generator, dtype=torch.float64
        )
        imaginary = torch.randn(
            (2, *shape), generator=generator, dtype=torch.float64
        )
        return torch.complex(real, imaginary)

    primal_input = random_complex()
    transpose_input = random_complex()
    left = torch.sum(
        torch.conj(pressure.apply_pressure_operator(primal_input))
        * transpose_input
    )
    right = torch.sum(
        torch.conj(primal_input)
        * pressure.apply_pressure_operator_transpose(transpose_input)
    )
    _assert_complex_close(left, right)

    projected = pressure._project_gauge(primal_input)
    null_modes = projected.masked_select(production.pressure_null_mask)
    assert bool((null_modes == 0).all())
    if shape[0] % 2 == 0:
        assert bool(
            (primal_input.masked_select(production.nyquist_mask) != 0).any()
        )
        assert bool(
            (projected.masked_select(production.nyquist_mask) == 0).all()
        )
        transpose_output = pressure.apply_pressure_operator_transpose(
            transpose_input
        )
        assert bool(
            (transpose_output.masked_select(production.nyquist_mask) == 0).all()
        )


def test_current_discretization_is_hermitian_only_after_explicit_validation():
    _, production, pressure = _solver()
    pressure_hat = production._project_pressure_gauge(_random_complex(61))
    primal = pressure.apply_pressure_operator(pressure_hat)
    transpose = pressure.apply_pressure_operator_transpose(pressure_hat)

    assert _relative_norm(transpose - primal, primal) < 4.0e-15


def test_manufactured_primal_and_transpose_pressure_solves_recover_state():
    spectral, production, pressure = _solver()
    exact = _random_pressure_hat(spectral, production, 71)

    primal_rhs = pressure.apply_pressure_operator(exact)
    primal = production._solve_pressure(primal_rhs)
    assert _relative_norm(primal - exact, exact) < 3.0e-12
    assert _relative_norm(
        pressure.apply_pressure_operator(primal) - primal_rhs,
        primal_rhs,
    ) < 2.0e-12

    warm_start = production.pressure_guess
    transpose_rhs = pressure.apply_pressure_operator_transpose(exact)
    transpose_rhs = transpose_rhs.detach().requires_grad_(True)
    transpose = pressure.solve_pressure_transpose(transpose_rhs)
    assert production.pressure_guess is warm_start
    assert transpose.requires_grad is False
    assert _relative_norm(transpose - exact, exact) < 3.0e-12
    assert _relative_norm(
        pressure.apply_pressure_operator_transpose(transpose) - transpose_rhs,
        transpose_rhs,
    ) < 2.0e-12
    assert pressure.last_transpose_iterations > 0
    assert pressure.last_transpose_relative_residual < 2.0e-12


def test_zero_transpose_rhs_is_exact_and_does_not_touch_production_warm_start():
    _, production, pressure = _solver()
    warm_start = _random_complex(83)
    production.pressure_guess = warm_start
    zero = torch.zeros_like(warm_start)

    actual = pressure.solve_pressure_transpose(zero)

    assert bool((actual == 0).all().item())
    assert production.pressure_guess is warm_start
    assert pressure.last_transpose_iterations == 0
    assert pressure.last_transpose_residual == 0.0
    assert pressure.last_transpose_relative_residual == 0.0
