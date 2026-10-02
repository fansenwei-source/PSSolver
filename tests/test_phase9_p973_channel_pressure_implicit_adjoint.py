"""P9.7.3 Channel pressure implicit-adjoint qualification."""

from __future__ import annotations

import math

import pytest
import torch

from pssolver import SpectralSolver
from pssolver.functional import (
    ChannelImplicitPressureAdjoint,
    ChannelImplicitPressureAdjointProtocol,
    ChannelPressureTransposeOperator,
    unrolled_channel_pressure_solve_oracle,
)
from pssolver.linear_solvers.stokes import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
)


SHAPE = (3, 2, 2)
LENGTHS = (3.0, 2.0, 2.0)


def _solver(
    *,
    shape=SHAPE,
    fixed_iterations=None,
) -> tuple[SpectralSolver, ChannelNoSlipModalStokesSolver]:
    lengths = LENGTHS if shape == SHAPE else tuple(float(value) for value in shape)
    spectral = SpectralSolver(
        shape,
        L=lengths,
        device="cpu",
        dtype=torch.float64,
    )
    pressure = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=1.0e-13,
        pressure_max_iterations=100,
        pressure_fixed_iterations=fixed_iterations,
    )
    return spectral, pressure


def _pressure_hat(
    spectral: SpectralSolver,
    pressure: ChannelNoSlipModalStokesSolver,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    physical = torch.randn(
        (1, *tuple(spectral.shape)),
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


def _graph_node_count(value: torch.Tensor) -> int:
    pending = [value.grad_fn]
    seen = set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        pending.extend(next_node for next_node, _ in node.next_functions)
    return len(seen)


def test_implicit_pressure_adjoint_protocol_and_nonclaims_are_explicit():
    _, pressure = _solver()
    implicit = ChannelImplicitPressureAdjoint(pressure)

    assert isinstance(implicit, ChannelImplicitPressureAdjointProtocol)
    assert implicit.implicit_adjoint_metadata() == {
        "version": "1",
        "batch_size": 1,
        "differentiable_input": "pressure_rhs_hat",
        "operator_coefficients_differentiable": False,
        "primal_solve": "zero_initial_guess_pcg",
        "backward_solve": "explicit_transpose_zero_initial_guess_pcg",
        "production_pressure_warm_start_read": False,
        "production_pressure_warm_start_written": False,
        "forward_iteration_tensors_saved": 0,
        "backward_iteration_tensors_saved": 0,
        "higher_order_derivatives": False,
        "unrolled_oracle": "small_grid_cpu_fixed_iteration_only",
        "functional_runtime_executable": False,
    }

    with pytest.raises(TypeError, match="ChannelNoSlipModalStokesSolver"):
        ChannelImplicitPressureAdjoint(object())


def test_custom_forward_recovers_pressure_without_touching_production_warm_start():
    spectral, pressure = _solver()
    operators = ChannelPressureTransposeOperator(pressure)
    implicit = ChannelImplicitPressureAdjoint(pressure)
    exact = _pressure_hat(spectral, pressure, 103)
    rhs = operators.apply_pressure_operator(exact).detach().requires_grad_(True)
    rhs_before = rhs.detach().clone()
    warm_start = _pressure_hat(spectral, pressure, 107)
    pressure.pressure_guess = warm_start

    actual = implicit.solve(rhs)

    assert pressure.pressure_guess is warm_start
    torch.testing.assert_close(rhs.detach(), rhs_before, rtol=0.0, atol=0.0)
    assert _relative_norm(actual.detach() - exact, exact) < 2.0e-13
    assert actual.requires_grad is True
    assert type(actual.grad_fn).__name__ == "_ImplicitPressureSolveBackward"
    assert tuple(actual.grad_fn.saved_tensors) == ()
    assert implicit.last_primal_diagnostics.iterations > 0
    assert implicit.last_primal_diagnostics.relative_residual < 1.0e-13


def test_custom_vjp_equals_the_explicit_manufactured_transpose_solve():
    spectral, pressure = _solver()
    operators = ChannelPressureTransposeOperator(pressure)
    implicit = ChannelImplicitPressureAdjoint(pressure)
    exact = _pressure_hat(spectral, pressure, 109)
    rhs = operators.apply_pressure_operator(exact).detach().requires_grad_(True)
    cotangent = _pressure_hat(spectral, pressure, 113)
    warm_start = _pressure_hat(spectral, pressure, 127)
    pressure.pressure_guess = warm_start

    pressure_hat = implicit.solve(rhs)
    actual_vjp = torch.autograd.grad(
        pressure_hat,
        rhs,
        grad_outputs=cotangent,
    )[0]
    expected_vjp = operators.solve_pressure_transpose(cotangent)

    assert pressure.pressure_guess is warm_start
    assert _relative_norm(actual_vjp - expected_vjp, expected_vjp) < 2.0e-13
    assert implicit.last_transpose_diagnostics.iterations > 0
    assert implicit.last_transpose_diagnostics.relative_residual < 1.0e-13


def test_implicit_vjp_matches_the_fixed_iteration_unrolled_cpu_oracle():
    spectral, pressure = _solver(fixed_iterations=5)
    operators = ChannelPressureTransposeOperator(pressure)
    implicit = ChannelImplicitPressureAdjoint(pressure)
    exact = _pressure_hat(spectral, pressure, 131)
    base_rhs = operators.apply_pressure_operator(exact).detach()
    cotangent = _pressure_hat(spectral, pressure, 137)
    implicit_rhs = base_rhs.clone().requires_grad_(True)
    oracle_rhs = base_rhs.clone().requires_grad_(True)

    implicit_pressure = implicit.solve(implicit_rhs)
    oracle_pressure = unrolled_channel_pressure_solve_oracle(
        oracle_rhs,
        pressure,
        iterations=5,
    )
    implicit_vjp = torch.autograd.grad(
        implicit_pressure,
        implicit_rhs,
        grad_outputs=cotangent,
    )[0]
    oracle_vjp = torch.autograd.grad(
        oracle_pressure,
        oracle_rhs,
        grad_outputs=cotangent,
    )[0]

    torch.testing.assert_close(
        implicit_pressure,
        oracle_pressure,
        rtol=0.0,
        atol=0.0,
    )
    assert _relative_norm(implicit_vjp - oracle_vjp, oracle_vjp) < 3.0e-13


def test_fixed_iteration_transpose_solve_is_scale_homogeneous():
    """A tiny cotangent must not trip an absolute PCG breakdown threshold."""

    spectral, pressure = _solver(fixed_iterations=12)
    implicit = ChannelImplicitPressureAdjoint(pressure)
    cotangent = _pressure_hat(spectral, pressure, 137)
    reference = implicit._solve_transpose_no_grad(cotangent)

    for scale in (1.0e-8, 1.0e-10, 1.0e-12):
        actual = implicit._solve_transpose_no_grad(scale * cotangent)
        torch.testing.assert_close(
            actual,
            scale * reference,
            rtol=3.0e-13,
            atol=1.0e-28,
        )
        assert implicit.last_transpose_diagnostics.relative_residual < 1.0e-13


@pytest.mark.parametrize("scale", (1.0e-155, 1.0e155))
@pytest.mark.parametrize("solve_kind", ("primal", "transpose"))
def test_pressure_solve_normalizes_extreme_rhs_scales(scale, solve_kind):
    """PCG quadratic products must not inherit the caller's RHS scale."""

    spectral, pressure = _solver(fixed_iterations=12)
    implicit = ChannelImplicitPressureAdjoint(pressure)
    rhs = _pressure_hat(spectral, pressure, 138)
    solve = getattr(implicit, f"_solve_{solve_kind}_no_grad")

    reference = solve(rhs)
    reference_diagnostics = getattr(
        implicit,
        f"last_{solve_kind}_diagnostics",
    )
    actual = solve(scale * rhs)
    actual_diagnostics = getattr(
        implicit,
        f"last_{solve_kind}_diagnostics",
    )

    assert bool(torch.isfinite(actual).all().item())
    torch.testing.assert_close(
        actual / scale,
        reference,
        rtol=5.0e-13,
        atol=5.0e-13,
    )
    assert actual_diagnostics.termination_reason == (
        reference_diagnostics.termination_reason
    )
    assert actual_diagnostics.iterations == reference_diagnostics.iterations
    assert actual_diagnostics.relative_residual < 1.0e-13
    assert actual_diagnostics.residual < abs(scale) * 1.0e-13


def test_custom_vjp_passes_directional_finite_difference_and_taylor_gate():
    spectral, pressure = _solver()
    operators = ChannelPressureTransposeOperator(pressure)
    implicit = ChannelImplicitPressureAdjoint(pressure)
    exact = _pressure_hat(spectral, pressure, 139)
    base = operators.apply_pressure_operator(exact).detach()
    direction = _pressure_hat(spectral, pressure, 149)
    direction = direction / torch.linalg.vector_norm(direction.reshape(-1))
    rhs = base.clone().requires_grad_(True)

    def objective(value):
        solution = implicit.solve(value)
        return 0.5 * torch.sum(torch.abs(solution).square())

    base_objective = objective(rhs)
    gradient = torch.autograd.grad(base_objective, rhs)[0]
    directional = torch.sum(torch.conj(gradient) * direction).real
    steps = (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4)
    first_remainders = []
    centered_values = []
    for step in steps:
        plus = objective(base + step * direction).detach()
        minus = objective(base - step * direction).detach()
        centered_values.append((plus - minus) / (2.0 * step))
        first_remainders.append(
            torch.abs(plus - base_objective.detach() - step * directional)
        )

    derivative_error = torch.abs(centered_values[-1] - directional)
    derivative_scale = max(float(torch.abs(directional)), 1.0e-14)
    assert float(derivative_error) / derivative_scale < 2.0e-7
    orders = [
        math.log2(float(left / right))
        for left, right in zip(
            first_remainders[:-1],
            first_remainders[1:],
            strict=True,
        )
    ]
    assert min(orders) > 1.95


def test_custom_graph_size_and_saved_tensors_do_not_depend_on_pcg_iterations():
    graph_sizes = []
    for iterations in (2, 5):
        spectral, pressure = _solver(fixed_iterations=iterations)
        operators = ChannelPressureTransposeOperator(pressure)
        exact = _pressure_hat(spectral, pressure, 151 + iterations)
        rhs = operators.apply_pressure_operator(exact).detach().requires_grad_(
            True
        )
        packed = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda tensor: packed.append(tensor) or tensor,
            lambda tensor: tensor,
        ):
            value = ChannelImplicitPressureAdjoint(pressure).solve(rhs)
        assert packed == []
        assert tuple(value.grad_fn.saved_tensors) == ()
        graph_sizes.append(_graph_node_count(value))

    assert graph_sizes[0] == graph_sizes[1]


def test_custom_backward_is_first_order_only():
    spectral, pressure = _solver()
    operators = ChannelPressureTransposeOperator(pressure)
    exact = _pressure_hat(spectral, pressure, 163)
    rhs = operators.apply_pressure_operator(exact).detach().requires_grad_(True)
    value = ChannelImplicitPressureAdjoint(pressure).solve(rhs)
    gradient = torch.autograd.grad(
        torch.sum(torch.abs(value).square()),
        rhs,
        create_graph=True,
    )[0]

    assert gradient.requires_grad is True
    assert type(gradient.grad_fn).__name__ == "Error"
    with pytest.raises(RuntimeError):
        torch.autograd.grad(torch.sum(torch.abs(gradient).square()), rhs)


def test_unrolled_oracle_and_implicit_input_contracts_fail_closed():
    _, pressure = _solver()
    implicit = ChannelImplicitPressureAdjoint(pressure)
    valid = torch.zeros(
        tuple(pressure.pressure_null_mask.shape),
        dtype=torch.complex128,
    )

    with pytest.raises(ValueError, match="shape"):
        implicit.solve(valid.expand(2, *valid.shape[1:]))
    with pytest.raises(TypeError, match="dtype"):
        implicit.solve(valid.real)
    invalid = valid.clone()
    invalid[:, 1, 0, 0] = complex(float("nan"), 0.0)
    with pytest.raises(ValueError, match="finite"):
        implicit.solve(invalid)
    with pytest.raises(ValueError, match="positive integer"):
        unrolled_channel_pressure_solve_oracle(
            valid,
            pressure,
            iterations=0,
        )

    _, large_pressure = _solver(shape=(9, 8, 8))
    large = torch.zeros(
        tuple(large_pressure.pressure_null_mask.shape),
        dtype=torch.complex128,
    )
    with pytest.raises(ValueError, match="small grids"):
        unrolled_channel_pressure_solve_oracle(
            large,
            large_pressure,
            iterations=2,
        )
