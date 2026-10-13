"""RC4.4.3 scale-safe Channel pressure-policy implementation gates."""

from __future__ import annotations

import json
import math

import pytest
import torch

from pssolver import SpectralSolver
from pssolver.functional import (
    CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION,
    CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION,
    CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION,
    ChannelImplicitPressureAdjoint,
    ChannelPressureTransposeOperator,
    FunctionalConvergenceError,
)
from pssolver.linear_solvers.stokes import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
    ChannelPressureConvergenceError,
)
from pssolver.linear_solvers.stokes.channel_no_slip import (
    _channel_pressure_operator_breakdown,
    _channel_pressure_working_breakdown,
    _channel_pressure_working_magnitude,
    _channel_pressure_working_residual_observation,
    channel_pressure_active_rms,
    channel_pressure_relative_breakdown,
    channel_pressure_scale_power_of_two,
    channel_pressure_working_vector,
)


SHAPE = (3, 2, 2)


def _solver(
    *,
    fixed_iterations: int | None = 5,
    relative_tolerance: float = 1.0e-13,
    max_iterations: int = 100,
):
    spectral = SpectralSolver(
        SHAPE,
        L=(3.0, 2.0, 2.0),
        device="cpu",
        dtype=torch.float64,
    )
    pressure = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=relative_tolerance,
        pressure_max_iterations=max_iterations,
        pressure_fixed_iterations=fixed_iterations,
    )
    return spectral, pressure


def _pressure_rhs(spectral, pressure, seed=4430):
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


def _ordinary_unscaled_fixed_work(pressure, rhs, *, transpose):
    operators = ChannelPressureTransposeOperator(pressure)
    action = (
        operators.apply_pressure_operator_transpose
        if transpose
        else operators.apply_pressure_operator
    )
    project = operators._project_gauge
    value = torch.zeros_like(rhs)
    residual = project(rhs).clone()
    preconditioned = residual / pressure.schur_diag_safe
    direction = preconditioned.clone()
    rz_old = torch.sum(torch.conj(residual) * preconditioned).real
    for index in range(pressure.pressure_fixed_iterations):
        operator_direction = action(direction)
        step = rz_old / torch.sum(
            torch.conj(direction) * operator_direction
        ).real
        value = project(value + step * direction)
        residual = project(residual - step * operator_direction)
        if index + 1 < pressure.pressure_fixed_iterations:
            preconditioned = residual / pressure.schur_diag_safe
            rz_new = torch.sum(
                torch.conj(residual) * preconditioned
            ).real
            direction = preconditioned + (rz_new / rz_old) * direction
            rz_old = rz_new
    return value


@pytest.mark.parametrize("shape", ((3, 2, 2), (4, 3, 2), (5, 4, 3)))
@pytest.mark.parametrize("exponent", (-1074, -1020, 0, 1018, 1023))
def test_active_rms_encodes_every_frozen_scale_without_mislabeling_zero(
    shape,
    exponent,
):
    value = torch.zeros((1, *shape), dtype=torch.complex128)
    mask = torch.zeros_like(value, dtype=torch.bool)
    mask[:, 0, 0, 0] = True
    value[:, 1, 0, 0] = complex(math.ldexp(1.0, exponent), 0.0)

    magnitude = channel_pressure_active_rms(value, mask)
    working, working_exponent, duplicate = channel_pressure_working_vector(
        value,
        mask,
    )
    restored = channel_pressure_scale_power_of_two(
        working,
        working_exponent,
    )

    assert magnitude == duplicate
    assert magnitude.is_zero is False
    assert math.isfinite(magnitude.log2_value)
    assert 0.5 <= magnitude.binary_mantissa < 1.0
    assert working_exponent == exponent
    assert float(working.real.abs().amax()) == 1.0
    assert torch.equal(restored, value)
    if exponent == -1074:
        assert magnitude.float64_value is None
        assert magnitude.representable_as_positive_float64 is False


def test_zero_power_of_two_scale_preserves_tensor_identity():
    value = torch.ones((2, 3), dtype=torch.complex128)
    assert channel_pressure_scale_power_of_two(value, 0) is value


def test_relative_breakdown_is_scale_invariant_and_has_positive_control():
    _, pressure = _solver()
    mask = pressure.pressure_inactive_mask
    left = torch.zeros(mask.shape, dtype=torch.complex128)
    right = torch.zeros_like(left)
    left[:, 1, 0, 0] = 1.0
    right[:, 2, 0, 0] = 1.0

    assert channel_pressure_relative_breakdown(left, right, mask) is True
    right.copy_(left)
    assert channel_pressure_relative_breakdown(left, right, mask) is False
    for exponent in (-1000, 1000):
        scaled_left = channel_pressure_scale_power_of_two(left, exponent)
        scaled_right = channel_pressure_scale_power_of_two(right, -exponent)
        assert (
            channel_pressure_relative_breakdown(
                scaled_left,
                scaled_right,
                mask,
            )
            is False
        )


def test_working_scale_fast_measurements_match_public_pressure_primitives():
    _, pressure = _solver()
    mask = pressure.pressure_inactive_mask
    generator = torch.Generator().manual_seed(4437)
    left = torch.complex(
        torch.randn(mask.shape, generator=generator, dtype=torch.float64),
        torch.randn(mask.shape, generator=generator, dtype=torch.float64),
    ).masked_fill(mask, 0)
    right = (0.75 * left).masked_fill(mask, 0)

    magnitude, finite = _channel_pressure_working_magnitude(
        left,
        mask,
        finite_companions=(right,),
    )
    breakdown, pair_finite = _channel_pressure_working_breakdown(
        left,
        right,
        mask,
    )

    assert finite is True
    assert pair_finite is True
    assert magnitude == channel_pressure_active_rms(left, mask)
    assert breakdown is channel_pressure_relative_breakdown(left, right, mask)

    active_count = int((~mask).sum().item())
    inner_product = torch.sum(torch.conj(left) * right).real
    observed, state_finite, residual_breakdown, residual_pair_finite = (
        _channel_pressure_working_residual_observation(
            left,
            right,
            mask,
            inner_product=inner_product,
            active_count=active_count,
            finite_companions=(right,),
        )
    )
    assert observed == channel_pressure_active_rms(left, mask)
    assert state_finite is True
    assert residual_pair_finite is True
    assert residual_breakdown is channel_pressure_relative_breakdown(
        left,
        right,
        mask,
    )


def test_working_scale_fast_measurements_fail_closed_on_nonfinite_values():
    _, pressure = _solver()
    mask = pressure.pressure_inactive_mask
    left = torch.ones(mask.shape, dtype=torch.complex128).masked_fill(mask, 0)
    nonfinite = left.clone()
    nonfinite[:, 1, 0, 0] = complex(float("nan"), 0.0)

    magnitude, finite = _channel_pressure_working_magnitude(
        left,
        mask,
        finite_companions=(nonfinite,),
    )
    breakdown, pair_finite = _channel_pressure_working_breakdown(
        left,
        nonfinite,
        mask,
    )

    assert magnitude.is_zero is True
    assert finite is False
    assert breakdown is False
    assert pair_finite is False

    (
        certified_magnitude,
        certified_state_finite,
        certified_breakdown,
        certified_pair_finite,
    ) = _channel_pressure_working_residual_observation(
        left,
        nonfinite,
        mask,
        inner_product=torch.tensor(float("inf"), dtype=torch.float64),
        active_count=int((~mask).sum().item()),
        breakdown_certified=True,
        preconditioner_inverse_max=float("inf"),
    )
    assert certified_magnitude == channel_pressure_active_rms(left, mask)
    assert certified_state_finite is True
    assert certified_breakdown is False
    assert certified_pair_finite is False


@pytest.mark.parametrize("shape", ((128, 20, 20), (512, 40, 40)))
def test_float64_channel_breakdown_has_a_margin_certified_spectral_bound(shape):
    spectral = SpectralSolver(
        shape,
        L=(32.0, 5.0, 5.0),
        device="cpu",
        dtype=torch.float64,
    )
    pressure = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=1.0,
        viscosity=1.0,
        pressure_relative_tolerance=1.0e-6,
        pressure_max_iterations=100,
    )

    floor = 4096.0 * torch.finfo(torch.float64).eps
    assert pressure._pressure_operator_angle_lower_bound > floor
    assert pressure._pressure_preconditioner_angle_lower_bound > floor
    assert pressure._pressure_operator_breakdown_certified is True
    assert pressure._pressure_preconditioner_breakdown_certified is True


def test_breakdown_certificate_fast_path_matches_full_vector_gate_and_solution():
    spectral, certified = _solver(
        fixed_iterations=None,
        relative_tolerance=1.0e-6,
    )
    rhs = _pressure_rhs(spectral, certified)
    fallback = _solver(
        fixed_iterations=None,
        relative_tolerance=1.0e-6,
    )[1]
    fallback._pressure_operator_breakdown_certified = False
    fallback._pressure_preconditioner_breakdown_certified = False

    actual = certified._solve_pressure(rhs)
    expected = fallback._solve_pressure(rhs)

    assert torch.equal(actual, expected)
    certified_diagnostics = certified.last_pressure_diagnostics
    fallback_diagnostics = fallback.last_pressure_diagnostics
    for name in (
        "achieved_iteration_count",
        "termination_reason",
        "postcondition_satisfied",
        "acceptable",
    ):
        assert certified_diagnostics[name] == fallback_diagnostics[name]
    assert certified_diagnostics["achieved_normalized_residual"] == pytest.approx(
        fallback_diagnostics["achieved_normalized_residual"],
        rel=2.0e-15,
        abs=0.0,
    )


def test_operator_upper_bound_fast_path_matches_full_relative_breakdown_gate():
    spectral, pressure = _solver()
    mask = pressure.pressure_inactive_mask
    generator = torch.Generator().manual_seed(4481)
    direction = torch.complex(
        torch.randn(mask.shape, generator=generator, dtype=torch.float64),
        torch.randn(mask.shape, generator=generator, dtype=torch.float64),
    ).masked_fill(mask, 0)
    operator_direction = pressure._pressure_operator(direction)
    inner_product = torch.sum(torch.conj(direction) * operator_direction)

    breakdown, finite = _channel_pressure_operator_breakdown(
        direction,
        operator_direction,
        mask,
        inner_product=inner_product,
        operator_norm_upper_bound=pressure._pressure_operator_norm_upper_bound,
    )

    assert finite is True
    assert breakdown is channel_pressure_relative_breakdown(
        direction,
        operator_direction,
        mask,
    )


def test_fixed_work_ignores_loose_tolerance_on_all_three_executable_loops():
    spectral, pressure = _solver(
        fixed_iterations=5,
        relative_tolerance=1.0,
    )
    rhs = _pressure_rhs(spectral, pressure)

    pressure._solve_pressure(rhs)
    assert pressure.last_pressure_iterations == 5
    assert pressure.last_pressure_termination_reason == "fixed_work_completed"

    implicit = ChannelImplicitPressureAdjoint(pressure)
    implicit.solve(rhs)
    assert implicit.last_primal_diagnostics.iterations == 5
    assert implicit.last_primal_diagnostics.termination_reason == (
        "fixed_work_completed"
    )

    transpose = ChannelPressureTransposeOperator(pressure)
    transpose.solve_pressure_transpose(rhs)
    assert transpose.last_transpose_iterations == 5
    assert transpose.last_transpose_diagnostics.termination_reason == (
        "fixed_work_completed"
    )


def test_tolerance_cap_fails_closed_and_preserves_production_warm_start():
    spectral, pressure = _solver(
        fixed_iterations=None,
        relative_tolerance=1.0e-30,
        max_iterations=1,
    )
    rhs = _pressure_rhs(spectral, pressure)
    warm_start = torch.ones_like(rhs)
    pressure.pressure_guess = warm_start

    with pytest.raises(ChannelPressureConvergenceError) as captured:
        pressure._solve_pressure(rhs)

    assert pressure.pressure_guess is warm_start
    diagnostics = captured.value.diagnostics
    assert diagnostics == pressure.last_pressure_diagnostics
    assert diagnostics["schema_version"] == 2
    assert diagnostics["termination_reason"] == (
        "iteration_limit_reached_without_postcondition"
    )
    assert diagnostics["acceptable"] is False


def test_production_pressure_solution_is_power_of_two_scale_covariant():
    spectral, pressure = _solver(fixed_iterations=5)
    rhs = _pressure_rhs(spectral, pressure)
    pressure.pressure_guess = None
    reference = pressure._solve_pressure(rhs)
    assert pressure.last_pressure_working_scale_binary_exponent == 0

    for exponent in (-1000, 1000):
        pressure.pressure_guess = None
        actual = pressure._solve_pressure(
            channel_pressure_scale_power_of_two(rhs, exponent)
        )
        expected = channel_pressure_scale_power_of_two(reference, exponent)
        torch.testing.assert_close(actual, expected, rtol=2.0e-15, atol=0.0)


def test_functional_failure_is_persisted_before_exception_and_zero_start():
    spectral, pressure = _solver(
        fixed_iterations=None,
        relative_tolerance=1.0e-30,
        max_iterations=1,
    )
    rhs = _pressure_rhs(spectral, pressure)
    warm_start = torch.ones_like(rhs)
    pressure.pressure_guess = warm_start
    implicit = ChannelImplicitPressureAdjoint(pressure)

    with pytest.raises(FunctionalConvergenceError) as captured:
        implicit.solve(rhs)

    assert pressure.pressure_guess is warm_start
    assert captured.value.details == implicit.last_primal_diagnostics.to_metadata()
    assert captured.value.details["acceptable"] is False
    assert captured.value.details["solve_phase"] == "functional_primal"
    assert implicit.last_transpose_diagnostics.solve_phase == "not_run"


@pytest.mark.parametrize("transpose", (False, True))
def test_ordinary_fixed_work_path_remains_byte_identical_to_unscaled_order(
    transpose,
):
    spectral, pressure = _solver(fixed_iterations=5)
    rhs = _pressure_rhs(spectral, pressure)
    expected = _ordinary_unscaled_fixed_work(
        pressure,
        rhs,
        transpose=transpose,
    )
    if transpose:
        actual = ChannelPressureTransposeOperator(
            pressure
        ).solve_pressure_transpose(rhs)
    else:
        actual = ChannelImplicitPressureAdjoint(pressure).solve(rhs)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("transpose", (False, True))
def test_functional_pressure_solution_is_power_of_two_scale_covariant(
    transpose,
):
    spectral, pressure = _solver(fixed_iterations=5)
    rhs = _pressure_rhs(spectral, pressure)
    if transpose:
        solve = ChannelPressureTransposeOperator(pressure).solve_pressure_transpose
    else:
        solve = ChannelImplicitPressureAdjoint(pressure).solve
    reference = solve(rhs)

    for exponent in (-1000, 1000):
        actual = solve(channel_pressure_scale_power_of_two(rhs, exponent))
        expected = channel_pressure_scale_power_of_two(reference, exponent)
        torch.testing.assert_close(actual, expected, rtol=2.0e-15, atol=0.0)


def test_schema_v2_phase_order_is_json_safe_and_invalidates_old_transpose():
    spectral, pressure = _solver(fixed_iterations=5)
    rhs = _pressure_rhs(spectral, pressure).requires_grad_(True)
    implicit = ChannelImplicitPressureAdjoint(pressure)
    first = implicit.solve(rhs)
    torch.autograd.grad(first, rhs, grad_outputs=torch.ones_like(first))
    first_metadata = implicit.pressure_solve_metadata()
    assert first_metadata["transpose"]["solve_phase"] == "functional_transpose"
    assert first_metadata["transpose"]["source_primal_sequence_id"] == 1

    implicit.solve(rhs.detach())
    second_metadata = implicit.pressure_solve_metadata()
    assert second_metadata["schema_version"] == 2
    assert second_metadata["solve_sequence_id"] == 2
    assert second_metadata["primal"]["solve_sequence_id"] == 2
    assert second_metadata["transpose"]["solve_phase"] == "not_run"
    assert second_metadata["transpose"]["source_primal_sequence_id"] == 2
    json.dumps(second_metadata, allow_nan=False, sort_keys=True)


def test_production_and_functional_schema_v2_fields_have_identical_meanings():
    spectral, pressure = _solver(fixed_iterations=5)
    rhs = _pressure_rhs(spectral, pressure)
    pressure._solve_pressure(rhs)
    production = pressure.last_pressure_diagnostics
    functional_solver = ChannelImplicitPressureAdjoint(pressure)
    functional_solver.solve(rhs)
    functional = functional_solver.last_primal_diagnostics.to_metadata()
    required = {
        "schema_version",
        "solve_sequence_id",
        "source_primal_sequence_id",
        "solver_identity",
        "operator_identity",
        "solve_phase",
        "initial_guess_policy",
        "convergence_mode",
        "requested_relative_tolerance",
        "requested_fixed_iterations",
        "requested_max_iterations",
        "authoritative_iteration_limit",
        "authoritative_iteration_limit_source",
        "max_iterations_role",
        "relative_tolerance_role",
        "active_pressure_degrees_of_freedom",
        "rhs_norm_definition",
        "rhs_norm",
        "initial_absolute_residual",
        "achieved_absolute_residual",
        "normalized_residual_definition",
        "achieved_normalized_residual",
        "working_vector_scaling",
        "working_scale_binary_exponent",
        "breakdown_definition",
        "achieved_iteration_count",
        "termination_reason",
        "required_postcondition",
        "postcondition_satisfied",
        "acceptable",
    }

    assert set(production) == required
    assert set(functional) == required
    for field in (
        "schema_version",
        "convergence_mode",
        "requested_relative_tolerance",
        "requested_fixed_iterations",
        "requested_max_iterations",
        "authoritative_iteration_limit",
        "authoritative_iteration_limit_source",
        "max_iterations_role",
        "relative_tolerance_role",
        "active_pressure_degrees_of_freedom",
        "rhs_norm_definition",
        "normalized_residual_definition",
        "working_vector_scaling",
        "breakdown_definition",
        "required_postcondition",
    ):
        assert production[field] == functional[field]
    json.dumps(production, allow_nan=False, sort_keys=True)
    json.dumps(functional, allow_nan=False, sort_keys=True)


def test_backward_from_an_older_primal_does_not_publish_a_stale_snapshot():
    spectral, pressure = _solver(fixed_iterations=5)
    rhs = _pressure_rhs(spectral, pressure).requires_grad_(True)
    implicit = ChannelImplicitPressureAdjoint(pressure)
    stale = implicit.solve(rhs)
    implicit.solve(rhs.detach())

    gradient = torch.autograd.grad(
        stale,
        rhs,
        grad_outputs=torch.ones_like(stale),
    )[0]

    assert bool(torch.isfinite(gradient).all())
    metadata = implicit.pressure_solve_metadata()
    assert metadata["solve_sequence_id"] == 2
    assert metadata["transpose"]["solve_phase"] == "not_run"
    assert metadata["transpose"]["source_primal_sequence_id"] == 2


def test_rc443_core_versions_are_explicit_without_public_identity_connection():
    assert CHANNEL_PRESSURE_CONVERGENCE_SCHEMA_VERSION == 2
    assert CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION == "2"
    assert CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION == "2"
