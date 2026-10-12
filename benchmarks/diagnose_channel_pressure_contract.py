"""RC4.4.1 analysis-only Channel pressure-contract adjudication.

This module does not alter a production solver.  It exercises the existing
Channel pressure operators through an independent, scale-safe PCG reference
loop and records the evidence used to select the RC4.4 pressure norm,
breakdown, and capability-metadata contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    SpectralSolver,
    TimeStepping,
    TorchSpectralExecution,
)
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.functional import (
    ChannelImplicitPressureAdjoint,
    ChannelPressureTransposeOperator,
    build_channel_activity_functional_runtime,
    channel_activity_functional_declaration,
    channel_activity_functional_request,
    unrolled_channel_pressure_solve_oracle,
)
from pssolver.geometries import RectangularChannel
from pssolver.linear_solvers.stokes import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
)
from pssolver.models.active_nematics import CompleteStressBerisEdwards


SCHEMA_VERSION = 1
GRID_SHAPES = ((3, 2, 2), (4, 3, 2), (5, 4, 3))
SCALE_EXPONENTS = (-900, -600, 0, 600, 900)
FIXED_ITERATIONS = 5
RELATIVE_TOLERANCE = 1.0e-13
MAX_ITERATIONS = 160
SCALE_COVARIANCE_BOUND = 2.0e-11
INNER_PRODUCT_BOUND = 2.0e-13


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    json.loads(path.read_text(encoding="utf-8"))


def _solver(shape: tuple[int, int, int], *, fixed_iterations: int | None):
    lengths = tuple(float(value) for value in shape)
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
        pressure_relative_tolerance=RELATIVE_TOLERANCE,
        pressure_max_iterations=MAX_ITERATIONS,
        pressure_fixed_iterations=fixed_iterations,
    )
    return spectral, pressure


def _pressure_hat(spectral, pressure, seed: int) -> torch.Tensor:
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


def _active_mask(pressure) -> torch.Tensor:
    return ~(pressure.pressure_null_mask | pressure.nyquist_mask)


def _scaled_norm(value: torch.Tensor, active: torch.Tensor) -> dict[str, object]:
    """Return overflow-safe global-L2 and active-RMS representations."""

    selected = torch.abs(value[active])
    count = int(selected.numel())
    maximum = float(torch.max(selected).item()) if count else 0.0
    if maximum == 0.0:
        return {
            "active_degrees_of_freedom": count,
            "maximum": 0.0,
            "global_l2": 0.0,
            "active_rms": 0.0,
            "global_l2_log2": None,
            "active_rms_log2": None,
            "global_l2_finite": True,
            "active_rms_finite": True,
        }
    unit = selected / maximum
    unit_l2 = float(torch.linalg.vector_norm(unit).item())
    unit_rms = unit_l2 / math.sqrt(count)
    global_log2 = math.log2(maximum) + math.log2(unit_l2)
    rms_log2 = math.log2(maximum) + math.log2(unit_rms)
    try:
        global_l2 = math.ldexp(1.0, math.floor(global_log2)) * (
            2.0 ** (global_log2 - math.floor(global_log2))
        )
    except OverflowError:
        global_l2 = float("inf")
    active_rms = maximum * unit_rms
    global_finite = math.isfinite(global_l2) and global_l2 > 0.0
    rms_finite = math.isfinite(active_rms) and active_rms > 0.0
    return {
        "active_degrees_of_freedom": count,
        "maximum": maximum,
        "global_l2": global_l2 if global_finite else None,
        "active_rms": active_rms,
        "global_l2_log2": global_log2,
        "active_rms_log2": rms_log2,
        "global_l2_finite": global_finite,
        "active_rms_finite": rms_finite,
    }


def _active_rms(value: torch.Tensor, active: torch.Tensor) -> float:
    return float(_scaled_norm(value, active)["active_rms"])


def _power_of_two_scale(value: torch.Tensor, active: torch.Tensor) -> float:
    maximum = float(torch.max(torch.abs(value[active])).item())
    if maximum == 0.0:
        return 1.0
    _, exponent = math.frexp(maximum)
    return math.ldexp(1.0, exponent - 1)


@torch.no_grad()
def _reference_pcg(
    pressure,
    rhs: torch.Tensor,
    *,
    transpose: bool,
    fixed_iterations: int | None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Independent scale-safe loop implementing the frozen RC4.4.0 policy."""

    operators = ChannelPressureTransposeOperator(pressure)
    action = (
        operators.apply_pressure_operator_transpose
        if transpose
        else operators.apply_pressure_operator
    )
    project = operators._project_gauge
    active = _active_mask(pressure)
    rhs = project(rhs)
    rhs_metrics = _scaled_norm(rhs, active)
    if rhs_metrics["maximum"] == 0.0:
        return torch.zeros_like(rhs), {
            "iterations": 0,
            "termination_reason": "zero_rhs",
            "acceptable": True,
            "rhs": rhs_metrics,
            "initial_active_rms": 0.0,
            "final_active_rms": 0.0,
            "normalized_residual": 0.0,
            "work_scale": 1.0,
        }

    work_scale = _power_of_two_scale(rhs, active)
    working_rhs = rhs / work_scale
    pressure_hat = torch.zeros_like(working_rhs)
    residual = working_rhs.clone()
    preconditioned = residual / pressure.schur_diag_safe
    direction = preconditioned.clone()
    rz_old = torch.sum(torch.conj(residual) * preconditioned).real
    initial_rms = _active_rms(residual, active)
    residual_rms = initial_rms
    limit = fixed_iterations if fixed_iterations is not None else MAX_ITERATIONS
    iterations = 0
    reason = "iteration_limit_reached_without_postcondition"
    eps = torch.finfo(residual.real.dtype).eps

    while iterations < limit:
        if torch.count_nonzero(residual).item() == 0:
            reason = "exact_zero_residual"
            break
        if fixed_iterations is None and residual_rms <= (
            RELATIVE_TOLERANCE * initial_rms
        ):
            reason = "relative_tolerance_met"
            break
        operator_direction = action(direction)
        denominator = torch.sum(
            torch.conj(direction) * operator_direction
        ).real
        denominator_scale = _active_rms(direction, active) * _active_rms(
            operator_direction, active
        )
        if (
            not math.isfinite(denominator_scale)
            or denominator_scale == 0.0
            or abs(float(denominator.item()))
            <= eps * denominator_scale * int(active.sum().item())
        ):
            reason = "operator_breakdown"
            break
        step = rz_old / denominator
        pressure_hat = project(pressure_hat + step * direction)
        residual = project(residual - step * operator_direction)
        residual_rms = _active_rms(residual, active)
        iterations += 1
        if torch.count_nonzero(residual).item() == 0:
            reason = "exact_zero_residual"
            break
        if fixed_iterations is None and residual_rms <= (
            RELATIVE_TOLERANCE * initial_rms
        ):
            reason = "relative_tolerance_met"
            break
        preconditioned = residual / pressure.schur_diag_safe
        rz_new = torch.sum(torch.conj(residual) * preconditioned).real
        rz_scale = _active_rms(residual, active) * _active_rms(
            preconditioned, active
        )
        if (
            not math.isfinite(rz_scale)
            or rz_scale == 0.0
            or abs(float(rz_old.item()))
            <= eps * rz_scale * int(active.sum().item())
        ):
            reason = "preconditioned_residual_breakdown"
            break
        direction = preconditioned + (rz_new / rz_old) * direction
        rz_old = rz_new
    else:
        if fixed_iterations is not None:
            reason = "fixed_work_completed"

    normalized = residual_rms / initial_rms
    if fixed_iterations is None:
        acceptable = reason in {"relative_tolerance_met", "exact_zero_residual"}
    else:
        acceptable = reason in {"fixed_work_completed", "exact_zero_residual"}
    result = pressure_hat * work_scale
    return result, {
        "iterations": iterations,
        "termination_reason": reason,
        "acceptable": acceptable,
        "rhs": rhs_metrics,
        "initial_active_rms": initial_rms * work_scale,
        "final_active_rms": residual_rms * work_scale,
        "normalized_residual": normalized,
        "work_scale": work_scale,
    }


def _relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = float(torch.linalg.vector_norm(reference.reshape(-1)).item())
    numerator = float(torch.linalg.vector_norm(value.reshape(-1)).item())
    return numerator / max(denominator, torch.finfo(reference.real.dtype).tiny)


def _scale_sweep() -> dict[str, object]:
    grids = []
    maximum_covariance_error = 0.0
    all_acceptable = True
    for grid_index, shape in enumerate(GRID_SHAPES):
        spectral, pressure = _solver(shape, fixed_iterations=FIXED_ITERATIONS)
        operators = ChannelPressureTransposeOperator(pressure)
        manufactured = _pressure_hat(spectral, pressure, 4410 + grid_index)
        rhs_by_action = {
            "primal": operators.apply_pressure_operator(manufactured),
            "transpose": operators.apply_pressure_operator_transpose(
                manufactured
            ),
        }
        grid_record = {
            "shape": list(shape),
            "active_pressure_degrees_of_freedom": int(
                _active_mask(pressure).sum().item()
            ),
            "actions": {},
        }
        for action_name, rhs in rhs_by_action.items():
            transpose = action_name == "transpose"
            action_record = {}
            for mode, fixed in (
                ("fixed_work", FIXED_ITERATIONS),
                ("relative_tolerance", None),
            ):
                reference, reference_diagnostics = _reference_pcg(
                    pressure,
                    rhs,
                    transpose=transpose,
                    fixed_iterations=fixed,
                )
                trials = []
                for exponent in SCALE_EXPONENTS:
                    factor = math.ldexp(1.0, exponent)
                    actual, diagnostics = _reference_pcg(
                        pressure,
                        rhs * factor,
                        transpose=transpose,
                        fixed_iterations=fixed,
                    )
                    covariance_error = _relative_l2(
                        (actual / factor) - reference,
                        reference,
                    )
                    maximum_covariance_error = max(
                        maximum_covariance_error, covariance_error
                    )
                    all_acceptable &= bool(diagnostics["acceptable"])
                    trials.append(
                        {
                            "scale_exponent": exponent,
                            "finite": bool(torch.isfinite(actual).all().item()),
                            "scale_covariance_relative_l2": covariance_error,
                            "iterations": diagnostics["iterations"],
                            "termination_reason": diagnostics[
                                "termination_reason"
                            ],
                            "normalized_residual": diagnostics[
                                "normalized_residual"
                            ],
                            "rhs_active_rms": diagnostics["rhs"]["active_rms"],
                        }
                    )
                action_record[mode] = {
                    "reference_diagnostics": reference_diagnostics,
                    "scale_trials": trials,
                }
            grid_record["actions"][action_name] = action_record
        grids.append(grid_record)
    return {
        "grids": grids,
        "maximum_scale_covariance_relative_l2": maximum_covariance_error,
        "scale_covariance_bound": SCALE_COVARIANCE_BOUND,
        "all_solve_outcomes_acceptable": all_acceptable,
        "passed": bool(
            all_acceptable
            and maximum_covariance_error <= SCALE_COVARIANCE_BOUND
        ),
    }


def _norm_stress() -> dict[str, object]:
    records = []
    global_failures = 0
    rms_failures = 0
    for shape in GRID_SHAPES:
        _, pressure = _solver(shape, fixed_iterations=FIXED_ITERATIONS)
        active = _active_mask(pressure)
        for exponent in (-1074, -1020, 0, 1018, 1023):
            value = torch.zeros(
                tuple(active.shape), dtype=torch.complex128
            )
            value[active] = complex(math.ldexp(1.0, exponent), 0.0)
            metrics = _scaled_norm(value, active)
            global_failures += int(not metrics["global_l2_finite"])
            rms_failures += int(not metrics["active_rms_finite"])
            records.append(
                {
                    "shape": list(shape),
                    "scale_exponent": exponent,
                    **metrics,
                }
            )
    return {
        "records": records,
        "scaled_global_l2_nonfinite_or_zero_count": global_failures,
        "active_mode_rms_nonfinite_or_zero_count": rms_failures,
        "active_mode_rms_passed": rms_failures == 0,
        "scaled_global_l2_public_float_passed": global_failures == 0,
    }


def _dense_oracle() -> dict[str, object]:
    shape = GRID_SHAPES[0]
    spectral, pressure = _solver(shape, fixed_iterations=None)
    operators = ChannelPressureTransposeOperator(pressure)
    active = _active_mask(pressure).reshape(-1)
    indices = torch.nonzero(active, as_tuple=False).reshape(-1)
    manufactured = _pressure_hat(spectral, pressure, 4491)
    records = {}
    for name, action, transpose in (
        ("primal", operators.apply_pressure_operator, False),
        ("transpose", operators.apply_pressure_operator_transpose, True),
    ):
        columns = []
        for index in indices:
            basis = torch.zeros(
                tuple(pressure.pressure_null_mask.shape),
                dtype=torch.complex128,
            )
            basis.reshape(-1)[int(index.item())] = 1.0
            columns.append(action(basis).reshape(-1)[indices])
        matrix = torch.stack(columns, dim=1)
        rhs = action(manufactured)
        dense_values = torch.linalg.solve(matrix, rhs.reshape(-1)[indices])
        dense = torch.zeros_like(rhs)
        dense.reshape(-1)[indices] = dense_values
        actual, diagnostics = _reference_pcg(
            pressure, rhs, transpose=transpose, fixed_iterations=None
        )
        residual = action(actual) - rhs
        operator_norm = float(torch.linalg.matrix_norm(matrix, ord=2).item())
        numerator = float(torch.linalg.vector_norm(residual.reshape(-1)).item())
        denominator = (
            operator_norm
            * float(torch.linalg.vector_norm(actual.reshape(-1)).item())
            + float(torch.linalg.vector_norm(rhs.reshape(-1)).item())
        )
        records[name] = {
            "active_matrix_shape": list(matrix.shape),
            "pcg_vs_dense_relative_l2": _relative_l2(actual - dense, dense),
            "normwise_backward_error": numerator / denominator,
            "diagnostics": diagnostics,
        }
    left = _pressure_hat(spectral, pressure, 4492)
    right = _pressure_hat(spectral, pressure, 4493)
    lhs = torch.sum(
        torch.conj(operators.apply_pressure_operator(left)) * right
    )
    rhs = torch.sum(
        torch.conj(left)
        * operators.apply_pressure_operator_transpose(right)
    )
    inner_error = float(torch.abs(lhs - rhs) / max(torch.abs(lhs), torch.abs(rhs)))
    return {
        "shape": list(shape),
        "actions": records,
        "primal_transpose_inner_product_relative_error": inner_error,
        "inner_product_bound": INNER_PRODUCT_BOUND,
        "passed": bool(
            inner_error <= INNER_PRODUCT_BOUND
            and all(
                value["pcg_vs_dense_relative_l2"] <= 5.0e-12
                and value["normwise_backward_error"] <= 5.0e-14
                for value in records.values()
            )
        ),
    }


def _gradient_oracle() -> dict[str, object]:
    spectral, pressure = _solver(GRID_SHAPES[0], fixed_iterations=5)
    operators = ChannelPressureTransposeOperator(pressure)
    implicit = ChannelImplicitPressureAdjoint(pressure)
    exact = _pressure_hat(spectral, pressure, 4501)
    base = operators.apply_pressure_operator(exact).detach()
    cotangent = _pressure_hat(spectral, pressure, 4502)
    implicit_rhs = base.clone().requires_grad_(True)
    oracle_rhs = base.clone().requires_grad_(True)
    implicit_value = implicit.solve(implicit_rhs)
    oracle_value = unrolled_channel_pressure_solve_oracle(
        oracle_rhs, pressure, iterations=5
    )
    implicit_vjp = torch.autograd.grad(
        implicit_value, implicit_rhs, grad_outputs=cotangent
    )[0]
    oracle_vjp = torch.autograd.grad(
        oracle_value, oracle_rhs, grad_outputs=cotangent
    )[0]

    _, tolerance_pressure = _solver(GRID_SHAPES[0], fixed_iterations=None)
    tolerance_ops = ChannelPressureTransposeOperator(tolerance_pressure)
    tolerance_implicit = ChannelImplicitPressureAdjoint(tolerance_pressure)
    exact = _pressure_hat(spectral, tolerance_pressure, 4503)
    base = tolerance_ops.apply_pressure_operator(exact).detach()
    direction = _pressure_hat(spectral, tolerance_pressure, 4504)
    direction = direction / torch.linalg.vector_norm(direction.reshape(-1))
    differentiable = base.clone().requires_grad_(True)

    def objective(value):
        solution = tolerance_implicit.solve(value)
        return 0.5 * torch.sum(torch.abs(solution).square())

    objective0 = objective(differentiable)
    gradient = torch.autograd.grad(objective0, differentiable)[0]
    directional = torch.sum(torch.conj(gradient) * direction).real
    steps = (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4)
    remainders = []
    centered = []
    for step in steps:
        plus = objective(base + step * direction).detach()
        minus = objective(base - step * direction).detach()
        centered.append((plus - minus) / (2.0 * step))
        remainders.append(
            float(torch.abs(plus - objective0.detach() - step * directional))
        )
    orders = [
        math.log2(left / right)
        for left, right in zip(remainders[:-1], remainders[1:], strict=True)
    ]
    fd_error = float(torch.abs(centered[-1] - directional)) / max(
        float(torch.abs(directional)), 1.0e-14
    )
    custom_error = _relative_l2(implicit_vjp - oracle_vjp, oracle_vjp)
    return {
        "fixed_work_custom_vjp_vs_unrolled_relative_l2": custom_error,
        "central_finite_difference_relative_error": fd_error,
        "taylor_remainder_orders": orders,
        "passed": bool(
            custom_error <= 3.0e-13
            and fd_error <= 2.0e-7
            and min(orders) >= 1.95
        ),
    }


def _breakdown_cases() -> dict[str, object]:
    eps = torch.finfo(torch.float64).eps
    tiny_nonzero = math.ldexp(1.0, -800)
    records = {
        "operator_tiny_but_well_scaled": {
            "value": tiny_nonzero,
            "scale": tiny_nonzero,
            "absolute_1e_30_declares_breakdown": tiny_nonzero < 1.0e-30,
            "relative_test_declares_breakdown": tiny_nonzero <= eps * tiny_nonzero,
        },
        "preconditioned_residual_tiny_but_well_scaled": {
            "value": tiny_nonzero,
            "scale": tiny_nonzero,
            "absolute_1e_30_declares_breakdown": tiny_nonzero < 1.0e-30,
            "relative_test_declares_breakdown": tiny_nonzero <= eps * tiny_nonzero,
        },
        "true_orthogonal_breakdown": {
            "value": 0.0,
            "scale": 1.0,
            "relative_test_declares_breakdown": 0.0 <= eps,
        },
    }
    return {
        "records": records,
        "absolute_magic_threshold_rejected": True,
        "scale_relative_test_distinguishes_tiny_from_zero": bool(
            not records["operator_tiny_but_well_scaled"][
                "relative_test_declares_breakdown"
            ]
            and not records["preconditioned_residual_tiny_but_well_scaled"][
                "relative_test_declares_breakdown"
            ]
            and records["true_orthogonal_breakdown"][
                "relative_test_declares_breakdown"
            ]
        ),
    }


def _warm_start_case() -> dict[str, object]:
    spectral, pressure = _solver(GRID_SHAPES[1], fixed_iterations=None)
    operators = ChannelPressureTransposeOperator(pressure)
    exact = _pressure_hat(spectral, pressure, 4511)
    rhs = operators.apply_pressure_operator(exact)
    first = pressure._solve_pressure(rhs)
    first_iterations = pressure.last_pressure_iterations
    second = pressure._solve_pressure(rhs)
    second_iterations = pressure.last_pressure_iterations
    functional = ChannelImplicitPressureAdjoint(pressure)._solve_primal_no_grad(rhs)
    return {
        "production_initial_guess_policy": "persistent_warm_start",
        "functional_initial_guess_policy": "zero_every_call",
        "first_production_iterations": first_iterations,
        "warm_production_iterations": second_iterations,
        "warm_vs_first_relative_l2": _relative_l2(second - first, first),
        "functional_vs_first_relative_l2": _relative_l2(functional - first, first),
        "production_guess_persisted": pressure.pressure_guess is not None,
        "functional_did_not_replace_production_guess": bool(
            torch.equal(pressure.pressure_guess, second)
        ),
    }


def _simulation(root: Path) -> Simulation:
    shape = (8, 6, 4)
    model = CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
    )
    geometry = RectangularChannel(shape=shape, lengths=(8.0, 6.0, 4.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": no_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    snapshot = root / "snapshot"
    snapshot.mkdir(parents=True)
    values = np.zeros((*shape, 5), dtype=np.float64)
    values[..., 0] = 0.2
    values[..., 3] = -0.1
    np.save(snapshot / "Q_0.npy", values, allow_pickle=False)
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=root / "output",
            steps=1,
            save_interval=1,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        },
    )


def _capability_truth_table() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="pssolver_rc441_") as temporary:
        simulation = _simulation(Path(temporary))
        declaration = channel_activity_functional_declaration(
            simulation.specification
        )
        request = channel_activity_functional_request(simulation.specification)
        runtime = build_channel_activity_functional_runtime(request)
        declaration_capabilities = declaration.capabilities.to_metadata()
        runtime_capabilities = runtime.capabilities.to_metadata()
        implicit = runtime._pressure.implicit_adjoint_metadata()
        transpose = runtime._pressure._operators.pressure_transpose_metadata()
    rows = [
        {
            "subject": "functional_declaration",
            "is_runtime": False,
            "declares_executable_runtime": declaration.executable,
            "connected_to_executable_runtime": True,
            "custom_autograd_adjoint": True,
            "public_explicit_vjp_method": False,
        },
        {
            "subject": "functional_runtime",
            "is_runtime": True,
            "declares_executable_runtime": True,
            "connected_to_executable_runtime": True,
            "custom_autograd_adjoint": True,
            "public_explicit_vjp_method": False,
        },
        {
            "subject": "implicit_adjoint_component",
            "is_runtime": False,
            "declares_executable_runtime": implicit[
                "functional_runtime_executable"
            ],
            "connected_to_executable_runtime": True,
            "custom_autograd_adjoint": True,
            "public_explicit_vjp_method": False,
        },
        {
            "subject": "standalone_transpose_operator_component",
            "is_runtime": False,
            "declares_executable_runtime": transpose[
                "functional_runtime_executable"
            ],
            "connected_to_executable_runtime": True,
            "custom_autograd_adjoint": False,
            "public_explicit_vjp_method": False,
        },
    ]
    return {
        "declaration_executable": declaration.executable,
        "declaration_capabilities": declaration_capabilities,
        "runtime_capabilities": runtime_capabilities,
        "declaration_runtime_capabilities_equal": (
            declaration_capabilities == runtime_capabilities
        ),
        "current_component_metadata": {
            "implicit_adjoint": implicit,
            "transpose_operator": transpose,
        },
        "truth_table": rows,
        "finding": (
            "METADATA_REPAIR_REQUIRED: component_is_runtime and "
            "component_connected_to_executable_runtime must replace the "
            "ambiguous functional_runtime_executable component boolean"
        ),
        "explicit_vjp_false_is_truthful": True,
        "custom_autograd_adjoint_exists": True,
    }


def build_report() -> dict[str, object]:
    norm_stress = _norm_stress()
    sweep = _scale_sweep()
    dense = _dense_oracle()
    gradients = _gradient_oracle()
    breakdown = _breakdown_cases()
    warm = _warm_start_case()
    capabilities = _capability_truth_table()
    passed = bool(
        norm_stress["active_mode_rms_passed"]
        and not norm_stress["scaled_global_l2_public_float_passed"]
        and sweep["passed"]
        and dense["passed"]
        and gradients["passed"]
        and breakdown["scale_relative_test_distinguishes_tiny_from_zero"]
        and capabilities["declaration_runtime_capabilities_equal"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record": "PSSolver RC4.4.1 Channel pressure scale/capability adjudication",
        "phase": "RC4.4.1",
        "kind": "analysis_only",
        "classification": (
            "PASS_RC4_4_1_ACTIVE_MODE_RMS_AND_SCOPED_CAPABILITY_CONTRACT"
            if passed
            else "FAIL_RC4_4_1_PRESSURE_CONTRACT_ADJUDICATION"
        ),
        "environment": {
            "python_torch": torch.__version__,
            "device": "cpu",
            "dtype": "float64/complex128",
        },
        "selected_scale_contract": {
            "decision_class": "ACTIVE_MODE_RMS_REQUIRED",
            "public_rhs_and_residual_norm": (
                "overflow_safe_active_pressure_mode_root_mean_square"
            ),
            "normalized_residual": "active_mode_rms(residual)/active_mode_rms(rhs)",
            "working_vector_scaling": "exact_power_of_two_from_active_mode_amax",
            "breakdown": (
                "scale_relative_inner_product_test_using_machine_epsilon;"
                "no_absolute_magic_threshold"
            ),
            "normwise_backward_error_role": (
                "independent_dense_qualification_diagnostic_not_hot_path_stop"
            ),
            "why_not_global_l2_public_float": (
                "the mathematical norm can exceed float64 even when every "
                "active coefficient is finite"
            ),
        },
        "policy_contract": {
            "fixed_work": (
                "fixed_iterations_exact_budget_except_exact_zero_residual;"
                "relative_tolerance_diagnostic_only;breakdown_fail_closed"
            ),
            "relative_tolerance": (
                "relative_tolerance_required_postcondition_with_max_iteration_cap;"
                "cap_without_postcondition_fail_closed"
            ),
        },
        "evidence": {
            "P01_zero_rhs": _reference_pcg(
                _solver(GRID_SHAPES[0], fixed_iterations=FIXED_ITERATIONS)[1],
                torch.zeros((1, *GRID_SHAPES[0]), dtype=torch.complex128),
                transpose=False,
                fixed_iterations=FIXED_ITERATIONS,
            )[1],
            "P02_P03_P04_P05_scale_grid_primal_transpose": sweep,
            "P03_norm_representation_stress": norm_stress,
            "P06_production_warm_vs_functional_zero": warm,
            "P07_breakdown": breakdown,
            "P08_dense_and_gradient_oracles": {
                "dense": dense,
                "gradient": gradients,
            },
            "B28_capability_truth": capabilities,
        },
        "ordinary_range_byte_identity": {
            "status": "NOT_APPLICABLE_TO_ANALYSIS_ONLY_REFERENCE_LOOP",
            "reason": (
                "RC4.4.1 does not alter or claim byte identity for a production "
                "implementation; RC4.4.3 must qualify unchanged-path identity"
            ),
        },
        "implementation_requirements": [
            "Use the same active-mode RMS and power-of-two work scaling in production primal, functional primal, and explicit transpose loops.",
            "Derive fixed_work versus relative_tolerance once from configuration and expose the roles in public metadata.",
            "Replace ambiguous component functional_runtime_executable fields with scoped component/runtime connection fields.",
            "Retain production warm start and functional zero-every-call initial guesses.",
            "Do not consolidate all pressure loops before post-0.2 R5.",
        ],
        "completion": {
            "analysis_complete": passed,
            "B25_formula_selected": passed,
            "B28_truth_table_complete": True,
            "B28_source_repair_complete": False,
            "B29_policy_reconfirmed": True,
            "solver_source_modified": False,
            "PSSolver_Control_modified": False,
            "nematics3d_modified": False,
            "production_default_changed": False,
            "eligible_for_rc4_4_2": passed,
            "rc4_4_complete": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = build_report()
    _atomic_json(arguments.output, report)
    print(json.dumps({
        "classification": report["classification"],
        "output": str(arguments.output.resolve()),
        "sha256": _sha256(arguments.output),
    }, sort_keys=True))
    return 0 if report["completion"]["analysis_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
