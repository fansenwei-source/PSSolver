#!/usr/bin/env python3
"""RC4.4.7 candidate Channel-pressure correctness qualification.

This qualification-only runner imports PSSolver and PSSolver-Control from an
installed environment.  A frozen Consumer source archive contributes only the
``tests/channel_runtime.py`` declaration helper; it never shadows either
installed package.  Results are written after every gate and the final report
is emitted only when every frozen gate passes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any

import torch


SCHEMA = "pssolver.rc4_4_7.channel_pressure_cuda.v1"
GATE_SCHEMA = "pssolver.rc4_4_7.channel_pressure_cuda.gate.v1"
PASS = "PASS_RC4_4_7_CHANNEL_PRESSURE_CUDA_CORRECTNESS"
PROVIDER_COMMIT = "5b4ee16c81a3ae452ffcf8992a84a738a76ff629"
CONSUMER_SOURCE_COMMIT = "0cf55b4fb699ff40433000f5b4efe52b1d33546b"
PROVIDER_WHEEL_SHA256 = "6f38996c31586ff7b2a0637de5ff479410bbd01bb53926828071affaf5d4cf26"
CONSUMER_WHEEL_SHA256 = "068dc2f3d3be5b8eee96169711878a71bb4e513b452cdcc2fa54dd67c124b96d"
SCALE_SHAPES = ((7, 6, 5), (8, 6, 4), (9, 8, 6))
SCALE_EXPONENTS = (-600, 0, 600)
SCALE_RELATIVE_L2_MAX = 2.0e-11
TRANSPOSE_RELATIVE_ERROR_MAX = 2.0e-13
CUSTOM_VJP_RELATIVE_L2_MAX = 3.0e-13
FD_RELATIVE_ERROR_MAX = 2.0e-5
CHECKPOINT_GRADIENT_RELATIVE_L2_MAX = 1.0e-12


def _check_finite_json(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite value at {path}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string key at {path}")
            _check_finite_json(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_finite_json(item, f"{path}[{index}]")
        return
    raise ValueError(f"non-JSON value at {path}: {type(value).__name__}")


def _atomic_json(path: Path, value: object) -> object:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_finite_json(value)
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    expected = json.loads(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if loaded != expected:
            raise RuntimeError(f"read-back of {path} differs")
        return loaded
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(expected.reshape(-1))), 1.0e-300)
    return float(torch.linalg.vector_norm((actual - expected).reshape(-1))) / denominator


def _tensor_sha256(values: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state_equal(left, right) -> bool:
    return len(left) == len(right) and all(
        torch.equal(a, b) for a, b in zip(left, right, strict=True)
    )


def _fields_byte_identical(left, right) -> bool:
    """Compare public ``Fields`` mappings without assuming they are iterable."""

    left_names = tuple(left.keys())
    right_names = tuple(right.keys())
    if len(left_names) != len(right_names) or set(left_names) != set(right_names):
        return False
    return all(torch.equal(left[name], right[name]) for name in left_names)


def _all_finite(values) -> bool:
    return all(bool(torch.isfinite(value).all().item()) for value in values)


@contextlib.contextmanager
def _device_override(module, device: str):
    real = module.TorchSpectralExecution

    def wrapped(*args, **kwargs):
        kwargs["device"] = device
        return real(*args, **kwargs)

    module.TorchSpectralExecution = wrapped
    try:
        yield
    finally:
        module.TorchSpectralExecution = real


def _load_channel_runtime(root: Path):
    path = (root / "tests" / "channel_runtime.py").resolve()
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise FileNotFoundError("frozen Consumer channel_runtime.py is missing")
    name = f"_rc447_channel_runtime_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError("Consumer helper import identity differs")
    return module


class Gate:
    def __init__(self, output: Path, name: str):
        self.path = output / f"gate_{name}.json"
        self.record: dict[str, Any] = {
            "schema": GATE_SCHEMA,
            "gate": name,
            "passed": False,
            "checks": [],
            "observations": {},
        }

    def check(self, name: str, passed: bool, observed=None, limit=None) -> None:
        item = {"name": name, "passed": bool(passed)}
        if observed is not None:
            item["observed"] = observed
        if limit is not None:
            item["limit"] = limit
        self.record["checks"].append(item)

    def observe(self, name: str, value: object) -> None:
        self.record["observations"][name] = value

    def finish(self) -> dict[str, Any]:
        if not self.record["checks"]:
            raise RuntimeError(f"gate {self.record['gate']} has no checks")
        self.record["passed"] = all(item["passed"] for item in self.record["checks"])
        return _atomic_json(self.path, self.record)


def _pressure_solver(shape, *, device: str, mode: str):
    from pssolver import SpectralSolver
    from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver

    spectral = SpectralSolver(
        shape,
        L=tuple(float(value) for value in shape),
        device=device,
        dtype=torch.float64,
    )
    pressure = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=1.0e-12,
        pressure_max_iterations=400,
        pressure_fixed_iterations=12 if mode == "fixed_work" else None,
    )
    return spectral, pressure


def _pressure_rhs(spectral, pressure, seed: int):
    from pssolver.functional import ChannelPressureTransposeOperator
    from pssolver.linear_solvers.stokes import CHANNEL_PRESSURE_BOUNDARY_CONDITIONS

    generator = torch.Generator().manual_seed(seed)
    physical = torch.randn(
        (1, *tuple(spectral.shape)),
        generator=generator,
        dtype=torch.float64,
    ).to(spectral.device)
    exact = spectral.transform_tensor(physical, CHANNEL_PRESSURE_BOUNDARY_CONDITIONS)
    exact = pressure._project_pressure_gauge(exact)
    return ChannelPressureTransposeOperator(pressure).apply_pressure_operator(exact)


def gate_environment(output: Path) -> dict[str, Any]:
    import pssolver
    import pssolver_control

    gate = Gate(output, "environment")
    before = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    after = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    allocated = str(torch.empty((), device="cuda").device) if torch.cuda.is_available() else None
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    gate.observe(
        "environment",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "pssolver_import": str(Path(pssolver.__file__).resolve()),
            "pssolver_control_import": str(Path(pssolver_control.__file__).resolve()),
            "cuda_available": bool(torch.cuda.is_available()),
            "visible_device_count": int(torch.cuda.device_count()),
            "allocated_device": allocated,
            "device_name": device_name,
            "tf32_before_policy": before,
            "tf32_after_policy": after,
        },
    )
    gate.check("CUDA available", torch.cuda.is_available())
    gate.check("one visible CUDA device", torch.cuda.device_count() == 1, torch.cuda.device_count(), 1)
    gate.check("allocated device", allocated == "cuda:0", allocated, "cuda:0")
    gate.check("H100", "H100" in str(device_name), device_name)
    gate.check("TF32 matmul off", after["matmul"] is False, after["matmul"])
    gate.check("TF32 cuDNN off", after["cudnn"] is False, after["cudnn"])
    return gate.finish()


def gate_scale_and_transpose(output: Path) -> dict[str, Any]:
    from pssolver.functional import (
        ChannelImplicitPressureAdjoint,
        ChannelPressureTransposeOperator,
    )
    from pssolver.linear_solvers.stokes.channel_no_slip import (
        channel_pressure_scale_power_of_two,
    )

    gate = Gate(output, "scale_transpose")
    scale_rows = []
    for shape_index, shape in enumerate(SCALE_SHAPES):
        for mode in ("fixed_work", "relative_tolerance"):
            spectral, pressure = _pressure_solver(shape, device="cuda", mode=mode)
            rhs = _pressure_rhs(spectral, pressure, 44700 + shape_index)
            actions = ChannelPressureTransposeOperator(pressure)
            results: dict[str, dict[int, torch.Tensor]] = {
                "production_primal": {},
                "functional_primal": {},
                "functional_transpose": {},
            }
            diagnostics: dict[str, dict[int, object]] = {name: {} for name in results}
            for exponent in SCALE_EXPONENTS:
                scaled = channel_pressure_scale_power_of_two(rhs, exponent)
                pressure.pressure_guess = None
                results["production_primal"][exponent] = pressure._solve_pressure(scaled)
                diagnostics["production_primal"][exponent] = dict(pressure.last_pressure_diagnostics)
                implicit = ChannelImplicitPressureAdjoint(pressure)
                results["functional_primal"][exponent] = implicit.solve(scaled)
                diagnostics["functional_primal"][exponent] = implicit.last_primal_diagnostics.to_metadata()
                results["functional_transpose"][exponent] = actions.solve_pressure_transpose(scaled)
                diagnostics["functional_transpose"][exponent] = actions.last_transpose_diagnostics.to_metadata()
            for action, values in results.items():
                reference = values[0]
                for exponent in SCALE_EXPONENTS:
                    unscaled = channel_pressure_scale_power_of_two(values[exponent], -exponent)
                    relative = _relative_l2(unscaled, reference)
                    metadata = diagnostics[action][exponent]
                    gate.check(
                        f"{shape}:{mode}:{action}:{exponent}:scale covariance",
                        relative <= SCALE_RELATIVE_L2_MAX,
                        relative,
                        SCALE_RELATIVE_L2_MAX,
                    )
                    gate.check(
                        f"{shape}:{mode}:{action}:{exponent}:acceptable",
                        metadata["acceptable"] is True,
                        metadata["termination_reason"],
                    )
                    gate.check(
                        f"{shape}:{mode}:{action}:{exponent}:finite",
                        bool(torch.isfinite(values[exponent]).all().item()),
                    )
                    scale_rows.append(
                        {
                            "shape": list(shape),
                            "mode": mode,
                            "action": action,
                            "exponent": exponent,
                            "relative_l2_after_unscale": relative,
                            "iterations": metadata["achieved_iteration_count"],
                            "termination_reason": metadata["termination_reason"],
                        }
                    )
    spectral, pressure = _pressure_solver((8, 6, 4), device="cuda", mode="relative_tolerance")
    actions = ChannelPressureTransposeOperator(pressure)
    generator = torch.Generator().manual_seed(44791)
    left = torch.complex(
        torch.randn((1, 8, 6, 4), generator=generator, dtype=torch.float64),
        torch.randn((1, 8, 6, 4), generator=generator, dtype=torch.float64),
    ).to("cuda")
    right = torch.complex(
        torch.randn((1, 8, 6, 4), generator=generator, dtype=torch.float64),
        torch.randn((1, 8, 6, 4), generator=generator, dtype=torch.float64),
    ).to("cuda")
    primal = actions.apply_pressure_operator(left)
    transpose = actions.apply_pressure_operator_transpose(right)
    lhs = torch.sum(torch.conj(primal) * right)
    rhs_value = torch.sum(torch.conj(left) * transpose)
    transpose_error = float(torch.abs(lhs - rhs_value)) / max(
        float(torch.abs(lhs)), float(torch.abs(rhs_value)), 1.0e-300
    )
    gate.check(
        "primal-transpose inner product",
        transpose_error <= TRANSPOSE_RELATIVE_ERROR_MAX,
        transpose_error,
        TRANSPOSE_RELATIVE_ERROR_MAX,
    )
    gate.observe("scale_matrix", scale_rows)
    gate.observe("transpose_relative_error", transpose_error)
    return gate.finish()


def _cpu_unrolled_oracle() -> dict[str, object]:
    from pssolver.functional import (
        ChannelImplicitPressureAdjoint,
        ChannelPressureTransposeOperator,
        unrolled_channel_pressure_solve_oracle,
    )

    spectral, pressure = _pressure_solver((3, 2, 2), device="cpu", mode="fixed_work")
    pressure.pressure_fixed_iterations = 5
    rhs = _pressure_rhs(spectral, pressure, 44732).detach()
    generator = torch.Generator().manual_seed(44733)
    cotangent = torch.complex(
        torch.randn(rhs.shape, generator=generator, dtype=torch.float64),
        torch.randn(rhs.shape, generator=generator, dtype=torch.float64),
    )
    cotangent = ChannelPressureTransposeOperator(pressure)._project_gauge(cotangent)
    implicit_rhs = rhs.clone().requires_grad_(True)
    oracle_rhs = rhs.clone().requires_grad_(True)
    implicit_solution = ChannelImplicitPressureAdjoint(pressure).solve(implicit_rhs)
    oracle_solution = unrolled_channel_pressure_solve_oracle(
        oracle_rhs, pressure, iterations=5
    )
    implicit_vjp = torch.autograd.grad(
        implicit_solution, implicit_rhs, grad_outputs=cotangent
    )[0]
    oracle_vjp = torch.autograd.grad(
        oracle_solution, oracle_rhs, grad_outputs=cotangent
    )[0]
    return {
        "execution_device": "cpu",
        "reason": "public unrolled oracle is intentionally CPU-only",
        "forward_byte_identical": torch.equal(implicit_solution, oracle_solution),
        "vjp_relative_l2": _relative_l2(implicit_vjp, oracle_vjp),
    }


def _consumer_problem(channel_runtime, directory: Path, *, mode: str):
    from pssolver_control import (
        BlockFieldControl,
        ControlProblem,
        DiscreteAdjointLoop,
        QuadraticTrackingObjective,
    )
    from pssolver_control.adapters.pssolver_channel import build_channel_activity_stepper

    solver = None if mode == "fixed_work" else channel_runtime.TOLERANCE_PRESSURE_SOLVER
    with _device_override(channel_runtime, "cuda"):
        simulation = channel_runtime.simulation(directory, pressure_solver=solver)
    stepper = build_channel_activity_stepper(simulation.specification)
    control = BlockFieldControl(
        field_shape=channel_runtime.SHAPE,
        num_steps=4,
        block_size=2,
        dtype=torch.float64,
        device="cuda",
    )
    theta = channel_runtime.activity_blocks().to("cuda")
    initial = stepper.initial_state()
    uniform = control.parameters_from_value(channel_runtime.ACTIVITY)
    with torch.no_grad():
        target = initial
        for step in range(4):
            target = stepper.step(target, control.field_for_step(uniform, step), step)
    objective = QuadraticTrackingObjective(
        target[0],
        control_spec=control.control_spec,
        observation_key="Q",
        terminal_weight=1.0,
        running_weight=1.0,
    )
    problem = ControlProblem(stepper, control, objective, initial)
    loop = DiscreteAdjointLoop(problem, checkpoint_stride=2, path_check="every_step")
    return stepper, problem, loop, theta


def gate_consumer(output: Path, channel_runtime) -> dict[str, Any]:
    from pssolver_control.adapters.pssolver_channel import (
        ADAPTER_SCHEMA,
        ADAPTER_VERSION,
        pressure_diagnostics_violations,
        pressure_identity_violations,
    )
    from pssolver_control.validation import (
        estimate_computational_noise,
        full_history_value_and_gradient,
        gradient_check_suite,
        random_directions,
        resolution_aware_taylor_test,
        taylor_epsilon_ladder,
    )

    gate = Gate(output, "consumer")
    oracle = _cpu_unrolled_oracle()
    gate.check("unrolled forward byte identity", oracle["forward_byte_identical"] is True)
    gate.check(
        "custom VJP versus unrolled oracle",
        oracle["vjp_relative_l2"] <= CUSTOM_VJP_RELATIVE_L2_MAX,
        oracle["vjp_relative_l2"],
        CUSTOM_VJP_RELATIVE_L2_MAX,
    )
    gate.observe("cpu_unrolled_oracle", oracle)
    mode_reports = {}
    for mode in ("fixed_work", "relative_tolerance"):
        work = output / f"consumer_{mode}"
        work.mkdir()
        stepper, problem, loop, theta = _consumer_problem(channel_runtime, work, mode=mode)
        identity = stepper.identity()
        pressure = identity["provider_identity"]["execution"]["functional_runtime"]["pressure_solver"]
        gate.check(f"{mode}:adapter schema", identity["adapter_schema"] == ADAPTER_SCHEMA)
        gate.check(f"{mode}:adapter version", identity["adapter_version"] == ADAPTER_VERSION)
        gate.check(f"{mode}:pressure identity", pressure_identity_violations(pressure) == [])
        before = stepper.pressure_diagnostics()
        gate.check(
            f"{mode}:before diagnostics",
            pressure_diagnostics_violations(before, pressure_identity=pressure) == [],
        )
        result = loop.gradient(theta)
        full_cost, full_gradient = full_history_value_and_gradient(problem, theta)
        gradient_l2 = _relative_l2(result.gradient, full_gradient)
        value = lambda parameters, loop=loop: loop.forward(parameters).cost  # noqa: E731
        fd = gradient_check_suite(
            value,
            theta,
            result.gradient,
            epsilons=(2.5e-4,),
            num_random_directions=2,
            seed=29,
        )
        direction = random_directions(
            theta.shape, 1, seed=31, dtype=torch.float64, device="cuda"
        )[0]
        epsilon_max = 0.5 * float(theta.min()) / float(direction.abs().max())
        ladder = taylor_epsilon_ladder(epsilon_max, 12)
        noise = estimate_computational_noise(value, theta, direction)
        true = resolution_aware_taylor_test(
            value, theta, result.cost, result.gradient, direction, ladder, noise=noise
        )
        flat = result.gradient.reshape(-1)
        wrong = {
            "first_quarter_negated": torch.cat(
                (-flat[: flat.numel() // 4], flat[flat.numel() // 4 :])
            ).reshape(result.gradient.shape),
            "x1.1": result.gradient * 1.1,
            "x1.01": result.gradient * 1.01,
        }
        negative = {
            name: resolution_aware_taylor_test(
                value, theta, result.cost, candidate, direction, ladder, noise=noise
            ).passed
            for name, candidate in wrong.items()
        }
        after = stepper.pressure_diagnostics()
        gate.check(f"{mode}:checkpoint gradient", gradient_l2 <= CHECKPOINT_GRADIENT_RELATIVE_L2_MAX, gradient_l2, CHECKPOINT_GRADIENT_RELATIVE_L2_MAX)
        gate.check(f"{mode}:cost identity", result.cost == full_cost, [result.cost, full_cost])
        gate.check(f"{mode}:finite difference", fd["max_relative_error"] < FD_RELATIVE_ERROR_MAX, fd["max_relative_error"], FD_RELATIVE_ERROR_MAX)
        gate.check(f"{mode}:Taylor pass", true.passed is True, true.adjudication.reason)
        gate.check(f"{mode}:Taylor resolved levels", len(true.adjudication.resolved_indices) >= 3, len(true.adjudication.resolved_indices), 3)
        gate.check(f"{mode}:Taylor R0 slope", 0.8 <= true.adjudication.slope_without_gradient <= 1.2, true.adjudication.slope_without_gradient, [0.8, 1.2])
        gate.check(f"{mode}:Taylor R1 slope", 1.8 <= true.adjudication.slope_with_gradient <= 2.2, true.adjudication.slope_with_gradient, [1.8, 2.2])
        gate.check(f"{mode}:Taylor negatives", all(value is False for value in negative.values()), negative)
        gate.check(f"{mode}:gradient finite", bool(torch.isfinite(result.gradient).all().item()))
        gate.check(
            f"{mode}:after diagnostics",
            pressure_diagnostics_violations(after, pressure_identity=pressure) == [],
        )
        mode_reports[mode] = {
            "identity_sha256": stepper.provider_identity_sha256,
            "cost": result.cost,
            "checkpointed_full_history_relative_l2": gradient_l2,
            "fd_max_relative_error": fd["max_relative_error"],
            "taylor": {
                "floor": true.floor,
                "resolved_levels": len(true.adjudication.resolved_indices),
                "r0_slope": true.adjudication.slope_without_gradient,
                "r1_slope": true.adjudication.slope_with_gradient,
                "negative_controls": negative,
            },
            "pressure_diagnostics": after,
        }
    gate.observe("modes", mode_reports)
    return gate.finish()


def _functional_restart(output: Path, channel_runtime, mode: str) -> dict[str, object]:
    from pssolver_control.adapters.pssolver_channel import build_channel_activity_stepper

    solver = None if mode == "fixed_work" else channel_runtime.TOLERANCE_PRESSURE_SOLVER

    def build(name: str):
        with _device_override(channel_runtime, "cuda"):
            declaration = channel_runtime.simulation(output / name, pressure_solver=solver)
        return build_channel_activity_stepper(declaration.specification)

    continuous = build(f"functional_{mode}_continuous")
    split = build(f"functional_{mode}_split")
    resumed = build(f"functional_{mode}_resumed")
    control = torch.full(
        continuous.control_spec.shape,
        channel_runtime.ACTIVITY,
        dtype=torch.float64,
        device="cuda",
    )
    continuous_state = continuous.initial_state()
    for step in range(4):
        continuous_state = continuous.step(continuous_state, control, step)
    split_state = split.initial_state()
    for step in range(2):
        split_state = split.step(split_state, control, step)
    checkpoint = output / f"functional_{mode}_checkpoint"
    split.checkpoint_bridge.export_checkpoint(checkpoint, split_state, completed_steps=2)
    imported = resumed.checkpoint_bridge.import_checkpoint(checkpoint)
    resumed_state = imported.state
    for step in range(imported.completed_steps, 4):
        resumed_state = resumed.step(resumed_state, control, step)
    return {
        "mode": mode,
        "completed_steps": imported.completed_steps,
        "byte_identical": _state_equal(continuous_state, resumed_state),
        "continuous_sha256": _tensor_sha256(continuous_state),
        "resumed_sha256": _tensor_sha256(resumed_state),
        "finite": _all_finite(resumed_state),
    }


def _production_restart(output: Path, channel_runtime, mode: str) -> dict[str, object]:
    from pssolver import compile_simulation
    from pssolver.applications.channel_beris_edwards import load_channel_initial_q
    from pssolver.runtime.channel_beris_edwards import (
        ChannelBerisEdwardsRuntimeBuildRequest,
        build_channel_beris_edwards_runtime,
    )
    from pssolver.workflows.channel_beris_edwards import (
        _load_checkpoint,
        write_channel_beris_edwards_checkpoint,
    )

    solver = None if mode == "fixed_work" else channel_runtime.TOLERANCE_PRESSURE_SOLVER

    def build(name: str):
        with _device_override(channel_runtime, "cuda"):
            simulation = channel_runtime.simulation(output / name, pressure_solver=solver)
        compiled = compile_simulation(simulation)
        values, _, _ = load_channel_initial_q(compiled.application_request)
        adapter = build_channel_beris_edwards_runtime(
            ChannelBerisEdwardsRuntimeBuildRequest(
                run_spec=compiled.application_request,
                initial_values=values,
                device="cuda",
            )
        )
        return compiled, adapter

    continuous_compiled, continuous = build(f"production_{mode}_continuous")
    split_compiled, split = build(f"production_{mode}_split")
    resumed_compiled, resumed = build(f"production_{mode}_resumed")
    continuous.advance(4)
    split.advance(2)
    checkpoint = output / f"production_{mode}_checkpoint"
    write_channel_beris_edwards_checkpoint(
        checkpoint, split, run_spec=split_compiled.application_request
    )
    _load_checkpoint(checkpoint, resumed, run_spec=resumed_compiled.application_request)
    resumed.advance(2)
    continuous.synchronize_for_observation()
    resumed.synchronize_for_observation()
    continuous_names = tuple(continuous.fields.keys())
    resumed_names = tuple(resumed.fields.keys())
    fields_equal = _fields_byte_identical(continuous.fields, resumed.fields)
    pressure_equal = torch.equal(
        continuous.capture_pressure_guess(), resumed.capture_pressure_guess()
    )
    values = tuple(continuous.fields[name] for name in continuous_names)
    resumed_values = tuple(resumed.fields[name] for name in resumed_names)
    return {
        "mode": mode,
        "continuous_steps": continuous.completed_steps,
        "resumed_steps": resumed.completed_steps,
        "continuous_field_names": list(continuous_names),
        "resumed_field_names": list(resumed_names),
        "fields_byte_identical": fields_equal,
        "pressure_guess_byte_identical": pressure_equal,
        "continuous_sha256": _tensor_sha256(values),
        "resumed_sha256": _tensor_sha256(resumed_values),
        "finite": _all_finite(resumed_values),
        "runtime_path": str(continuous_compiled.application_request.runtime_path),
    }


def gate_restart_and_negative(
    output: Path,
    channel_runtime,
    provider_test_root: Path,
) -> dict[str, Any]:
    gate = Gate(output, "restart_negative")
    restart = {"functional": {}, "production": {}}
    for mode in ("fixed_work", "relative_tolerance"):
        functional = _functional_restart(output, channel_runtime, mode)
        production = _production_restart(output, channel_runtime, mode)
        restart["functional"][mode] = functional
        restart["production"][mode] = production
        gate.check(f"functional {mode} byte identity", functional["byte_identical"] is True)
        gate.check(f"functional {mode} finite", functional["finite"] is True)
        gate.check(f"production {mode} field identity", production["fields_byte_identical"] is True)
        gate.check(f"production {mode} pressure identity", production["pressure_guess_byte_identical"] is True)
        gate.check(f"production {mode} finite", production["finite"] is True)
    gate.observe("same_version_restart", restart)

    test_file = (provider_test_root / "tests" / "test_rc4_pressure_metadata_identity_connection.py").resolve()
    if not test_file.is_file() or not test_file.is_relative_to(provider_test_root.resolve()):
        raise FileNotFoundError("Provider negative-gate test file is missing")
    work = output / "negative_pytest"
    work.mkdir()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--import-mode=importlib",
        f"--rootdir={work}",
        str(test_file),
        "-k",
        "stale_or_tampered_identity_rejects_before_mutation or functional_stale_or_tampered_identity_rejects_before_tensor_load or production_v3_import_to_functional_rejects_before_tensor_load",
    ]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PSSOLVER_REFERENCE_ROOT"}
    }
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(
        command,
        cwd=work,
        env=env,
        text=True,
        capture_output=True,
        timeout=1800,
    )
    negative = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "expected_cases": 6,
    }
    gate.observe("negative_identity_rejection", negative)
    gate.check("six negative identity cases pass", completed.returncode == 0 and "6 passed" in completed.stdout, negative)
    return gate.finish()


def _graph_breaks() -> int:
    try:
        from torch._dynamo.utils import counters
    except Exception:
        return 0
    value = counters.get("graph_break", {})
    if isinstance(value, dict):
        return sum(int(item) for item in value.values())
    return 0


def run(output: Path, consumer_root: Path, provider_test_root: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    channel_runtime = _load_channel_runtime(consumer_root)
    gates: dict[str, dict[str, Any]] = {}
    try:
        gates["environment"] = gate_environment(output)
        if not gates["environment"]["passed"]:
            raise RuntimeError("environment gate failed")
        gates["scale_transpose"] = gate_scale_and_transpose(output)
        if not gates["scale_transpose"]["passed"]:
            raise RuntimeError("scale/transpose gate failed")
        gates["consumer"] = gate_consumer(output, channel_runtime)
        if not gates["consumer"]["passed"]:
            raise RuntimeError("consumer gate failed")
        gates["restart_negative"] = gate_restart_and_negative(
            output, channel_runtime, provider_test_root
        )
        if not gates["restart_negative"]["passed"]:
            raise RuntimeError("restart/negative gate failed")
    except BaseException as error:  # noqa: BLE001
        _atomic_json(
            output / "failure.json",
            {
                "schema": SCHEMA,
                "classification": "FAIL_RC4_4_7_CHANNEL_PRESSURE_CUDA_CORRECTNESS",
                "error": "".join(traceback.format_exception(error))[-12000:],
                "completed_gates": sorted(gates),
            },
        )
        raise
    consumer_modes = gates["consumer"]["observations"]["modes"]
    restart = gates["restart_negative"]["observations"]["same_version_restart"]
    public_identity = all(
        consumer_modes[mode]["identity_sha256"]
        for mode in ("fixed_work", "relative_tolerance")
    )
    final_gates = {
        "public_identity": {"passed": bool(public_identity)},
        "scale_matrix": {"passed": gates["scale_transpose"]["passed"]},
        "transpose_identity": {"passed": gates["scale_transpose"]["passed"]},
        "custom_adjoint": {"passed": gates["consumer"]["passed"]},
        "finite_difference": {"passed": gates["consumer"]["passed"]},
        "resolution_aware_taylor": {"passed": gates["consumer"]["passed"]},
        "checkpointed_full_history": {"passed": gates["consumer"]["passed"]},
        "same_version_restart": {
            "passed": all(
                value["finite"]
                and (
                    value.get("byte_identical") is True
                    or (
                        value.get("fields_byte_identical") is True
                        and value.get("pressure_guess_byte_identical") is True
                    )
                )
                for family in restart.values()
                for value in family.values()
            )
        },
        "negative_identity_rejection": {
            "passed": gates["restart_negative"]["passed"]
        },
    }
    report = {
        "schema": SCHEMA,
        "classification": PASS,
        "qualification_complete": all(item["passed"] for item in final_gates.values()),
        "environment": gates["environment"]["observations"]["environment"],
        "identities": {
            "provider_commit": PROVIDER_COMMIT,
            "provider_wheel_sha256": PROVIDER_WHEEL_SHA256,
            "consumer_source_commit": CONSUMER_SOURCE_COMMIT,
            "consumer_wheel_sha256": CONSUMER_WHEEL_SHA256,
        },
        "gates": final_gates,
        "gate_files": {name: f"gate_{name}.json" for name in gates},
        "runtime_fallback_used": False,
        "compile_fallback_used": False,
        "compile_execution_requested": False,
        "graph_breaks": _graph_breaks(),
        "production_default_changed": False,
        "larger_batch_qualified": False,
        "optimizer_campaign_authorized": False,
        "scientific_control_result": False,
    }
    if not report["qualification_complete"]:
        raise RuntimeError("one or more final correctness gates failed")
    return _atomic_json(output / "report.json", report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--consumer-root", type=Path, required=True)
    parser.add_argument("--provider-test-root", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.output.resolve(), args.consumer_root.resolve(), args.provider_test_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
