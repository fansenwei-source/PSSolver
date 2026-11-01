#!/usr/bin/env python3
"""Compare the four Channel pressure PCG paths byte for byte.

The child process records returned tensors, public diagnostics, pressure warm
state, operator-call counts, and the ordered Torch-dispatch trace.  The parent
runs the same matrix from two source trees and requires exact record equality.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

try:
    from benchmarks._artifact_io import (
        atomic_write_json,
        atomic_write_text,
        git_identity,
        runtime_environment,
    )
except ModuleNotFoundError:  # direct execution sets sys.path[0] to benchmarks/
    from _artifact_io import (  # type: ignore[no-redef]
        atomic_write_json,
        atomic_write_text,
        git_identity,
        runtime_environment,
    )


SCHEMA = "pssolver.v0_4.s18.r5.pcg_equivalence.v1"
CHILD_SCHEMA = "pssolver.v0_4.s18.r5.pcg_equivalence_child.v1"
PATHS = ("production", "transpose", "implicit", "oracle")
CASES = (
    {
        "name": "tolerance_float64",
        "shape": (3, 2, 2),
        "dtype": "float64",
        "fixed_iterations": None,
        "max_iterations": 100,
        "relative_tolerance": 1.0e-13,
        "friction": 0.2,
        "viscosity": 0.73,
        "length_scale": 1.0,
        "seed": 109,
    },
    {
        "name": "fixed_float64",
        "shape": (3, 2, 2),
        "dtype": "float64",
        "fixed_iterations": 5,
        "max_iterations": 100,
        "relative_tolerance": 1.0e-13,
        "friction": 0.2,
        "viscosity": 0.73,
        "length_scale": 1.0,
        "seed": 131,
    },
    {
        "name": "fixed_float32_roundoff_floor",
        "shape": (8, 6, 4),
        "dtype": "float32",
        "fixed_iterations": 50,
        "max_iterations": 50,
        "relative_tolerance": 1.0e-6,
        "friction": 0.2,
        "viscosity": 0.73,
        "length_scale": 1.0,
        "seed": 9,
    },
    {
        "name": "fixed_float32_floor_crossing",
        "shape": (12, 8, 32),
        "dtype": "float32",
        "fixed_iterations": 200,
        "max_iterations": 200,
        "relative_tolerance": 1.0e-6,
        "friction": 1.0e-3,
        "viscosity": 0.73,
        "length_scale": 8.0,
        "seed": 0,
    },
)


def _tensor_record(value: object) -> dict[str, object]:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("pressure result must be a tensor")
    tensor = value.detach().contiguous().cpu()
    payload = tensor.numpy().tobytes(order="C")
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "finite": bool(tensor.isfinite().all().item()),
    }


class _DispatchTrace:
    """Collect an ordered Torch-operation digest without retaining tensors."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def mode(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        trace = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(
                self,
                func,
                types,
                args=(),
                kwargs=None,
            ):
                trace.events.append(str(func))
                return func(*args, **(kwargs or {}))

        return _Mode()

    def record(self) -> dict[str, object]:
        payload = "\n".join(self.events).encode("utf-8")
        counts = Counter(self.events)
        return {
            "count": len(self.events),
            "ordered_sha256": hashlib.sha256(payload).hexdigest(),
            "operation_counts": dict(sorted(counts.items())),
            "host_scalar_sync_count": sum(
                count
                for name, count in counts.items()
                if "_local_scalar_dense" in name or ".item" in name
            ),
        }


def _build_solver(case: dict[str, object]):
    import torch
    from pssolver import SpectralSolver
    from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver

    shape = tuple(int(value) for value in case["shape"])
    dtype = getattr(torch, str(case["dtype"]))
    length_scale = float(case["length_scale"])
    spectral = SpectralSolver(
        shape,
        L=tuple(length_scale * (1.0 + 0.37 * value) for value in shape),
        device="cpu",
        dtype=dtype,
    )
    pressure = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=float(case["friction"]),
        viscosity=float(case["viscosity"]),
        pressure_relative_tolerance=float(case["relative_tolerance"]),
        pressure_max_iterations=int(case["max_iterations"]),
        pressure_fixed_iterations=(
            None if case["fixed_iterations"] is None else int(case["fixed_iterations"])
        ),
    )
    return spectral, pressure


def _rhs(pressure, case: dict[str, object], *, seed_offset: int = 0):
    import torch
    from pssolver.functional import ChannelPressureTransposeOperator

    shape = tuple(int(value) for value in case["shape"])
    real_dtype = getattr(torch, str(case["dtype"]))
    generator = torch.Generator().manual_seed(int(case["seed"]) + seed_offset)
    real = torch.randn((1, *shape), generator=generator, dtype=real_dtype)
    imag = torch.randn((1, *shape), generator=generator, dtype=real_dtype)
    return ChannelPressureTransposeOperator(pressure)._project_gauge(
        torch.complex(real, imag)
    )


def _diagnostics(value: object) -> object:
    if value is None:
        return None
    to_metadata = getattr(value, "to_metadata", None)
    if callable(to_metadata):
        return to_metadata()
    if isinstance(value, dict):
        return value
    raise TypeError(f"unsupported diagnostics type {type(value)!r}")


def _run_path(case: dict[str, object], path: str) -> dict[str, object]:
    import torch
    import pssolver.functional.channel_pressure_adjoint as adjoint_module
    from pssolver.functional import (
        ChannelImplicitPressureAdjoint,
        ChannelPressureTransposeOperator,
        unrolled_channel_pressure_solve_oracle,
    )
    from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver

    _, pressure = _build_solver(case)
    rhs = _rhs(pressure, case)
    cotangent = _rhs(pressure, case, seed_offset=1000)
    counts = {"primal": 0, "transpose": 0}
    original_primal = ChannelNoSlipModalStokesSolver._pressure_operator
    original_transpose = (
        ChannelPressureTransposeOperator.apply_pressure_operator_transpose
    )

    def counted_primal(instance, value):
        counts["primal"] += 1
        return original_primal(instance, value)

    def counted_transpose(instance, value):
        counts["transpose"] += 1
        return original_transpose(instance, value)

    ChannelNoSlipModalStokesSolver._pressure_operator = counted_primal
    ChannelPressureTransposeOperator.apply_pressure_operator_transpose = (
        counted_transpose
    )
    trace = _DispatchTrace()
    try:
        with trace.mode():
            if path == "production":
                result = pressure._solve_pressure(rhs.clone())
                gradient = None
                diagnostics = {"primal": pressure.last_pressure_diagnostics}
            elif path == "transpose":
                operator = ChannelPressureTransposeOperator(pressure)
                result = operator.solve_pressure_transpose(rhs.clone())
                gradient = None
                diagnostics = {
                    "transpose": _diagnostics(operator.last_transpose_diagnostics)
                }
            elif path == "implicit":
                implicit_rhs = rhs.clone().requires_grad_(True)
                implicit = ChannelImplicitPressureAdjoint(pressure)
                result = implicit.solve(implicit_rhs)
                gradient = torch.autograd.grad(
                    result,
                    implicit_rhs,
                    grad_outputs=cotangent,
                )[0]
                diagnostics = {
                    "primal": _diagnostics(implicit.last_primal_diagnostics),
                    "transpose": _diagnostics(implicit.last_transpose_diagnostics),
                }
            else:
                if case["fixed_iterations"] is None:
                    return {"skipped": "oracle_requires_fixed_work"}
                adjoint_module.CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES = max(
                    adjoint_module.CHANNEL_PRESSURE_UNROLLED_ORACLE_MAX_MODES,
                    rhs.numel(),
                )
                oracle_rhs = rhs.clone().requires_grad_(True)
                result = unrolled_channel_pressure_solve_oracle(
                    oracle_rhs,
                    pressure,
                    iterations=int(case["fixed_iterations"]),
                )
                gradient = torch.autograd.grad(
                    result,
                    oracle_rhs,
                    grad_outputs=cotangent,
                )[0]
                diagnostics = None
    finally:
        ChannelNoSlipModalStokesSolver._pressure_operator = original_primal
        ChannelPressureTransposeOperator.apply_pressure_operator_transpose = (
            original_transpose
        )
    return {
        "result": _tensor_record(result),
        "gradient": None if gradient is None else _tensor_record(gradient),
        "diagnostics": diagnostics,
        "operator_calls": counts,
        "torch_dispatch": trace.record(),
        "warm_pressure": (
            None
            if pressure.pressure_guess is None
            else _tensor_record(pressure.pressure_guess)
        ),
        "warm_operator_action": (
            None
            if pressure._pressure_guess_operator_action is None
            else _tensor_record(pressure._pressure_guess_operator_action)
        ),
    }


def _child(args: argparse.Namespace) -> None:
    import pssolver
    import torch

    records = []
    for case in CASES:
        paths = {path: _run_path(case, path) for path in PATHS}
        records.append({"case": case, "paths": paths})
    imported = Path(pssolver.__file__).resolve()
    source_root = args.source_root.resolve()
    if not imported.is_relative_to(source_root):
        raise RuntimeError(f"pssolver import {imported} is outside {source_root}")
    atomic_write_json(
        args.output,
        {
            "schema": CHILD_SCHEMA,
            "role": args.role,
            "commit": args.commit,
            "source_root": str(source_root),
            "pssolver_import": str(imported),
            "torch_version": torch.__version__,
            "git": git_identity(source_root),
            "environment": runtime_environment(),
            "cases": records,
        },
    )


def _run_child(
    *,
    python: Path,
    script: Path,
    source_root: Path,
    output_root: Path,
    role: str,
    commit: str,
) -> dict[str, object]:
    output = output_root / "children" / f"{role}.json"
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(source_root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    command = [
        str(python),
        str(script),
        "--internal-child",
        "--source-root",
        str(source_root),
        "--role",
        role,
        "--commit",
        commit,
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=output_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    atomic_write_text(
        output_root / "logs" / f"{role}.stdout",
        completed.stdout,
    )
    atomic_write_text(
        output_root / "logs" / f"{role}.stderr",
        completed.stderr,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{role} child failed with {completed.returncode}")
    return json.loads(output.read_text(encoding="utf-8"))


def _parent(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    baseline = _run_child(
        python=args.python.resolve(),
        script=script,
        source_root=args.baseline_root.resolve(),
        output_root=output_root,
        role="baseline",
        commit=args.baseline_commit,
    )
    candidate = _run_child(
        python=args.python.resolve(),
        script=script,
        source_root=args.candidate_root.resolve(),
        output_root=output_root,
        role="candidate",
        commit=args.candidate_commit,
    )
    if baseline["cases"] != candidate["cases"]:
        raise RuntimeError("Channel pressure PCG records are not bitwise equal")
    atomic_write_json(
        output_root / "pcg_equivalence.json",
        {
            "schema": SCHEMA,
            "baseline_commit": args.baseline_commit,
            "candidate_commit": args.candidate_commit,
            "python": str(args.python.resolve()),
            "case_count": len(CASES),
            "paths": list(PATHS),
            "comparisons": baseline["cases"],
            "state_diagnostics_gradients_calls_and_dispatch_equal": True,
            "passed": True,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-child", action="store_true")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--candidate-commit")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--role", choices=("baseline", "candidate"))
    parser.add_argument("--commit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    required = (
        ("source_root", "role", "commit", "output")
        if args.internal_child
        else (
            "python",
            "baseline_root",
            "candidate_root",
            "baseline_commit",
            "candidate_commit",
            "output_root",
        )
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def main() -> None:
    args = _parse_args()
    if args.internal_child:
        _child(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
