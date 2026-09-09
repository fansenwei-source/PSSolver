"""Benchmark free-slip Stokes solves with and without residual diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from pssolver.transforms import (
    FreeSlipModalStokesSolver,
    TensorProductTransformBackend,
)


TANGENTIAL_BCS = ("periodic", "periodic", "neumann")
NORMAL_BCS = ("periodic", "periodic", "dirichlet")
PRESSURE_BCS = ("periodic", "periodic", "neumann")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs that define one reproducible Stokes microbenchmark."""

    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    batch_size: int
    device: str
    dtype: str
    viscosity: float
    warmup: int
    repeats: int
    seed: int


def _tuple_of_ints(value):
    values = tuple(int(item) for item in value.split(","))
    if len(values) != 3 or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("shape must contain three positive integers")
    return values


def _tuple_of_floats(value):
    values = tuple(float(item) for item in value.split(","))
    if len(values) != 3 or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("lengths must contain three positive numbers")
    return values


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(operation: Callable[[], tuple[torch.Tensor, ...]], *, config, device):
    with torch.inference_mode():
        for _ in range(config.warmup):
            operation()
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(config.repeats):
            operation()
        _synchronize(device)
    return (time.perf_counter() - start) / config.repeats


def _git_head():
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _make_solver(backend, config, *, diagnostics):
    return FreeSlipModalStokesSolver(
        backend,
        tangential_boundary_conditions=TANGENTIAL_BCS,
        normal_boundary_conditions=NORMAL_BCS,
        pressure_boundary_conditions=PRESSURE_BCS,
        friction=0.0,
        viscosity=config.viscosity,
        zero_mode_policy="zero_mean",
        pressure_diagnostics=diagnostics,
    )


def run_benchmark(config):
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    backend = TensorProductTransformBackend(
        shape=config.shape,
        lengths=config.lengths,
        device=device,
        dtype=dtype,
    )
    force = torch.randn(
        3,
        config.batch_size,
        *config.shape,
        device=device,
        dtype=dtype,
    )
    tangential_hat = backend.forward(force[:2], TANGENTIAL_BCS)
    normal_hat = backend.forward(force[2], NORMAL_BCS)
    force_hats = (tangential_hat[0], tangential_hat[1], normal_hat)

    tracked = _make_solver(backend, config, diagnostics=True)
    production = _make_solver(backend, config, diagnostics=False)

    with torch.inference_mode():
        tracked_solution = tracked.solve_force_hats(*force_hats)
        production_solution = production.solve_force_hats(*force_hats)
    max_abs = max(
        float((actual - expected).abs().max().item())
        for actual, expected in zip(production_solution, tracked_solution, strict=True)
    )
    if max_abs != 0.0:
        raise RuntimeError(f"diagnostic mode changed the Stokes solution: max_abs={max_abs}")

    tracked_seconds = _measure(
        lambda: tracked.solve_force_hats(*force_hats),
        config=config,
        device=device,
    )
    production_seconds = _measure(
        lambda: production.solve_force_hats(*force_hats),
        config=config,
        device=device,
    )

    return {
        "schema_version": 1,
        "config": asdict(config),
        "environment": {
            "git_head": _git_head(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
        "correctness": {
            "solution_max_abs_difference": max_abs,
            "tracked_relative_residual": tracked.last_pressure_relative_residual,
            "production_relative_residual_recorded": (
                not math.isnan(production.last_pressure_relative_residual)
            ),
        },
        "timing": {
            "diagnostics_enabled_seconds": tracked_seconds,
            "production_seconds": production_seconds,
            "speedup": tracked_seconds / production_seconds,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_tuple_of_ints, default=(64, 64, 32))
    parser.add_argument("--lengths", type=_tuple_of_floats, default=(64.0, 64.0, 32.0))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--viscosity", type=float, default=2.0 / 3.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if not math.isfinite(args.viscosity) or args.viscosity <= 0.0:
        parser.error("--viscosity must be positive and finite")
    return args


def main():
    args = parse_args()
    result = run_benchmark(
        BenchmarkConfig(
            shape=args.shape,
            lengths=args.lengths,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            viscosity=args.viscosity,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
