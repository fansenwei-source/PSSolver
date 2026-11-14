"""Benchmark independent periodic FFT and transform-group indexing fast paths."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pssolver import SpectralSolver


VARIANTS = {
    "A_advanced_axiswise": ("advanced", "axiswise"),
    "B_contiguous_axiswise": ("contiguous_slice", "axiswise"),
    "C_advanced_multidim": ("advanced", "multidim"),
    "D_contiguous_multidim": ("contiguous_slice", "multidim"),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    shape: tuple[int, ...]
    lengths: tuple[float, ...]
    field_count: int
    batch_size: int
    device: str
    dtype: str
    warmup_steps: int
    profile_steps: int
    trials: int
    seed: int


class _ZeroNonlinear(torch.nn.Module):
    def forward(self, fields, parameters):
        del parameters
        return torch.zeros_like(fields.spectral[: fields.dyn_count])


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm((actual - reference).reshape(-1))
    scale = torch.linalg.vector_norm(reference.reshape(-1))
    tiny = torch.finfo(reference.real.dtype).tiny
    return float((difference / scale.clamp_min(tiny)).item())


def _git_head() -> str | None:
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


def _initial_fields(config: BenchmarkConfig) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    dtype = _dtype(config.dtype)
    return tuple(
        torch.randn(
            config.batch_size,
            *config.shape,
            generator=generator,
            dtype=dtype,
        )
        for _ in range(config.field_count)
    )


def _build_solver(
    config: BenchmarkConfig,
    initial_fields: tuple[torch.Tensor, ...],
    *,
    transform_group_indexing: str | None,
    periodic_transform_execution: str | None,
) -> SpectralSolver:
    device = torch.device(config.device)
    dtype = _dtype(config.dtype)
    selector_kwargs = {}
    if transform_group_indexing is not None:
        selector_kwargs["transform_group_indexing"] = transform_group_indexing
    if periodic_transform_execution is not None:
        selector_kwargs["periodic_transform_execution"] = (
            periodic_transform_execution
        )
    solver = SpectralSolver(
        config.shape,
        L=config.lengths,
        dt=0.01,
        batchsize=config.batch_size,
        device=device,
        dtype=dtype,
        **selector_kwargs,
    )
    linear_operator = -0.1 * solver.q2_raw
    for index, initial in enumerate(initial_fields):
        solver.model.add_dynamic_field(
            f"q{index}",
            initial.to(device=device, dtype=dtype),
            linear_operator,
        )
    solver.model.set_nonlinear_model(_ZeroNonlinear())
    solver.build()
    return solver


def _run_variant(
    config: BenchmarkConfig,
    initial_fields: tuple[torch.Tensor, ...],
    variant: str,
) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    group_indexing, periodic_execution = VARIANTS[variant]
    device = torch.device(config.device)
    solver = _build_solver(
        config,
        initial_fields,
        transform_group_indexing=group_indexing,
        periodic_transform_execution=periodic_execution,
    )
    solver.run(config.warmup_steps)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    solver.run(config.profile_steps)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    group = solver.integrator.dynamic_transform_groups[0]
    metadata = {
        **solver.optimization_metadata(),
        "transform_group_indexing": (
            solver.fields.transform_group_indexing_metadata(group)
        ),
    }
    result: dict[str, object] = {
        "variant": variant,
        "elapsed_seconds": elapsed,
        "mean_timestep_seconds": elapsed / config.profile_steps,
        "steps_per_second": config.profile_steps / elapsed,
        "optimization": metadata,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else None
        ),
    }
    return (
        result,
        solver.fields.spatial.detach().cpu(),
        solver.fields.spectral.detach().cpu(),
    )


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    values = [float(record["mean_timestep_seconds"]) for record in records]
    return {
        "trials": len(values),
        "mean_timestep_seconds": statistics.fmean(values),
        "median_timestep_seconds": statistics.median(values),
        "sample_standard_deviation_seconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "mean_steps_per_second": 1.0 / statistics.fmean(values),
        "peak_allocated_bytes": max(
            (record["peak_allocated_bytes"] or 0 for record in records),
            default=0,
        ),
        "peak_reserved_bytes": max(
            (record["peak_reserved_bytes"] or 0 for record in records),
            default=0,
        ),
    }


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    if config.field_count <= 0 or config.batch_size <= 0:
        raise ValueError("field_count and batch_size must be positive")
    if config.warmup_steps < 0 or config.profile_steps <= 0 or config.trials <= 0:
        raise ValueError("invalid warmup, profile step, or trial count")
    if len(config.shape) != len(config.lengths):
        raise ValueError("shape and lengths must have the same dimension")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    initial_fields = _initial_fields(config)
    records = {name: [] for name in VARIANTS}
    correctness = []
    names = tuple(VARIANTS)
    for trial in range(config.trials):
        order = names[trial % len(names) :] + names[: trial % len(names)]
        outputs = {}
        for variant in order:
            record, spatial, spectral = _run_variant(
                config,
                initial_fields,
                variant,
            )
            record["trial"] = trial + 1
            record["order"] = order.index(variant) + 1
            records[variant].append(record)
            outputs[variant] = (spatial, spectral)
        reference_spatial, reference_spectral = outputs[names[0]]
        for variant in names[1:]:
            spatial, spectral = outputs[variant]
            correctness.append(
                {
                    "trial": trial + 1,
                    "reference": names[0],
                    "candidate": variant,
                    "spatial_relative_l2": _relative_l2(
                        spatial,
                        reference_spatial,
                    ),
                    "spectral_relative_l2": _relative_l2(
                        spectral,
                        reference_spectral,
                    ),
                    "finite": bool(
                        torch.isfinite(spatial).all()
                        and torch.isfinite(spectral).all()
                    ),
                }
            )

    aggregate = {name: _aggregate(values) for name, values in records.items()}
    baseline = float(aggregate[names[0]]["mean_timestep_seconds"])
    speedups = {
        name: baseline / float(values["mean_timestep_seconds"])
        for name, values in aggregate.items()
    }
    max_relative_l2 = max(
        max(
            float(item["spatial_relative_l2"]),
            float(item["spectral_relative_l2"]),
        )
        for item in correctness
    )
    tolerance = 5.0e-6 if config.dtype == "float32" else 1.0e-12
    if not all(item["finite"] for item in correctness):
        raise RuntimeError("a periodic fast-path result contains NaN or Inf")
    if max_relative_l2 > tolerance:
        raise RuntimeError(
            "periodic fast-path numerical mismatch: "
            f"max_relative_l2={max_relative_l2:.6e}, "
            f"tolerance={tolerance:.6e}"
        )

    return {
        "schema_version": 1,
        "config": asdict(config),
        "environment": {
            "git_head": _git_head(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
        },
        "variants": {
            name: {
                "transform_group_indexing": selectors[0],
                "periodic_transform_execution": selectors[1],
            }
            for name, selectors in VARIANTS.items()
        },
        "records": records,
        "aggregate": aggregate,
        "speedup_over_A": speedups,
        "correctness": {
            "tolerance_relative_l2": tolerance,
            "maximum_relative_l2": max_relative_l2,
            "comparisons": correctness,
        },
    }


def _ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("shape entries must be positive")
    return values


def _floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(","))
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("length entries must be positive and finite")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_ints, default=(512, 512))
    parser.add_argument("--lengths", type=_floats)
    parser.add_argument("--field-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.lengths is None:
        args.lengths = tuple(float(item) for item in args.shape)
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite")
    return args


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        BenchmarkConfig(
            shape=args.shape,
            lengths=args.lengths,
            field_count=args.field_count,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            trials=args.trials,
            seed=args.seed,
        )
    )
    serialized = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
