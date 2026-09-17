"""Qualify the implicit periodic fast-path defaults and explicit rollback."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from benchmarks.benchmark_periodic_fast_path import (
    BenchmarkConfig,
    _build_solver,
    _floats,
    _initial_fields,
    _ints,
    _relative_l2,
    _synchronize,
)


ROLES = {
    "implicit_default": (None, None),
    "explicit_candidate": ("contiguous_slice", "multidim"),
    "explicit_rollback": ("advanced", "axiswise"),
}


@dataclass(frozen=True)
class DefaultSmokeConfig:
    shape: tuple[int, ...]
    lengths: tuple[float, ...]
    field_count: int
    batch_size: int
    device: str
    dtype: str
    warmup_steps: int
    profile_steps: int
    trajectory_steps: int
    trials: int
    seed: int


def _benchmark_config(config: DefaultSmokeConfig) -> BenchmarkConfig:
    return BenchmarkConfig(
        shape=config.shape,
        lengths=config.lengths,
        field_count=config.field_count,
        batch_size=config.batch_size,
        device=config.device,
        dtype=config.dtype,
        warmup_steps=config.warmup_steps,
        profile_steps=config.profile_steps,
        trials=config.trials,
        seed=config.seed,
    )


def _solver_for_role(config, initial_fields, role):
    group_indexing, periodic_execution = ROLES[role]
    return _build_solver(
        _benchmark_config(config),
        initial_fields,
        transform_group_indexing=group_indexing,
        periodic_transform_execution=periodic_execution,
    )


def _metadata(solver) -> dict[str, object]:
    group = solver.integrator.dynamic_transform_groups[0]
    return {
        **solver.optimization_metadata(),
        "transform_group_indexing": (
            solver.fields.transform_group_indexing_metadata(group)
        ),
    }


def _trajectory(config, initial_fields, role):
    solver = _solver_for_role(config, initial_fields, role)
    solver.run(config.trajectory_steps)
    _synchronize(torch.device(config.device))
    return {
        "metadata": _metadata(solver),
        "spatial": solver.fields.spatial.detach().cpu(),
        "spectral": solver.fields.spectral.detach().cpu(),
    }


def _profile(config, initial_fields, role):
    device = torch.device(config.device)
    solver = _solver_for_role(config, initial_fields, role)
    solver.run(config.warmup_steps)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    solver.run(config.profile_steps)
    _synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "role": role,
        "elapsed_seconds": elapsed,
        "mean_timestep_seconds": elapsed / config.profile_steps,
        "steps_per_second": config.profile_steps / elapsed,
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
        "metadata": _metadata(solver),
    }


def _aggregate(records):
    times = [float(record["mean_timestep_seconds"]) for record in records]
    return {
        "trials": len(times),
        "mean_timestep_seconds": statistics.fmean(times),
        "median_timestep_seconds": statistics.median(times),
        "sample_standard_deviation_seconds": (
            statistics.stdev(times) if len(times) > 1 else 0.0
        ),
        "peak_allocated_bytes": max(
            (record["peak_allocated_bytes"] or 0 for record in records),
            default=0,
        ),
        "peak_reserved_bytes": max(
            (record["peak_reserved_bytes"] or 0 for record in records),
            default=0,
        ),
    }


def run_default_smoke(config: DefaultSmokeConfig) -> dict[str, object]:
    if config.trajectory_steps <= 0:
        raise ValueError("trajectory_steps must be positive")
    benchmark_config = _benchmark_config(config)
    if config.field_count <= 0 or config.batch_size <= 0:
        raise ValueError("field_count and batch_size must be positive")
    if config.warmup_steps < 0 or config.profile_steps <= 0 or config.trials <= 0:
        raise ValueError("invalid warmup, profile step, or trial count")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    initial_fields = _initial_fields(benchmark_config)

    trajectories = {
        role: _trajectory(config, initial_fields, role) for role in ROLES
    }
    implicit = trajectories["implicit_default"]
    candidate = trajectories["explicit_candidate"]
    rollback = trajectories["explicit_rollback"]
    implicit_candidate = {
        "spatial_bytewise_equal": torch.equal(
            implicit["spatial"], candidate["spatial"]
        ),
        "spectral_bytewise_equal": torch.equal(
            implicit["spectral"], candidate["spectral"]
        ),
        "spatial_relative_l2": _relative_l2(
            implicit["spatial"], candidate["spatial"]
        ),
        "spectral_relative_l2": _relative_l2(
            implicit["spectral"], candidate["spectral"]
        ),
    }
    rollback_candidate = {
        "spatial_relative_l2": _relative_l2(
            rollback["spatial"], candidate["spatial"]
        ),
        "spectral_relative_l2": _relative_l2(
            rollback["spectral"], candidate["spectral"]
        ),
    }
    tolerance = 5.0e-6 if config.dtype == "float32" else 1.0e-12
    maximum_rollback_relative_l2 = max(rollback_candidate.values())
    if not all(
        torch.isfinite(result[name]).all()
        for result in trajectories.values()
        for name in ("spatial", "spectral")
    ):
        raise RuntimeError("a default smoke trajectory contains NaN or Inf")
    if not all(
        implicit_candidate[name]
        for name in ("spatial_bytewise_equal", "spectral_bytewise_equal")
    ):
        raise RuntimeError("implicit default differs from explicit candidate")
    if maximum_rollback_relative_l2 > tolerance:
        raise RuntimeError(
            "rollback and candidate trajectories disagree: "
            f"relative_l2={maximum_rollback_relative_l2:.6e}, "
            f"tolerance={tolerance:.6e}"
        )

    records = {role: [] for role in ROLES}
    roles = tuple(ROLES)
    for trial in range(config.trials):
        order = roles[trial % len(roles) :] + roles[: trial % len(roles)]
        for position, role in enumerate(order, start=1):
            record = _profile(config, initial_fields, role)
            record["trial"] = trial + 1
            record["order"] = position
            records[role].append(record)
    aggregate = {role: _aggregate(values) for role, values in records.items()}
    implicit_mean = float(
        aggregate["implicit_default"]["mean_timestep_seconds"]
    )
    candidate_mean = float(
        aggregate["explicit_candidate"]["mean_timestep_seconds"]
    )
    rollback_mean = float(
        aggregate["explicit_rollback"]["mean_timestep_seconds"]
    )

    return {
        "schema_version": 1,
        "config": asdict(config),
        "roles": {
            role: {
                "transform_group_indexing": selectors[0],
                "periodic_transform_execution": selectors[1],
            }
            for role, selectors in ROLES.items()
        },
        "trajectory": {
            "implicit_default_metadata": implicit["metadata"],
            "explicit_candidate_metadata": candidate["metadata"],
            "explicit_rollback_metadata": rollback["metadata"],
            "implicit_candidate": implicit_candidate,
            "rollback_candidate": rollback_candidate,
            "tolerance_relative_l2": tolerance,
        },
        "records": records,
        "aggregate": aggregate,
        "performance": {
            "implicit_over_explicit_candidate_ratio": (
                implicit_mean / candidate_mean
            ),
            "rollback_over_implicit_default_speedup": (
                rollback_mean / implicit_mean
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_ints, default=(512, 512))
    parser.add_argument("--lengths", type=_floats)
    parser.add_argument("--field-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("--trajectory-steps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
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
    result = run_default_smoke(
        DefaultSmokeConfig(
            shape=args.shape,
            lengths=args.lengths,
            field_count=args.field_count,
            batch_size=args.batch_size,
            device=args.device,
            dtype=args.dtype,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            trajectory_steps=args.trajectory_steps,
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
