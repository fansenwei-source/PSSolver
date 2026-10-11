#!/usr/bin/env python3
"""RC4.3.7 Plane production profiler with explicit live-memory evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import torch

if __package__:
    from benchmarks.profile_beris_edwards_timestep import (
        RegionTimer,
        _counter_delta,
        _dynamo_counter_snapshot,
        _git_provenance,
        _install_transform_timers,
    )
    from benchmarks.profile_plane_runtime_timestep import (
        RuntimeProfileConfig,
        _build_runtime,
        _initial_q,
        _state_sha256,
        _tensor_mapping_sha256,
        _validate_config,
    )
else:
    from profile_beris_edwards_timestep import (
        RegionTimer,
        _counter_delta,
        _dynamo_counter_snapshot,
        _git_provenance,
        _install_transform_timers,
    )
    from profile_plane_runtime_timestep import (
        RuntimeProfileConfig,
        _build_runtime,
        _initial_q,
        _state_sha256,
        _tensor_mapping_sha256,
        _validate_config,
    )


SCHEMA = "pssolver.rc4_3_7.plane_runtime_profile.v1"


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("profile JSON round-trip differs")
    finally:
        if temporary.exists():
            temporary.unlink()


def run_profile(config: RuntimeProfileConfig) -> dict[str, object]:
    """Measure one frozen production path and persist observed compiler state."""

    _validate_config(config)
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RC4.3.7 profiles require an allocated CUDA device")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    initial = _initial_q(config)
    initial_sha256 = _tensor_mapping_sha256(initial)
    timer = RegionTimer(device, timed_regions=frozenset())
    counters_before = _dynamo_counter_snapshot()

    with torch.no_grad():
        spec, adapter = _build_runtime(config, initial)
        del initial
        _install_transform_timers(adapter.solver, timer)
        torch.cuda.synchronize(device)
        counters_after_build = _dynamo_counter_snapshot()
        adapter.advance(config.warmup_steps)
        torch.cuda.synchronize(device)
        counters_after_warmup = _dynamo_counter_snapshot()
        torch.cuda.reset_peak_memory_stats(device)
        timer.reset()
        samples: list[float] = []
        for _ in range(config.profile_steps):
            started = time.perf_counter()
            adapter.advance(1)
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
        transform_summary = timer.summarize()
        counters_after_profile = _dynamo_counter_snapshot()
        adapter.synchronize_for_observation()
        torch.cuda.synchronize(device)
        final_sha256 = _state_sha256(adapter)
        memory_stats = torch.cuda.memory_stats(device)
        pointwise = adapter.solver.model.nlmodel.pointwise_kernels.metadata()
        finite = bool(
            torch.isfinite(adapter.fields.spatial).all().item()
            and torch.isfinite(adapter.fields.spectral).all().item()
        )
        completed_steps = adapter.completed_steps
        runtime = adapter.to_metadata()

    if runtime["requested"] != runtime["effective"] or runtime["fallback_used"]:
        raise RuntimeError("runtime fallback or identity mismatch")
    if pointwise["requested"] != "compile" or pointwise["effective"] != "compile":
        raise RuntimeError("compiled pointwise execution was not effective")
    if pointwise["fallback_allowed"] is not False:
        raise RuntimeError("pointwise fallback was allowed")
    forward = int(transform_summary.get("transform_forward", {}).get("calls", 0))
    inverse = int(transform_summary.get("transform_inverse", {}).get("calls", 0))
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "schema": SCHEMA,
        "config": asdict(config),
        "configuration_identity": spec.identity_metadata(),
        "runtime_identity": runtime,
        "environment": {
            "git": _git_provenance(),
            "pssolver_import": str(Path(__import__("pssolver").__file__).resolve()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "requested_device": "cuda",
            "allocated_device": str(torch.empty((), device=device).device),
            "device_name": torch.cuda.get_device_name(device),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "timing": {
            "samples_seconds": samples,
            "mean_timestep_seconds": mean,
            "median_timestep_seconds": median,
            "sample_standard_deviation_seconds": stdev,
            "coefficient_of_variation": stdev / mean,
        },
        "memory": {
            "peak_allocated_bytes": int(memory_stats["allocated_bytes.all.peak"]),
            "peak_active_bytes": int(memory_stats["active_bytes.all.peak"]),
            "peak_reserved_bytes": int(memory_stats["reserved_bytes.all.peak"]),
        },
        "transform_calls": {
            "forward": forward,
            "inverse": inverse,
            "forward_per_step": forward / config.profile_steps,
            "inverse_per_step": inverse / config.profile_steps,
        },
        "pointwise_compile": {
            "execution": pointwise,
            "during_build": _counter_delta(counters_after_build, counters_before),
            "during_warmup": _counter_delta(counters_after_warmup, counters_after_build),
            "during_profile": _counter_delta(counters_after_profile, counters_after_warmup),
        },
        "initial_q_sha256": initial_sha256,
        "final_state_sha256": final_sha256,
        "completed_steps": completed_steps,
        "finite": finite,
    }


def _shape(value: str) -> tuple[int, int, int]:
    result = tuple(int(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("shape requires three integers")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", type=_shape, required=True)
    parser.add_argument("--initial-q-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = RuntimeProfileConfig(
        runtime_path="legacy_production",
        trial=args.trial,
        shape=args.shape,
        lengths=(100.0, 100.0, 20.0),
        device="cuda",
        dtype="float64",
        dt=0.005,
        activity_number=18.0,
        dealias_rule="cubic_half",
        projected_transform_execution="truncated",
        transform_execution_order="real_first",
        spectral_storage="hermitian_half",
        molecular_field_linear_space="spectral",
        stress_divergence_sum_space="spectral",
        pointwise_execution="compile",
        reuse_q_gradients=True,
        spectral_refresh_interval=None,
        warmup_steps=10,
        profile_steps=50,
        seed=24,
        initial_q_path=args.initial_q_path,
    )
    _atomic_json(args.output, run_profile(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
