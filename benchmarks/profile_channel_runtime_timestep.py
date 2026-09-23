#!/usr/bin/env python3
"""Profile one P7.6 Channel runtime with an identical outer timer."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.profile_beris_edwards_timestep import (
    RegionTimer,
    _git_provenance,
    _install_transform_timers,
)
from pssolver.channel import build_active_nematic_channel
from pssolver.configuration.channel_active_nematics import ChannelActiveNematicRunSpec
from pssolver.configuration.channel_active_nematics_declarations import ChannelRuntimePath
from pssolver.diagnostics import cuda_memory_snapshot, cuda_memory_window
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.channel_active_nematics import (
    ChannelRuntimeBuildRequest,
    build_channel_active_nematic_runtime,
)
from pssolver.runtime.channel_application_bridge import build_package_compiled_channel_runtime


RUNTIMES = tuple(item.value for item in ChannelRuntimePath)


@dataclass(frozen=True, slots=True)
class ChannelRuntimeProfileConfig:
    runtime_path: str = "legacy_channel"
    trial: int = 1
    shape: tuple[int, int, int] = (128, 20, 20)
    lengths: tuple[float, float, float] = (32.0, 5.0, 5.0)
    device: str = "cpu"
    dt: float = 0.01
    activity: float = 5.0
    warmup_steps: int = 10
    profile_steps: int = 50
    seed: int = 24


def _validate(config: ChannelRuntimeProfileConfig) -> None:
    if config.runtime_path not in RUNTIMES:
        raise ValueError("unsupported Channel runtime path")
    if not isinstance(config.trial, int) or isinstance(config.trial, bool) or config.trial <= 0:
        raise ValueError("trial must be positive")
    if len(config.shape) != 3 or any(value <= 0 for value in config.shape):
        raise ValueError("shape must contain three positive values")
    if len(config.lengths) != 3 or any(not math.isfinite(value) or value <= 0 for value in config.lengths):
        raise ValueError("lengths must contain three positive finite values")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not math.isfinite(config.dt) or config.dt <= 0:
        raise ValueError("dt must be positive and finite")
    if config.warmup_steps < 0 or config.profile_steps <= 0:
        raise ValueError("invalid warmup/profile step counts")


def _initial_q(config: ChannelRuntimeProfileConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    values = 0.01 * torch.randn(5, *config.shape, generator=generator, dtype=torch.float32)
    values[0].add_(1.0 / 3.0)
    values[3].add_(-1.0 / 6.0)
    return {name: values[index] for index, name in enumerate(Q_COMPONENTS)}


def _hash_tensors(values) -> str:
    digest = hashlib.sha256()
    for name, tensor in values:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _scheduled_refreshes(
    *,
    start_step_count: int,
    interval: int | None,
    steps: int,
) -> int:
    """Return refreshes that must occur in a measured integration window."""
    if interval is None:
        return 0
    if not 0 <= start_step_count < interval:
        raise ValueError("spectral-refresh phase is outside its interval")
    return (start_step_count + steps) // interval


def _build(config: ChannelRuntimeProfileConfig, initial_q):
    spec = ChannelActiveNematicRunSpec(
        shape=config.shape,
        lengths=config.lengths,
        dt=config.dt,
        steps=config.warmup_steps + config.profile_steps,
        save_interval=config.warmup_steps + config.profile_steps,
        diagnostic_interval=config.warmup_steps + config.profile_steps,
        activity=config.activity,
        device=config.device,
        pressure_relative_tolerance=1.0e-6,
        pressure_max_iterations=80,
        runtime_path=config.runtime_path,
    )
    metadata = {
        "configuration": spec.identity_metadata(),
        "runtime_selection": spec.runtime_selection_metadata(),
    }
    request = ChannelRuntimeBuildRequest(spec, metadata, initial_q, config.device)
    material, pressure = spec.components.material, spec.components.pressure_solver

    def legacy_builder():
        solver = build_active_nematic_channel(
            spec.shape, spec.lengths, spec.dt, initial_q,
            device=config.device, batchsize=1, rho=material.rho,
            elastic_constant=material.elastic_constant, beta=material.beta,
            friction=material.friction, viscosity=material.viscosity,
            pressure_rel_tol=pressure.relative_tolerance,
            pressure_max_iter=pressure.max_iterations,
        )
        solver.parameters["alpha"] = torch.full(
            (1, *spec.shape), material.activity,
            dtype=solver.dtype, device=solver.device,
        )
        return solver

    adapter = build_channel_active_nematic_runtime(
        request,
        legacy_builder=legacy_builder if spec.runtime_path is ChannelRuntimePath.LEGACY_CHANNEL else None,
        compiled_builder=(lambda: build_package_compiled_channel_runtime(request)) if spec.runtime_path is ChannelRuntimePath.COMPILED_CHANNEL_V2 else None,
    )
    return spec, adapter


def run_profile(config: ChannelRuntimeProfileConfig) -> dict[str, object]:
    _validate(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    initial_q = _initial_q(config)
    initial_hash = _hash_tensors((name, initial_q[name]) for name in Q_COMPONENTS)
    timer = RegionTimer(device)
    with torch.no_grad():
        before_build = cuda_memory_snapshot(device)
        build_started = time.perf_counter()
        spec, adapter = _build(config, initial_q)
        _install_transform_timers(adapter.solver, timer)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        build_seconds = time.perf_counter() - build_started
        after_build = cuda_memory_snapshot(device)
        adapter.advance(config.warmup_steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        after_warmup = cuda_memory_snapshot(device)
        integrator = adapter.solver.integrator
        refresh_interval = integrator.spectral_refresh_interval
        refresh_step_count_before = int(integrator.step_count)
        refresh_count_before = int(integrator.refresh_count)
        dynamic_transform_group_count = len(integrator.dynamic_transform_groups)
        expected_scheduled_refreshes = _scheduled_refreshes(
            start_step_count=refresh_step_count_before,
            interval=refresh_interval,
            steps=config.profile_steps,
        )
        pressure_iterations = []
        timer.reset()
        for _ in range(config.profile_steps):
            with timer.region("whole_timestep"):
                adapter.advance(1)
            pressure_iterations.append(int(adapter.solver.model.static_model.last_pressure_iterations))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        after_window = cuda_memory_snapshot(device)
        refresh_step_count_after = int(integrator.step_count)
        refresh_count_after = int(integrator.refresh_count)
        timings = timer.summarize()
        peak = cuda_memory_snapshot(device)
        adapter.synchronize_for_observation()
        final_hash = _hash_tensors((
            ("spatial", adapter.fields.spatial),
            ("spectral", adapter.fields.spectral),
        ))
    memory = cuda_memory_window(after_warmup, after_window, peak)
    whole = timings["whole_timestep"]["total_seconds"]
    runtime = adapter.to_metadata()
    if runtime["requested"] != runtime["effective"] or runtime["fallback_used"] is not False:
        raise RuntimeError("Channel runtime fallback or identity mismatch")
    forward = int(timings.get("transform_forward", {}).get("calls", 0))
    inverse = int(timings.get("transform_inverse", {}).get("calls", 0))
    observed_scheduled_refreshes = refresh_count_after - refresh_count_before
    if observed_scheduled_refreshes != expected_scheduled_refreshes:
        raise RuntimeError("spectral-refresh count differs from the integration clock")
    scheduled_refresh_forward_calls = (
        expected_scheduled_refreshes * dynamic_transform_group_count
    )
    base_forward = forward - scheduled_refresh_forward_calls
    if base_forward < 0:
        raise RuntimeError("scheduled refresh calls exceed observed forward transforms")
    return {
        "schema_version": 1,
        "config": asdict(config),
        "configuration_identity": spec.identity_metadata(),
        "runtime_identity": runtime,
        "environment": {
            "git": _git_provenance(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "timings": timings,
        "throughput": {
            "mean_timestep_seconds": whole / config.profile_steps,
            "timesteps_per_second": config.profile_steps / whole,
        },
        "transform_calls": {
            "forward_total": forward,
            "inverse_total": inverse,
            "forward_per_step": forward / config.profile_steps,
            "inverse_per_step": inverse / config.profile_steps,
            "base_forward_per_step": base_forward / config.profile_steps,
            "scheduled_refresh_forward_calls": scheduled_refresh_forward_calls,
        },
        "spectral_refresh": {
            "interval": refresh_interval,
            "step_count_before": refresh_step_count_before,
            "step_count_after": refresh_step_count_after,
            "count_before": refresh_count_before,
            "count_after": refresh_count_after,
            "expected_in_window": expected_scheduled_refreshes,
            "observed_in_window": observed_scheduled_refreshes,
            "dynamic_transform_group_count": dynamic_transform_group_count,
        },
        "pressure_iterations": {
            "values": pressure_iterations,
            "mean": statistics.fmean(pressure_iterations),
            "maximum": max(pressure_iterations),
        },
        "memory": {**memory, "before_build": before_build, "after_build": after_build},
        "build_wall_seconds": build_seconds,
        "initial_q_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "completed_steps": adapter.completed_steps,
        "finite": bool(torch.isfinite(adapter.fields.spatial).all() and torch.isfinite(adapter.fields.spectral).all()),
    }


def _triplet(value: str, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-path", choices=RUNTIMES, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", type=lambda value: _triplet(value, int), required=True)
    parser.add_argument("--lengths", type=lambda value: _triplet(value, float), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    result = run_profile(ChannelRuntimeProfileConfig(
        runtime_path=args.runtime_path, trial=args.trial, shape=args.shape,
        lengths=args.lengths, device=args.device,
        warmup_steps=args.warmup_steps, profile_steps=args.profile_steps,
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
