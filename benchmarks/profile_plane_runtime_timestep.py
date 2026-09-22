#!/usr/bin/env python3
"""Profile one production Plane runtime path with an identical outer timer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .profile_beris_edwards_timestep import (
        RegionTimer,
        _counter_delta,
        _dynamo_counter_snapshot,
        _git_provenance,
        _install_transform_timers,
    )
else:
    from profile_beris_edwards_timestep import (
        RegionTimer,
        _counter_delta,
        _dynamo_counter_snapshot,
        _git_provenance,
        _install_transform_timers,
    )
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)
from pssolver.diagnostics import cuda_memory_snapshot, cuda_memory_window
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.plane_beris_edwards import LegacyPlaneRuntimeAdapter
from pssolver.runtime.plane_legacy import build_legacy_plane_runtime
from pssolver.workflows.plane_compiled_v2 import (
    build_plane_compiled_v2_runtime,
)


SUPPORTED_RUNTIME_PATHS = (
    PlaneRuntimePath.LEGACY_PRODUCTION.value,
    PlaneRuntimePath.COMPILED_V2.value,
)


@dataclass(frozen=True, slots=True)
class RuntimeProfileConfig:
    """One complete production-runtime profile identity."""

    runtime_path: str = PlaneRuntimePath.LEGACY_PRODUCTION.value
    trial: int = 1
    shape: tuple[int, int, int] = (64, 64, 32)
    lengths: tuple[float, float, float] = (100.0, 100.0, 20.0)
    device: str = "cpu"
    dtype: str = "float64"
    dt: float = 0.005
    activity_number: float = 18.0
    dealias_rule: str = "cubic_half"
    projected_transform_execution: str = "truncated"
    transform_execution_order: str = "real_first"
    spectral_storage: str = "hermitian_half"
    molecular_field_linear_space: str = "spectral"
    stress_divergence_sum_space: str = "spectral"
    pointwise_execution: str = "compile"
    reuse_q_gradients: bool = True
    spectral_refresh_interval: int | None = None
    warmup_steps: int = 10
    profile_steps: int = 50
    seed: int = 20260908
    initial_q_path: str | None = None


def _validate_config(config: RuntimeProfileConfig) -> None:
    if config.runtime_path not in SUPPORTED_RUNTIME_PATHS:
        raise ValueError("runtime_path must select legacy_production or compiled_v2")
    if not isinstance(config.trial, int) or isinstance(config.trial, bool) or config.trial <= 0:
        raise ValueError("trial must be a positive integer")
    if len(config.shape) != 3 or any(value <= 0 for value in config.shape):
        raise ValueError("shape must contain three positive integers")
    if len(config.lengths) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in config.lengths
    ):
        raise ValueError("lengths must contain three positive finite values")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    if not math.isfinite(config.dt) or config.dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not math.isfinite(config.activity_number) or config.activity_number <= 0.0:
        raise ValueError("activity_number must be positive and finite")
    if config.warmup_steps < 0 or config.profile_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and profile_steps positive")
    if config.pointwise_execution == "compile" and config.warmup_steps < 1:
        raise ValueError("compiled pointwise execution requires warmup")
    if (
        config.spectral_refresh_interval is not None
        and config.spectral_refresh_interval <= 0
    ):
        raise ValueError("spectral_refresh_interval must be positive or None")
    if config.initial_q_path is not None and not Path(
        config.initial_q_path
    ).expanduser().is_file():
        raise FileNotFoundError(f"initial Q file is missing: {config.initial_q_path}")


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _initial_q(config: RuntimeProfileConfig) -> dict[str, torch.Tensor]:
    dtype = _torch_dtype(config.dtype)
    if config.initial_q_path is not None:
        path = Path(config.initial_q_path).expanduser().resolve()
        values = np.load(path, allow_pickle=False)
        expected_shape = (*config.shape, len(Q_COMPONENTS))
        if values.shape != expected_shape:
            raise ValueError(
                f"initial Q shape must be {expected_shape}, got {values.shape}"
            )
        if values.dtype != np.dtype(config.dtype):
            raise ValueError("initial Q dtype does not match profile dtype")
        if not np.isfinite(values).all():
            raise ValueError("initial Q contains NaN or Inf")
        return {
            name: torch.from_numpy(np.array(values[..., index], copy=True))
            for index, name in enumerate(Q_COMPONENTS)
        }

    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    noise = 0.01 * torch.randn(
        len(Q_COMPONENTS),
        *config.shape,
        generator=generator,
        dtype=dtype,
    )
    q_amplitude = 0.5
    noise[0].add_(2.0 * q_amplitude / 3.0)
    noise[3].add_(-q_amplitude / 3.0)
    return {name: noise[index] for index, name in enumerate(Q_COMPONENTS)}


def _tensor_mapping_sha256(values: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in Q_COMPONENTS:
        array = values[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state_sha256(adapter) -> str:
    digest = hashlib.sha256()
    for tensor in (adapter.fields.spatial, adapter.fields.spectral):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _run_spec(config: RuntimeProfileConfig) -> object:
    nx, ny, nz = config.shape
    lx, ly, height = config.lengths
    return create_plane_beris_edwards_run_spec(
        activity_number=config.activity_number,
        output_dir=Path.cwd() / ".phase6_profile_no_output",
        height=height,
        parameterization="fixed-k",
        frank_k=1.0 / 81.0,
        lx=lx,
        ly=ly,
        nx=nx,
        ny=ny,
        nz=nz,
        dt=config.dt,
        steps=config.warmup_steps + config.profile_steps,
        save_start_step=0,
        save_interval=config.warmup_steps + config.profile_steps,
        diagnostic_interval=config.warmup_steps + config.profile_steps,
        seed=config.seed,
        dealias_rule=config.dealias_rule,
        projected_transform_execution=config.projected_transform_execution,
        device=config.device,
        dtype=config.dtype,
        molecular_field_linear_space=config.molecular_field_linear_space,
        stress_divergence_sum_space=config.stress_divergence_sum_space,
        pointwise_execution=config.pointwise_execution,
        transform_execution_order=config.transform_execution_order,
        spectral_storage=config.spectral_storage,
        tf32="off",
        spectral_refresh_steps=config.spectral_refresh_interval,
        disable_spectral_refresh=config.spectral_refresh_interval is None,
        disable_q_gradient_reuse=not config.reuse_q_gradients,
        save_hydrodynamics=True,
        runtime_path=config.runtime_path,
    )


def _build_runtime(config: RuntimeProfileConfig, initial_values):
    spec = _run_spec(config)
    if config.runtime_path == PlaneRuntimePath.LEGACY_PRODUCTION.value:
        solver, projector = build_legacy_plane_runtime(
            spec,
            device=config.device,
            initial_values=initial_values,
        )
        return spec, LegacyPlaneRuntimeAdapter(solver, projector)
    return spec, build_plane_compiled_v2_runtime(
        spec,
        device=config.device,
        initial_values=initial_values,
    )


def run_profile(config: RuntimeProfileConfig) -> dict[str, object]:
    """Measure one runtime path without changing its timestep implementation."""

    _validate_config(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    initial_values = _initial_q(config)
    initial_q_sha256 = _tensor_mapping_sha256(initial_values)
    timer = RegionTimer(device)
    counters_before = _dynamo_counter_snapshot()

    with torch.no_grad():
        memory_phases = {"before_build": cuda_memory_snapshot(device)}
        build_start = time.perf_counter()
        spec, adapter = _build_runtime(config, initial_values)
        _install_transform_timers(adapter.solver, timer)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        build_seconds = time.perf_counter() - build_start
        memory_phases["after_build"] = cuda_memory_snapshot(device)
        counters_after_build = _dynamo_counter_snapshot()

        warmup_start = time.perf_counter()
        adapter.advance(config.warmup_steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start
        memory_phases["after_warmup"] = cuda_memory_snapshot(device)
        counters_after_warmup = _dynamo_counter_snapshot()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        timer.reset()
        profile_start = time.perf_counter()
        for _ in range(config.profile_steps):
            with timer.region("whole_timestep"):
                adapter.advance(1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile_wall_seconds = time.perf_counter() - profile_start
        memory_phases["after_timestep_window"] = cuda_memory_snapshot(device)
        peak = cuda_memory_snapshot(device)
        timings = timer.summarize()
        counters_after_profile = _dynamo_counter_snapshot()
        adapter.synchronize_for_observation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        final_state_sha256 = _state_sha256(adapter)

    window = cuda_memory_window(
        memory_phases["after_warmup"],
        memory_phases["after_timestep_window"],
        peak,
    )
    whole = timings["whole_timestep"]["total_seconds"]
    transform_calls = {
        name: int(timings.get(name, {}).get("calls", 0))
        for name in ("transform_forward", "transform_inverse")
    }
    runtime_metadata = adapter.to_metadata()
    if runtime_metadata["requested"] != runtime_metadata["effective"]:
        raise RuntimeError("runtime fallback or identity mismatch detected")
    if runtime_metadata["fallback_used"] is not False:
        raise RuntimeError("runtime fallback detected")

    return {
        "schema_version": 1,
        "config": asdict(config),
        "configuration_identity": spec.identity_metadata(),
        "runtime_identity": runtime_metadata,
        "environment": {
            "git": _git_provenance(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "timings": timings,
        "throughput": {
            "profile_wall_seconds": profile_wall_seconds,
            "whole_timestep_seconds": whole,
            "mean_timestep_seconds": whole / config.profile_steps,
            "timesteps_per_second": config.profile_steps / whole,
        },
        "transform_calls": {
            **transform_calls,
            "forward_per_step": (
                transform_calls["transform_forward"] / config.profile_steps
            ),
            "inverse_per_step": (
                transform_calls["transform_inverse"] / config.profile_steps
            ),
        },
        "memory": {
            "peak_allocated_bytes": window["peak_allocated_bytes"],
            "peak_reserved_bytes": window["peak_reserved_bytes"],
            "phases": memory_phases,
        },
        "pointwise_compile": {
            "build_wall_seconds": build_seconds,
            "warmup_wall_seconds": warmup_seconds,
            "dynamo_during_build": _counter_delta(
                counters_after_build,
                counters_before,
            ),
            "dynamo_during_warmup": _counter_delta(
                counters_after_warmup,
                counters_after_build,
            ),
            "dynamo_during_profile": _counter_delta(
                counters_after_profile,
                counters_after_warmup,
            ),
        },
        "initial_q_sha256": initial_q_sha256,
        "final_state_sha256": final_state_sha256,
        "completed_steps": adapter.completed_steps,
        "finite": bool(
            torch.isfinite(adapter.fields.spatial).all().item()
            and torch.isfinite(adapter.fields.spectral).all().item()
        ),
    }


def _three_ints(value: str) -> tuple[int, int, int]:
    parsed = tuple(int(item) for item in value.split(","))
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return parsed


def _three_floats(value: str) -> tuple[float, float, float]:
    parsed = tuple(float(item) for item in value.split(","))
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated floats")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-path", choices=SUPPORTED_RUNTIME_PATHS, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", type=_three_ints, default=(64, 64, 32))
    parser.add_argument("--lengths", type=_three_floats, default=(100.0, 100.0, 20.0))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--activity-number", type=float, default=18.0)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=50)
    parser.add_argument("--spectral-refresh-interval", type=int)
    parser.add_argument("--pointwise-execution", choices=("eager", "compile"), default="compile")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--initial-q-path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        parser.error(f"output exists: {args.output}; pass --overwrite to replace it")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_profile(
        RuntimeProfileConfig(
            runtime_path=args.runtime_path,
            trial=args.trial,
            shape=args.shape,
            lengths=args.lengths,
            device=args.device,
            dtype=args.dtype,
            dt=args.dt,
            activity_number=args.activity_number,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            spectral_refresh_interval=args.spectral_refresh_interval,
            pointwise_execution=args.pointwise_execution,
            seed=args.seed,
            initial_q_path=args.initial_q_path,
        )
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
