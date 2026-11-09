"""Bounded H100 qualification utilities for the experimental Plane runtime.

This module is deliberately outside production entry points.  It replays a
completed production reference, profiles the migrated runtime independently,
and combines already-written numerical and performance evidence without
changing any solver default.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch

from ._shadow_support import file_sha256, ordered_tensor_sha256
from .plane_shadow_driver import (
    build_h100_plane_shadow_runtime_from_production_metadata,
    load_h100_production_plane_reference,
)
from .shadow_run import ExperimentalPlaneShadowRun


def _require_positive_integer(value: int, description: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{description} must be a positive integer")
    return value


def _write_new_json(path: Path, value: Mapping[str, object]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(value),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _cuda_identity(
    *,
    expected_gpu_name: str,
    device: torch.device,
) -> dict[str, object]:
    if device.type != "cuda":
        raise ValueError("Stage M requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    name = torch.cuda.get_device_name(device)
    if expected_gpu_name.lower() not in name.lower():
        raise RuntimeError(
            f"execution GPU {name!r} does not contain {expected_gpu_name!r}"
        )
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "device": str(device),
        "device_name": name,
        "device_count": torch.cuda.device_count(),
        "total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _timing_summary(samples: Sequence[float]) -> dict[str, float | int]:
    values = tuple(float(value) for value in samples)
    if not values or any(
        not math.isfinite(value) or value <= 0.0 for value in values
    ):
        raise RuntimeError("timing samples must be positive and finite")
    total = math.fsum(values)
    return {
        "sample_count": len(values),
        "total_seconds": total,
        "mean_timestep_seconds": total / len(values),
        "median_timestep_seconds": statistics.median(values),
        "sample_std_seconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "timesteps_per_second": len(values) / total,
    }


def _cuda_step_samples(runtime, steps: int) -> tuple[float, ...]:
    events = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        runtime.solver.run(1)
        stop.record()
        events.append((start, stop))
    torch.cuda.synchronize(runtime.context.device)
    return tuple(start.elapsed_time(stop) / 1000.0 for start, stop in events)


def run_h100_plane_shadow_from_production_reference(
    production_directory: str | Path,
    output_directory: str | Path,
    *,
    confirmed_steps: int,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Run the Stage M numerical trajectory on the same H100 as its source."""

    _require_positive_integer(confirmed_steps, "confirmed_steps")
    reference = load_h100_production_plane_reference(
        production_directory,
        expected_gpu_name=expected_gpu_name,
    )
    if reference.metadata.get("completed_steps") != confirmed_steps:
        raise ValueError(
            "--confirm-steps must exactly equal production completed_steps"
        )
    save_interval = reference.metadata["solver"]["save_interval"]
    _require_positive_integer(save_interval, "production save_interval")
    device = torch.device("cuda")
    environment = _cuda_identity(
        expected_gpu_name=expected_gpu_name,
        device=device,
    )
    runtime, comparison = (
        build_h100_plane_shadow_runtime_from_production_metadata(
            reference.metadata,
            expected_gpu_name=expected_gpu_name,
            device=device,
        )
    )
    initial_values = {
        name: value.to(device=device)
        for name, value in reference.initial_values.items()
    }
    run = ExperimentalPlaneShadowRun(
        runtime,
        output_directory,
        initial_values=initial_values,
        initial_condition_metadata={
            "name": "production_projected_q_snapshot",
            "source_run_directory": str(reference.directory),
            "source_step": 0,
            "source_metadata_sha256": reference.metadata_sha256,
            "source_q_file": reference.initial_q_path.name,
            "source_q_file_sha256": reference.initial_q_file_sha256,
            "source_projected_q_sha256": reference.metadata[
                "initial_condition"
            ]["projected_q_sha256"],
            "configuration_comparison": comparison.to_metadata(),
            "qualification_stage": "M",
        },
    )
    saved_steps = [run.save_observation().step]
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    step_samples = []
    host_start = time.perf_counter()
    for step in range(1, confirmed_steps + 1):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        run.advance(1)
        stop.record()
        step_samples.append((start, stop))
        if step % save_interval == 0 or step == confirmed_steps:
            saved_steps.append(run.save_observation().step)
    torch.cuda.synchronize(device)
    host_seconds = time.perf_counter() - host_start
    samples = tuple(
        start.elapsed_time(stop) / 1000.0 for start, stop in step_samples
    )
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    final = run.complete()
    metrics = {
        "schema_version": 1,
        "qualification_stage": "M",
        "measurement_role": "numerical_trajectory",
        "implementation": "experimental_plane_model_runtime",
        "production_directory": str(reference.directory),
        "shadow_directory": str(run.output_directory),
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "completed_steps": final.step,
        "saved_steps": saved_steps,
        "configuration": comparison.to_metadata(),
        "environment": environment,
        "timing": {
            **_timing_summary(samples),
            "host_wall_seconds_including_snapshot_transfers": host_seconds,
            "includes_compile_warmup": True,
            "performance_qualification": False,
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
    }
    metrics_path = run.output_directory / "stage_m_metrics.json"
    _write_new_json(metrics_path, metrics)
    return metrics


def profile_h100_plane_shadow_from_production_reference(
    production_directory: str | Path,
    *,
    warmup_steps: int,
    profile_steps: int,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Profile only the migrated runtime, with I/O outside the timed region."""

    _require_positive_integer(warmup_steps, "warmup_steps")
    _require_positive_integer(profile_steps, "profile_steps")
    reference = load_h100_production_plane_reference(
        production_directory,
        expected_gpu_name=expected_gpu_name,
    )
    device = torch.device("cuda")
    environment = _cuda_identity(
        expected_gpu_name=expected_gpu_name,
        device=device,
    )
    build_start = time.perf_counter()
    runtime, comparison = (
        build_h100_plane_shadow_runtime_from_production_metadata(
            reference.metadata,
            expected_gpu_name=expected_gpu_name,
            device=device,
        )
    )
    initial_values = {
        name: value.to(device=device)
        for name, value in reference.initial_values.items()
    }
    runtime.reset(initial_values)
    torch.cuda.synchronize(device)
    build_seconds = time.perf_counter() - build_start
    _cuda_step_samples(runtime, warmup_steps)
    torch.cuda.reset_peak_memory_stats(device)
    samples = _cuda_step_samples(runtime, profile_steps)
    runtime.synchronize_algebraic_for_observation()
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    fields = runtime.solver.fields
    if not bool(torch.isfinite(fields.spatial).all().item()) or not bool(
        torch.isfinite(fields.spectral).all().item()
    ):
        raise RuntimeError("shadow profile ended with non-finite fields")
    q_hash = ordered_tensor_sha256(
        {
            name: fields[name]
            for name in reference.initial_values
        }
    )
    production_signature = comparison.production_signature
    numerical = production_signature["numerics"]
    solver = production_signature["solver"]
    return {
        "schema_version": 1,
        "qualification_stage": "M",
        "measurement_role": "performance_profile",
        "implementation": "experimental_plane_model_runtime",
        "production_directory": str(reference.directory),
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "configuration": {
            "shape": solver["shape"],
            "lengths": solver["lengths"],
            "device": "cuda",
            "dtype": solver["real_dtype"],
            "dt": solver["dt"],
            "dealias_rule": numerical["dealias_rule"],
            "projected_transform_execution": numerical[
                "projected_transform_execution"
            ],
            "warmup_steps": warmup_steps,
            "profile_steps": profile_steps,
            "spectral_refresh_interval": numerical[
                "spectral_refresh_interval_steps"
            ],
            "reuse_q_gradients": True,
            "molecular_field_linear_space": numerical[
                "molecular_field_linear_space"
            ],
            "stress_divergence_sum_space": numerical[
                "stress_divergence_sum_space"
            ],
            "pointwise_execution": numerical["pointwise_execution"],
            "transform_execution_order": solver[
                "transform_execution_order"
            ],
            "spectral_storage": solver["spectral_storage"],
            "initial_q_path": str(reference.initial_q_path),
        },
        "configuration_comparison": comparison.to_metadata(),
        "environment": environment,
        "build_wall_seconds": build_seconds,
        "throughput": _timing_summary(samples),
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "final_q_sha256": q_hash,
        "finite": True,
    }


_PROFILE_CONFIG_KEYS = (
    "shape",
    "lengths",
    "device",
    "dtype",
    "dt",
    "dealias_rule",
    "projected_transform_execution",
    "warmup_steps",
    "profile_steps",
    "spectral_refresh_interval",
    "reuse_q_gradients",
    "molecular_field_linear_space",
    "stress_divergence_sum_space",
    "pointwise_execution",
    "transform_execution_order",
    "spectral_storage",
    "initial_q_path",
)


def _load_json(path: str | Path, description: str) -> dict[str, object]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must contain a JSON object")
    return dict(value)


def _profile_identity(profile: Mapping[str, object]) -> dict[str, object]:
    config = profile.get("config", profile.get("configuration"))
    if not isinstance(config, Mapping):
        raise ValueError("profile lacks a configuration mapping")
    try:
        return {name: config[name] for name in _PROFILE_CONFIG_KEYS}
    except KeyError as exc:
        raise ValueError(f"profile lacks configuration key {exc.args[0]!r}") from exc


def _profile_measurements(profile: Mapping[str, object]) -> tuple[float, int, int]:
    try:
        throughput = profile["throughput"]
        mean = float(throughput["mean_timestep_seconds"])
        memory = profile["memory"]
        allocated = int(memory["peak_allocated_bytes"])
        reserved = int(memory["peak_reserved_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("profile lacks timing or memory measurements") from exc
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError("profile mean timestep must be positive and finite")
    if allocated <= 0 or reserved <= 0 or allocated > reserved:
        raise ValueError("profile CUDA peak memory is invalid")
    return mean, allocated, reserved


def analyze_stage_m_h100_qualification(
    trajectory_comparison_path: str | Path,
    production_profile_paths: Sequence[str | Path],
    shadow_profile_paths: Sequence[str | Path],
    *,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Combine strict numerical evidence with separately measured profiles."""

    production_paths = tuple(
        Path(path).expanduser().resolve()
        for path in production_profile_paths
    )
    shadow_paths = tuple(
        Path(path).expanduser().resolve() for path in shadow_profile_paths
    )
    if not production_paths or len(production_paths) != len(shadow_paths):
        raise ValueError("production and shadow profiles need equal nonzero counts")
    comparison_path = Path(trajectory_comparison_path).expanduser().resolve()
    comparison = _load_json(comparison_path, "trajectory comparison")
    if comparison.get("classification") != "PASS":
        raise ValueError("trajectory comparison did not pass")
    if comparison.get("configuration", {}).get("compatible") is not True:
        raise ValueError("trajectory configuration comparison did not pass")
    maximum_error = float(comparison["maximum_gate_relative_l2"])
    tolerance = float(comparison["relative_l2_tolerance"])
    if not math.isfinite(maximum_error) or maximum_error > tolerance:
        raise ValueError("trajectory relative-L2 gate did not pass")

    production_profiles = [
        _load_json(path, "production profile") for path in production_paths
    ]
    shadow_profiles = [
        _load_json(path, "shadow profile") for path in shadow_paths
    ]
    identities = [
        _profile_identity(profile)
        for profile in (*production_profiles, *shadow_profiles)
    ]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("production and shadow profile configurations differ")
    if identities[0]["device"] != "cuda" or identities[0]["dtype"] != "float64":
        raise ValueError("Stage M profiles must use CUDA float64")

    def validate_environment(profile: Mapping[str, object]) -> None:
        try:
            environment = profile["environment"]
            available = environment["cuda_available"]
            device_name = environment["device_name"]
            tf32 = environment["cuda_matmul_allow_tf32"]
        except (KeyError, TypeError) as exc:
            raise ValueError("profile lacks CUDA environment provenance") from exc
        if available is not True or tf32 is not False:
            raise ValueError("Stage M requires CUDA with TF32 disabled")
        if (
            not isinstance(device_name, str)
            or expected_gpu_name.lower() not in device_name.lower()
        ):
            raise ValueError("profile GPU does not match expected_gpu_name")

    for profile in (*production_profiles, *shadow_profiles):
        validate_environment(profile)

    try:
        production_q_hashes = {
            profile["profile_input"]["initial_q_sha256"]
            for profile in production_profiles
        }
        shadow_q_hashes = {
            profile["production_initial_q_sha256"]
            for profile in shadow_profiles
        }
        trajectory_q_hashes = {
            value["production_sha256"]
            for value in comparison["arrays"]
            if value["field"] == "Q" and value["step"] == 0
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("qualification lacks initial-Q identity") from exc
    if (
        len(production_q_hashes) != 1
        or production_q_hashes != shadow_q_hashes
        or production_q_hashes != trajectory_q_hashes
    ):
        raise ValueError("production, shadow, and trajectory Q_0 identities differ")

    production_values = [
        _profile_measurements(profile) for profile in production_profiles
    ]
    shadow_values = [
        _profile_measurements(profile) for profile in shadow_profiles
    ]
    production_times = [value[0] for value in production_values]
    shadow_times = [value[0] for value in shadow_values]
    production_allocated = max(value[1] for value in production_values)
    shadow_allocated = max(value[1] for value in shadow_values)
    production_reserved = max(value[2] for value in production_values)
    shadow_reserved = max(value[2] for value in shadow_values)
    paired_speedups = [
        production / shadow
        for production, shadow in zip(
            production_times,
            shadow_times,
            strict=True,
        )
    ]
    return {
        "schema_version": 1,
        "qualification_stage": "M",
        "classification": "PASS",
        "numerical_equivalence_passed": True,
        "eligible_for_stage_n_architecture_decision": True,
        "eligible_for_production_promotion": False,
        "production_path_changed": False,
        "trajectory": {
            "comparison_path": str(comparison_path),
            "comparison_sha256": file_sha256(comparison_path),
            "relative_l2_tolerance": tolerance,
            "maximum_gate_relative_l2": maximum_error,
            "array_count": comparison["array_count"],
            "initial_q_sha256": next(iter(production_q_hashes)),
        },
        "profile_configuration": identities[0],
        "profile_count_per_implementation": len(production_profiles),
        "production": {
            "mean_timestep_seconds": (
                math.fsum(production_times) / len(production_times)
            ),
            "median_timestep_seconds": statistics.median(production_times),
            "peak_allocated_bytes": production_allocated,
            "peak_reserved_bytes": production_reserved,
        },
        "shadow": {
            "mean_timestep_seconds": math.fsum(shadow_times) / len(shadow_times),
            "median_timestep_seconds": statistics.median(shadow_times),
            "peak_allocated_bytes": shadow_allocated,
            "peak_reserved_bytes": shadow_reserved,
        },
        "comparison": {
            "paired_speedups_production_over_shadow": paired_speedups,
            "mean_speedup_production_over_shadow": (
                math.fsum(production_times) / math.fsum(shadow_times)
            ),
            "median_speedup_production_over_shadow": (
                statistics.median(production_times)
                / statistics.median(shadow_times)
            ),
            "peak_allocated_ratio_shadow_over_production": (
                shadow_allocated / production_allocated
            ),
            "peak_reserved_ratio_shadow_over_production": (
                shadow_reserved / production_reserved
            ),
            "performance_is_observational_not_a_promotion_gate": True,
        },
        "inputs": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in (comparison_path, *production_paths, *shadow_paths)
        ],
    }


def _trajectory_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded Stage M H100 Plane shadow trajectory."
    )
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-steps", type=int, required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    return parser


def trajectory_main(argv: Sequence[str] | None = None) -> int:
    args = _trajectory_parser().parse_args(argv)
    result = run_h100_plane_shadow_from_production_reference(
        args.production_reference_dir,
        args.output_dir,
        confirmed_steps=args.confirm_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile the Stage M H100 Plane shadow runtime."
    )
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def profile_main(argv: Sequence[str] | None = None) -> int:
    args = _profile_parser().parse_args(argv)
    result = profile_h100_plane_shadow_from_production_reference(
        args.production_reference_dir,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _analysis_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze bounded Stage M H100 trajectory/profile evidence."
    )
    parser.add_argument("--trajectory-comparison", type=Path, required=True)
    parser.add_argument(
        "--production-profile",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--shadow-profile", type=Path, action="append", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def analysis_main(argv: Sequence[str] | None = None) -> int:
    args = _analysis_parser().parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"analysis output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = analyze_stage_m_h100_qualification(
            args.trajectory_comparison,
            args.production_profile,
            args.shadow_profile,
            expected_gpu_name=args.expected_gpu_name,
        )
    except BaseException:
        (output / "FAILED").write_text("fail\n", encoding="utf-8")
        raise
    report = output / "stage_m_qualification.json"
    _write_new_json(report, result)
    (output / "COMPLETE").write_text("pass\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "analysis_main",
    "analyze_stage_m_h100_qualification",
    "profile_h100_plane_shadow_from_production_reference",
    "profile_main",
    "run_h100_plane_shadow_from_production_reference",
    "trajectory_main",
]
