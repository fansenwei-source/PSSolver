"""Bounded qualification for Stage N.1 representation reuse.

The candidate remains opt-in.  This module runs a short H100 shadow trajectory,
profiles control and candidate independently, audits transform-call counts, and
combines already-written evidence without promoting any production default.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics

import torch

from ._shadow_support import file_sha256, ordered_tensor_sha256
from .h100_shadow_qualification import (
    _cuda_identity,
    _cuda_step_samples,
    _require_positive_integer,
    _timing_summary,
    _write_new_json,
)
from .plane_shadow_driver import (
    build_h100_plane_shadow_runtime_from_production_metadata,
    load_h100_production_plane_reference,
)
from .shadow_run import ExperimentalPlaneShadowRun


_MODES = ("control", "candidate")


def _reuse_enabled(mode: str) -> bool:
    if mode not in _MODES:
        raise ValueError("mode must be 'control' or 'candidate'")
    return mode == "candidate"


def _initial_values(reference, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device)
        for name, value in reference.initial_values.items()
    }


def run_stage_n1_h100_candidate_trajectory(
    production_directory: str | Path,
    output_directory: str | Path,
    *,
    confirmed_steps: int,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Write one short candidate trajectory from the qualified production Q0."""

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
            enable_algebraic_representation_reuse=True,
        )
    )
    initial_values = _initial_values(reference, device)
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
            "qualification_stage": "N.1",
            "algebraic_representation_reuse": True,
        },
    )
    saved_steps = [run.save_observation().step]
    for step in range(1, confirmed_steps + 1):
        run.advance(1)
        if step % save_interval == 0 or step == confirmed_steps:
            saved_steps.append(run.save_observation().step)
    final = run.complete()
    diagnostics = runtime.algebraic_representation_reuse_diagnostics()
    if diagnostics is None or diagnostics.get("forward_hits", 0) <= 0:
        raise RuntimeError("Stage N.1 candidate did not reuse a representation")
    result = {
        "schema_version": 1,
        "qualification_stage": "N.1",
        "measurement_role": "candidate_numerical_trajectory",
        "classification": "TRAJECTORY_COMPLETE",
        "production_path_changed": False,
        "production_promotion_authorized": False,
        "production_directory": str(reference.directory),
        "candidate_directory": str(run.output_directory),
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "completed_steps": final.step,
        "saved_steps": saved_steps,
        "configuration": comparison.to_metadata(),
        "representation_reuse": dict(diagnostics),
        "environment": environment,
    }
    _write_new_json(run.output_directory / "stage_n1_metrics.json", result)
    return result


def _transform_call_audit(
    reference,
    *,
    mode: str,
    steps: int,
    expected_gpu_name: str,
    device: torch.device,
) -> dict[str, object]:
    runtime, _ = build_h100_plane_shadow_runtime_from_production_metadata(
        reference.metadata,
        expected_gpu_name=expected_gpu_name,
        device=device,
        enable_performance_instrumentation=True,
        enable_algebraic_representation_reuse=_reuse_enabled(mode),
    )
    runtime.reset(_initial_values(reference, device))
    recorder = runtime.performance_recorder
    if recorder is None:
        raise RuntimeError("transform audit lacks instrumentation")
    # Reset leaves the initial algebraic state synchronized.  Advance once so
    # every audited step exercises the steady pre-RHS algebraic update.
    runtime.solver.run(1)
    recorder.reset()
    runtime.solver.run(steps)
    snapshot = recorder.snapshot()
    regions = snapshot["regions"]
    reuse = runtime.algebraic_representation_reuse_diagnostics()
    return {
        "steps": steps,
        "forward_calls": regions["transform.forward"]["calls"],
        "inverse_calls": regions["transform.inverse"]["calls"],
        "forward_calls_per_step": (
            regions["transform.forward"]["calls"] / steps
        ),
        "inverse_calls_per_step": (
            regions["transform.inverse"]["calls"] / steps
        ),
        "representation_reuse": dict(reuse) if reuse is not None else None,
        "separate_from_throughput_timing": True,
    }


def profile_stage_n1_h100_shadow(
    production_directory: str | Path,
    *,
    mode: str,
    warmup_steps: int,
    profile_steps: int,
    transform_audit_steps: int,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Measure an uninstrumented control or candidate on one H100."""

    enabled = _reuse_enabled(mode)
    _require_positive_integer(warmup_steps, "warmup_steps")
    _require_positive_integer(profile_steps, "profile_steps")
    _require_positive_integer(transform_audit_steps, "transform_audit_steps")
    reference = load_h100_production_plane_reference(
        production_directory,
        expected_gpu_name=expected_gpu_name,
    )
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
            enable_algebraic_representation_reuse=enabled,
        )
    )
    runtime.reset(_initial_values(reference, device))
    _cuda_step_samples(runtime, warmup_steps)
    torch.cuda.reset_peak_memory_stats(device)
    samples = _cuda_step_samples(runtime, profile_steps)
    runtime.synchronize_algebraic_for_observation()
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    fields = runtime.solver.fields
    finite = bool(torch.isfinite(fields.spatial).all().item()) and bool(
        torch.isfinite(fields.spectral).all().item()
    )
    if not finite:
        raise RuntimeError("Stage N.1 profile ended with non-finite fields")
    final_q_sha256 = ordered_tensor_sha256(
        {name: fields[name] for name in reference.initial_values}
    )
    signature = comparison.production_signature
    numerical = signature["numerics"]
    solver = signature["solver"]
    return {
        "schema_version": 1,
        "qualification_stage": "N.1",
        "measurement_role": "representation_reuse_performance_profile",
        "classification": "PROFILE_COMPLETE",
        "mode": mode,
        "algebraic_representation_reuse": enabled,
        "production_path_changed": False,
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "configuration": {
            "shape": solver["shape"],
            "lengths": solver["lengths"],
            "dtype": solver["real_dtype"],
            "dt": solver["dt"],
            "dealias_rule": numerical["dealias_rule"],
            "projected_transform_execution": numerical[
                "projected_transform_execution"
            ],
            "spectral_storage": solver["spectral_storage"],
            "spectral_refresh_interval": numerical[
                "spectral_refresh_interval_steps"
            ],
            "warmup_steps": warmup_steps,
            "profile_steps": profile_steps,
            "transform_audit_steps": transform_audit_steps,
        },
        "configuration_comparison": comparison.to_metadata(),
        "environment": environment,
        "throughput": _timing_summary(samples),
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "transform_call_audit": _transform_call_audit(
            reference,
            mode=mode,
            steps=transform_audit_steps,
            expected_gpu_name=expected_gpu_name,
            device=device,
        ),
        "final_q_sha256": final_q_sha256,
        "finite": True,
    }


def _load_json(path: str | Path, description: str) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} is missing: {resolved}")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{description} must contain a JSON object")
    return value


def _profile_measurement(profile: Mapping[str, object]) -> tuple[float, int, int]:
    try:
        mean = float(profile["throughput"]["mean_timestep_seconds"])
        allocated = int(profile["memory"]["peak_allocated_bytes"])
        reserved = int(profile["memory"]["peak_reserved_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("profile lacks timing or memory measurements") from exc
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError("profile mean timestep must be positive and finite")
    if allocated <= 0 or reserved <= 0 or allocated > reserved:
        raise ValueError("profile peak memory is invalid")
    return mean, allocated, reserved


def analyze_stage_n1_qualification(
    trajectory_comparison_path: str | Path,
    control_profile_paths: Sequence[str | Path],
    candidate_profile_paths: Sequence[str | Path],
    *,
    minimum_mean_speedup: float = 1.02,
    maximum_memory_ratio: float = 1.05,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Apply numerical, transform-count, performance, and memory gates."""

    comparison_path = Path(trajectory_comparison_path).expanduser().resolve()
    comparison = _load_json(comparison_path, "trajectory comparison")
    if comparison.get("classification") != "PASS":
        raise ValueError("candidate trajectory comparison did not pass")
    try:
        tolerance = float(comparison["relative_l2_tolerance"])
        maximum_error = float(comparison["maximum_gate_relative_l2"])
        array_count = int(comparison["array_count"])
        saved_steps = list(comparison["saved_steps"])
        comparison_arrays = comparison["arrays"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("trajectory comparison lacks the fixed gate") from exc
    if (
        tolerance != 1.0e-10
        or not math.isfinite(maximum_error)
        or maximum_error > tolerance
        or array_count != 21
        or saved_steps != list(range(7))
        or not isinstance(comparison_arrays, list)
        or len(comparison_arrays) != 21
        or not all(isinstance(value, Mapping) for value in comparison_arrays)
    ):
        raise ValueError("trajectory comparison violates the fixed N.1 gate")
    if (
        not isinstance(minimum_mean_speedup, (int, float))
        or isinstance(minimum_mean_speedup, bool)
        or not math.isfinite(float(minimum_mean_speedup))
        or float(minimum_mean_speedup) <= 0.0
    ):
        raise ValueError("minimum_mean_speedup must be positive and finite")
    if (
        not isinstance(maximum_memory_ratio, (int, float))
        or isinstance(maximum_memory_ratio, bool)
        or not math.isfinite(float(maximum_memory_ratio))
        or float(maximum_memory_ratio) <= 0.0
    ):
        raise ValueError("maximum_memory_ratio must be positive and finite")
    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")
    control_paths = tuple(Path(path).expanduser().resolve() for path in control_profile_paths)
    candidate_paths = tuple(
        Path(path).expanduser().resolve() for path in candidate_profile_paths
    )
    if not control_paths or len(control_paths) != len(candidate_paths):
        raise ValueError("control and candidate need equal nonzero profile counts")
    controls = [_load_json(path, "control profile") for path in control_paths]
    candidates = [
        _load_json(path, "candidate profile") for path in candidate_paths
    ]
    for profile, mode in (
        *((profile, "control") for profile in controls),
        *((profile, "candidate") for profile in candidates),
    ):
        if (
            profile.get("qualification_stage") != "N.1"
            or profile.get("classification") != "PROFILE_COMPLETE"
            or profile.get("mode") != mode
            or profile.get("finite") is not True
            or profile.get("algebraic_representation_reuse")
            is not (mode == "candidate")
        ):
            raise ValueError(f"{mode} profile is incomplete")
        try:
            environment = profile["environment"]
            cuda_available = environment["cuda_available"]
            device_name = environment["device_name"]
            tf32 = environment["cuda_matmul_allow_tf32"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{mode} profile lacks GPU provenance") from exc
        if (
            cuda_available is not True
            or tf32 is not False
            or not isinstance(device_name, str)
            or expected_gpu_name.lower() not in device_name.lower()
        ):
            raise ValueError(f"{mode} profile has incompatible GPU provenance")
    identities = [profile["configuration"] for profile in (*controls, *candidates)]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("control and candidate configurations differ")
    expected_configuration = {
        "shape": [128, 128, 32],
        "lengths": [100.0, 100.0, 20.0],
        "dtype": "float64",
        "dt": 0.005,
        "dealias_rule": "cubic_half",
        "projected_transform_execution": "truncated",
        "spectral_storage": "hermitian_half",
        "spectral_refresh_interval": 2,
        "warmup_steps": 10,
        "profile_steps": 20,
        "transform_audit_steps": 2,
    }
    if identities[0] != expected_configuration:
        raise ValueError("profile configuration violates the fixed N.1 gate")
    input_identities = {
        (
            profile["production_metadata_sha256"],
            profile["production_initial_q_sha256"],
        )
        for profile in (*controls, *candidates)
    }
    if len(input_identities) != 1:
        raise ValueError("profile production inputs differ")
    q0_hashes = {
        value.get("production_sha256")
        for value in comparison_arrays
        if value.get("field") == "Q" and value.get("step") == 0
    }
    profile_q0_hash = next(iter(input_identities))[1]
    if q0_hashes != {profile_q0_hash}:
        raise ValueError("trajectory and profile Q0 identities differ")

    control_values = [_profile_measurement(profile) for profile in controls]
    candidate_values = [_profile_measurement(profile) for profile in candidates]
    control_times = [value[0] for value in control_values]
    candidate_times = [value[0] for value in candidate_values]
    paired_speedups = [
        control / candidate
        for control, candidate in zip(
            control_times,
            candidate_times,
            strict=True,
        )
    ]
    mean_control = math.fsum(control_times) / len(control_times)
    mean_candidate = math.fsum(candidate_times) / len(candidate_times)
    mean_speedup = mean_control / mean_candidate
    allocated_ratio = max(value[1] for value in candidate_values) / max(
        value[1] for value in control_values
    )
    reserved_ratio = max(value[2] for value in candidate_values) / max(
        value[2] for value in control_values
    )
    control_forward = [
        float(profile["transform_call_audit"]["forward_calls_per_step"])
        for profile in controls
    ]
    candidate_forward = [
        float(profile["transform_call_audit"]["forward_calls_per_step"])
        for profile in candidates
    ]
    control_inverse = [
        float(profile["transform_call_audit"]["inverse_calls_per_step"])
        for profile in controls
    ]
    candidate_inverse = [
        float(profile["transform_call_audit"]["inverse_calls_per_step"])
        for profile in candidates
    ]
    transform_counts = (
        *control_forward,
        *candidate_forward,
        *control_inverse,
        *candidate_inverse,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in transform_counts):
        raise ValueError("transform audit counts must be positive and finite")
    if any(
        profile["transform_call_audit"]["representation_reuse"] is not None
        for profile in controls
    ):
        raise ValueError("control profile unexpectedly enabled representation reuse")
    reuse_valid = all(
        profile["transform_call_audit"]["representation_reuse"] is not None
        and profile["transform_call_audit"]["representation_reuse"][
            "forward_hits"
        ]
        > 0
        and profile["transform_call_audit"]["representation_reuse"][
            "retained_pairs_after_generation"
        ]
        == 0
        for profile in candidates
    )
    transform_gate = all(
        candidate < control
        for control, candidate in zip(
            control_forward,
            candidate_forward,
            strict=True,
        )
    ) and all(
        candidate <= control
        for control, candidate in zip(
            control_inverse,
            candidate_inverse,
            strict=True,
        )
    )
    faster_count = sum(speedup > 1.0 for speedup in paired_speedups)
    performance_gate = (
        mean_speedup >= minimum_mean_speedup
        and faster_count >= math.ceil(len(paired_speedups) * 2 / 3)
    )
    memory_gate = (
        allocated_ratio <= maximum_memory_ratio
        and reserved_ratio <= maximum_memory_ratio
    )
    accepted = transform_gate and reuse_valid and performance_gate and memory_gate
    classification = "A_recommended" if accepted else "B_neutral"
    return {
        "schema_version": 1,
        "qualification_stage": "N.1",
        "classification": classification,
        "numerical_equivalence_passed": True,
        "architecture_decision": "retain_opt_in_representation_reuse_candidate",
        "eligible_for_stage_n2_optimization_design": accepted,
        "eligible_for_production_promotion": False,
        "production_path_changed": False,
        "trajectory": {
            "comparison_path": str(comparison_path),
            "comparison_sha256": file_sha256(comparison_path),
            "maximum_gate_relative_l2": comparison[
                "maximum_gate_relative_l2"
            ],
            "relative_l2_tolerance": comparison["relative_l2_tolerance"],
            "array_count": comparison["array_count"],
        },
        "profile_count_per_mode": len(controls),
        "control": {
            "mean_timestep_seconds": mean_control,
            "median_timestep_seconds": statistics.median(control_times),
            "peak_allocated_bytes": max(value[1] for value in control_values),
            "peak_reserved_bytes": max(value[2] for value in control_values),
            "mean_forward_calls_per_step": math.fsum(control_forward)
            / len(control_forward),
            "mean_inverse_calls_per_step": math.fsum(control_inverse)
            / len(control_inverse),
        },
        "candidate": {
            "mean_timestep_seconds": mean_candidate,
            "median_timestep_seconds": statistics.median(candidate_times),
            "peak_allocated_bytes": max(value[1] for value in candidate_values),
            "peak_reserved_bytes": max(value[2] for value in candidate_values),
            "mean_forward_calls_per_step": math.fsum(candidate_forward)
            / len(candidate_forward),
            "mean_inverse_calls_per_step": math.fsum(candidate_inverse)
            / len(candidate_inverse),
        },
        "comparison": {
            "mean_speedup_control_over_candidate": mean_speedup,
            "median_paired_speedup": statistics.median(paired_speedups),
            "paired_speedups": paired_speedups,
            "candidate_faster_count": faster_count,
            "peak_allocated_ratio_candidate_over_control": allocated_ratio,
            "peak_reserved_ratio_candidate_over_control": reserved_ratio,
            "forward_transform_reduction_gate": transform_gate,
            "generation_lifetime_gate": reuse_valid,
            "performance_gate": performance_gate,
            "memory_gate": memory_gate,
            "minimum_mean_speedup": minimum_mean_speedup,
            "maximum_memory_ratio": maximum_memory_ratio,
        },
    }


def _trajectory_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage N.1 candidate trajectory")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-steps", type=int, required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    return parser


def trajectory_main(argv: Sequence[str] | None = None) -> int:
    args = _trajectory_parser().parse_args(argv)
    result = run_stage_n1_h100_candidate_trajectory(
        args.production_reference_dir,
        args.output_dir,
        confirmed_steps=args.confirm_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile Stage N.1 on H100")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--transform-audit-steps", type=int, default=2)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def profile_main(argv: Sequence[str] | None = None) -> int:
    args = _profile_parser().parse_args(argv)
    result = profile_stage_n1_h100_shadow(
        args.production_reference_dir,
        mode=args.mode,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        transform_audit_steps=args.transform_audit_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _analysis_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Stage N.1 evidence")
    parser.add_argument("--trajectory-comparison", type=Path, required=True)
    parser.add_argument("--control-profile", type=Path, action="append", required=True)
    parser.add_argument("--candidate-profile", type=Path, action="append", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def analysis_main(argv: Sequence[str] | None = None) -> int:
    args = _analysis_parser().parse_args(argv)
    result = analyze_stage_n1_qualification(
        args.trajectory_comparison,
        args.control_profile,
        args.candidate_profile,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "analysis_main",
    "analyze_stage_n1_qualification",
    "profile_main",
    "profile_stage_n1_h100_shadow",
    "run_stage_n1_h100_candidate_trajectory",
    "trajectory_main",
]
