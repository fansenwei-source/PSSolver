"""Bounded Stage N.4 qualification for architectural consolidation.

The control uses the frozen Stage N.3 compatibility flags.  The candidate
selects the same execution semantics through one immutable policy and the
extracted scheduler service.  This stage must preserve transform counts and
field-lifecycle diagnostics while avoiding a material performance regression.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics

import torch

from pssolver.diagnostics import cuda_memory_snapshot, cuda_memory_window
from pssolver.execution import AlgebraicExecutionPolicy

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
from .stage_n1_qualification import _load_json, _profile_measurement


_MODES = ("control", "candidate")
_LIFECYCLE_KEYS = (
    "physical_materializations",
    "on_demand_physical_materializations",
    "physical_materialization_batches",
    "batched_physical_components",
    "singleton_materialization_batches",
    "maximum_materialization_batch_size",
    "physical_island_prefetches",
    "physical_island_requested_components",
    "unmaterialized_published_components",
)


def _require_mode(mode: str) -> str:
    if mode not in _MODES:
        raise ValueError("mode must be 'control' or 'candidate'")
    return mode


def _initial_values(reference, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device=device)
        for name, value in reference.initial_values.items()
    }


def _build_runtime(
    reference,
    *,
    mode: str,
    expected_gpu_name: str,
    device: torch.device,
    instrument: bool = False,
):
    mode = _require_mode(mode)
    common = dict(
        expected_gpu_name=expected_gpu_name,
        device=device,
        enable_performance_instrumentation=instrument,
    )
    if mode == "control":
        common.update(
            enable_algebraic_representation_reuse=True,
            enable_lazy_algebraic_materialization=True,
            enable_batched_physical_islands=True,
        )
    else:
        common["algebraic_execution_policy"] = (
            AlgebraicExecutionPolicy.batched()
        )
    return build_h100_plane_shadow_runtime_from_production_metadata(
        reference.metadata,
        **common,
    )


def run_stage_n4_h100_candidate_trajectory(
    production_directory: str | Path,
    output_directory: str | Path,
    *,
    confirmed_steps: int,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Write one short unified-policy trajectory from production Q0."""

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
    runtime, comparison = _build_runtime(
        reference,
        mode="candidate",
        expected_gpu_name=expected_gpu_name,
        device=device,
    )
    run = ExperimentalPlaneShadowRun(
        runtime,
        output_directory,
        initial_values=_initial_values(reference, device),
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
            "qualification_stage": "N.4",
            "configuration_authority": "unified_execution_policy",
            "algebraic_execution_policy": (
                runtime.algebraic_execution_policy.to_metadata()
            ),
        },
    )
    saved_steps = [run.save_observation().step]
    for step in range(1, confirmed_steps + 1):
        run.advance(1)
        if step % save_interval == 0 or step == confirmed_steps:
            saved_steps.append(run.save_observation().step)
    final = run.complete()
    reuse = runtime.algebraic_representation_reuse_diagnostics()
    materialization = runtime.algebraic_physical_materialization_diagnostics()
    if reuse is None or reuse.get("retained_pairs_after_generation") != 0:
        raise RuntimeError("Stage N.4 representation lifecycle is invalid")
    if (
        materialization is None
        or materialization.get("physical_materialization_batches", math.inf)
        >= materialization.get("physical_materializations", 0)
        or materialization.get("on_demand_physical_materializations") != 0
    ):
        raise RuntimeError("Stage N.4 scheduler lifecycle is invalid")
    result = {
        "schema_version": 1,
        "qualification_stage": "N.4",
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
        "algebraic_execution_policy": (
            runtime.algebraic_execution_policy.to_metadata()
        ),
        "representation_reuse": dict(reuse),
        "physical_materialization": dict(materialization),
        "environment": environment,
    }
    _write_new_json(run.output_directory / "stage_n4_metrics.json", result)
    return result


def _transform_call_audit(
    reference,
    *,
    mode: str,
    steps: int,
    expected_gpu_name: str,
    device: torch.device,
) -> dict[str, object]:
    runtime, _ = _build_runtime(
        reference,
        mode=mode,
        expected_gpu_name=expected_gpu_name,
        device=device,
        instrument=True,
    )
    runtime.reset(_initial_values(reference, device))
    recorder = runtime.performance_recorder
    if recorder is None:
        raise RuntimeError("transform audit lacks instrumentation")
    runtime.solver.run(1)
    recorder.reset()
    runtime.solver.run(steps)
    regions = recorder.snapshot()["regions"]
    reuse = runtime.algebraic_representation_reuse_diagnostics()
    materialization = runtime.algebraic_physical_materialization_diagnostics()
    return {
        "steps": steps,
        "forward_calls": regions["transform.forward"]["calls"],
        "inverse_calls": regions["transform.inverse"]["calls"],
        "forward_calls_per_step": regions["transform.forward"]["calls"] / steps,
        "inverse_calls_per_step": regions["transform.inverse"]["calls"] / steps,
        "execution_policy": runtime.algebraic_execution_policy.to_metadata(),
        "representation_reuse": dict(reuse) if reuse is not None else None,
        "physical_materialization": (
            dict(materialization) if materialization is not None else None
        ),
        "separate_from_throughput_timing": True,
    }


def profile_stage_n4_h100_shadow(
    production_directory: str | Path,
    *,
    mode: str,
    warmup_steps: int,
    profile_steps: int,
    transform_audit_steps: int,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Measure the frozen N.3 facade or consolidated N.4 policy path."""

    mode = _require_mode(mode)
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
    memory_phases = {"before_build": cuda_memory_snapshot(device)}
    runtime, comparison = _build_runtime(
        reference,
        mode=mode,
        expected_gpu_name=expected_gpu_name,
        device=device,
    )
    torch.cuda.synchronize(device)
    memory_phases["after_build"] = cuda_memory_snapshot(device)
    runtime.reset(_initial_values(reference, device))
    torch.cuda.synchronize(device)
    memory_phases["after_reset"] = cuda_memory_snapshot(device)
    _cuda_step_samples(runtime, warmup_steps)
    memory_phases["after_warmup"] = cuda_memory_snapshot(device)
    torch.cuda.reset_peak_memory_stats(device)
    samples = _cuda_step_samples(runtime, profile_steps)
    memory_phases["after_timestep_window"] = cuda_memory_snapshot(device)
    timestep_peak = cuda_memory_snapshot(device)
    torch.cuda.reset_peak_memory_stats(device)
    runtime.synchronize_algebraic_for_observation()
    torch.cuda.synchronize(device)
    memory_phases["after_observation"] = cuda_memory_snapshot(device)
    observation_peak = cuda_memory_snapshot(device)
    timestep_window = cuda_memory_window(
        memory_phases["after_warmup"],
        memory_phases["after_timestep_window"],
        timestep_peak,
    )
    observation_window = cuda_memory_window(
        memory_phases["after_timestep_window"],
        memory_phases["after_observation"],
        observation_peak,
    )
    peak_allocated = max(
        int(timestep_window["peak_allocated_bytes"]),
        int(observation_window["peak_allocated_bytes"]),
    )
    peak_reserved = max(
        int(timestep_window["peak_reserved_bytes"]),
        int(observation_window["peak_reserved_bytes"]),
    )
    fields = runtime.solver.fields
    finite = bool(torch.isfinite(fields.spatial).all().item()) and bool(
        torch.isfinite(fields.spectral).all().item()
    )
    if not finite:
        raise RuntimeError("Stage N.4 profile ended with non-finite fields")
    signature = comparison.production_signature
    numerical = signature["numerics"]
    solver = signature["solver"]
    return {
        "schema_version": 1,
        "qualification_stage": "N.4",
        "measurement_role": "architecture_consolidation_profile",
        "classification": "PROFILE_COMPLETE",
        "mode": mode,
        "configuration_authority": (
            "legacy_stage_flags"
            if mode == "control"
            else "unified_execution_policy"
        ),
        "algebraic_execution_policy": (
            runtime.algebraic_execution_policy.to_metadata()
        ),
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
        "memory_attribution": {
            "schema_version": 1,
            "allocator_scope": "current_process_cuda_allocator",
            "peak_reset_after_warmup": True,
            "observation_peak_reset_after_timestep_window": True,
            "phases": memory_phases,
            "windows": {
                "timestep": timestep_window,
                "observation": observation_window,
            },
        },
        "transform_call_audit": _transform_call_audit(
            reference,
            mode=mode,
            steps=transform_audit_steps,
            expected_gpu_name=expected_gpu_name,
            device=device,
        ),
        "final_q_sha256": ordered_tensor_sha256(
            {name: fields[name] for name in reference.initial_values}
        ),
        "finite": True,
    }


def analyze_stage_n4_qualification(
    trajectory_comparison_path: str | Path,
    control_profile_paths: Sequence[str | Path],
    candidate_profile_paths: Sequence[str | Path],
    *,
    maximum_mean_timestep_ratio: float = 1.03,
    maximum_paired_timestep_ratio: float = 1.05,
    maximum_memory_ratio: float = 1.03,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Apply numerical, structural, lifecycle, timing, and memory gates."""

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
    ):
        raise ValueError("trajectory comparison violates the fixed N.4 gate")
    for value, description in (
        (maximum_mean_timestep_ratio, "maximum_mean_timestep_ratio"),
        (maximum_paired_timestep_ratio, "maximum_paired_timestep_ratio"),
        (maximum_memory_ratio, "maximum_memory_ratio"),
    ):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{description} must be positive and finite")

    control_paths = tuple(
        Path(path).expanduser().resolve() for path in control_profile_paths
    )
    candidate_paths = tuple(
        Path(path).expanduser().resolve() for path in candidate_profile_paths
    )
    controls = [_load_json(path, "control profile") for path in control_paths]
    candidates = [
        _load_json(path, "candidate profile") for path in candidate_paths
    ]
    if not controls or len(controls) != len(candidates):
        raise ValueError("control and candidate profile counts must match")
    for profile, mode in (
        *((profile, "control") for profile in controls),
        *((profile, "candidate") for profile in candidates),
    ):
        if mode == "control":
            identity_valid = (
                profile.get("qualification_stage") == "N.3"
                and profile.get("mode") == "candidate"
                and profile.get("algebraic_representation_reuse") is True
                and profile.get("lazy_algebraic_materialization") is True
                and profile.get("batched_physical_islands") is True
            )
        else:
            policy = profile.get("algebraic_execution_policy")
            identity_valid = (
                profile.get("qualification_stage") == "N.4"
                and profile.get("mode") == "candidate"
                and profile.get("configuration_authority")
                == "unified_execution_policy"
                and isinstance(policy, Mapping)
                and policy.get("mode") == "batched_physical_islands"
            )
        if (
            not identity_valid
            or profile.get("classification") != "PROFILE_COMPLETE"
            or profile.get("finite") is not True
        ):
            raise ValueError(f"{mode} profile is incomplete")
        environment = profile.get("environment")
        if not isinstance(environment, Mapping) or (
            environment.get("cuda_available") is not True
            or environment.get("cuda_matmul_allow_tf32") is not False
            or not isinstance(environment.get("device_name"), str)
            or expected_gpu_name.lower() not in environment["device_name"].lower()
        ):
            raise ValueError(f"{mode} profile has incompatible GPU provenance")
        audit = profile.get("transform_call_audit")
        if (
            not isinstance(audit, Mapping)
            or not isinstance(audit.get("representation_reuse"), Mapping)
            or not isinstance(audit.get("physical_materialization"), Mapping)
            or (
                mode == "candidate"
                and not isinstance(audit.get("execution_policy"), Mapping)
            )
        ):
            raise ValueError(f"{mode} profile lacks lifecycle diagnostics")

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
        raise ValueError("profile configuration violates the fixed N.4 gate")
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
    if q0_hashes != {next(iter(input_identities))[1]}:
        raise ValueError("trajectory and profile Q0 identities differ")
    final_q_identity_gate = len(
        {profile.get("final_q_sha256") for profile in (*controls, *candidates)}
    ) == 1

    control_values = [_profile_measurement(profile) for profile in controls]
    candidate_values = [_profile_measurement(profile) for profile in candidates]
    control_times = [value[0] for value in control_values]
    candidate_times = [value[0] for value in candidate_values]
    paired_ratios = [
        candidate / control
        for control, candidate in zip(control_times, candidate_times, strict=True)
    ]
    mean_control = math.fsum(control_times) / len(control_times)
    mean_candidate = math.fsum(candidate_times) / len(candidate_times)
    mean_ratio = mean_candidate / mean_control
    paired_timing_gate = sum(
        ratio <= maximum_paired_timestep_ratio for ratio in paired_ratios
    ) >= math.ceil(len(paired_ratios) * 2 / 3)
    performance_gate = (
        mean_ratio <= maximum_mean_timestep_ratio and paired_timing_gate
    )
    allocated_ratio = max(value[1] for value in candidate_values) / max(
        value[1] for value in control_values
    )
    reserved_ratio = max(value[2] for value in candidate_values) / max(
        value[2] for value in control_values
    )
    memory_gate = (
        allocated_ratio <= maximum_memory_ratio
        and reserved_ratio <= maximum_memory_ratio
    )

    def audit_values(profiles, key):
        return [
            float(profile["transform_call_audit"][key]) for profile in profiles
        ]

    control_forward = audit_values(controls, "forward_calls_per_step")
    candidate_forward = audit_values(candidates, "forward_calls_per_step")
    control_inverse = audit_values(controls, "inverse_calls_per_step")
    candidate_inverse = audit_values(candidates, "inverse_calls_per_step")
    transform_count_gate = (
        control_forward == candidate_forward
        and control_inverse == candidate_inverse
    )

    control_materializations = [
        profile["transform_call_audit"]["physical_materialization"]
        for profile in controls
    ]
    candidate_materializations = [
        profile["transform_call_audit"]["physical_materialization"]
        for profile in candidates
    ]
    lifecycle_identity_gate = all(
        all(control.get(key) == candidate.get(key) for key in _LIFECYCLE_KEYS)
        and control.get("on_demand_physical_materializations") == 0
        and control.get("physical_materialization_batches", math.inf)
        < control.get("physical_materializations", 0)
        and control_profile["transform_call_audit"]["representation_reuse"].get(
            "retained_pairs_after_generation"
        )
        == 0
        and candidate_profile["transform_call_audit"][
            "representation_reuse"
        ].get("retained_pairs_after_generation")
        == 0
        for control_profile, candidate_profile, control, candidate in zip(
            controls,
            candidates,
            control_materializations,
            candidate_materializations,
            strict=True,
        )
    )
    accepted = (
        final_q_identity_gate
        and transform_count_gate
        and lifecycle_identity_gate
        and performance_gate
        and memory_gate
    )
    return {
        "schema_version": 1,
        "qualification_stage": "N.4",
        "classification": "A_recommended" if accepted else "B_neutral",
        "numerical_equivalence_passed": True,
        "architecture_decision": (
            "adopt_unified_execution_policy_and_scheduler_in_shadow_runtime"
            if accepted
            else "retain_stage_n3_compatibility_path"
        ),
        "eligible_for_stage_o_production_migration_design": accepted,
        "eligible_for_production_promotion": False,
        "production_path_changed": False,
        "trajectory": {
            "comparison_path": str(comparison_path),
            "comparison_sha256": file_sha256(comparison_path),
            "maximum_gate_relative_l2": maximum_error,
            "relative_l2_tolerance": tolerance,
            "array_count": array_count,
        },
        "profile_count_per_mode": len(controls),
        "control": {
            "configuration_authority": "frozen_stage_n3_parent_commit",
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
            "configuration_authority": "unified_execution_policy",
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
            "mean_timestep_ratio_candidate_over_control": mean_ratio,
            "median_paired_timestep_ratio": statistics.median(paired_ratios),
            "paired_timestep_ratios": paired_ratios,
            "peak_allocated_ratio_candidate_over_control": allocated_ratio,
            "peak_reserved_ratio_candidate_over_control": reserved_ratio,
            "final_q_identity_gate": final_q_identity_gate,
            "transform_count_identity_gate": transform_count_gate,
            "lifecycle_identity_gate": lifecycle_identity_gate,
            "performance_non_regression_gate": performance_gate,
            "memory_gate": memory_gate,
            "maximum_mean_timestep_ratio": maximum_mean_timestep_ratio,
            "maximum_paired_timestep_ratio": maximum_paired_timestep_ratio,
            "maximum_memory_ratio": maximum_memory_ratio,
        },
    }


def trajectory_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage N.4 trajectory")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-steps", type=int, required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    result = run_stage_n4_h100_candidate_trajectory(
        args.production_reference_dir,
        args.output_dir,
        confirmed_steps=args.confirm_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def profile_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile Stage N.4 on H100")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--transform-audit-steps", type=int, default=2)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = profile_stage_n4_h100_shadow(
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


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage N.4 evidence")
    parser.add_argument("--trajectory-comparison", type=Path, required=True)
    parser.add_argument(
        "--control-profile", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--candidate-profile", type=Path, action="append", required=True
    )
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze_stage_n4_qualification(
        args.trajectory_comparison,
        args.control_profile,
        args.candidate_profile,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "analyze_stage_n4_qualification",
    "analysis_main",
    "profile_main",
    "profile_stage_n4_h100_shadow",
    "run_stage_n4_h100_candidate_trajectory",
    "trajectory_main",
]
