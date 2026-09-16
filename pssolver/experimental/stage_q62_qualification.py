"""Bounded Stage Q.6.2 qualification for native-segment handoff."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

import torch

from pssolver.execution import AlgebraicOutputPublicationPolicy

from ._shadow_support import file_sha256
from .h100_shadow_qualification import (
    _cuda_identity,
    _require_positive_integer,
    _write_new_json,
)
from .plane_shadow_driver import load_h100_production_plane_reference
from .projected_scheduler import ProjectedBatchAssemblyPolicy
from .shadow_run import ExperimentalPlaneShadowRun
from .stage_n1_qualification import _load_json
from .stage_n4_qualification import _initial_values
from .stage_o43_qualification import (
    _StageO43Variant,
    _array_comparison,
    _build_runtime,
    _mean,
    _producer_output_snapshot,
    _profile_stage_o43_variant_h100_runtime,
)
from .stage_q4_qualification import _validate_compiled_rhs
from .stage_q61_design import (
    STAGE_Q61_ARCHITECTURE_DECISION,
    STAGE_Q61_CLASSIFICATION,
    STAGE_Q61_REQUIRED_SOURCES,
)


STAGE_Q62_PROFILE_TRIALS = 3
STAGE_Q62_DECISION_SHAPE = (320, 320, 80)
STAGE_Q62_TRAJECTORY_STEPS = 6
STAGE_Q62_RELATIVE_L2_TOLERANCE = 1.0e-12
STAGE_Q62_MAXIMUM_MEAN_TIMESTEP_RATIO = 0.98
STAGE_Q62_MAXIMUM_PAIRED_TIMESTEP_RATIO = 1.02
STAGE_Q62_MAXIMUM_SAFETY_TIMESTEP_RATIO = 1.05
STAGE_Q62_MAXIMUM_PEAK_ALLOCATED_RATIO = 1.05
STAGE_Q62_MAXIMUM_PEAK_RESERVED_RATIO = 1.05

_ROLES = ("baseline", "candidate")
_FIELDS = ("Q", "u", "p")
_VARIANT = _StageO43Variant(
    qualification_stage="Q.6.2",
    baseline_output_policy=AlgebraicOutputPublicationPolicy.deferred_stack(),
    baseline_batch_policy=(
        ProjectedBatchAssemblyPolicy.contiguous_storage_view()
    ),
    baseline_producer_output_layout="boundary_packed",
    candidate_output_policy=AlgebraicOutputPublicationPolicy.deferred_stack(),
    candidate_batch_policy=ProjectedBatchAssemblyPolicy.native_segments(
        source_names=STAGE_Q61_REQUIRED_SOURCES
    ),
    candidate_producer_output_layout="boundary_packed",
    architecture_decision="producer_owned_boundary_signature_native_segments",
    metrics_filename="stage_q62_metrics.json",
    structural_contract="three_source_native_segment_handoff",
)


def _require_role(role: str) -> str:
    if role not in _ROLES:
        raise ValueError("role must be 'baseline' or 'candidate'")
    return role


def _read_q61_evidence(
    path: str | Path,
    expected_sha256: str,
) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    if file_sha256(resolved) != expected_sha256:
        raise ValueError("Stage Q.6.1 evidence SHA-256 differs")
    report = _load_json(resolved, "Stage Q.6.1 evidence")
    try:
        candidate = report["stage_q62_candidate"]
        valid = bool(
            report["qualification_stage"] == "Q.6.1"
            and report["classification"] == STAGE_Q61_CLASSIFICATION
            and report["architecture_decision"]
            == STAGE_Q61_ARCHITECTURE_DECISION
            and report["eligible_for_stage_q62_candidate_implementation"]
            is True
            and report["eligible_for_stage_q62_production_promotion"] is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
            and report["changes_equations"] is False
            and report["changes_runtime_implementation"] is False
            and candidate["name"]
            == "plane_native_segment_handoff_shadow_candidate"
            and candidate["scope"]
            == "experimental_plane_shadow_runtime_only"
            and set(candidate["required_sources"])
            == set(STAGE_Q61_REQUIRED_SOURCES)
            and candidate["fallback_policy"]
            == "fail_closed_no_copy_cat_fallback"
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Stage Q.6.1 does not authorize Q.6.2")
    return resolved, report


def profile_stage_q62_h100_runtime(
    production_directory: str | Path,
    *,
    role: str,
    warmup_steps: int = 10,
    profile_steps: int = 20,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Profile the frozen Q.6 baseline or the native-segment candidate."""

    role = _require_role(role)
    report = _profile_stage_o43_variant_h100_runtime(
        production_directory,
        role=role,
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        expected_gpu_name=expected_gpu_name,
        variant=_VARIANT,
    )
    report["runtime_variant"] = (
        "q6_boundary_packed_copy_cat_baseline"
        if role == "baseline"
        else "q62_three_source_native_segments"
    )
    report["eligible_for_production_promotion"] = False
    return report


def run_stage_q62_h100_trajectory(
    production_directory: str | Path,
    output_directory: str | Path,
    *,
    role: str,
    confirmed_steps: int = STAGE_Q62_TRAJECTORY_STEPS,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Write one endpoint-complete six-step Q.6.2 trajectory."""

    role = _require_role(role)
    _require_positive_integer(confirmed_steps, "confirmed_steps")
    if confirmed_steps != STAGE_Q62_TRAJECTORY_STEPS:
        raise ValueError("Stage Q.6.2 trajectory must contain exactly six steps")
    reference = load_h100_production_plane_reference(
        production_directory,
        expected_gpu_name=expected_gpu_name,
    )
    device = torch.device("cuda")
    environment = _cuda_identity(
        expected_gpu_name=expected_gpu_name,
        device=device,
    )
    runtime, comparison = _build_runtime(
        reference,
        role=role,
        expected_gpu_name=expected_gpu_name,
        device=device,
        instrument=False,
        variant=_VARIANT,
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
            "qualification_stage": "Q.6.2",
            "measurement_role": role,
        },
    )
    runtime.reset_projected_batch_assembly_diagnostics()
    saved_steps = [run.save_observation().step]
    for _ in range(confirmed_steps):
        run.advance(1)
        saved_steps.append(run.save_observation().step)
    final = run.complete()
    result = {
        "schema_version": 1,
        "qualification_stage": "Q.6.2",
        "classification": "TRAJECTORY_COMPLETE",
        "measurement_role": role,
        "runtime_variant": (
            "q6_boundary_packed_copy_cat_baseline"
            if role == "baseline"
            else "q62_three_source_native_segments"
        ),
        "production_default_changed": False,
        "eligible_for_production_promotion": False,
        "production_directory": str(reference.directory),
        "trajectory_directory": str(run.output_directory),
        "production_metadata_sha256": reference.metadata_sha256,
        "production_initial_q_sha256": reference.initial_q_file_sha256,
        "completed_steps": final.step,
        "saved_steps": saved_steps,
        "projected_batch_assembly_policy": (
            runtime.projected_batch_assembly_policy.to_metadata()
        ),
        "batch_assembly_diagnostics": dict(
            runtime.projected_batch_assembly_diagnostics()
        ),
        "producer_outputs": _producer_output_snapshot(runtime),
        "explicit_rhs_execution": runtime.to_metadata()[
            "explicit_rhs_execution"
        ],
        "environment": environment,
    }
    _write_new_json(run.output_directory / _VARIANT.metrics_filename, result)
    return result


def _validate_native_sources(
    diagnostics: Mapping[str, object],
    *,
    role: str,
) -> bool:
    try:
        source_attribution = diagnostics["source_attribution"]
        if not isinstance(source_attribution, Mapping):
            return False
        for source in STAGE_Q61_REQUIRED_SOURCES:
            counters = source_attribution[source]
            if role == "baseline":
                if not (
                    counters["copy_cat_batches"] > 0
                    and counters["copy_cat_materialized_output_bytes"] > 0
                    and counters["native_segment_batches"] == 0
                ):
                    return False
            elif not (
                counters["native_segment_groups"] > 0
                and counters["native_segment_batches"] > 0
                and counters["native_segment_components"] > 0
                and counters["native_segment_logical_input_bytes"] > 0
                and counters["native_segment_materialized_output_bytes"] == 0
                and counters["copy_cat_batches"] == 0
                and counters["copy_cat_materialized_output_bytes"] == 0
                and counters["fallback_reasons"] == {}
            ):
                return False
        return bool(
            diagnostics["workspace_count"] == 0
            and diagnostics["workspace_allocated_bytes"] == 0
            and diagnostics["workspace_active_count"] == 0
            and diagnostics["workspace_retains_timestep_inputs"] is False
            and diagnostics["retained_tensor_references"] == 0
        )
    except (KeyError, TypeError):
        return False


def _validate_boundary_packed_producers(report: Mapping[str, object]) -> bool:
    try:
        outputs = report["producer_outputs"]
        return bool(
            isinstance(outputs, Mapping)
            and set(outputs) == {"molecular_field", "nematic_stress"}
            and all(
                output["producer_output_layout"] == "boundary_packed"
                and output["producer_packing"]["ownership"] == "producer"
                and output["producer_packing"]["packing_site"]
                == "inside_pointwise_kernel"
                and output["producer_packing"]["pointwise_execution"]
                == "compile"
                and output["producer_packing"]["post_kernel_stack"] is False
                and output["producer_packing"]["post_kernel_cat"] is False
                and output["producer_packing"]["cross_generation_reuse"]
                is False
                for output in outputs.values()
            )
        )
    except (KeyError, TypeError):
        return False


def _validate_trajectory(directory: Path, role: str) -> dict[str, object]:
    if not directory.is_dir() or not (directory / "COMPLETE").is_file():
        raise ValueError(f"Stage Q.6.2 {role} trajectory is incomplete")
    report = _load_json(
        directory / _VARIANT.metrics_filename,
        f"Stage Q.6.2 {role} trajectory",
    )
    expected_mode = (
        "contiguous_storage_view" if role == "baseline" else "native_segments"
    )
    try:
        valid = bool(
            report["qualification_stage"] == "Q.6.2"
            and report["classification"] == "TRAJECTORY_COMPLETE"
            and report["measurement_role"] == role
            and report["completed_steps"] == STAGE_Q62_TRAJECTORY_STEPS
            and report["saved_steps"]
            == list(range(STAGE_Q62_TRAJECTORY_STEPS + 1))
            and report["projected_batch_assembly_policy"]["mode"]
            == expected_mode
            and report["production_default_changed"] is False
            and _validate_compiled_rhs(report["explicit_rhs_execution"])
            and _validate_boundary_packed_producers(report)
            and _validate_native_sources(
                report["batch_assembly_diagnostics"], role=role
            )
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"Stage Q.6.2 {role} trajectory contract differs")
    return report


def _validate_profiles(
    paths: Sequence[str | Path],
    *,
    role: str,
    expected_gpu_name: str,
) -> list[dict[str, object]]:
    if len(paths) != STAGE_Q62_PROFILE_TRIALS:
        raise ValueError("Stage Q.6.2 requires three profiles per role")
    reports = [
        _load_json(Path(path).expanduser().resolve(), f"Q.6.2 {role} profile")
        for path in paths
    ]
    expected_mode = (
        "contiguous_storage_view" if role == "baseline" else "native_segments"
    )
    for report in reports:
        try:
            valid = bool(
                report["qualification_stage"] == "Q.6.2"
                and report["classification"] == "PROFILE_COMPLETE"
                and report["measurement_role"] == role
                and tuple(report["configuration"]["shape"])
                == STAGE_Q62_DECISION_SHAPE
                and report["configuration"]["warmup_steps"] == 10
                and report["configuration"]["profile_steps"] == 20
                and report["projected_batch_assembly_policy"]["mode"]
                == expected_mode
                and report["finite"] is True
                and report["production_default_changed"] is False
                and expected_gpu_name.lower()
                in report["environment"]["device_name"].lower()
                and report["environment"]["cuda_matmul_allow_tf32"] is False
                and _validate_compiled_rhs(report["explicit_rhs_execution"])
                and _validate_boundary_packed_producers(report)
                and _validate_native_sources(
                    report["batch_assembly_diagnostics"], role=role
                )
            )
        except (KeyError, TypeError):
            valid = False
        if not valid:
            raise ValueError(f"Stage Q.6.2 {role} profile contract differs")
    return reports


def _sum_source_counter(
    diagnostics: Mapping[str, object],
    counter: str,
) -> int:
    source_attribution = diagnostics["source_attribution"]
    if not isinstance(source_attribution, Mapping):
        raise ValueError("Stage Q.6.2 source attribution is invalid")
    return sum(
        int(source_attribution[source][counter])
        for source in STAGE_Q61_REQUIRED_SOURCES
    )


def analyze_stage_q62_qualification(
    baseline_trajectory: str | Path,
    candidate_trajectory: str | Path,
    baseline_profiles: Sequence[str | Path],
    candidate_profiles: Sequence[str | Path],
    *,
    stage_q61_report: str | Path,
    expected_stage_q61_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Apply Q.6.1's numerical, copy, memory, and H100 timing gates."""

    q61_path, _ = _read_q61_evidence(
        stage_q61_report,
        expected_stage_q61_sha256,
    )
    trajectory_paths = {
        "baseline": Path(baseline_trajectory).expanduser().resolve(),
        "candidate": Path(candidate_trajectory).expanduser().resolve(),
    }
    trajectories = {
        role: _validate_trajectory(path, role)
        for role, path in trajectory_paths.items()
    }
    trajectory_inputs = {
        (
            report["production_metadata_sha256"],
            report["production_initial_q_sha256"],
        )
        for report in trajectories.values()
    }
    if len(trajectory_inputs) != 1:
        raise ValueError("Stage Q.6.2 trajectory inputs differ")

    arrays = [
        _array_comparison(
            trajectory_paths["baseline"] / f"{field}_{step}.npy",
            trajectory_paths["candidate"] / f"{field}_{step}.npy",
            field=field,
            step=step,
        )
        for step in range(STAGE_Q62_TRAJECTORY_STEPS + 1)
        for field in _FIELDS
    ]
    maximum_relative_l2 = max(float(row["relative_l2"]) for row in arrays)
    q0_identical = all(
        bool(row["byte_identical"]) for row in arrays if row["step"] == 0
    )

    profiles = {
        "baseline": _validate_profiles(
            baseline_profiles,
            role="baseline",
            expected_gpu_name=expected_gpu_name,
        ),
        "candidate": _validate_profiles(
            candidate_profiles,
            role="candidate",
            expected_gpu_name=expected_gpu_name,
        ),
    }
    profile_inputs = {
        (
            report["production_metadata_sha256"],
            report["production_initial_q_sha256"],
        )
        for reports in profiles.values()
        for report in reports
    }
    if profile_inputs != trajectory_inputs:
        raise ValueError("Stage Q.6.2 profile and trajectory inputs differ")

    baseline_times = [
        float(report["throughput"]["mean_timestep_seconds"])
        for report in profiles["baseline"]
    ]
    candidate_times = [
        float(report["throughput"]["mean_timestep_seconds"])
        for report in profiles["candidate"]
    ]
    paired_ratios = [
        candidate / baseline
        for baseline, candidate in zip(
            baseline_times, candidate_times, strict=True
        )
    ]
    baseline_allocated = max(
        int(report["memory"]["peak_allocated_bytes"])
        for report in profiles["baseline"]
    )
    candidate_allocated = max(
        int(report["memory"]["peak_allocated_bytes"])
        for report in profiles["candidate"]
    )
    baseline_reserved = max(
        int(report["memory"]["peak_reserved_bytes"])
        for report in profiles["baseline"]
    )
    candidate_reserved = max(
        int(report["memory"]["peak_reserved_bytes"])
        for report in profiles["candidate"]
    )
    mean_ratio = _mean(candidate_times) / _mean(baseline_times)
    allocated_ratio = candidate_allocated / baseline_allocated
    reserved_ratio = candidate_reserved / baseline_reserved
    candidate_diagnostics = [
        report["batch_assembly_diagnostics"]
        for report in profiles["candidate"]
    ]

    numerical_gate = bool(
        q0_identical
        and maximum_relative_l2 <= STAGE_Q62_RELATIVE_L2_TOLERANCE
    )
    structural_gate = bool(
        all(
            _validate_native_sources(
                report["batch_assembly_diagnostics"], role=role
            )
            for role, reports in profiles.items()
            for report in reports
        )
        and all(
            _validate_native_sources(
                report["batch_assembly_diagnostics"], role=role
            )
            for role, report in trajectories.items()
        )
    )
    memory_gate = bool(
        allocated_ratio <= STAGE_Q62_MAXIMUM_PEAK_ALLOCATED_RATIO
        and reserved_ratio <= STAGE_Q62_MAXIMUM_PEAK_RESERVED_RATIO
    )
    safety_gate = bool(
        mean_ratio <= STAGE_Q62_MAXIMUM_SAFETY_TIMESTEP_RATIO
        and memory_gate
    )
    performance_gate = bool(
        mean_ratio <= STAGE_Q62_MAXIMUM_MEAN_TIMESTEP_RATIO
        and max(paired_ratios) <= STAGE_Q62_MAXIMUM_PAIRED_TIMESTEP_RATIO
        and sum(value < 1.0 for value in paired_ratios) >= 2
    )
    if not (numerical_gate and structural_gate and safety_gate):
        classification = "C_rejected"
    elif performance_gate:
        classification = "A_recommended"
    else:
        classification = "B_neutral"

    fragmentation = [
        {
            "native_segment_batches_per_step": (
                _sum_source_counter(value, "native_segment_batches") / 20.0
            ),
            "extra_transform_batches_per_step": (
                _sum_source_counter(
                    value, "native_segment_extra_transform_batches"
                )
                / 20.0
            ),
            "singleton_native_batches_per_step": (
                _sum_source_counter(
                    value, "native_segment_singleton_batches"
                )
                / 20.0
            ),
        }
        for value in candidate_diagnostics
    ]
    return {
        "schema_version": 1,
        "qualification_stage": "Q.6.2",
        "classification": classification,
        "architecture_decision": (
            "producer_owned_boundary_signature_native_segments"
        ),
        "eligible_for_stage_q63_architecture_decision": (
            classification == "A_recommended"
        ),
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "stage_q61_evidence": {
            "path": str(q61_path),
            "sha256": expected_stage_q61_sha256,
        },
        "trajectory": {
            "steps": STAGE_Q62_TRAJECTORY_STEPS,
            "saved_steps": list(range(STAGE_Q62_TRAJECTORY_STEPS + 1)),
            "relative_l2_tolerance": STAGE_Q62_RELATIVE_L2_TOLERANCE,
            "maximum_relative_l2": maximum_relative_l2,
            "q0_byte_identical": q0_identical,
            "arrays": arrays,
        },
        "performance": {
            "baseline_mean_timestep_seconds": _mean(baseline_times),
            "candidate_mean_timestep_seconds": _mean(candidate_times),
            "mean_timestep_ratio_candidate_over_baseline": mean_ratio,
            "paired_timestep_ratios": paired_ratios,
            "paired_candidate_faster_count": sum(
                value < 1.0 for value in paired_ratios
            ),
            "baseline_peak_allocated_bytes": baseline_allocated,
            "candidate_peak_allocated_bytes": candidate_allocated,
            "peak_allocated_ratio_candidate_over_baseline": allocated_ratio,
            "baseline_peak_reserved_bytes": baseline_reserved,
            "candidate_peak_reserved_bytes": candidate_reserved,
            "peak_reserved_ratio_candidate_over_baseline": reserved_ratio,
        },
        "native_segment_fragmentation": fragmentation,
        "source_diagnostics": {
            "baseline": [
                report["batch_assembly_diagnostics"]
                for report in profiles["baseline"]
            ],
            "candidate": candidate_diagnostics,
        },
        "gates": {
            "numerical_equivalence": numerical_gate,
            "three_source_copy_elimination": structural_gate,
            "r320_performance_improvement": performance_gate,
            "r320_memory_non_regression": memory_gate,
            "candidate_safety_non_regression": safety_gate,
        },
        "thresholds": {
            "maximum_relative_l2": STAGE_Q62_RELATIVE_L2_TOLERANCE,
            "maximum_mean_timestep_ratio": (
                STAGE_Q62_MAXIMUM_MEAN_TIMESTEP_RATIO
            ),
            "maximum_paired_timestep_ratio": (
                STAGE_Q62_MAXIMUM_PAIRED_TIMESTEP_RATIO
            ),
            "maximum_safety_timestep_ratio": (
                STAGE_Q62_MAXIMUM_SAFETY_TIMESTEP_RATIO
            ),
            "maximum_peak_allocated_ratio": (
                STAGE_Q62_MAXIMUM_PEAK_ALLOCATED_RATIO
            ),
            "maximum_peak_reserved_ratio": (
                STAGE_Q62_MAXIMUM_PEAK_RESERVED_RATIO
            ),
        },
    }


def profile_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile Stage Q.6.2")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = profile_stage_q62_h100_runtime(
        args.production_reference_dir,
        role=args.role,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    return 0


def trajectory_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage Q.6.2 trajectory")
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=_ROLES, required=True)
    parser.add_argument("--confirm-steps", type=int, required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    report = run_stage_q62_h100_trajectory(
        args.production_reference_dir,
        args.output_dir,
        role=args.role,
        confirmed_steps=args.confirm_steps,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage Q.6.2")
    parser.add_argument("--baseline-trajectory", type=Path, required=True)
    parser.add_argument("--candidate-trajectory", type=Path, required=True)
    parser.add_argument(
        "--baseline-profile", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--candidate-profile", type=Path, action="append", required=True
    )
    parser.add_argument("--stage-q61-report", type=Path, required=True)
    parser.add_argument("--expected-stage-q61-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_q62_qualification(
        args.baseline_trajectory,
        args.candidate_trajectory,
        args.baseline_profile,
        args.candidate_profile,
        stage_q61_report=args.stage_q61_report,
        expected_stage_q61_sha256=args.expected_stage_q61_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new_json(args.output, report)
    return 0


__all__ = [
    "STAGE_Q62_DECISION_SHAPE",
    "STAGE_Q62_PROFILE_TRIALS",
    "STAGE_Q62_TRAJECTORY_STEPS",
    "analysis_main",
    "analyze_stage_q62_qualification",
    "profile_main",
    "profile_stage_q62_h100_runtime",
    "run_stage_q62_h100_trajectory",
    "trajectory_main",
]
