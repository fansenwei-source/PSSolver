#!/usr/bin/env python3
"""Pre-register the rc5 G-1 measurement-window recovery contract.

This plan is qualification support only.  It separates post-``empty_cache``
allocation recovery from the formal timing, transform-counter, compiler-counter,
and steady-state memory windows.  It does not relax a threshold, delete a timing
sample, replace a trial, modify a production runtime, or authorize an H100 job.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from benchmarks.plan_rc5_g1_focused_performance_h100 import (
    TIMED_STEPS,
    TIMING_CV_MAX,
    WARMUP_STEPS,
    build_plan as build_focused_plan,
)
from benchmarks.plan_rc5_g1_h100 import _scheduled_refreshes


SCHEMA = "pssolver.rc5.g1.measurement_window_recovery_plan.v1"
SOURCE_ANALYSIS_RECORD = (
    "notes/PSSolver_v0_2_0rc5_g1_timing_variability_analysis_v8.json"
)
SOURCE_ANALYSIS_CLASSIFICATION = (
    "HARNESS_FIRST_SAMPLE_CONTAMINATION_AND_RESIDUAL_IN_WINDOW_VARIABILITY"
)
POST_FLUSH_REWARM_STEPS = 3
RUNTIME_COMPLETED_STEPS = WARMUP_STEPS + POST_FLUSH_REWARM_STEPS + TIMED_STEPS


def _measurement_entry(entry: dict[str, object]) -> dict[str, object]:
    result = dict(entry)
    family = str(result["family"])
    measurement_start_step = WARMUP_STEPS + POST_FLUSH_REWARM_STEPS
    refreshes = _scheduled_refreshes(
        family=family,
        warmup_steps=measurement_start_step,
        timed_steps=TIMED_STEPS,
    )
    scheduled_forward = refreshes if family in {
        "channel_legacy",
        "channel_compiled_v2",
    } else 0
    result.update(
        {
            "post_flush_rewarm_steps": POST_FLUSH_REWARM_STEPS,
            "measurement_start_completed_steps": measurement_start_step,
            "expected_runtime_completed_steps": RUNTIME_COMPLETED_STEPS,
            "scheduled_refreshes_in_timed_window": refreshes,
            "scheduled_forward_calls_in_timed_window": scheduled_forward,
            "scheduled_inverse_calls_in_timed_window": 0,
            "expected_forward_raw_calls": (
                int(result["forward_transforms_per_step"]) * TIMED_STEPS
                + scheduled_forward
            ),
            "expected_inverse_raw_calls": (
                int(result["inverse_transforms_per_step"]) * TIMED_STEPS
            ),
        }
    )
    return result


def build_plan() -> dict[str, object]:
    focused = build_focused_plan()
    profiles = [_measurement_entry(entry) for entry in focused["profile_matrix"]]
    thresholds = dict(focused["thresholds"])
    timing_validity = dict(focused["timing_validity"])
    return {
        "schema": SCHEMA,
        "classification": "READY_FOR_QUALIFICATION_SUPPORT_IMPLEMENTATION",
        "purpose": (
            "replace the contaminated measurement window, not any scientific "
            "runtime or frozen qualification threshold"
        ),
        "source_evidence": {
            "analysis_record": SOURCE_ANALYSIS_RECORD,
            "analysis_classification": SOURCE_ANALYSIS_CLASSIFICATION,
            "v7_failure_must_remain_failed": True,
            "v8_is_analysis_only": True,
            "source_profiles_must_remain_unchanged": True,
        },
        "provider": focused["provider"],
        "immutable_inputs": focused["immutable_inputs"],
        "focused_cells": focused["focused_cells"],
        "grid_contracts": focused["grid_contracts"],
        "profile_matrix": profiles,
        "profile_count": len(profiles),
        "measurement_window": {
            "phase_order": [
                "build_runtime",
                "warmup_10_steps",
                "synchronize_after_warmup",
                "gc_collect",
                "cuda_empty_cache",
                "capture_post_flush_memory_snapshot",
                "post_flush_rewarm_3_steps_untimed",
                "synchronize_after_rewarm",
                "capture_cold_recovery_diagnostics",
                "reset_formal_peak_memory_stats",
                "reset_formal_transform_timer",
                "snapshot_formal_compiler_counters",
                "time_60_steps_with_cuda_events",
                "synchronize_after_formal_window",
                "capture_formal_results",
            ],
            "warmup_steps": WARMUP_STEPS,
            "post_flush_rewarm_steps": POST_FLUSH_REWARM_STEPS,
            "timed_steps": TIMED_STEPS,
            "expected_runtime_completed_steps": RUNTIME_COMPLETED_STEPS,
            "rewarm_is_outside_formal_timing_window": True,
            "rewarm_is_outside_formal_transform_counter_window": True,
            "rewarm_is_outside_formal_compiler_counter_window": True,
            "rewarm_is_outside_formal_memory_peak_window": True,
            "formal_memory_semantics": "steady_state_after_allocation_recovery",
            "cold_recovery_memory_is_preserved_as_diagnostic": True,
            "formal_state_window_changes_from_steps_11_70_to_steps_14_73": True,
            "baseline_and_candidate_use_identical_phase_order": True,
        },
        "timing_validity": timing_validity,
        "thresholds": thresholds,
        "diagnostic_replay": {
            "required": True,
            "separate_fresh_runtime": True,
            "same_seed_configuration_and_initial_state_sha": True,
            "never_used_for_performance_or_memory_adjudication": True,
            "steps": TIMED_STEPS,
            "raw_per_step_timing_samples_preserved": True,
            "public_step_diagnostics_when_applicable": [
                "pressure_iteration_count",
                "pressure_termination_reason",
                "pressure_absolute_residual",
                "pressure_relative_residual",
            ],
            "unavailable_diagnostic_encoding": {
                "applicable": False,
                "reason_required": True,
                "fabricated_numeric_placeholder_allowed": False,
            },
            "device_telemetry_outside_timed_window": [
                "pstate",
                "sm_clock_mhz",
                "memory_clock_mhz",
                "power_draw_watts",
                "temperature_celsius",
            ],
            "memory_snapshots": [
                "after_warmup",
                "after_empty_cache",
                "after_post_flush_rewarm",
                "after_formal_timed_window",
            ],
        },
        "execution_order": {
            **focused["execution_order"],
            "post_flush_rewarm_steps": POST_FLUSH_REWARM_STEPS,
            "expected_runtime_completed_steps": RUNTIME_COMPLETED_STEPS,
        },
        "adjudication": {
            "all_48_profiles_are_new": True,
            "all_four_cells_must_pass": True,
            "coefficient_of_variation_is_computed_over_all_60_formal_samples": True,
            "no_outlier_removal": True,
            "no_first_sample_deletion": True,
            "no_trial_replacement": True,
            "no_threshold_is_relaxed": True,
            "failed_v7_result_is_not_reclassified": True,
            "focused_pass_is_not_g1_complete": True,
            "cumulative_reassembly_required_after_focused_pass": True,
        },
        "implementation_boundary": {
            "production_solver_modified": False,
            "production_runtime_modified": False,
            "scientific_equations_modified": False,
            "qualification_profiler_change_required_next": True,
            "qualification_runner_change_required_next": True,
            "qualification_analyzer_threshold_change_required": False,
            "h100_execution_part_of_this_plan_generation": False,
        },
        "versioned_support": {
            "plan": "benchmarks/plan_rc5_g1_measurement_window_recovery.py",
            "target_profiler": "benchmarks/profile_rc458_runtime_family.py",
            "target_focused_plan": (
                "benchmarks/plan_rc5_g1_focused_performance_h100.py"
            ),
            "target_runner": (
                "benchmarks/run_rc5_g1_focused_profile_matrix.py"
            ),
            "target_analyzer": (
                "benchmarks/analyze_rc5_g1_focused_performance_h100.py"
            ),
            "tests": "tests/test_rc5_g1_measurement_window_recovery.py",
            "implementation_status": "not_started_by_this_plan",
        },
        "authorization": {
            "plan_commit_and_push_authorized": True,
            "h100_submission_authorized": False,
            "automatic_retry_authorized": False,
            "profile_threshold_change_authorized": False,
            "trial_replacement_authorized": False,
            "production_default_changed": False,
        },
    }


def _atomic_json(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    _atomic_json(args.output, build_plan())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
