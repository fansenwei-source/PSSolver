#!/usr/bin/env python3
"""Pre-register the analysis-only rc5 G-1 timing instrumentation matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from benchmarks.plan_rc5_g1_focused_performance_h100 import FOCUSED_CELLS
from benchmarks.plan_rc5_g1_h100 import (
    GRID_CONTRACTS,
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
)


SCHEMA = "pssolver.rc5.g1.timing_instrumentation_plan.v1"
FAMILIES = tuple(str(cell["family"]) for cell in FOCUSED_CELLS)
DIAGNOSTIC_TRIALS = 3
WARMUP_STEPS = 10
POST_FLUSH_REWARM_STEPS = 3
TIMED_STEPS = 60
TELEMETRY_INTERVAL_MS = 100
V10_CONTROL_MANIFEST_SHA256 = (
    "747609e354ff2cc01c68032b4685b09fb96cc43059c1e6c7817d2c4961bbf7b9"
)
V10_SCRATCH_MANIFEST_SHA256 = (
    "092950645960bab8f2fd9bbfa02483cd37728461bf31206064b2cb5eeeb05df8"
)
V11_ANALYSIS_MANIFEST_SHA256 = (
    "82b88089904c3cf54685101d22c0c3f4ddfaf353ee0fa522b29db28a78f6bb2c"
)


def _family_dtype(family: str) -> str:
    return "float32" if family == "channel_compiled_v2" else "float64"


def build_profile_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell_sequence, cell in enumerate(FOCUSED_CELLS, start=1):
        family = str(cell["family"])
        grid = str(cell["grid"])
        for trial in range(1, DIAGNOSTIC_TRIALS + 1):
            rows.append(
                {
                    "sequence": len(rows) + 1,
                    "cell_sequence": cell_sequence,
                    "key": f"candidate_{family}_{grid}_diagnostic_trial{trial}",
                    "provider_role": "candidate",
                    "provider_commit": PROVIDER_CANDIDATE,
                    "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
                    "family": family,
                    "grid": grid,
                    "trial": trial,
                    "shape": list(GRID_CONTRACTS[grid]["shape"]),
                    "lengths": list(GRID_CONTRACTS[grid]["lengths"]),
                    "dtype": _family_dtype(family),
                    "device": "cuda",
                    "warmup_steps": WARMUP_STEPS,
                    "post_flush_rewarm_steps": POST_FLUSH_REWARM_STEPS,
                    "timed_steps": TIMED_STEPS,
                }
            )
    return rows


def build_plan() -> dict[str, object]:
    profiles = build_profile_matrix()
    return {
        "schema": SCHEMA,
        "classification": "READY_FOR_ANALYSIS_ONLY_INSTRUMENTATION",
        "purpose": (
            "distinguish process-periodic events from position-drifting GPU or "
            "host-dispatch bursts without changing formal G-1 adjudication"
        ),
        "source_evidence": {
            "formal_v10_classification": (
                "FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE"
            ),
            "formal_v10_reclassified": False,
            "v10_control_manifest_sha256": V10_CONTROL_MANIFEST_SHA256,
            "v10_scratch_manifest_sha256": V10_SCRATCH_MANIFEST_SHA256,
            "v11_analysis_classification": (
                "MIXED_PHASE_SHIFTED_PERIODIC_AND_POSITION_DRIFTING_TIMING_VARIABILITY"
            ),
            "v11_analysis_manifest_sha256": V11_ANALYSIS_MANIFEST_SHA256,
        },
        "provider": {
            "role": "candidate",
            "commit": PROVIDER_CANDIDATE,
            "wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
        },
        "families": list(FAMILIES),
        "profile_matrix": profiles,
        "profile_count": len(profiles),
        "fresh_child_process_per_profile": True,
        "windows": {
            "asynchronous_compatibility": {
                "fresh_runtime": True,
                "warmup_steps": WARMUP_STEPS,
                "post_flush_rewarm_steps": POST_FLUSH_REWARM_STEPS,
                "timed_steps": TIMED_STEPS,
                "synchronize_each_step": False,
                "records": [
                    "cuda_event_seconds",
                    "host_submit_seconds",
                    "host_step_unix_nanoseconds",
                    "gc_counts_before_after",
                    "gc_collection_events",
                    "scheduled_refresh_due",
                ],
            },
            "synchronized_attribution_replay": {
                "fresh_runtime": True,
                "same_seed_configuration_and_initial_state": True,
                "warmup_steps": WARMUP_STEPS,
                "post_flush_rewarm_steps": POST_FLUSH_REWARM_STEPS,
                "timed_steps": TIMED_STEPS,
                "synchronize_each_step": True,
                "records": [
                    "cuda_event_seconds",
                    "host_synchronized_seconds",
                    "host_step_unix_nanoseconds",
                    "gc_counts_before_after",
                    "gc_collection_events",
                    "scheduled_refresh_due",
                    "public_pressure_diagnostics_when_available",
                ],
                "used_for_formal_performance_adjudication": False,
            },
        },
        "background_gpu_telemetry": {
            "tool": "nvidia-smi",
            "subprocess_is_separate_from_profile_python": True,
            "interval_ms": TELEMETRY_INTERVAL_MS,
            "query_fields": [
                "timestamp",
                "index",
                "uuid",
                "pci.bus_id",
                "pstate",
                "clocks.current.sm",
                "clocks.current.memory",
                "power.draw",
                "temperature.gpu",
                "utilization.gpu",
                "utilization.memory",
            ],
            "all_physical_rows_preserved": True,
            "visible_device_binding_must_be_explicit_or_unresolved": True,
        },
        "frozen_boundaries": {
            "formal_raw_sample_count": 60,
            "formal_cv_threshold": 0.05,
            "formal_threshold_changed": False,
            "outlier_removal_allowed": False,
            "trial_replacement_allowed": False,
            "formal_v10_reclassification_allowed": False,
            "production_solver_modified": False,
            "production_runtime_modified": False,
            "production_default_changed": False,
            "tf32_child_policy": "explicit_highest_false_false",
        },
        "interpretation": {
            "instrumentation_changes_execution_timing": True,
            "diagnostic_result_is_not_formal_requalification": True,
            "gc_correlation_does_not_by_itself_prove_causality": True,
            "telemetry_correlation_does_not_by_itself_prove_causality": True,
            "root_cause_may_remain_unresolved": True,
        },
        "authorization": {
            "implementation_and_cpu_tests_authorized": True,
            "push_authorized": False,
            "h100_submission_authorized": False,
            "automatic_retry_authorized": False,
            "production_default_changed": False,
        },
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, build_plan())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
