#!/usr/bin/env python3
"""Pre-register the CPU-only rc5 G-1 paired-block timing contract.

The contract is a proposed successor to the per-profile CV validity gate.  It
does not reclassify the failed v10 run and does not authorize an H100 job.  All
60 samples remain in the record and are assigned, in order, to six immutable
ten-sample blocks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc5_g1_measurement_window_recovery import (
    build_plan as build_measurement_window_plan,
)


SCHEMA = "pssolver.rc5.g1.paired_block_measurement_contract.v1"
CLASSIFICATION = "READY_FOR_G1_PAIRED_BLOCK_CONTRACT_REVIEW"
BLOCK_SIZE = 10
BLOCK_COUNT = 6


def build_plan() -> dict[str, object]:
    source = build_measurement_window_plan()
    profiles = [dict(entry) for entry in source["profile_matrix"]]
    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "purpose": (
            "replace the unsupported universal per-profile CV rejection with "
            "a deterministic paired-block estimator while retaining every raw "
            "sample and every frozen numerical threshold"
        ),
        "source_evidence": {
            "formal_v10_classification": (
                "FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE"
            ),
            "formal_v10_manifest_sha256": (
                "747609e354ff2cc01c68032b4685b09fb96cc43059c1e6c7817d2c4961bbf7b9"
            ),
            "v11_classification": (
                "MIXED_PHASE_SHIFTED_PERIODIC_AND_POSITION_DRIFTING_TIMING_VARIABILITY"
            ),
            "v11_manifest_sha256": (
                "82b88089904c3cf54685101d22c0c3f4ddfaf353ee0fa522b29db28a78f6bb2c"
            ),
            "v12_classification": (
                "ANALYSIS_ONLY_INSTRUMENTATION_EVIDENCE_COMPLETE"
            ),
            "v12_attribution_classification": (
                "NO_SINGLE_GC_REFRESH_OR_PRESSURE_TERMINATION_CAUSE_SUPPORTED"
            ),
            "v12_manifest_sha256": (
                "da7ddb92658572b76889123c56f08e7bdb6ef863b009409c42a1ab6379e58ee1"
            ),
            "historical_results_must_remain_unchanged": True,
            "post_hoc_replay_cannot_grant_qualification": True,
        },
        "provider": source["provider"],
        "focused_cells": source["focused_cells"],
        "grid_contracts": source["grid_contracts"],
        "profile_matrix": profiles,
        "profile_count": len(profiles),
        "execution_order": source["execution_order"],
        "raw_sample_contract": {
            "samples_per_profile": 60,
            "block_size": BLOCK_SIZE,
            "block_count": BLOCK_COUNT,
            "block_boundaries_zero_based_half_open": [
                [index * BLOCK_SIZE, (index + 1) * BLOCK_SIZE]
                for index in range(BLOCK_COUNT)
            ],
            "sample_assignment": "each sample belongs to exactly one block",
            "block_statistic": "arithmetic_mean",
            "trial_statistic": "median_of_six_candidate_over_baseline_block_ratios",
            "cell_statistic": "median_of_all_36_fixed_block_ratios",
            "reported_profile_summaries_must_match_raw_samples": True,
            "outlier_removal_allowed": False,
            "sample_deletion_allowed": False,
            "block_reordering_allowed": False,
            "trial_replacement_allowed": False,
        },
        "timing_diagnostics": {
            "coefficient_of_variation_reference": 0.05,
            "coefficient_of_variation_is_reported": True,
            "coefficient_of_variation_role": "environment_diagnostic_only",
            "standalone_cv_rejection_authorized": False,
            "reason": (
                "v12 found position-drifting spikes and did not support GC, "
                "scheduled refresh, pressure termination, or synchronization "
                "as a single dominant cause"
            ),
        },
        "thresholds": {
            "aggregate_all_sample_mean_ratio_max": 1.03,
            "median_of_profile_means_ratio_max": 1.03,
            "median_of_all_fixed_block_ratios_max": 1.03,
            "every_trial_median_fixed_block_ratio_max": 1.05,
            "peak_allocated_group_ratio_max": 1.03,
            "peak_reserved_group_ratio_max": 1.03,
        },
        "threshold_provenance": {
            "numeric_performance_thresholds_changed": False,
            "numeric_memory_thresholds_changed": False,
            "old_mean_ratio_limit": 1.03,
            "old_median_ratio_limit": 1.03,
            "old_paired_limit": 1.05,
            "old_memory_limits": [1.03, 1.03],
            "estimator_semantics_changed": True,
            "estimator_change_requires_explicit_formal_authorization": True,
        },
        "adjudication": {
            "all_four_cells_must_pass": True,
            "all_six_trials_per_cell_are_required": True,
            "odd_even_role_order_must_remain_balanced": True,
            "all_raw_samples_are_retained": True,
            "non_finite_or_missing_data_fail_closed": True,
            "identity_numerical_transform_and_memory_gates_remain_required": True,
            "post_hoc_v10_replay_is_analysis_only": True,
            "new_formal_evidence_required_after_contract_authorization": True,
        },
        "scope": {
            "production_solver_modified": False,
            "production_runtime_modified": False,
            "scientific_equations_modified": False,
            "formal_v10_reclassified": False,
            "formal_requalification_performed": False,
            "qualification_complete": False,
            "g1_complete": False,
            "production_default_changed": False,
        },
        "authorization": {
            "cpu_only_contract_implementation_authorized": True,
            "post_hoc_replay_authorized": True,
            "formal_contract_adoption_authorized": False,
            "push_authorized": False,
            "h100_submission_authorized": False,
            "automatic_retry_authorized": False,
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
