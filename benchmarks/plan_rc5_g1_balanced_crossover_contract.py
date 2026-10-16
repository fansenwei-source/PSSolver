#!/usr/bin/env python3
"""Build the proposed rc5 G-1 balanced-crossover timing contract.

This CPU-only proposal responds to the v13 maximum-per-trial failure-cell
migration.  It retains all raw samples and the existing 1.03/1.05 numeric
limits, but treats the six alternating-order trials as replicated crossover
evidence instead of allowing one trial maximum to veto a complete cell.
Formal adoption, push, and H100 execution require separate authorization.
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

from benchmarks.plan_rc5_g1_paired_block_contract import (
    BLOCK_COUNT,
    BLOCK_SIZE,
    build_plan as build_paired_block_plan,
)


SCHEMA = "pssolver.rc5.g1.balanced_crossover_contract.v1"
CLASSIFICATION = "READY_FOR_G1_BALANCED_CROSSOVER_CONTRACT_REVIEW"


def build_plan() -> dict[str, object]:
    source = build_paired_block_plan()
    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "purpose": (
            "replace the non-repeatable maximum-per-trial veto with a "
            "replicated, order-balanced crossover estimator while retaining "
            "all raw samples and every existing numeric threshold"
        ),
        "source_evidence": {
            "v10_manifest_sha256": (
                "747609e354ff2cc01c68032b4685b09fb96cc43059c1e6c7817d2c4961bbf7b9"
            ),
            "v13_manifest_sha256": (
                "fb2e9e3cf16fd5a40497b2077fa57436f446cf66baafc87308424305f0d8e60a"
            ),
            "v13_formal_classification": (
                "FAIL_RC5_G1_PAIRED_BLOCK_PERFORMANCE_MEMORY_GATE"
            ),
            "v13_failure_analysis_classification": (
                "ANALYSIS_ONLY_V13_MAX_TRIAL_FAILURE_CELL_MIGRATION_SUPPORTED"
            ),
            "historical_results_must_remain_unchanged": True,
            "post_hoc_replay_cannot_grant_qualification": True,
        },
        "provider": source["provider"],
        "focused_cells": source["focused_cells"],
        "grid_contracts": source["grid_contracts"],
        "profile_matrix": source["profile_matrix"],
        "profile_count": source["profile_count"],
        "execution_order": source["execution_order"],
        "raw_sample_contract": {
            "samples_per_profile": 60,
            "block_size": BLOCK_SIZE,
            "block_count": BLOCK_COUNT,
            "block_boundaries_zero_based_half_open": [
                [index * BLOCK_SIZE, (index + 1) * BLOCK_SIZE]
                for index in range(BLOCK_COUNT)
            ],
            "sample_assignment": "each raw sample belongs to exactly one block",
            "block_statistic": "arithmetic_mean_of_ten_raw_samples",
            "profile_center": "median_of_six_fixed_block_means",
            "trial_ratio": "candidate_profile_center_over_baseline_profile_center",
            "cross_profile_block_index_pairing_used": False,
            "all_raw_samples_retained": True,
            "reported_profile_summaries_must_match_raw_samples": True,
            "outlier_removal_allowed": False,
            "sample_deletion_allowed": False,
            "block_reordering_allowed": False,
            "trial_replacement_allowed": False,
        },
        "replicated_crossover_contract": {
            "trials_per_cell": 6,
            "baseline_then_candidate_trials": [1, 3, 5],
            "candidate_then_baseline_trials": [2, 4, 6],
            "central_trial_statistic": "median_of_six_trial_ratios",
            "order_stratum_statistic": "median_of_three_trial_ratios",
            "order_imbalance_statistic": (
                "maximum_order_stratum_median_over_minimum_order_stratum_median"
            ),
            "minimum_trials_at_or_below_paired_limit": 4,
            "maximum_single_trial_used_as_standalone_veto": False,
        },
        "thresholds": {
            "aggregate_all_sample_mean_ratio_max": 1.03,
            "median_of_profile_means_ratio_max": 1.03,
            "median_of_six_trial_ratios_max": 1.03,
            "baseline_then_candidate_stratum_median_ratio_max": 1.05,
            "candidate_then_baseline_stratum_median_ratio_max": 1.05,
            "order_stratum_imbalance_ratio_max": 1.05,
            "individual_trial_ratio_reference_max": 1.05,
            "minimum_trials_at_or_below_individual_reference": 4,
            "peak_allocated_group_ratio_max": 1.03,
            "peak_reserved_group_ratio_max": 1.03,
        },
        "threshold_provenance": {
            "numeric_performance_thresholds_changed": False,
            "numeric_memory_thresholds_changed": False,
            "existing_central_limit": 1.03,
            "existing_paired_limit": 1.05,
            "existing_memory_limits": [1.03, 1.03],
            "estimator_semantics_changed": True,
            "formal_adoption_requires_explicit_authorization": True,
        },
        "adjudication": {
            "all_four_cells_must_pass": True,
            "all_six_trials_per_cell_are_required": True,
            "both_order_strata_must_pass": True,
            "order_imbalance_must_pass": True,
            "at_least_four_trials_must_meet_individual_reference": True,
            "aggregate_profile_and_memory_gates_remain_required": True,
            "non_finite_or_missing_data_fail_closed": True,
            "post_hoc_v10_v13_replay_is_analysis_only": True,
            "new_formal_evidence_required_after_contract_authorization": True,
        },
        "scope": {
            "production_solver_modified": False,
            "production_runtime_modified": False,
            "scientific_equations_modified": False,
            "formal_v13_reclassified": False,
            "formal_requalification_performed": False,
            "qualification_complete": False,
            "g1_complete": False,
            "production_default_changed": False,
        },
        "authorization": {
            "cpu_only_contract_design_authorized": True,
            "post_hoc_replay_authorized": True,
            "local_commit_authorized": True,
            "push_authorized": False,
            "formal_contract_adoption_authorized": False,
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
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("JSON round-trip differs")
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
