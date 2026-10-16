"""CPU-only tests for the proposed rc5 G-1 balanced-crossover contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.analyze_rc5_g1_balanced_crossover_replay import adjudicate_cell
from benchmarks.plan_rc5_g1_balanced_crossover_contract import (
    CLASSIFICATION,
    build_plan,
    main as plan_main,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adjudicate(
    ratios: list[float],
    *,
    aggregate: float = 1.0,
    median_profile: float = 1.0,
    allocated: float = 1.0,
    reserved: float = 1.0,
):
    return adjudicate_cell(
        trial_ratios=ratios,
        aggregate_all_sample_mean_ratio=aggregate,
        median_of_profile_means_ratio=median_profile,
        peak_allocated_group_ratio=allocated,
        peak_reserved_group_ratio=reserved,
    )


def test_plan_freezes_raw_samples_order_balance_and_existing_limits():
    plan = build_plan()
    assert plan["classification"] == CLASSIFICATION
    assert plan["profile_count"] == 48
    assert plan["raw_sample_contract"] == {
        "samples_per_profile": 60,
        "block_size": 10,
        "block_count": 6,
        "block_boundaries_zero_based_half_open": [
            [0, 10],
            [10, 20],
            [20, 30],
            [30, 40],
            [40, 50],
            [50, 60],
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
    }
    assert plan["thresholds"] == {
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
    }
    assert plan["threshold_provenance"]["numeric_performance_thresholds_changed"] is False
    assert plan["threshold_provenance"]["estimator_semantics_changed"] is True
    assert plan["authorization"] == {
        "cpu_only_contract_design_authorized": True,
        "post_hoc_replay_authorized": True,
        "local_commit_authorized": True,
        "push_authorized": False,
        "formal_contract_adoption_authorized": False,
        "h100_submission_authorized": False,
        "automatic_retry_authorized": False,
    }


def test_balanced_non_regression_passes():
    result = _adjudicate([1.02] * 6, aggregate=1.02, median_profile=1.02)
    assert result["passed"] is True
    assert all(result["gates"].values())


@pytest.mark.parametrize("ratio", [1.031, 1.06])
def test_uniform_regression_is_rejected(ratio):
    result = _adjudicate([ratio] * 6)
    assert result["passed"] is False
    assert result["gates"]["median_of_six_trial_ratios"] is False


def test_one_isolated_high_trial_with_reverse_drift_does_not_veto_cell():
    result = _adjudicate(
        [0.997817, 1.001939, 1.063249, 0.958185, 0.935247, 1.012930],
        aggregate=0.994,
        median_profile=1.001,
    )
    assert result["passed"] is True
    assert result["metrics"]["trials_at_or_below_individual_reference"] == 5
    assert result["metrics"]["individual_trial_ratios"][2] > 1.05


def test_order_bias_is_rejected():
    result = _adjudicate([1.06, 0.95, 1.06, 0.95, 1.06, 0.95])
    assert result["passed"] is False
    assert result["gates"]["baseline_then_candidate_stratum"] is False
    assert result["gates"]["order_stratum_imbalance"] is False


def test_fewer_than_four_individually_acceptable_trials_is_rejected():
    result = _adjudicate([1.051, 1.051, 1.051, 0.99, 0.99, 0.99])
    assert result["passed"] is False
    assert result["metrics"]["trials_at_or_below_individual_reference"] == 3
    assert result["gates"]["minimum_individual_trial_count"] is False


@pytest.mark.parametrize("name", ["allocated", "reserved"])
def test_memory_regression_is_rejected(name):
    arguments = {"allocated": 1.0, "reserved": 1.0, name: 1.031}
    result = _adjudicate([1.0] * 6, **arguments)
    assert result["passed"] is False
    assert result["gates"][f"peak_{name}"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_nonfinite_or_nonpositive_ratio_fails_closed(value):
    with pytest.raises(ValueError):
        _adjudicate([1.0, 1.0, value, 1.0, 1.0, 1.0])


def test_plan_cli_is_atomic_and_refuses_overwrite(tmp_path):
    output = tmp_path / "plan.json"
    assert plan_main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == build_plan()
    with pytest.raises(FileExistsError):
        plan_main(["--output", str(output)])


def test_versioned_record_preserves_failures_and_binds_analysis_only_evidence():
    record = json.loads(
        (
            ROOT
            / "notes"
            / "PSSolver_v0_2_0rc5_g1_balanced_crossover_contract_v14.json"
        ).read_text(encoding="utf-8")
    )
    assert record["classification"] == (
        "ANALYSIS_ONLY_V10_V13_REPLAY_PASSES_PROPOSED_V14_BALANCED_CROSSOVER_CONTRACT"
    )
    assert record["contract_status"] == CLASSIFICATION
    assert record["frozen_replay"]["both_runs_passed_proposed_contract"] is True
    assert record["frozen_replay"]["v10"]["formal_classification_preserved"] == (
        "FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE"
    )
    assert record["frozen_replay"]["v13"]["formal_classification_preserved"] == (
        "FAIL_RC5_G1_PAIRED_BLOCK_PERFORMANCE_MEMORY_GATE"
    )
    for run in ("v10", "v13"):
        cells = record["frozen_replay"][run]["cells"]
        assert len(cells) == 4
        assert all(cell["passed"] for cell in cells)
        assert all(len(cell["trial_ratios"]) == 6 for cell in cells)
    assert record["negative_controls"] == {
        "uniform_1_031_rejected": True,
        "uniform_1_06_rejected": True,
        "order_bias_rejected": True,
        "fewer_than_four_individual_trials_rejected": True,
        "memory_ratio_1_031_rejected": True,
        "nonfinite_and_nonpositive_values_rejected": True,
    }
    hpcc = record["hpcc_analysis"]
    assert hpcc["formal_h100_submission_count"] == 0
    assert hpcc["slurm_job_submitted"] is False
    assert hpcc["v10_checksums"] == "41/41 PASS"
    assert hpcc["v13_checksums"] == "1803/1803 PASS"
    assert hpcc["archive_checksums"] == "19/19 PASS"
    scope = record["scope"]
    assert scope["formal_contract_adoption_authorized"] is False
    assert scope["push_authorized"] is False
    assert scope["h100_submission_authorized"] is False
    assert scope["formal_v10_reclassified"] is False
    assert scope["formal_v13_reclassified"] is False
    assert scope["formal_v13_pass_granted"] is False
    assert scope["qualification_complete"] is False
    assert scope["g1_complete"] is False
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected
