from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.analyze_v0_3_s09_cpu_6x60_stop import (
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_CAUSAL_SHA256,
    EXPECTED_SELF_SHA256,
    _read_bound,
    analyze_existing_evidence,
)


EVIDENCE = Path("/home/fansenwei/pssolver_v0_3_s09_cpu_6x60_2573dec_20261023_v3")
PRIOR = Path("/home/fansenwei/pssolver_v0_3_s09_cpu_diagnosis_2573dec_20261023_v2")


def _inputs():
    return (
        _read_bound(EVIDENCE / "results/self_control/matrix.json", EXPECTED_SELF_SHA256),
        _read_bound(EVIDENCE / "results/candidate_bound/matrix.json", EXPECTED_CANDIDATE_SHA256),
        _read_bound(PRIOR / "results/s09_diagnosis_c105_cost_5x30/matrix.json", EXPECTED_CAUSAL_SHA256),
    )


def test_s09_stop_analysis_preserves_failure_and_uses_no_new_profiles():
    result = analyze_existing_evidence(*_inputs())
    assert result["classification"] == "TRANSIENT_PROFILE_SLOWDOWN_SUPPORTED_CAUSE_UNRESOLVED"
    assert result["new_profile_execution_count"] == 0
    assert result["input_verdicts_preserved"]["self_control_passed"] is True
    assert result["input_verdicts_preserved"]["candidate_bound_passed"] is False
    assert result["adjudication"]["qualification_failure_remains_binding"] is True


def test_s09_stop_analysis_separates_transient_from_iteration_count():
    failed = analyze_existing_evidence(*_inputs())["failed_profile"]
    assert failed["trial"] == 1
    assert failed["paired_profile_center_ratio"] > 1.05
    assert failed["six_trial_median_ratio"] <= 1.03
    assert failed["first_half_ratio"] > failed["second_half_ratio"]
    assert failed["second_half_ratio"] < 1.03
    assert failed["same_pressure_iteration_count"] is True
    assert failed["remaining_trials_all_below_1_05"] is True


def test_s09_stop_analysis_does_not_overclaim_unique_cause():
    result = analyze_existing_evidence(*_inputs())
    assert result["prior_intervention"]["sufficient_to_explain_1_070771_failure"] is False
    assert result["adjudication"]["single_deleted_outlier_explanation_supported"] is False
    assert result["adjudication"]["transient_first_half_slowdown_supported"] is True
    assert result["adjudication"]["unique_runtime_root_cause_supported"] is False


def test_s09_stop_analysis_rejects_unbound_input(tmp_path):
    copied = tmp_path / "matrix.json"
    copied.write_text(json.dumps({"not": "the matrix"}), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix hash mismatch"):
        _read_bound(copied, EXPECTED_SELF_SHA256)
