"""Frozen record checks for the rc5 G-1 timing analysis v8."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / "notes/PSSolver_v0_2_0rc5_g1_timing_variability_analysis_v8.json"
REPORT_PATH = ROOT / "notes/PSSolver_v0_2_0rc5_g1_timing_variability_analysis_v8_zh.md"


def _record() -> dict[str, object]:
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


def test_record_preserves_the_formal_v7_failure_and_scope():
    record = _record()
    assert record["classification"] == (
        "HARNESS_FIRST_SAMPLE_CONTAMINATION_AND_RESIDUAL_IN_WINDOW_VARIABILITY"
    )
    assert record["analysis_complete"] is True
    assert record["analysis_only"] is True
    assert record["evidence"]["v7_classification_preserved"] == (
        "FAIL_RC5_G1_FOCUSED_PERFORMANCE_MEMORY_GATE"
    )
    assert record["formal_state"] == {
        "eligible_for_automatic_merge": False,
        "eligible_for_cumulative_reassembly": False,
        "eligible_for_default_promotion": False,
        "g1_complete": False,
        "new_h100_submission_count": 0,
        "production_default_changed": False,
        "qualification_complete": False,
    }
    assert not any(record["authorization"].values())


def test_record_binds_the_analyzer_and_read_only_evidence():
    record = _record()
    assert record["analysis_source"]["commit"] == (
        "5a64129db775ed476997d17d291493105594a0cc"
    )
    assert record["analysis_source"]["qualification_support_archive_sha256"] == (
        "7f7f2b12ecd562c74071f164bdc047abc661b5501e56062376429fd4a7811ccb"
    )
    assert record["analysis_source"]["timing_analysis_sha256"] == (
        "7e3e45e4f7701ba78398d045427e77443d27681d0bc6aa9ac71bd2f29cf28a75"
    )
    assert record["evidence"]["profile_count"] == 48
    assert record["evidence"]["v7_control_checksums"] == "36/36 PASS"
    assert record["evidence"]["v7_scratch_checksums"] == "1119/1119 PASS"
    assert record["findings"]["full_window_cv_failure_count"] == 32
    assert record["findings"]["diagnostic_excluding_first_sample_cv_failure_count"] == 21
    assert record["findings"]["profiles_with_first_sample_overhead"] == 37
    assert sum(row["diagnostic_spike_count"] for row in record["per_family"]) == 132


def test_record_does_not_overstate_order_observation_or_recovery_authority():
    record = _record()
    order = record["execution_order_observation"]
    assert order["candidate_first_ratio_mean_higher_family_count"] == 4
    assert order["family_count"] == 4
    assert order["causal_conclusion_supported"] is False
    recommendation = record["recommended_next_step"]
    assert recommendation["h100_submission_authorized"] is False
    assert recommendation["must_preserve_raw_samples"] is True
    assert recommendation["must_preserve_v7_failure"] is True
    assert recommendation["outlier_removal_allowed"] is False
    assert recommendation["threshold_relaxation_authorized"] is False
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "不能据此唯一归因" in report
    assert "不应直接重跑同一合同" in report
