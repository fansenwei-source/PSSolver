"""Frozen record for the rc5 G-1 focused performance recovery plan."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.analyze_rc5_g1_focused_performance_h100 import (
    PASS_CLASSIFICATION,
)
from benchmarks.plan_rc5_g1_focused_performance_h100 import build_plan


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_focused_performance_recovery_v7.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_focused_recovery_record_matches_versioned_contract():
    record = _record()
    plan = build_plan()
    assert record["classification"] == "READY_FOR_EXPLICIT_H100_AUTHORIZATION"
    assert record["new_contract"]["profile_count"] == plan["profile_count"] == 48
    assert record["new_contract"]["timing_cv_max"] == (
        plan["timing_validity"]["coefficient_of_variation_max"]
    )
    assert record["unchanged_thresholds"] == {
        "mean_timestep_ratio_max": 1.03,
        "median_of_profile_means_ratio_max": 1.03,
        "every_paired_mean_timestep_ratio_max": 1.05,
        "peak_allocated_group_ratio_max": 1.03,
        "peak_reserved_group_ratio_max": 1.03,
        "graph_breaks": 0,
        "runtime_fallback": False,
        "compile_fallback": False,
        "transform_counts_exact": True,
    }
    assert record["versioned_support"]["pass_classification"] == PASS_CLASSIFICATION


def test_focused_recovery_record_preserves_failure_and_authorization_boundaries():
    record = _record()
    assert record["source_failure"]["historical_result_reclassified"] is False
    assert record["source_failure"]["thresholds_changed"] is False
    assert record["adjudication_boundary"]["focused_pass_is_g1_complete"] is False
    assert record["adjudication_boundary"][
        "focused_pass_only_authorizes_cumulative_reassembly"
    ] is True
    assert record["adjudication_boundary"]["production_default_changed"] is False
    assert not any(record["authorization"].values())
