"""Static contracts for the Phase 6 H100 qualification support slice."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def test_phase6_support_record_freezes_authorization_boundary():
    value = json.loads(
        (NOTES / "phase_6_p61_h100_qualification_support.json").read_text(
            encoding="utf-8"
        )
    )

    assert value["classification"] == "READY_P6_1_H100_CANDIDATE_GATE"
    assert value["authorization"] == {
        "phase_6_execution_authorized_by_user": True,
        "formal_h100_candidate_gate_authorized": True,
        "long_run_contingent_on_candidate_pass": True,
        "production_default_changed": False,
        "default_promotion_authorized": False,
    }
    assert value["profile_matrix"]["profile_count"] == 12
    assert value["workflow_matrix"]["comparison_count"] == 6
    assert value["eligibility"]["ready_for_h100_candidate_gate"] is True
    assert value["eligibility"]["long_run_stage_eligible"] is False
    assert value["eligibility"]["default_promotion_eligible"] is False


def test_phase6_tools_are_analysis_and_measurement_only():
    profiler = (ROOT / "benchmarks" / "profile_plane_runtime_timestep.py").read_text(
        encoding="utf-8"
    )
    analyzer = (
        ROOT / "scripts_plane" / "analyze_phase6_compiled_v2_qualification.py"
    ).read_text(encoding="utf-8")

    assert "LegacyPlaneRuntimeAdapter" in profiler
    assert "build_plane_compiled_v2_runtime" in profiler
    assert "adapter.advance(1)" in profiler
    assert "PASS_PHASE6_H100_CANDIDATE_PERFORMANCE_EQUIVALENT" in analyzer
    assert "eligible_for_default_promotion\": False" in analyzer
    assert "production_default_changed\": False" in analyzer
    assert "build_legacy_plane_runtime" not in analyzer
    assert "build_plane_compiled_v2_runtime" not in analyzer


def test_phase6_recovery_record_preserves_failure_and_changes_only_adjudication():
    value = json.loads(
        (NOTES / "phase_6_p61_performance_equivalence_recovery.json").read_text(
            encoding="utf-8"
        )
    )

    assert value["classification"] == (
        "READY_PHASE6_PERFORMANCE_EQUIVALENCE_RECOVERY"
    )
    assert value["profile_source"]["original_classification"] == (
        "FAIL_PHASE6_H100_CANDIDATE_GATE"
    )
    assert value["corrected_performance_contract"][
        "candidate_faster_count_role"
    ] == "informational"
    assert value["measured_results"]["R128"]["performance_class"] == (
        "performance_equivalent"
    )
    assert value["measured_results"]["R320"]["performance_class"] == (
        "performance_equivalent"
    )
    assert value["authorization"]["long_run_authorized"] is False
    assert value["authorization"]["production_default_changed"] is False
    assert value["recovery_scope"]["rerun_profiles"] is False
