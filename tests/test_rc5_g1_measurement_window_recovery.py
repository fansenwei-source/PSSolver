"""Contracts for the rc5 G-1 measurement-window recovery plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.plan_rc5_g1_focused_performance_h100 import (
    build_plan as build_focused_plan,
)
from benchmarks.plan_rc5_g1_measurement_window_recovery import (
    POST_FLUSH_REWARM_STEPS,
    RUNTIME_COMPLETED_STEPS,
    SCHEMA,
    build_plan,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_measurement_window_recovery_v9.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_preserves_matrix_pair_order_and_every_frozen_threshold():
    focused = build_focused_plan()
    plan = build_plan()
    assert plan["schema"] == SCHEMA
    assert plan["profile_count"] == 48
    assert plan["thresholds"] == focused["thresholds"]
    assert plan["timing_validity"] == focused["timing_validity"]
    assert [row["key"] for row in plan["profile_matrix"]] == [
        row["key"] for row in focused["profile_matrix"]
    ]
    assert [row["provider_role"] for row in plan["profile_matrix"]] == [
        row["provider_role"] for row in focused["profile_matrix"]
    ]


def test_measurement_window_separates_recovery_from_all_formal_windows():
    plan = build_plan()
    window = plan["measurement_window"]
    assert POST_FLUSH_REWARM_STEPS == 3
    assert window["post_flush_rewarm_steps"] == 3
    assert window["timed_steps"] == 60
    assert window["expected_runtime_completed_steps"] == RUNTIME_COMPLETED_STEPS == 73
    assert window["rewarm_is_outside_formal_timing_window"] is True
    assert window["rewarm_is_outside_formal_transform_counter_window"] is True
    assert window["rewarm_is_outside_formal_compiler_counter_window"] is True
    assert window["rewarm_is_outside_formal_memory_peak_window"] is True
    assert window["cold_recovery_memory_is_preserved_as_diagnostic"] is True
    assert window["phase_order"].index("cuda_empty_cache") < window[
        "phase_order"
    ].index("post_flush_rewarm_3_steps_untimed")
    assert window["phase_order"].index(
        "post_flush_rewarm_3_steps_untimed"
    ) < window["phase_order"].index("reset_formal_peak_memory_stats")
    assert window["phase_order"].index(
        "reset_formal_transform_timer"
    ) < window["phase_order"].index("time_60_steps_with_cuda_events")


def test_every_profile_freezes_new_window_and_recomputes_exact_call_contract():
    plan = build_plan()
    assert len({row["key"] for row in plan["profile_matrix"]}) == 48
    for row in plan["profile_matrix"]:
        assert row["post_flush_rewarm_steps"] == 3
        assert row["measurement_start_completed_steps"] == 13
        assert row["expected_runtime_completed_steps"] == 73
        assert row["expected_inverse_raw_calls"] == (
            row["inverse_transforms_per_step"] * 60
        )
        assert row["expected_forward_raw_calls"] == (
            row["forward_transforms_per_step"] * 60
            + row["scheduled_forward_calls_in_timed_window"]
        )


def test_no_sample_trial_threshold_or_h100_escape_hatch_is_authorized():
    plan = build_plan()
    assert plan["timing_validity"]["raw_sample_count"] == 60
    assert plan["timing_validity"]["coefficient_of_variation_max"] == 0.05
    assert plan["adjudication"]["no_outlier_removal"] is True
    assert plan["adjudication"]["no_first_sample_deletion"] is True
    assert plan["adjudication"]["no_trial_replacement"] is True
    assert plan["adjudication"]["no_threshold_is_relaxed"] is True
    assert plan["authorization"]["h100_submission_authorized"] is False
    assert plan["authorization"]["automatic_retry_authorized"] is False
    assert plan["authorization"]["profile_threshold_change_authorized"] is False
    assert plan["authorization"]["production_default_changed"] is False


def test_diagnostic_replay_is_independent_and_cannot_adjudicate_performance():
    replay = build_plan()["diagnostic_replay"]
    assert replay["required"] is True
    assert replay["separate_fresh_runtime"] is True
    assert replay["same_seed_configuration_and_initial_state_sha"] is True
    assert replay["never_used_for_performance_or_memory_adjudication"] is True
    assert replay["raw_per_step_timing_samples_preserved"] is True
    assert replay["unavailable_diagnostic_encoding"] == {
        "applicable": False,
        "reason_required": True,
        "fabricated_numeric_placeholder_allowed": False,
    }


def test_plan_cli_is_strict_atomic_and_non_overwriting(tmp_path):
    output = tmp_path / "plan.json"
    assert main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == build_plan()
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_versioned_record_binds_the_plan_analysis_and_authorization_boundary():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    support = record["versioned_support"]
    source = record["source_analysis"]
    assert record["classification"] == "READY_FOR_QUALIFICATION_SUPPORT_IMPLEMENTATION"
    assert _sha256(ROOT / support["plan"]) == support["plan_sha256"]
    assert _sha256(ROOT / source["record"]) == source["record_sha256"]
    assert source["v7_failure_preserved"] is True
    assert record["formal_state"]["qualification_complete"] is False
    assert record["formal_state"]["g1_complete"] is False
    assert record["authorization"]["h100_submission_authorized"] is False
    assert record["authorization"]["profile_threshold_change_authorized"] is False
    assert record["next_step"]["h100_submission_part_of_next_step"] is False
