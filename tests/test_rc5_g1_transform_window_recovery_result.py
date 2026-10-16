"""Frozen result contract for the rc5 G-1 transform-window replay."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_transform_window_recovery_v6_result.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_transform_window_replay_passes_without_mutating_frozen_profiles():
    record = _record()
    assert record["classification"] == (
        "STOPPED_RC5_G1_TRANSFORM_WINDOW_RECOVERY_V6_AT_"
        "FROZEN_PROFILE_PERFORMANCE_MEMORY_GATE"
    )
    assert record["transform_measurement_window"] == {
        "passed": True,
        "profiles_passed": 96,
        "profiles_failed": 0,
        "scheduled_refresh_profiles": 24,
        "channel_raw_forward_calls": 663,
        "channel_scheduled_forward_calls": 3,
        "channel_steady_forward_calls": 660,
        "channel_steady_forward_per_step": 11.0,
        "channel_raw_inverse_calls": 2160,
        "channel_steady_inverse_per_step": 36.0,
        "source_profiles_modified": False,
    }
    assert record["execution"]["new_h100_submission_count"] == 0
    assert record["execution"]["scientific_timestep_executed"] is False


def test_transform_window_replay_records_exact_performance_memory_stops():
    gate = _record()["performance_memory_gate"]
    assert gate["passed"] is False
    assert gate["group_count"] == 16
    assert gate["failed_group_count"] == 4
    assert {
        (entry["family"], entry["grid"], entry["gate"])
        for entry in gate["failures"]
    } == {
        ("plane_compiled_v2", "plane_R128", "paired_timestep_ratio"),
        (
            "channel_compiled_v2",
            "channel_R128x20x20",
            "paired_timestep_ratio",
        ),
        ("periodic_functional", "periodic_R128", "paired_timestep_ratio"),
        (
            "channel_functional",
            "channel_R512x40x40",
            "peak_reserved_group_ratio",
        ),
    }


def test_transform_window_replay_is_not_a_qualification_pass():
    record = _record()
    assert record["analysis_complete"] is True
    assert record["qualification_complete"] is False
    assert record["g1_complete"] is False
    assert record["archive"]["checksum_entries_passed"] == 15
    assert record["archive"]["checksum_entries_failed"] == 0
    assert record["archive"]["complete_present"] is False
    assert record["archive"]["failed_marker_present"] is True
    assert not any(record["authorization"].values())
