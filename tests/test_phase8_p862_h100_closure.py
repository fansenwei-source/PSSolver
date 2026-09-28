"""Audit the imported P8.6.2 integrated H100 closure evidence."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _record() -> dict[str, object]:
    return json.loads(
        (NOTES / "phase_8_p862_h100_closure.json").read_text(encoding="utf-8")
    )


def test_p862_h100_closure_binds_source_wheel_job_and_archive():
    record = _record()
    assert record["classification"] == (
        "PASS_P8_6_2_INTEGRATED_INSTALLED_WHEEL_H100_CLOSURE"
    )
    assert record["imported_report"]["sha256"] == (
        "2cce0f3c368ba05ba06f95218ea77f0d5f7ebedea7100cdaf1e1b130a68c9e2c"
    )
    assert record["identity"]["qualification_source_commit"] == (
        "24739c772fb6f046b6e9d5a40ae32fb54f525246"
    )
    assert record["identity"]["installed_wheel_only"] is True
    assert record["h100"]["job_id"] == "10844682"
    assert record["h100"]["state"] == "COMPLETED"
    assert record["h100"]["formal_submission_count"] == 1
    assert record["archive"]["manifest_passed"] == 1204
    assert record["archive"]["manifest_sha256"] == (
        "70d65ad5a79a7282fd3653157442f4e443efe3788d1c2c83c2874ed71184203d"
    )


def test_p862_uses_the_exact_channel_checkpoint_schema():
    schema = _record()["canonical_channel_checkpoint_schema"]
    assert schema["kind"] == "channel_complete_stress_pressure_pcg"
    assert schema["state_keys"] == ["pressure_guess"]
    assert schema["reconstructed_state_keys"] == ["q_gradient_cache"]
    assert schema["exact_match_required"] is True
    assert schema["q_gradient_cache_persisted_as_backend_tensor"] is False


def test_p862_closes_all_eight_cases_and_restart_gates():
    record = _record()
    matrix = record["runtime_matrix"]
    assert len(matrix) == 8
    assert [item["case"] for item in matrix] == [
        "plane_legacy",
        "plane_compiled",
        "channel_legacy",
        "channel_compiled",
        "periodic_complete_stress",
        "channel_complete_stress",
        "plane_static_q_lifting",
        "finite_q_robin_relaxation_pilot",
    ]
    assert all(item["finite"] and item["restart"] for item in matrix)
    assert all(item["public"] for item in matrix[:7])
    assert matrix[7]["public"] is False
    assert record["restart"]["passed_gates"] == 8
    assert record["restart"]["all_required_complete"] is True
    assert record["negative_gates"]["all_required_complete"] is True


def test_p862_preserves_defaults_and_explicitly_limits_claims():
    record = _record()
    compatibility = record["compatibility"]
    assert compatibility["plane_default"] == "legacy_production"
    assert compatibility["channel_default"] == "legacy_channel"
    assert compatibility["compiled_runtime_promoted"] is False
    assert compatibility["robin_publicly_executable"] is False
    assert compatibility["production_default_changed"] is False
    assert all(value is False for value in record["non_claims"].values())
    assert record["eligibility"] == {
        "qualification_complete": True,
        "p8_6_2_complete": True,
        "eligible_for_p8_6_3_local_finalization": True,
        "phase_8_complete": False,
        "phase_9_authorized": False,
        "production_default_changed": False,
        "compiled_runtime_promoted": False,
    }
