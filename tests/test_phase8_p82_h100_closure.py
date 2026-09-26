"""Audit the imported P8.2 H100 closure without erasing local history."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
LOCAL_RECORD = NOTES / "phase_8_p82_periodic_complete_stress.json"
CLOSURE_RECORD = NOTES / "phase_8_p82_h100_closure.json"
ARCHIVE_BUILDER = NOTES / "build_verbatim_archive_pdf.py"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_p82_closure_preserves_local_candidate_history():
    local = _json(LOCAL_RECORD)
    closure = _json(CLOSURE_RECORD)

    assert local["status"] == "local_candidate"
    assert local["verification"]["h100_completed"] is False
    assert local["eligibility"]["p8_2_complete"] is False
    assert closure["status"] == "complete"
    assert closure["classification"] == (
        "PASS_P8_2_PERIODIC_COMPLETE_STRESS_H100_QUALIFICATION"
    )
    assert closure["qualified_commit"] == (
        "44a76b36a86e221fde3cfe0ef59d5a9a17a091c7"
    )


def test_p82_final_execution_and_environment_are_bound():
    record = _json(CLOSURE_RECORD)
    execution = record["execution"]
    environment = record["environment"]

    assert execution["job_id"] == 10842576
    assert execution["state"] == "COMPLETED"
    assert execution["exit_code"] == "0:0"
    assert execution["formal_recovery_v3_h100_submissions"] == 1
    assert execution["automatic_retry"] is False
    assert environment["gpu"] == "NVIDIA H100 PCIe"
    assert environment["runtime_requested"] == "periodic_spectral"
    assert environment["runtime_effective"] == "periodic_spectral"
    assert environment["fallback_used"] is False
    assert environment["source_shadow_import"] is False


def test_p82_profiles_and_transform_window_pass_frozen_contract():
    record = _json(CLOSURE_RECORD)
    profiles = record["profiles"]
    counter = record["transform_counter"]

    assert max(profiles["r128"]["trial_cv_percent"]) / 100 <= profiles[
        "single_trial_cv_limit"
    ]
    assert max(profiles["r320"]["trial_cv_percent"]) / 100 <= profiles[
        "single_trial_cv_limit"
    ]
    assert profiles["r128"]["trial_mean_max_min_ratio"] <= profiles[
        "trial_mean_max_min_limit"
    ]
    assert profiles["r320"]["trial_mean_max_min_ratio"] <= profiles[
        "trial_mean_max_min_limit"
    ]
    assert profiles["speedup_claim"] is False
    assert profiles["benchmark_claim"] is False
    assert counter["measured_forward_raw_delta"] == 700
    assert counter["measured_inverse_raw_delta"] == 3100
    assert counter["forward_calls_per_step"] == 7.0
    assert counter["inverse_calls_per_step"] == 31.0
    assert counter["post_measurement_observation"] == {
        "forward": 6,
        "inverse": 21,
    }
    assert counter["post_measurement_calls_excluded"] is True


def test_p82_restart_and_all_negative_gates_are_complete():
    record = _json(CLOSURE_RECORD)

    assert record["production_trajectories"]["completed"] == 4
    assert record["restart"]["r128_continuous_resumed_byte_identical"] is True
    gates = record["negative_gates"]
    assert gates["passed"] == gates["expected"] == 6
    assert all(
        gates[name] == "PASS"
        for name in (
            "tensor_raw_bytes",
            "checksum_metadata",
            "shape_dtype_record",
            "runtime_path",
            "runtime_identity",
            "backend_identity",
        )
    )
    assert gates["all_rejected_before_timestep"] is True
    assert gates["target_state_unchanged"] is True


def test_p82_all_archives_are_bound_and_final_archive_is_complete():
    archives = _json(CLOSURE_RECORD)["evidence_chain"]

    assert [item["manifest_entries"] for item in archives] == [
        363,
        32,
        141,
        8,
        265,
    ]
    assert [item["manifest_sha256"] for item in archives] == [
        "b00ec613897da3f80c365cbabce7542de4aef42a4b44fd71366604c4877d6738",
        "540505c1a9150be1588d595396f42a0e9a2448b9ec694f135235c0dcea11470c",
        "6d212eb1201f8f0761014e20995010f3a024a28c9dc66d402edc13ab94e81a84",
        "d4d8a100981408806e61b217e60de97edefabb780b6c88f794b539425463469c",
        "42d16ad9fcd30f0470e20752b09ce0bc5fe3e278f712eb10a8f07becc79572b7",
    ]
    assert archives[-1]["complete_marker"] is True
    assert archives[-1]["complete_content"] == (
        "PASS_P8_2_PERIODIC_COMPLETE_STRESS_H100_QUALIFICATION"
    )


def test_p82_qualification_closes_only_the_registered_slice():
    qualification = _json(CLOSURE_RECORD)["qualification"]

    assert qualification == {
        "complete": True,
        "p8_2_complete": True,
        "eligible_for_p8_3_planning": True,
        "p8_3_started": False,
        "p8_4_started": False,
        "production_default_changed": False,
        "plane_default": "legacy_production",
        "channel_default": "legacy_channel",
        "compiled_runtime_promoted": False,
        "benchmark_claim_made": False,
        "phase_9_authorized": False,
    }


def test_archive_source_order_includes_closure_after_local_candidate():
    source = ARCHIVE_BUILDER.read_text(encoding="utf-8")
    candidate = '"phase_8_p82_periodic_complete_stress.json"'
    closure_md = '"phase_8_p82_h100_closure.md"'
    closure_json = '"phase_8_p82_h100_closure.json"'

    assert source.count(closure_md) == 1
    assert source.count(closure_json) == 1
    assert source.index(candidate) < source.index(closure_md)
    assert source.index(closure_md) < source.index(closure_json)
