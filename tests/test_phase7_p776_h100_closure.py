"""Audit the imported P7.7.6 H100/CPU closure evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
CLOSURE_PATH = NOTES / "phase_7_p776_h100_closure.json"
LOCAL_PATH = NOTES / "phase_7_p776_production_connection.json"
ARCHIVE_BUILDER = NOTES / "build_verbatim_archive_pdf.py"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(CLOSURE_PATH.read_text(encoding="utf-8"))


def test_closure_preserves_local_candidate_history_and_closes_qualification():
    local = json.loads(LOCAL_PATH.read_text(encoding="utf-8"))
    closure = _record()

    assert local["status"] == "local_candidate_ready_for_h100"
    assert local["qualification_complete"] is False
    assert local["p7_7_6_complete"] is False
    assert closure["status"] == "complete"
    assert closure["classification"] == (
        "PASS_P7_7_6_PRODUCTION_CONNECTION_H100_NON_REGRESSION_WITH_"
        "TARGETED_ADJUDICATION_AND_FINAL_CPU_GUARD_RECOVERY"
    )
    assert closure["qualification"]["complete"] is True
    assert closure["eligibility"] == {
        "p7_7_6_complete": True,
        "phase_8_planning_eligible": True,
        "phase_8_authorized": False,
        "phase_9_authorized": False,
        "plane_default": "legacy_production",
        "channel_default": "legacy_channel",
    }


def test_closure_binds_all_four_archives_and_final_complete_marker():
    archives = _record()["archive_evidence"]
    assert [item["entries"] for item in archives] == [244, 268, 19, 78]
    assert [item["manifest_sha256"] for item in archives] == [
        "d172818ab84cc97a4b1c01fab5e1942b5faeae7592cf2d689fe4e834a59e6af7",
        "94d4dbf2440136e9bc134e78e56cce3c05159224558b0a27d99d59784ab1c7e2",
        "f7e9e4f98f3e0530bb75e181624286f9abe9774c9d1942c77084a1701e47c63a",
        "14593163794bfed3f65445630ce3783268ddab16612c7e6c659774a4dd7db068",
    ]
    assert [item["complete_marker"] for item in archives] == [
        False,
        False,
        False,
        True,
    ]
    assert archives[-1]["complete_path"].endswith("/COMPLETE")


def test_targeted_performance_satisfies_the_unchanged_contract():
    profile = _record()["profiles"]
    assert profile["targeted_pairs"] == 12
    assert profile["measured_steps"] == 200
    assert profile["arithmetic_mean_ratio"] <= profile["mean_ratio_limit"]
    assert profile["median_ratio"] <= profile["median_ratio_limit"]
    assert profile["bootstrap_95_percent_ci"] == [
        0.9845594354899496,
        1.0248376844826659,
    ]
    assert profile["peak_allocated_ratio"] == 1.0
    assert profile["peak_reserved_ratio"] == 1.0
    assert profile["forward_transforms_per_step"] == 11
    assert profile["inverse_transforms_per_step"] == 36


def test_numerical_restart_and_negative_gates_are_complete():
    record = _record()
    numerical = record["numerical_equivalence"]
    gates = record["negative_gates"]

    assert numerical["production_paths"] == 5
    assert numerical["complete_continuous_segment_resumed_directories"] == 40
    assert numerical["baseline_candidate_q_u_p"] == "byte_for_byte"
    assert numerical["runtime_identity_drift"] is False
    assert all(value == "PASS" for key, value in numerical.items() if key.endswith("restart"))
    assert all(value == "PASS" for value in gates.values())


def test_closure_source_hashes_match_local_candidate_sources():
    assert _record()["source_sha256"] == {
        "notes/architecture_v0_2/phase_7_p776_production_connection.json": _sha256(
            "notes/architecture_v0_2/phase_7_p776_production_connection.json"
        ),
        "pssolver/applications/channel_active_nematics.py": _sha256(
            "pssolver/applications/channel_active_nematics.py"
        ),
        "pssolver/applications/plane_beris_edwards.py": _sha256(
            "pssolver/applications/plane_beris_edwards.py"
        ),
        "tests/test_phase7_p776_production_connection.py": _sha256(
            "tests/test_phase7_p776_production_connection.py"
        ),
    }


def test_archive_source_list_includes_closure_without_regenerating_pdf():
    source = ARCHIVE_BUILDER.read_text(encoding="utf-8")
    candidate = '"phase_7_p776_production_connection.json"'
    closure_md = '"phase_7_p776_h100_closure.md"'
    closure_json = '"phase_7_p776_h100_closure.json"'

    assert source.count(closure_md) == 1
    assert source.count(closure_json) == 1
    assert source.index(candidate) < source.index(closure_md)
    assert source.index(closure_md) < source.index(closure_json)
