"""Static record gate for the R5-1 released-checkpoint repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
)
R5_2_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
R5_3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
)
R5_5_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
R5_6_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
R5_7_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
R5_8_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
R5_9_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
R5_10_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
R5_11_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
R5_12_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)
R5_13_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
G1_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"
G1_GLIBC_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_checkpoint_glibc_compatibility.json"
)
RC6_R1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
)
RC6_R2_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
)
RC6_R3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
)


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _latest_successor(relative: str, historical: str) -> str:
    resolved = historical
    for path in (
        R5_2_RECORD,
        R5_3_RECORD,
        R5_5_RECORD,
        R5_6_RECORD,
        R5_7_RECORD,
        R5_8_RECORD,
        R5_9_RECORD,
        R5_10_RECORD,
        R5_11_RECORD,
        R5_12_RECORD,
        R5_13_RECORD,
        G1_RECORD,
        G1_GLIBC_RECORD,
        RC6_R1_RECORD,
        RC6_R2_RECORD,
        RC6_R3_RECORD,
    ):
        successor = json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        accepted = (
            []
            if successor is None
            else successor.get(
                "accepted_predecessor_sha256",
                [successor["historical_sha256"]],
            )
        )
        if successor is not None and resolved in accepted:
            resolved = successor["current_sha256"]
    return resolved


def test_r51_record_binds_implementation_fixture_and_regressions():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_R5_1_REAL_RC1_RC2_RC3_FUNCTIONAL_CHECKPOINT_"
        "COMPATIBILITY_CPU_CLOSURE"
    )
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == _latest_successor(relative, expected)
    fixture = record["fixture"]
    assert _sha256(fixture["manifest"]) == fixture["manifest_sha256"]
    assert fixture["checkpoint_families"] == 6
    assert fixture["verbatim_payload_copy"] is True


def test_r51_successors_are_exact_and_do_not_rewrite_historical_records():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    r5_2_successors = json.loads(R5_2_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_3_successors = json.loads(R5_3_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_5_successors = json.loads(R5_5_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_6_successors = json.loads(R5_6_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_7_successors = json.loads(R5_7_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_8_successors = json.loads(R5_8_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_9_successors = json.loads(R5_9_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_10_successors = json.loads(R5_10_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_11_successors = json.loads(R5_11_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_12_successors = json.loads(R5_12_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_13_successors = json.loads(R5_13_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    g1_successors = json.loads(G1_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    g1_glibc_successors = json.loads(
        G1_GLIBC_RECORD.read_text(encoding="utf-8")
    )["historical_successor_sources"]
    rc6_successors = json.loads(RC6_R1_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r2_successors = json.loads(RC6_R2_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc6_r3_successors = json.loads(RC6_R3_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        assert registration["historical_sha256"] != registration[
            "current_sha256"
        ]
        resolved = registration["current_sha256"]
        successor = r5_2_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_3_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_5_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_6_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_7_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_8_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_9_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_10_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_11_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_12_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = r5_13_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = g1_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = g1_glibc_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_successors.get(relative)
        accepted = (
            []
            if successor is None
            else successor.get(
                "accepted_predecessor_sha256",
                [successor["historical_sha256"]],
            )
        )
        if successor is not None and resolved in accepted:
            resolved = successor["current_sha256"]
        successor = rc6_r2_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc6_r3_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert _sha256(relative) == resolved
    assert record["scope"]["historical_records_rewritten"] is False
    assert record["scope"]["tag_or_release_modified"] is False


def test_r51_scope_does_not_claim_batch_or_release_completion():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["issue"]["matrix_updated"] is False
    assert record["scope"]["H100_required"] is False
    assert record["scope"]["H100_executed"] is False
    assert record["scope"]["PSSolver_Control_modified"] is False
    assert record["scope"]["nematics3d_modified"] is False
    assert record["scope"]["tests_relaxed"] is False
    assert record["scope"]["skip_or_xfail_added"] is False
    assert record["authorization"]["r5_1_complete"] is True
    assert record["authorization"]["eligible_for_next_rc5_must_item"] is True
    assert record["authorization"]["push_authorized"] is False
    assert record["authorization"]["eligible_for_rc5_release"] is False
    assert record["authorization"]["rc5_must_batch_complete"] is False


def test_r51_consumer_impact_review_uses_the_public_bridge_without_migration():
    review = json.loads(RECORD.read_text(encoding="utf-8"))[
        "consumer_impact_review"
    ]
    assert review["required"] is True
    assert review["targeted_cpu"] == "68 passed, 0 failed, 0 skipped"
    assert review["consumer_parses_legacy_checkpoint_schema"] is False
    assert review["consumer_forwards_public_checkpoint_bridge"] is True
    assert review["functional_api_version_changed"] is False
    assert review["current_runtime_identity_changed"] is False
    assert review["consumer_migration_required"] is False
    assert review["PSSolver_Control_modified"] is False
