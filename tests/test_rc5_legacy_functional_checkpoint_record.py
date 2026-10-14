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


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_r51_record_binds_implementation_fixture_and_regressions():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_R5_1_REAL_RC1_RC2_RC3_FUNCTIONAL_CHECKPOINT_"
        "COMPATIBILITY_CPU_CLOSURE"
    )
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == expected
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
