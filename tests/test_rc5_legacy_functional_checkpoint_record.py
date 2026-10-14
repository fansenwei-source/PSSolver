"""Static record gate for the R5-1 released-checkpoint repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
)


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
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        assert registration["historical_sha256"] != registration[
            "current_sha256"
        ]
        assert _sha256(relative) == registration["current_sha256"]
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
