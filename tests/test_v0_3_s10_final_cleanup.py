"""Bindings for the S-10 final-cleanup candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_53_s10_final_cleanup.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s10_candidate_binds_exact_local_implementation() -> None:
    record = _read(RECORD)
    assert record["plan_step"] == "S-10"
    assert record["git_identity"] == {
        "branch": "fix/v0.3-s10-final-cleanup",
        "base_commit": "f3620b87581de5f29356ed1da89673cf08ffc8a7",
        "implementation_commit": "e75ef77c2dc03237db1ddae8b275f9c8055fe06b",
        "record_commit_is_qualified_source": False,
        "pushed": False,
    }
    assert record["governing_decisions"] == ["D-0010", "D-0011"]


def test_v33_is_claimed_at_schema_two_without_payload_changes() -> None:
    full_record = _read(RECORD)
    record = full_record["v33"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["fingerprint_schema_version"] == 2
    assert record["d0008_visible_hidden_checks"]["visible_equals_hidden"] is True
    assert record["checkpoint_payloads_changed"] is False
    assert record["continuation_payloads_changed"] is False
    assert full_record["git_identity"]["implementation_commit"] == (
        "e75ef77c2dc03237db1ddae8b275f9c8055fe06b"
    )
    assert issues["V-33"]["status"] in {"claimed_fixed", "verified"}


def test_r1_r9_completed_subset_and_deferred_remainder_are_explicit() -> None:
    disposition = _read(RECORD)["r1_r9_disposition"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert set(disposition["completed_in_s10"]) == {"R7", "R9"}
    assert set(disposition["remaining_deferred"]) == {
        "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"
    }
    assert issues["R1-R9"]["status"] == "deferred"
    assert issues["R1-R9"]["decision"] == "D-0011"
    assert issues["B8/S2-07"]["decision"] == "D-0007"
    assert issues["A2-11"]["status"] == "accepted"


def test_s10_successors_are_forward_only_and_complete() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_s10_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["local_only"] is True
    assert record["scope"]["s09_branch_modified"] is False
    assert all(value is False for value in record["authorization"].values())
    assert record["next_request"] == "R-0032"
