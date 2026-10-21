"""Evidence bindings for the independent R-0009 S-03 PASS result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_03_r0009_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0009_import_binds_the_pass_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_R0009_S03_REWORK_REVIEW_IMPORTED"
    assert record["git_identity"] == {
        "reviewed_candidate": "372ffa080084a8abbcb4e9dfe1700c237bc32953",
        "implementation_commit": "a10785853736b2458eec80680b9ced72a495775f",
        "record_commit_is_qualified_source": False,
        "pushed": False,
    }
    result = record["sync_result"]
    assert result["id"] == "R-0009"
    assert result["commit"] == "34c26d5a386f49e34814d6521c758fb490302a84"
    assert result["sha256"] == (
        "1a2ae6297d4ca55cb61674611c4be1f7e278cbba86b83bb195df523fb10e89fe"
    )
    assert result["classification"] == "PASS"
    assert result["release_blockers"] == result["new_findings"] == []
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 2413
    assert archive["checksums_sha256"] == (
        "9afa823c8c668e35dcfe21c8bf2a9b1f9564c21083c83615c9692c6b31c97ff8"
    )
    assert archive["complete_present"] is True


def test_r0009_import_applies_exactly_three_verified_updates() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    verified = set(record["registry_updates"]["verified_by_R_0009"])
    assert verified == {"C1-08", "A2-24/C1-15", "V-24"}
    for issue_id in verified:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == record["registry_updates"]["fixed_in"]
        assert issues[issue_id]["verified_by"] == "R-0009"
    for issue_id in record["registry_updates"]["preserved_verified_by_R_0008"]:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0008"
    assert record["registry_updates"]["new_findings_registered"] == []


def test_r0009_import_is_a_forward_only_successor_layer() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == (
            successor["current_sha256"]
        )
        assert _sha256(ROOT / relative) == successor["current_sha256"]


def test_r0009_import_is_evidence_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert record["authorization"]["local_import_authorized"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_import_authorized"
    )
