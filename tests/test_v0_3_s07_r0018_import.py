"""Evidence bindings for importing the passing R-0018 S-07 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_22_r0018_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0018_import_binds_the_passing_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_R0018_S07_FINAL_REVIEW_IMPORTED"
    assert record["git_identity"]["reviewed_candidate"] == (
        "d86f2f647b53819702751d70c3798f441021d33b"
    )
    assert record["sync_result"] == {
        "id": "R-0018",
        "path": "/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0018.result.json",
        "sha256": "28d3efe579bfd7bc3f0bc392a2afddce73e1c37580b224e7d5a4d6479b88baff",
        "classification": "PASS",
        "new_findings": [],
    }
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 40
    assert archive["checksums_sha256"] == (
        "58a6b7a584b62f3c11f0a368016c4385ef59a420ada2d2bde8714b06ac85394d"
    )
    assert archive["complete_present"] is True


def test_r0018_verifies_all_s07_items_at_exact_fix_sources() -> None:
    imported = _read(RECORD)["registry_import"]
    registry = _read(REGISTRY)
    issues = {item["id"]: item for item in registry["issues"]}
    expected = {
        **{
            issue_id: "b7cf4ca02095e8b41ed720809880628d2375cfbb"
            for issue_id in imported["fixed_at_b7cf4ca"]
        },
        **{
            issue_id: "8aceaab446e57a13866090ec06ecdf30e33ef490"
            for issue_id in imported["fixed_at_8aceaab"]
        },
    }
    assert set(expected) == {
        "A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30", "V-26", "V-30"
    }
    for issue_id, fixed_in in expected.items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id]["verified_by"] == "R-0018"
    assert registry["status_counts"] == {
        "open": 1,
        "registered": 3,
        "claimed_fixed": 0,
        "verified": 33,
        "deferred": 2,
        "accepted": 1,
        "coverage_gap": 1,
    }
    assert imported["s07_complete"] is True


def test_r0018_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0018_import_is_evidence_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    validation = record["validation"]
    assert validation["environment_clone_method"] == "cp -al"
    assert validation["environment_clone_pruned"] is True
    assert validation["focused_pytest"]["passed"] > 0
    assert validation["focused_pytest"]["failed"] == 0
    assert validation["archive_checksums_entries"] > 0
    assert validation["archive_complete_present"] is True
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
    assert record["next_plan_step"] == "S-08"
    assert record["next_request"] == "R-0019"
