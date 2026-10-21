"""Evidence bindings for the independent R-0008 S-03 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_01_r0008_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0008_import_binds_the_rework_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["stage"] == "S-03"
    assert record["classification"] == "REWORK_REQUIRED_R0008_REVIEW_IMPORTED"
    assert record["sync_result"] == {
        "id": "R-0008",
        "path": "/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0008.result.json",
        "sha256": "1c56f6f59de0996c42a76ddf7ab03f448d3f884c8e03defca2fa1a2b4418211c",
        "classification": "REWORK_REQUIRED",
        "release_blockers": [],
    }
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 4893
    assert archive["checksums_sha256"] == (
        "957b3de79bc71780990714ec47e3c4d8961369a4fdb433cfa0b0f15fa1c43c0a"
    )
    assert archive["complete_present"] is True


def test_r0008_import_applies_the_exact_registry_verdicts() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    for issue_id in record["verified_issue_ids"]:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0008"
    for issue_id in record["reopened_issue_ids"]:
        assert issues[issue_id]["status"] == "reopened"
    assert record["new_finding_ids"] == ["V-24"]
    assert issues["V-24"]["status"] == "registered"
    assert issues["V-24"]["affects_control_paths"] is True


def test_r0008_import_is_one_forward_only_successor_layer() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == (
            successor["current_sha256"]
        )
        assert _sha256(ROOT / relative) == successor["current_sha256"]


def test_r0008_import_authorizes_no_external_or_release_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert all(value is False for value in record["authorization"].values())
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["next_request"] == "R-0009"
