"""Bindings for importing the R-0022 S-09 pre-H100 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_33_r0022_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0022.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0022_import_binds_the_rework_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "REWORK_REQUIRED_R0022_S09_PRE_H100_REVIEW_IMPORTED"
    )
    assert record["git_identity"]["reviewed_candidate"] == (
        "6b659b1ad45855877310ccc3e3a1718a1109ca84"
    )
    sync_result = record["sync_result"]
    assert sync_result["classification"] == "REWORK_REQUIRED"
    assert sync_result["new_findings"] == ["V-35", "V-36"]
    assert _sha256(RESULT) == sync_result["sha256"]
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 14880
    assert archive["checksums_sha256"] == (
        "ea1cbbedb5d952ac9881d57d1b01a7e629f3769953b2b20b17df3bb8a556cb30"
    )
    assert archive["complete_present"] is True


def test_r0022_record_is_preserved_while_s09_status_moves_forward() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["registry_import"]["registered"] == ["V-35", "V-36"]
    for issue_id in ("V-35", "V-36"):
        issue = issues[issue_id]
        assert issue["status"] == "verified"
        assert issue["fixed_in"] == (
            "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
        )
        assert issue["target_release"] == "0.3"
        assert issue["source_report"] == "PSSolver_sync/results/R-0022.result.json"
        assert issue["verified_by"] == "R-0024"
    assert issues["V-37"]["status"] == "verified"
    assert issues["V-37"]["fixed_in"] == (
        "87cf34cf4e17096a701f5cc5bc81076d14a9d237"
    )
    assert issues["V-37"]["verified_by"] == "R-0024"
    assert issues["A3-18"]["status"] == "coverage_gap"
    assert issues["NEW-14"]["status"] == "open"


def test_r0022_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0022_import_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_import_authorized"
    )
    assert record["next_plan_step"] == "S-09"
    assert record["next_request"] == "R-0023"
