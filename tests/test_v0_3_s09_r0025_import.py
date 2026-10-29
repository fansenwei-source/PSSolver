"""Bindings for the evidence-only R-0025 S-09 import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_39_r0025_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0025.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0025_result_and_archive_are_bound() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_r0025_import.py::"
        "test_r0025_result_and_archive_are_bound"
    )
    assert record["classification"] == "REWORK_REQUIRED_R0025_S09_H100_FAILURE_REVIEW_IMPORTED"
    assert len(record["sync_result"]["sha256"]) == 64
    assert record["sync_result"]["classification"] == "REWORK_REQUIRED"
    archive = record["independent_archive"]
    assert archive["complete_present"] is True
    assert archive["checksums_entries"] == 5846
    assert len(archive["checksums_sha256"]) == 64


def test_r0025_registry_updates_are_exact() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    record = _read(RECORD)["registry_import"]
    for issue_id, fixed_in in record["verified"].items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id]["verified_by"] == "R-0025"
    assert record["registered"] == ["V-39", "V-40"]
    assert all(
        issues[issue_id]["status"] in {"registered", "claimed_fixed", "verified"}
        for issue_id in record["registered"]
    )
    assert issues["NEW-14"]["status"] == "verified"
    assert issues["NEW-14"]["fixed_in"] == "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    assert issues["NEW-14"]["verified_by"] == "R-0027"
    assert record["s09_complete"] is False
    assert record["h100_resubmission_authorized"] is False


def test_r0025_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        live = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == live
        assert resolve_s03_successor(relative, successor["current_sha256"]) == live


def test_r0025_import_authorizes_no_external_action() -> None:
    authorization = _read(RECORD)["authorization"]
    assert authorization["local_evidence_import"] is True
    assert all(value is False for key, value in authorization.items() if key != "local_evidence_import")
