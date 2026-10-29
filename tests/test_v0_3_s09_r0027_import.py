"""Bindings for the evidence-only R-0027 S-09 import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_42_r0027_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0027.result.json")
DECISION = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/decisions/D-0012.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0027_result_archive_and_decision_are_bound() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_r0027_import.py::"
        "test_r0027_result_archive_and_decision_are_bound"
    )
    assert record["classification"] == (
        "PASS_R0027_H100_CORRECTNESS_AND_FAILURE_REVIEW_IMPORTED"
    )
    assert len(record["sync_result"]["sha256"]) == 64
    assert record["sync_result"]["classification"] == "REWORK_REQUIRED"
    archive = record["independent_archive"]
    assert archive["complete_present"] is True
    assert archive["checksums_entries"] == 4056
    assert len(archive["checksums_sha256"]) == 64
    assert len(record["decision"]["sha256"]) == 64
    assert record["decision"]["id"] == "D-0012"
    assert record["decision"]["diagnostic_h100_authorized"] is False


def test_r0027_registry_updates_are_exact() -> None:
    record = _read(RECORD)["registry_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["verified"] == {
        "NEW-14": "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    }
    assert issues["NEW-14"]["status"] == "verified"
    assert issues["NEW-14"]["fixed_in"] == record["verified"]["NEW-14"]
    assert issues["NEW-14"]["verified_by"] == "R-0027"
    assert record["registered"] == ["V-41"]
    assert issues["V-41"]["status"] in {"registered", "accepted"}
    assert issues["V-41"]["source_report"] == (
        "PSSolver_sync/results/R-0027.result.json"
    )
    assert issues["V-41"]["affects_control_paths"] is False
    assert record["s09_complete"] is False


def test_r0027_import_is_a_forward_only_complete_successor() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        live = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == live
        assert resolve_s03_successor(relative, successor["current_sha256"]) == live


def test_r0027_import_authorizes_no_external_action() -> None:
    authorization = _read(RECORD)["authorization"]
    assert authorization["local_evidence_import"] is True
    assert authorization["local_diagnostic_support"] is True
    assert authorization["push"] is False
    assert authorization["hpcc_write"] is False
    assert authorization["slurm_or_h100"] is False
    assert authorization["tag_or_release"] is False
