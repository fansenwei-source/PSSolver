"""Bindings for the evidence-only R-0026 S-09 import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_41_r0026_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0026.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0026_result_and_archive_are_bound() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_R0026_S09_RECOVERY_REVIEW_IMPORTED"
    assert _sha256(RESULT) == record["sync_result"]["sha256"]
    assert record["sync_result"]["classification"] == "PASS"
    archive = record["independent_archive"]
    root = Path(archive["path"])
    assert archive["complete_present"] is True
    assert (root / "COMPLETE").is_file()
    assert sum(1 for _ in (root / "checksums.sha256").open(encoding="utf-8")) == 7658
    assert _sha256(root / "checksums.sha256") == archive["checksums_sha256"]


def test_r0026_registry_updates_are_exact() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    imported = _read(RECORD)["registry_import"]
    for issue_id, fixed_in in imported["verified"].items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id]["verified_by"] == "R-0026"
    assert issues["NEW-14"]["status"] == "verified"
    assert issues["NEW-14"]["fixed_in"] == "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    assert issues["NEW-14"]["verified_by"] == "R-0027"
    assert imported["s09_complete"] is False
    assert imported["recovery_h100_submission_authorized"] is False


def test_r0026_h100_citations_are_exact() -> None:
    expected = _read(RESULT)["scope_observed"]["citable_for_d0009_h100"]
    assert _read(RECORD)["h100_authorization_inputs"] == expected


def test_r0026_import_is_a_forward_only_complete_successor() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        live = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == live
        assert resolve_s03_successor(relative, successor["current_sha256"]) == live


def test_r0026_import_authorizes_no_external_action() -> None:
    authorization = _read(RECORD)["authorization"]
    assert authorization["local_evidence_import"] is True
    assert all(value is False for key, value in authorization.items() if key != "local_evidence_import")
