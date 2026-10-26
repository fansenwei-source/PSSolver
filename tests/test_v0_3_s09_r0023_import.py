"""Bindings for the evidence-only R-0023 S-09 import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_35_r0023_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0023.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0023_import_binds_the_exact_review_result() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "REWORK_REQUIRED_R0023_S09_FINAL_PRE_H100_REVIEW_IMPORTED"
    )
    assert record["git_identity"]["reviewed_candidate"] == (
        "ee1f9e0107373fdee92b357a9ffc0518bab8435c"
    )
    assert record["git_identity"]["result_commit"] == "30ad331"
    assert _sha256(RESULT) == record["sync_result"]["sha256"] == (
        "4570f85423eb0c6008eeaa141889afcb78e751f9cad003f887c222ae121ca3fc"
    )
    assert record["sync_result"]["new_findings"] == ["V-37"]


def test_r0023_import_advances_only_the_reviewed_registry_boundary() -> None:
    record = _read(RECORD)["registry_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["claimed_fixed"] == {
        "V-35": "85ba3aef75548d549d9db453ca4a5ab604b17a9d",
        "V-36": "85ba3aef75548d549d9db453ca4a5ab604b17a9d",
    }
    assert record["registered"] == ["V-37"]
    for issue_id, fixed_in in record["claimed_fixed"].items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id]["verified_by"] == "R-0024"
    assert issues["V-37"]["status"] == "verified"
    assert issues["V-37"]["fixed_in"] == (
        "87cf34cf4e17096a701f5cc5bc81076d14a9d237"
    )
    assert issues["V-37"]["verified_by"] == "R-0024"
    assert issues["A3-18"]["status"] == "verified"
    assert issues["A3-18"]["verified_by"] == "R-0025"
    assert issues["NEW-14"]["status"] == "verified"
    assert issues["NEW-14"]["fixed_in"] == "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    assert issues["NEW-14"]["verified_by"] == "R-0027"


def test_r0023_import_is_evidence_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    assert not any(
        value
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert record["authorization"]["local_import_authorized"] is True
    assert not any(
        value
        for key, value in record["authorization"].items()
        if key != "local_import_authorized"
    )
    assert record["next_request"] == "R-0024"
