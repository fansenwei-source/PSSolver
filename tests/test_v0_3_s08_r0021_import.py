"""Bindings for the R-0021 S-08 final review import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_27_r0021_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0021.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0021_result_and_independent_archive_are_exact() -> None:
    record = _read(RECORD)
    result = _read(RESULT)
    assert record["classification"] == "PASS_R0021_S08_FINAL_REVIEW_IMPORTED"
    assert record["sync_result"] == {
        "id": "R-0021",
        "path": str(RESULT),
        "sha256": "5f46e5991639f9c014ccff3cf8c43dabad8dcf59b426ec68dbd2b0add30c98f5",
        "classification": "PASS",
        "new_findings": [],
    }
    assert result["classification"] == "PASS"
    assert result["target"]["commit"] == (
        "dea728f1a926284420e6b1a17edb56eb56ea3db1"
    )
    assert record["independent_archive"] == {
        "path": "/home/fansenwei/pssolver_s08rec_claude_review_20261023_v1",
        "checksums_entries": 13,
        "checksums_sha256": (
            "7543c806fc4e898ae37eb1a5d91fa8b9753c038ecaf79820950357723dc0ba2c"
        ),
        "complete_present": True,
        "final_report_sha256": (
            "d26514de94e66111b11b12e401cda99cce89374e6fcbded520a5470ff18b0db0"
        ),
        "summary_sha256": (
            "3c5b8ee7300627bca3ec2958ddbb97c9c09846b87ffca874bc0263d97d2c535c"
        ),
    }


def test_r0021_registry_updates_are_exact_and_v33_stays_registered() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    expected = {
        "A2-25": "8bf898f5b10056c0b70e1d3acbeaad2c0d8c3851",
        "N5": "8bf898f5b10056c0b70e1d3acbeaad2c0d8c3851",
        "V-31": "e2343d19301de146ca632a1a12f7c9659e598f73",
        "V-32": "e2343d19301de146ca632a1a12f7c9659e598f73",
        "V-34": "dea728f1a926284420e6b1a17edb56eb56ea3db1",
    }
    assert record["registry_import"]["verified"] == expected
    for issue_id, fixed_in in expected.items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id]["verified_by"] == "R-0021"
    assert issues["V-33"]["status"] == "registered"
    assert "fixed_in" not in issues["V-33"]
    assert "verified_by" not in issues["V-33"]
    assert record["registry_import"]["kept_registered"] == ["V-33"]
    assert {
        item["id"] for item in issues.values() if item["status"] == "claimed_fixed"
    } == {"V-39", "V-40"}


def test_r0021_import_is_a_forward_only_complete_successor() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0021_import_is_evidence_only_and_authorizes_no_external_action() -> None:
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
    assert record["next_plan_step"] == "S-09"
    assert record["next_request"] == "R-0022"
