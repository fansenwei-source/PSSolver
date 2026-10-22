"""Evidence bindings for importing the R-0015 S-06 rework result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_15_r0015_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0015_import_binds_rework_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "REWORK_REQUIRED_R0015_S06_REVIEW_IMPORTED"
    assert record["git_identity"]["result_commit"] == (
        "d2f7850af12a3119a5b2e83253c29e4852e9d7db"
    )
    assert record["sync_result"]["sha256"] == (
        "ce54a688fc5ffd0427544be1b404379db84a6a0049f873f5a39d1624c6a0945a"
    )
    assert record["sync_result"]["classification"] == "REWORK_REQUIRED"
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 41
    assert archive["checksums_sha256"] == (
        "de58e0900b4169f77f65a83c510e290a222fe3d8afeb22a922fa78dfd8cb9d8b"
    )
    assert archive["complete_present"] is True


def test_r0015_registers_v29_without_verifying_s06_claims() -> None:
    record = _read(RECORD)
    imported = record["registry_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert imported["new_finding"] == {
        "id": "V-29",
        "severity": "low",
        "status": "registered",
        "target_release": "0.3",
        "plan_step": "S-06",
    }
    v29 = issues["V-29"]
    assert v29["status"] == "verified"
    assert v29["fixed_in"] == (
        "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
    )
    assert v29["severity"] == "low"
    assert v29["target_release"] == "0.3"
    assert v29["verified_by"] == "R-0016"
    assert imported["verified_by_r0015"] == []
    assert set(imported["passed_claims"]) == {"C1-05", "C2-18", "V-27", "V-28"}
    assert imported["s06_complete"] is False
    for issue_id in imported["kept_claimed_fixed"]:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0016"


def test_r0015_rework_contract_is_exact() -> None:
    contract = _read(RECORD)["rework_contract"]
    assert contract == {
        "rollback_candidate": "minimum_recurrence_residual_at_or_below_floor",
        "actual_rollback_requires_strict_represented_residual_postcondition": True,
        "normal_completion_preserves_pre_s06_acceptance": True,
        "scan_fail_closed_count_must_be_reported": True,
    }


def test_r0015_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0015_import_is_evidence_only_and_authorizes_no_external_action() -> None:
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
    assert record["next_request"] == "R-0016"
