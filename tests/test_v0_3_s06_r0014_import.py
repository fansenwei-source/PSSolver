"""Evidence bindings for importing the R-0014 S-06 rework result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_13_r0014_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0014_import_binds_rework_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "REWORK_REQUIRED_R0014_S06_REVIEW_IMPORTED"
    assert record["git_identity"]["result_commit"] == (
        "8d08f640ce6b85663eec2db3d945fa4916f5e4b0"
    )
    assert record["sync_result"]["sha256"] == (
        "19166e4eb6addae653598e953a3da544e040f11a14ccebdcc525d8231f6493e6"
    )
    assert record["sync_result"]["classification"] == "REWORK_REQUIRED"
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 56
    assert archive["checksums_sha256"] == (
        "2a8d810290e7fa4f371980dffc810847599b550b65fcbaa8e4aee40f190c2006"
    )
    assert archive["complete_present"] is True


def test_r0014_registers_v28_without_verifying_s06_claims() -> None:
    record = _read(RECORD)
    imported = record["registry_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert imported["new_finding"] == {
        "id": "V-28",
        "severity": "medium",
        "status": "registered",
        "target_release": "0.3",
        "plan_step": "S-06",
    }
    v28 = issues["V-28"]
    assert v28["severity"] == "medium"
    assert v28["status"] == "verified"
    assert v28["fixed_in"] == "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
    assert v28["target_release"] == "0.3"
    assert "recurrence-minimum" in v28["notes"]
    assert v28["verified_by"] == "R-0016"
    assert imported["verified_by_r0014"] == []
    assert imported["s06_complete"] is False
    for issue_id in imported["kept_claimed_fixed"]:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0016"


def test_r0014_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0014_import_is_evidence_only_and_authorizes_no_external_action() -> None:
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
    assert record["next_request"] == "R-0015"
