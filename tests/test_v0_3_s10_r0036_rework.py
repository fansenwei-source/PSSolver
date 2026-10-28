"""Evidence-only bindings for the R-0036 S-10 integration rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_59_r0036_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0036.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0036_rework_binds_the_exact_independent_review() -> None:
    record = _read(RECORD)
    assert record["plan_step"] == "S-10"
    assert record["sync_result"] == {
        "id": "R-0036",
        "path": "PSSolver_sync/results/R-0036.result.json",
        "sha256": "d7572772516ba1d6adbe0e34359cb1a8d9778425cdac47dcf75977964288580e",
        "sync_commit": "57c8dce915dc033480e49ceebed3ad09f19a9322",
        "classification": "REWORK_REQUIRED",
    }
    assert _sha256(RESULT) == record["sync_result"]["sha256"]
    archive = record["independent_archive"]
    root = Path(archive["path"])
    assert archive["complete_present"] is True
    assert (root / "COMPLETE").is_file()
    assert sum(1 for _ in (root / "checksums.sha256").open(encoding="utf-8")) == 57
    assert _sha256(root / "checksums.sha256") == archive["checksums_sha256"]
    assert _sha256(root / "final_report.md") == archive["final_report_sha256"]
    assert _sha256(root / "summary.json") == archive["summary_sha256"]


def test_r0036_registry_updates_and_new_finding_are_exact() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    for issue_id in ("V-33", "V-43"):
        expected = record["registry_import"][issue_id]
        assert issues[issue_id]["status"] == expected["status"] == "verified"
        assert issues[issue_id]["verified_by"] == expected["verified_by"] == "R-0033"
        assert issues[issue_id]["fixed_in"] == expected["fixed_in"]
        assert expected["reviewed_original"] in issues[issue_id]["notes"]
    assert issues["V-46"]["status"] == "claimed_fixed"
    assert issues["V-46"]["source_report"] == (
        "PSSolver_sync/results/R-0036.result.json"
    )
    assert issues["V-46"]["target_release"] == "0.3.0 (S-10 integration)"


def test_r0036_rework_is_a_forward_only_complete_successor() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        live = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == live
        assert resolve_s03_successor(relative, successor["current_sha256"]) == live


def test_r0036_rework_preserves_local_only_scope() -> None:
    record = _read(RECORD)
    assert record["replay_identity_correction"]["expected_mapping_entries"] == 10
    assert record["replay_identity_correction"]["archive_ref_must_not_be_pushed"] is True
    assert record["replay_identity_correction"][
        "fresh_clone_gate_required_after_replay_rebase_or_amend"
    ] is True
    assert record["scope"]["records_and_tests_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "records_and_tests_only"
    )
    assert record["authorization"]["local_rework"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_rework"
    )
    assert record["next_request"] == "R-0037"
