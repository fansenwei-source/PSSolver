"""Evidence bindings for importing the passing R-0016 S-06 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_17_r0016_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0016_import_binds_the_passing_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == "PASS_R0016_S06_FINAL_REVIEW_IMPORTED"
    assert record["git_identity"]["reviewed_candidate"] == (
        "0c2889f092ea25f9aa2b6f67ebc6c82cd51a76a9"
    )
    assert record["git_identity"]["qualified_runtime_source"] == (
        "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
    )
    assert record["sync_result"] == {
        "id": "R-0016",
        "path": "/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0016.result.json",
        "sha256": "9c09af359166bf6f688a138c5ae2e0efbaf30b97fe61b53964bcf92502b888ba",
        "classification": "PASS",
        "new_findings": [],
    }
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 44
    assert archive["checksums_sha256"] == (
        "be701ad8e16e6405ebaab5b276fd51e09fb918de26a568c0217cf573d02e6db8"
    )
    assert archive["complete_present"] is True


def test_r0016_verifies_all_s06_items_at_exact_fix_sources() -> None:
    record = _read(RECORD)["registry_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    expected = {
        **{issue_id: "bbb7077fa0f0cc7206a0939f6bff34ddc1ff823a"
           for issue_id in record["fixed_at_bbb7077"]},
        **{issue_id: "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
           for issue_id in record["fixed_at_f6da20f"]},
    }
    assert set(expected) == {"C1-05", "C2-18", "V-27", "S2-17", "V-28", "V-29"}
    for issue_id, fixed_in in expected.items():
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert issues[issue_id]["verified_by"] == "R-0016"
    assert record["s06_complete"] is True


def test_r0016_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0016_import_is_evidence_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    validation = record["validation"]
    assert validation["focused_pytest"]["passed"] == 71
    assert validation["environment_clone_pruned"] is True
    assert validation["archive_checksums_entries"] == 5
    assert validation["archive_checksums_sha256"] == (
        "54338166862fcef36068f502eabe467922ed4cc39fca911ebf87e97f656fc862"
    )
    assert validation["archive_complete_present"] is True
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
    assert record["next_plan_step"] == "S-07"
    assert record["next_request"] == "R-0017"
