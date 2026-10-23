"""Bindings for the S-01 migration of the issue registry to dev/0.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s01_registry_migration.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _issues() -> dict[str, dict[str, object]]:
    return {item["id"]: item for item in _read(REGISTRY)["issues"]}


def test_s01_record_binds_the_registry_successor_and_roadmap() -> None:
    record = _read(RECORD)
    registry = _read(REGISTRY)
    assert record["classification"] == "S01_REGISTRY_MIGRATED_TO_DEV_0_3_CANDIDATE"
    assert record["plan_step"] == "S-01"
    assert record["decision"] == "D-0010"
    assert record["git_identity"]["parent_commit"] == (
        "7301bd4bf0b49b27e1c07d248ea3ca3f483dc147"
    )
    assert _sha256(REGISTRY) == resolve_s03_successor(
        "notes/pssolver_v0_2_x_open_issues.json",
        record["registry"]["successor_sha256"],
    )
    assert registry["primary_ref"] == "dev/0.3"
    assert registry["roadmap"] == {
        "decision": "D-0010",
        "path": "PSSolver_sync/plan/roadmap.json",
        "step": "S-01",
    }


def test_s01_assigns_every_registered_item_to_0_3_exactly() -> None:
    record = _read(RECORD)["registry"]
    issues = _issues()
    tracked = set(record["registered_ids"])
    assert {issue_id for issue_id in tracked if issues[issue_id]["status"] == "verified"} == {
        "C1-10",
        "C1-11",
        "C2-17",
        "V-22",
        "V-23",
        "V-16",
        "V-13",
        "V-14",
        "V-21",
        "A2-24/C1-15",
        "C1-08",
        "A2-21/S2-12",
        "C1-05",
        "C2-18",
        "S2-17",
    }
    assert {
        issue_id
        for issue_id in tracked
        if issues[issue_id]["status"] == "claimed_fixed"
    } == {"A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30"}
    assert all(issues[issue_id]["target_release"] == "0.3" for issue_id in tracked)
    assert set(record["changed_target_ids"]) | set(
        record["already_targeted_to_0_3"]
    ) == tracked
    assert set(record["changed_target_ids"]).isdisjoint(
        record["already_targeted_to_0_3"]
    )
    assert issues["V-24"]["status"] == "verified"
    assert issues["V-24"]["verified_by"] == "R-0009"
    assert issues["V-24"]["target_release"] == "0.3"


def test_s01_preserves_v17_and_schedules_new14_and_b30() -> None:
    record = _read(RECORD)["registry"]
    issues = _issues()
    assert issues["V-17"]["status"] == record["v17_preserved"]["status"]
    assert issues["V-17"]["fixed_in"] == record["v17_preserved"]["fixed_in"]
    assert issues["V-17"]["verified_by"] == record["v17_preserved"]["verified_by"]
    assert issues["V-17"]["target_release"] == record["v17_preserved"][
        "target_release"
    ]
    assert issues["NEW-14"]["target_release"] == record["new_14_target"] == (
        "0.3 GPU verification"
    )
    assert record["b30_plan_step"] == "S-07"
    assert issues["B30"]["target_release"] == "0.3"
    assert "S-07" in issues["B30"]["notes"]


def test_s01_preserves_historical_registry_hashes_without_rewriting_records() -> None:
    record = _read(RECORD)
    assert record["registry"]["historical_sha256"] == (
        "ffdd4fe41ddc140857081febf634b28bed4e1d3345f9e434982ef11ec2f67cc2"
    )
    contracts = record["historical_import_contracts"]
    assert contracts["records_rewritten"] is False
    assert contracts["historical_registry_sha256"] == record["registry"][
        "historical_sha256"
    ]
    assert set(contracts["tests_now_bind_the_recorded_historical_hash_instead_of_the_live_successor"]) == {
        "tests/test_v0_2_2_r0003_import.py",
        "tests/test_v0_2_2_r0004_import.py",
        "tests/test_v0_3_n1_v17_r0005_import.py",
    }


def test_s01_historical_successors_bind_the_parent_and_current_files() -> None:
    record = _read(RECORD)
    parent = record["git_identity"]["parent_commit"]
    for relative, successor in record["historical_successor_sources"].items():
        historical = subprocess.run(
            ["git", "show", f"{parent}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == successor[
            "historical_sha256"
        ]
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, successor["current_sha256"]
        )
        assert successor["reason"]


def test_s01_does_not_modify_runtime_or_authorize_external_actions() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert record["authorization"]["local_registry_update_authorized"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_registry_update_authorized"
    )
    assert record["git_identity"]["record_commit_is_runtime_qualification_source"] is False
    assert record["git_identity"]["pushed"] is False
