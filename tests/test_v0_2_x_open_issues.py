"""Contract for the forward-maintained v0.2.x issue registry."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "pssolver_v0_2_x_open_issues.json"
V020 = ROOT / "notes" / "pssolver_v0_2_0.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _issues() -> dict[str, dict[str, object]]:
    return {item["id"]: item for item in _read(RECORD)["issues"]}


def test_registry_has_unique_complete_entries_and_exact_status_counts():
    record = _read(RECORD)
    entries = record["issues"]
    required = {
        "id",
        "title",
        "severity",
        "source_report",
        "first_found",
        "affects_v0_2_0",
        "affects_v0_2_1",
        "affects_control_paths",
        "exercised_by_current_control_research",
        "status",
        "target_release",
        "notes",
    }
    optional = {"decision", "fixed_in", "verified_by"}
    assert len(entries) == record["issue_count"] == 28
    assert len({entry["id"] for entry in entries}) == 28
    assert all(required <= set(entry) <= required | optional for entry in entries)
    counts = Counter(entry["status"] for entry in entries)
    assert dict(counts) == record["status_counts"] == {
        "open": 1,
        "registered": 14,
        "claimed_fixed": 6,
        "verified": 1,
        "deferred": 2,
        "accepted": 1,
        "coverage_gap": 3,
    }


def test_registry_is_a_successor_not_a_rewrite_of_v020_tracking():
    record = _read(RECORD)
    source = record["historical_sources"]["v0_2_0_release_record"]
    assert source["rewritten"] is False
    assert _sha256(V020) == source["sha256"] == (
        "1bdbc695b6a2f57ca1fd68061f884ac7f2cfd66eaf8fb708b65efc1db74d0ac6"
    )
    scope = record["record_scope"]
    assert scope["evidence_only"] is True
    assert scope["historical_records_rewritten"] is False
    assert scope["runtime_changed"] is False
    assert scope["tag_or_release_modified"] is False
    assert scope["living_reference_hash_bound"] is False


def test_five_v020_deferred_items_are_migrated_with_control_classification():
    issues = _issues()
    expected = {
        "C1-05": True,
        "C1-10": True,
        "C1-11": True,
        "C2-17": False,
        "C2-18": True,
    }
    claimed = {"C1-10", "C1-11"}
    for issue_id, affects_control in expected.items():
        item = issues[issue_id]
        assert item["status"] == (
            "claimed_fixed" if issue_id in claimed else "registered"
        )
        if issue_id in claimed:
            assert item["fixed_in"] == (
                "a654e1c1d97fa364ac32b6ccbf93070056066d21"
            )
        assert item["affects_v0_2_0"] is True
        assert item["affects_v0_2_1"] is True
        assert item["affects_control_paths"] is affects_control
        assert item["exercised_by_current_control_research"] is False


def test_v021_limitations_preserve_version_and_control_boundaries():
    issues = _issues()
    assert issues["V-17"]["status"] == "verified"
    assert issues["V-17"]["fixed_in"] == (
        "611d167d1b2c2d634222f608f4c9cd4dc671636d"
    )
    assert issues["V-17"]["verified_by"] == "R-0002"
    assert issues["V-17"]["target_release"] == "0.2.x priority"
    for issue_id in ("V-17", "V-13", "V-14", "OBS-4"):
        assert issues[issue_id]["affects_v0_2_0"] is False
        assert issues[issue_id]["affects_v0_2_1"] is True
    for issue_id in ("V-16", "V-21", "V-22", "V-23"):
        assert issues[issue_id]["affects_v0_2_0"] is True
        assert issues[issue_id]["affects_v0_2_1"] is True
    for issue_id in (
        "V-17",
        "V-16",
        "V-13",
        "V-14",
        "V-21",
        "V-22",
        "V-23",
        "OBS-4",
    ):
        assert issues[issue_id]["affects_control_paths"] is True
        assert issues[issue_id]["exercised_by_current_control_research"] is False
    assert issues["V-22"]["status"] == "registered"
    assert issues["V-23"]["status"] == "claimed_fixed"
    assert issues["V-23"]["fixed_in"] == (
        "a654e1c1d97fa364ac32b6ccbf93070056066d21"
    )
    assert issues["V-22"]["target_release"] == "0.3"
    assert issues["V-23"]["target_release"] == "0.3"
    assert "retarget" in issues["V-22"]["notes"]
    assert "PermissionError" in issues["V-23"]["notes"]


def test_rc5_rc6_residuals_are_registered_without_claiming_fixes():
    issues = _issues()
    registered = {
        "A2-21/S2-12",
        "A2-24/C1-15",
        "C1-08",
        "S1-15/NEW-13",
        "S2-17",
        "A1-07",
        "B30",
        "S1-04",
    }
    claimed = {"A2-24/C1-15", "C1-08"}
    assert all(
        issues[issue_id]["status"]
        == ("claimed_fixed" if issue_id in claimed else "registered")
        for issue_id in registered
    )
    assert all(
        issues[issue_id]["fixed_in"]
        == "a654e1c1d97fa364ac32b6ccbf93070056066d21"
        for issue_id in claimed
    )
    assert issues["A2-11"]["status"] == "accepted"
    assert issues["NEW-14"]["status"] == "open"
    assert issues["NEW-14"]["affects_control_paths"] is True
    assert issues["NEW-14"]["target_release"] == "0.3 GPU verification"
    assert all(
        issues[issue_id]["exercised_by_current_control_research"] is False
        for issue_id in registered | {"A2-11", "NEW-14"}
    )


def test_remaining_registered_items_are_assigned_to_the_0_3_roadmap():
    record = _read(RECORD)
    issues = _issues()
    assert record["primary_ref"] == "dev/0.3"
    assert record["roadmap"] == {
        "decision": "D-0010",
        "path": "PSSolver_sync/plan/roadmap.json",
        "step": "S-01",
    }
    registered = {
        issue_id for issue_id, item in issues.items() if item["status"] == "registered"
    }
    assert len(registered) == 14
    assert all(issues[issue_id]["target_release"] == "0.3" for issue_id in registered)
    assert issues["A2-21/S2-12"]["target_release"] == "0.3"
    assert "S-07" in issues["B30"]["notes"]
    assert issues["V-17"]["status"] == "verified"
    assert issues["V-17"]["fixed_in"] == (
        "611d167d1b2c2d634222f608f4c9cd4dc671636d"
    )
    assert issues["V-17"]["verified_by"] == "R-0002"
    assert issues["V-17"]["target_release"] == "0.2.x priority"


def test_deferred_and_coverage_items_match_the_frozen_boundary():
    issues = _issues()
    assert {key for key, item in issues.items() if item["status"] == "deferred"} == {
        "B8/S2-07",
        "R1-R9",
    }
    assert {
        key for key, item in issues.items() if item["status"] == "coverage_gap"
    } == {"A2-25", "A3-18", "N5"}
    assert issues["B8/S2-07"]["target_release"] == "before lifting is reopened"
    assert issues["R1-R9"]["target_release"] == "after 0.3"
    assert issues["B8/S2-07"]["decision"] == "D-0007"
    assert issues["R1-R9"]["decision"] == "D-0007"
    assert issues["A2-11"]["decision"] == "D-0007"
    assert issues["A3-18"]["affects_control_paths"] is True


def test_registry_does_not_authorize_implementation_or_external_actions():
    authorization = _read(RECORD)["authorization"]
    assert authorization["local_registration_authorized"] is True
    assert all(
        authorization[key] is False
        for key in (
            "runtime_fix_authorized",
            "push_authorized",
            "main_or_maint_fast_forward_authorized",
            "dev_cherry_pick_authorized",
            "h100_authorized",
        )
    )
