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
    assert len(entries) == record["issue_count"] == 34
    assert len({entry["id"] for entry in entries}) == 34
    assert all(required <= set(entry) <= required | optional for entry in entries)
    counts = Counter(entry["status"] for entry in entries)
    counts.setdefault("registered", 0)
    counts.setdefault("claimed_fixed", 0)
    assert dict(counts) == record["status_counts"] == {
        "open": 1,
        "registered": 0,
        "claimed_fixed": 6,
        "verified": 21,
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
    verified = set(expected)
    for issue_id, affects_control in expected.items():
        item = issues[issue_id]
        expected_status = (
            "verified"
            if issue_id in verified
            else "registered"
        )
        assert item["status"] == expected_status
        if issue_id in verified:
            expected_fix = {
                "C1-05": "bbb7077fa0f0cc7206a0939f6bff34ddc1ff823a",
                "C1-10": "a654e1c1d97fa364ac32b6ccbf93070056066d21",
                "C1-11": "a654e1c1d97fa364ac32b6ccbf93070056066d21",
                "C2-17": "468e06710e1b4881a557f0228dcb16be38791094",
                "C2-18": "bbb7077fa0f0cc7206a0939f6bff34ddc1ff823a",
            }[issue_id]
            expected_review = {
                "C1-05": "R-0016",
                "C1-10": "R-0008",
                "C1-11": "R-0008",
                "C2-17": "R-0012",
                "C2-18": "R-0016",
            }[issue_id]
            assert item["fixed_in"] == expected_fix
            assert item["verified_by"] == expected_review
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
    assert issues["V-22"]["status"] == "verified"
    assert issues["V-22"]["verified_by"] == "R-0012"
    assert issues["V-23"]["status"] == "verified"
    assert issues["V-23"]["fixed_in"] == (
        "a654e1c1d97fa364ac32b6ccbf93070056066d21"
    )
    assert issues["V-23"]["verified_by"] == "R-0008"
    assert issues["V-22"]["target_release"] == "0.3"
    assert issues["V-23"]["target_release"] == "0.3"
    assert "retarget" in issues["V-22"]["notes"]
    assert "PermissionError" in issues["V-23"]["notes"]


def test_s04_checkpoint_robustness_items_are_verified_by_r0012():
    issues = _issues()
    implementation = "468e06710e1b4881a557f0228dcb16be38791094"
    for issue_id in ("V-22", "V-16", "V-13", "V-14", "C2-17"):
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == implementation
        assert issues[issue_id]["verified_by"] == "R-0012"


def test_v25_rework_is_verified_without_changing_its_fix_identity():
    issues = _issues()
    v25 = issues["V-25"]
    assert v25["status"] == "verified"
    assert v25["fixed_in"] == "44a6fc3d88d94ccf939e78e23be3ffbac542def5"
    assert v25["verified_by"] == "R-0012"
    assert v25["severity"] == "medium-low"
    assert v25["source_report"] == "PSSolver_sync/results/R-0010.result.json"
    assert v25["affects_control_paths"] is True
    assert v25["exercised_by_current_control_research"] is False
    assert v25["target_release"] == "0.3"
    assert "ENOSPC" in v25["notes"]
    assert "EDQUOT" in v25["notes"]


def test_rc5_rc6_residuals_track_verified_s06_fixes():
    issues = _issues()
    tracked = {
        "A2-24/C1-15",
        "C1-08",
        "S1-15/NEW-13",
        "S2-17",
        "A1-07",
        "B30",
        "S1-04",
    }
    verified = {"A2-24/C1-15", "C1-08"}
    s06_verified = {"S2-17"}
    s07_claimed = {"S1-15/NEW-13", "A1-07", "B30", "S1-04"}
    assert all(
        issues[issue_id]["status"]
        == (
            "verified"
            if issue_id in verified | s06_verified
            else "claimed_fixed"
        )
        for issue_id in tracked
    )
    assert all(
        issues[issue_id]["fixed_in"]
        == "a10785853736b2458eec80680b9ced72a495775f"
        for issue_id in verified
    )
    assert all(issues[issue_id]["verified_by"] == "R-0009" for issue_id in verified)
    assert issues["S2-17"]["fixed_in"] == (
        "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
    )
    assert issues["S2-17"]["verified_by"] == "R-0016"
    assert issues["V-24"]["status"] == "verified"
    assert issues["V-24"]["fixed_in"] == (
        "a10785853736b2458eec80680b9ced72a495775f"
    )
    assert issues["V-24"]["verified_by"] == "R-0009"
    assert issues["A2-21/S2-12"]["status"] == "verified"
    assert issues["A2-21/S2-12"]["fixed_in"] == (
        "65ef1080cb0215131efe90143863de7014fe4b5b"
    )
    assert issues["A2-21/S2-12"]["verified_by"] == "R-0013"
    assert "not a MAC or signature" in issues["A2-21/S2-12"]["notes"]
    assert issues["A2-11"]["status"] == "accepted"
    assert issues["NEW-14"]["status"] == "open"
    assert issues["NEW-14"]["affects_control_paths"] is True
    assert issues["NEW-14"]["target_release"] == "0.3 GPU verification"
    assert all(
        issues[issue_id]["exercised_by_current_control_research"] is False
        for issue_id in tracked | {"A2-11", "NEW-14"}
    )
    for issue_id in s07_claimed:
        assert issues[issue_id]["fixed_in"] == (
            "b7cf4ca02095e8b41ed720809880628d2375cfbb"
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
    claimed = {
        issue_id
        for issue_id, item in issues.items()
        if item["status"] == "claimed_fixed"
    }
    assert claimed == {
        "A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30", "V-26"
    }
    assert all(issues[issue_id]["target_release"] == "0.3" for issue_id in claimed)
    assert all(
        issues[issue_id]["fixed_in"]
        == "b7cf4ca02095e8b41ed720809880628d2375cfbb"
        for issue_id in claimed
    )
    assert issues["A2-21/S2-12"]["target_release"] == "0.3"
    assert issues["V-26"]["target_release"] == "0.3"
    assert "S-07" in issues["V-26"]["notes"]
    assert issues["V-27"]["status"] == "verified"
    assert issues["V-27"]["fixed_in"] == (
        "bbb7077fa0f0cc7206a0939f6bff34ddc1ff823a"
    )
    assert issues["V-27"]["target_release"] == "0.3"
    assert issues["V-27"]["verified_by"] == "R-0016"
    assert "R-0016" in issues["V-27"]["notes"]
    assert issues["V-28"]["status"] == "verified"
    assert issues["V-28"]["fixed_in"] == (
        "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
    )
    assert issues["V-28"]["severity"] == "medium"
    assert issues["V-28"]["source_report"] == (
        "PSSolver_sync/results/R-0014.result.json"
    )
    assert issues["V-28"]["target_release"] == "0.3"
    assert issues["V-28"]["affects_control_paths"] is True
    assert issues["V-28"]["exercised_by_current_control_research"] is False
    assert "recurrence-minimum" in issues["V-28"]["notes"]
    assert issues["V-28"]["verified_by"] == "R-0016"
    assert issues["S2-17"]["fixed_in"] == (
        "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
    )
    assert issues["V-29"]["status"] == "verified"
    assert issues["V-29"]["fixed_in"] == (
        "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
    )
    assert issues["V-29"]["severity"] == "low"
    assert issues["V-29"]["source_report"] == (
        "PSSolver_sync/results/R-0015.result.json"
    )
    assert issues["V-29"]["target_release"] == "0.3"
    assert issues["V-29"]["affects_v0_2_0"] is False
    assert issues["V-29"]["affects_v0_2_1"] is False
    assert issues["V-29"]["affects_control_paths"] is True
    assert issues["V-29"]["exercised_by_current_control_research"] is False
    assert "R-0016" in issues["V-29"]["notes"]
    assert issues["V-29"]["verified_by"] == "R-0016"
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
