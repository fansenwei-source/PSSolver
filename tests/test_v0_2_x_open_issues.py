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
    assert len(entries) == record["issue_count"] == 51
    assert len({entry["id"] for entry in entries}) == 51
    assert all(required <= set(entry) <= required | optional for entry in entries)
    counts = Counter(entry["status"] for entry in entries)
    counts.setdefault("open", 0)
    counts.setdefault("registered", 0)
    counts.setdefault("claimed_fixed", 0)
    counts.setdefault("coverage_gap", 0)
    assert dict(counts) == record["status_counts"] == {
        "open": 0,
        "registered": 0,
        "claimed_fixed": 1,
        "verified": 46,
        "deferred": 2,
        "accepted": 2,
        "coverage_gap": 0,
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
    s07_verified = {"S1-15/NEW-13", "A1-07", "B30", "S1-04"}
    assert all(
        issues[issue_id]["status"]
        == (
            "verified"
            if issue_id in verified | s06_verified | s07_verified
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
    assert issues["NEW-14"]["status"] == "verified"
    assert issues["NEW-14"]["fixed_in"] == (
        "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    )
    assert issues["NEW-14"]["verified_by"] == "R-0027"
    assert issues["NEW-14"]["affects_control_paths"] is True
    assert issues["NEW-14"]["target_release"] == "0.3 GPU verification"
    assert all(
        issues[issue_id]["exercised_by_current_control_research"] is False
        for issue_id in tracked | {"A2-11", "NEW-14"}
    )
    for issue_id in s07_verified:
        assert issues[issue_id]["fixed_in"] == (
            "b7cf4ca02095e8b41ed720809880628d2375cfbb"
        )
        assert issues[issue_id]["verified_by"] == "R-0018"


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
    assert claimed == {"V-44"}
    s07_verified = {
        "A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30", "V-26", "V-30"
    }
    assert all(issues[issue_id]["target_release"] == "0.3" for issue_id in s07_verified)
    assert all(
        issues[issue_id]["fixed_in"]
        == "b7cf4ca02095e8b41ed720809880628d2375cfbb"
        for issue_id in s07_verified - {"V-30"}
    )
    assert all(issues[issue_id]["verified_by"] == "R-0018" for issue_id in s07_verified)
    assert issues["A2-21/S2-12"]["target_release"] == "0.3"
    assert issues["V-26"]["target_release"] == "0.3"
    assert "S-07" in issues["V-26"]["notes"]
    for issue_id in ("V-35", "V-36"):
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == (
            "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
        )
        assert issues[issue_id]["target_release"] == "0.3"
        assert issues[issue_id]["source_report"] == (
            "PSSolver_sync/results/R-0022.result.json"
        )
        assert issues[issue_id]["affects_control_paths"] is False
        assert issues[issue_id]["exercised_by_current_control_research"] is False
        assert issues[issue_id]["verified_by"] == "R-0024"
    assert issues["V-37"]["status"] == "verified"
    assert issues["V-37"]["fixed_in"] == (
        "87cf34cf4e17096a701f5cc5bc81076d14a9d237"
    )
    assert issues["V-37"]["severity"] == "low"
    assert issues["V-37"]["source_report"] == (
        "PSSolver_sync/results/R-0023.result.json"
    )
    assert issues["V-37"]["target_release"] == "0.3"
    assert issues["V-37"]["verified_by"] == "R-0024"
    assert "R-0024" in issues["V-37"]["notes"]
    assert issues["V-38"] == {
        "id": "V-38",
        "title": (
            "Historical import tests pin live registry counts and fail after "
            "forward S-09 status changes"
        ),
        "severity": "low",
        "source_report": "PSSolver_sync/results/R-0024.result.json",
        "first_found": "S-09 final pre-H100 independent review R-0024",
        "affects_v0_2_0": False,
        "affects_v0_2_1": False,
        "affects_control_paths": False,
        "exercised_by_current_control_research": False,
        "status": "verified",
        "fixed_in": "07b0ea69af08f20f2183262b39fde8a356cff029",
        "verified_by": "R-0025",
        "target_release": "0.3",
        "notes": issues["V-38"]["notes"],
    }
    assert "immutable snapshots" in issues["V-38"]["notes"]
    for issue_id in ("V-39", "V-40"):
        issue = issues[issue_id]
        assert issue["status"] == "verified"
        assert issue["fixed_in"] == (
            "72286694c43ee15e5859c0658c989431404fe505"
        )
        assert issue["verified_by"] == "R-0026"
        assert issue["source_report"] == "PSSolver_sync/results/R-0025.result.json"
        assert issue["target_release"] == "0.3.0 (S-09 recovery)"
        assert issue["affects_control_paths"] is False
        assert issue["exercised_by_current_control_research"] is False
    assert issues["V-41"] == {
        "id": "V-41",
        "title": (
            "Channel production warm-start A.p cache retains one pressure field "
            "and exceeds the frozen peak-reserved gate at R512"
        ),
        "severity": "medium-low",
        "source_report": "PSSolver_sync/results/R-0027.result.json",
        "first_found": "S-09 recovery_v2 independent review R-0027",
        "affects_v0_2_0": False,
        "affects_v0_2_1": False,
        "affects_control_paths": False,
        "exercised_by_current_control_research": False,
        "status": "accepted",
        "decision": "D-0013",
        "target_release": "0.3.0 (S-09 bounded memory exception)",
        "notes": issues["V-41"]["notes"],
    }
    assert "1.0307692307692307" in issues["V-41"]["notes"]
    assert "562036736 / 545259520" in issues["V-41"]["notes"]
    assert "1.0150638559607594" in issues["V-41"]["notes"]
    assert "D-0013" in issues["V-41"]["notes"]
    assert "compute_nonlinear" in issues["V-41"]["notes"]
    assert "No other family, grid, dtype, runtime or future version" in issues["V-41"]["notes"]
    assert issues["V-42"] == {
        "id": "V-42",
        "title": (
            "Historical records tests pin the pre-V-41 registry or reject "
            "forward successors"
        ),
        "severity": "low",
        "source_report": "PSSolver_sync/results/R-0030.result.json",
        "first_found": "S-09 diagnostic environment rework review R-0030",
        "affects_v0_2_0": False,
        "affects_v0_2_1": False,
        "affects_control_paths": False,
        "exercised_by_current_control_research": False,
        "status": "verified",
        "fixed_in": "f3620b87581de5f29356ed1da89673cf08ffc8a7",
        "verified_by": "R-0031",
        "target_release": "0.3.0 (S-09, R-0030 import)",
        "notes": issues["V-42"]["notes"],
    }
    assert "immutable snapshot" in issues["V-42"]["notes"]
    assert "skip" in issues["V-42"]["notes"]
    assert issues["V-44"] == {
        "id": "V-44",
        "title": (
            "The sealed S-09 diagnostic summary records a nonexistent manually "
            "completed support commit"
        ),
        "severity": "low",
        "source_report": "PSSolver_sync/results/R-0031.result.json",
        "first_found": "S-09 A.p cache diagnostic independent review R-0031",
        "affects_v0_2_0": False,
        "affects_v0_2_1": False,
        "affects_control_paths": False,
        "exercised_by_current_control_research": False,
        "status": "claimed_fixed",
        "fixed_in": "4baa26c8c67d5e01c7615fa1dae29679f450645b",
        "target_release": "0.3.0 (S-09 evidence erratum)",
        "notes": issues["V-44"]["notes"],
    }
    assert "616bc453a38b8f1459b1c278b76596f9e83e8687" in issues["V-44"]["notes"]
    assert "616bc453d50040b20d1ef448cf2647358aeee1cc" in issues["V-44"]["notes"]
    assert "R-0038" in issues["V-44"]["notes"]
    assert issues["V-45"] == {
        "id": "V-45",
        "title": (
            "S-09 closure noninterrupting recompute was bound to the runtime-tree "
            "plan instead of the job's frozen support plan"
        ),
        "severity": "low",
        "source_report": "PSSolver_sync/results/R-0034.result.json",
        "first_found": "S-09 D-0013 closure independent review R-0034",
        "affects_v0_2_0": False,
        "affects_v0_2_1": False,
        "affects_control_paths": False,
        "exercised_by_current_control_research": False,
        "status": "verified",
        "fixed_in": "e052f5b87e1424fbd8d9bd097a893469f9c72658",
        "verified_by": "R-0035",
        "target_release": "0.3.0 (S-09 closure rework)",
        "notes": issues["V-45"]["notes"],
    }
    assert "210 requirements" in issues["V-45"]["notes"]
    assert "a21b211b" in issues["V-45"]["notes"]
    assert "R-0035" in issues["V-45"]["notes"]
    assert "1757374241e0" in issues["V-45"]["notes"]
    assert issues["V-46"] == {
        "id": "V-46",
        "title": (
            "S-10 replay leaves registry, successor records and integration tests "
            "bound to original commits that are unreachable from the integrated branch"
        ),
        "severity": "low",
        "source_report": "PSSolver_sync/results/R-0036.result.json",
        "first_found": "S-09/S-10 integration independent review R-0036",
        "affects_v0_2_0": False,
        "affects_v0_2_1": False,
        "affects_control_paths": False,
        "exercised_by_current_control_research": False,
        "status": "verified",
        "fixed_in": "5254ff32af5d68471f90f5fa8055d040598090f0",
        "verified_by": "R-0037",
        "target_release": "0.3.0 (S-10 integration)",
        "notes": issues["V-46"]["notes"],
    }
    assert "a54074013e5f0fb9bfe278baf944f805bf86eef4" in issues["V-46"]["notes"]
    assert "refs/archive/s10-pre-integration" in issues["V-46"]["notes"]
    assert issues["V-43"] == {
        "id": "V-43",
        "title": (
            "Schema-2 numerical-build fingerprint requires a NumPy "
            "configuration API newer than the declared minimum"
        ),
        "severity": "low",
        "source_report": "PSSolver_sync/results/R-0032.result.json",
        "first_found": "S-10 independent review R-0032",
        "affects_v0_2_0": False,
        "affects_v0_2_1": False,
        "affects_control_paths": False,
        "exercised_by_current_control_research": False,
        "status": "verified",
        "fixed_in": "a54074013e5f0fb9bfe278baf944f805bf86eef4",
        "verified_by": "R-0033",
        "target_release": "0.3.0 (S-10 rework)",
        "notes": issues["V-43"]["notes"],
    }
    assert "NumPy 1.22.3" in issues["V-43"]["notes"]
    assert "R-0033" in issues["V-43"]["notes"]
    assert issues["V-30"] == {
        "id": "V-30",
        "title": (
            "The S-07 wheel cannot build either Channel runtime because retained "
            "pssolver.channel imports the excluded pssolver.control package"
        ),
        "severity": "high",
        "source_report": "PSSolver_sync/results/R-0017.result.json",
        "first_found": (
            "S-07 independent review R-0017; regression introduced by the dev/0.3 "
            "S-07 candidate b7cf4ca"
        ),
        "affects_v0_2_0": False,
        "affects_v0_2_1": False,
        "affects_control_paths": False,
        "exercised_by_current_control_research": False,
        "status": "verified",
        "fixed_in": "8aceaab446e57a13866090ec06ecdf30e33ef490",
        "verified_by": "R-0018",
        "target_release": "0.3",
        "notes": issues["V-30"]["notes"],
    }
    assert "S-07" in issues["V-30"]["notes"]
    assert "installed wheels" in issues["V-30"]["notes"]
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
    } == set()
    assert {
        key for key, item in issues.items() if item["status"] == "claimed_fixed"
    } == {"V-44"}
    for issue_id in ("A2-25", "N5"):
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == (
            "8bf898f5b10056c0b70e1d3acbeaad2c0d8c3851"
        )
        assert issues[issue_id]["verified_by"] == "R-0021"
    assert issues["B8/S2-07"]["target_release"] == "before lifting is reopened"
    assert issues["R1-R9"]["target_release"] == (
        "S-10 partial; remaining work after 0.3"
    )
    assert issues["B8/S2-07"]["decision"] == "D-0007"
    assert issues["R1-R9"]["decision"] == "D-0011"
    assert "R7 partial" in issues["R1-R9"]["notes"]
    assert "R9 partial" in issues["R1-R9"]["notes"]
    assert issues["A2-11"]["decision"] == "D-0007"
    assert issues["A3-18"]["affects_control_paths"] is True
    assert issues["A3-18"]["status"] == "verified"
    assert issues["A3-18"]["fixed_in"] == (
        "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    )
    assert issues["A3-18"]["verified_by"] == "R-0025"


def test_r0021_verifies_s08_claims_and_leaves_v33_for_s10():
    issues = _issues()
    for issue_id in ("V-31", "V-32"):
        item = issues[issue_id]
        assert item["status"] == "verified"
        assert item["fixed_in"] == (
            "e2343d19301de146ca632a1a12f7c9659e598f73"
        )
        assert item["verified_by"] == "R-0021"
        assert item["severity"] == "low"
        assert item["source_report"] == "PSSolver_sync/results/R-0019.result.json"
        assert item["target_release"] == "0.3"
        assert item["affects_v0_2_0"] is False
        assert item["affects_v0_2_1"] is False
        assert item["affects_control_paths"] is False
        assert item["exercised_by_current_control_research"] is False
    assert "fingerprint" in issues["V-31"]["notes"]
    assert "8*eps" in issues["V-31"]["notes"]
    assert "re-migrate" in issues["V-32"]["notes"]
    assert all(issues[issue_id]["status"] == "verified" for issue_id in ("A2-25", "N5"))
    assert all(issues[issue_id]["verified_by"] == "R-0021" for issue_id in ("A2-25", "N5"))
    assert issues["V-33"]["status"] == "verified"
    assert issues["V-33"]["fixed_in"] == (
        "a54074013e5f0fb9bfe278baf944f805bf86eef4"
    )
    assert issues["V-33"]["target_release"] == "0.3"
    assert issues["V-33"]["verified_by"] == "R-0033"
    assert "S-10" in issues["V-33"]["notes"]
    assert "R-0033" in issues["V-33"]["notes"]
    assert issues["V-34"]["status"] == "verified"
    assert issues["V-34"]["fixed_in"] == (
        "dea728f1a926284420e6b1a17edb56eb56ea3db1"
    )
    assert issues["V-34"]["verified_by"] == "R-0021"
    assert issues["V-34"]["target_release"] == "0.3"
    assert "S-08" in issues["V-34"]["notes"]


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
