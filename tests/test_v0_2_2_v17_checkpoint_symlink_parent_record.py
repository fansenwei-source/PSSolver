"""Evidence contract for the local V-17 CPU candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_2_v17_checkpoint_symlink_parent.json"
RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_2.json"
ISSUES = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_v17_record_binds_the_minimal_implementation_commit() -> None:
    record = _read(RECORD)
    identity = record["git_identity"]
    assert identity == {
        "branch": "fix/v0.2.2-v17",
        "implementation_commit": "611d167d1b2c2d634222f608f4c9cd4dc671636d",
        "implementation_parent": "c0934ed20af3b33ad19c8872d1b3a520a1e9e245",
        "record_commit_is_qualified_source": False,
        "pushed": False,
    }
    changed = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            identity["implementation_parent"],
            identity["implementation_commit"],
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert changed == [
        "pssolver/io/checkpoint.py",
        "tests/test_v0_2_2_v17_checkpoint_symlink_parent.py",
    ]


def test_v17_record_binds_current_and_historical_source_bytes() -> None:
    record = _read(RECORD)
    release_successors = _read(RELEASE_RECORD)["historical_successor_sources"]
    for relative, expected in record["source_sha256"].items():
        assert _sha256(relative) == expected
    parent = record["git_identity"]["implementation_parent"]
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        historical = subprocess.run(
            ["git", "show", f"{parent}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == registration[
            "historical_sha256"
        ]
        expected = registration["current_sha256"]
        successor = release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == expected:
            expected = successor["current_sha256"]
        assert _sha256(relative) == expected
        assert registration["reason"]


def test_v17_contract_covers_both_control_visible_functional_paths() -> None:
    contract = _read(RECORD)["contract"]
    for key in (
        "parent_symlink_followed_for_device_identity",
        "same_resolved_filesystem_required",
        "distinct_resolved_filesystems_rejected",
        "periodic_functional_export_covered",
        "channel_functional_export_covered",
        "requested_absolute_return_path_preserved",
        "checkpoint_round_trip_bitwise_equal",
    ):
        assert contract[key] is True
    assert contract["staging_residue_allowed"] is False


def test_v17_scope_is_narrow_and_does_not_claim_external_qualification() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_V0_2_2_V17_CHECKPOINT_PARENT_SYMLINK_LOCAL_CPU_CANDIDATE"
    )
    scope = record["scope"]
    assert scope["checkpoint_parent_device_guard_changed"] is True
    for key, value in scope.items():
        if key != "checkpoint_parent_device_guard_changed":
            assert value is False
    authorization = record["authorization"]
    assert authorization["implementation_complete"] is True
    assert authorization["broad_cpu_qualification_complete"] is True
    assert authorization["eligible_for_external_qualification"] is True
    for key in (
        "push_authorized",
        "h100_authorized",
        "tag_authorized",
        "release_authorized",
        "automatic_retry",
    ):
        assert authorization[key] is False
    evidence = record["test_evidence"]
    assert evidence["v17_and_n1_targeted"] == {"passed": 16, "failed": 0}
    assert evidence["checkpoint_functional_broad"] == {
        "passed": 96,
        "failed": 0,
    }
    assert evidence["historical_successor_records"] == {
        "registered_sources": 34,
        "registered_test_sources": 33,
        "targeted_passed": 144,
        "targeted_failed": 0,
    }
    assert evidence["full_cpu_suite"] == {
        "passed": 4168,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 7,
        "warnings": 6,
        "subtests_passed": 8,
    }


def test_v17_record_preserves_the_prequalification_status_and_current_verdict() -> None:
    issue = {
        item["id"]: item for item in _read(ISSUES)["issues"]
    }["V-17"]
    registration = _read(RECORD)["issue_registration"]
    assert registration["status_before_qualification"] == "open"
    assert issue["status"] == "verified"
    assert issue["fixed_in"] == (
        "611d167d1b2c2d634222f608f4c9cd4dc671636d"
    )
    assert issue["verified_by"] == "R-0002"
    assert issue["severity"] == registration["severity"] == "medium-low"
    assert issue["affects_control_paths"] is True
    assert issue["exercised_by_current_control_research"] is False
