"""Forward-only commit-identity corrections and resolution guards for S-08."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_26_s08_commit_identity_corrections.json"
SUCCESSORS = ROOT / "notes"
FULL_COMMIT = re.compile(r"[0-9a-f]{40}")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_commit(value: str) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _does_not_resolve(value: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{value}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
    ).returncode != 0


def test_current_registry_fixed_in_values_are_real_pssolver_commits() -> None:
    for issue in _read(REGISTRY)["issues"]:
        fixed_in = issue.get("fixed_in")
        if fixed_in is not None:
            assert FULL_COMMIT.fullmatch(fixed_in)
            assert _resolve_commit(fixed_in) == fixed_in


def test_successor_git_identities_resolve_or_have_an_explicit_forward_correction() -> None:
    correction = _read(RECORD)
    mappings = correction["commit_identity_corrections"]
    exceptions = correction["external_git_identity_fields"]
    encountered_corrections: set[str] = set()
    for path in sorted(SUCCESSORS.glob("PSSolver_v0_3_0_s03_successor_*.json")):
        for field, value in _read(path).get("git_identity", {}).items():
            if not isinstance(value, str) or FULL_COMMIT.fullmatch(value) is None:
                continue
            if field in exceptions:
                assert exceptions[field]["repository"] == "PSSolver_sync"
                continue
            authoritative = mappings.get(value, value)
            assert _resolve_commit(authoritative) == authoritative
            if authoritative != value:
                encountered_corrections.add(value)
                assert _does_not_resolve(value)
    assert encountered_corrections == set(correction["forward_successor_corrections"])


def test_all_historical_errata_bind_nonexistent_values_to_real_git_commits() -> None:
    record = _read(RECORD)
    assert len(record["historical_record_errata"]) == 10
    for item in record["historical_record_errata"]:
        assert _does_not_resolve(item["wrong_hash"])
        assert _resolve_commit(item["real_commit"]) == item["real_commit"]
        assert item["wrong_hash"][:7] == item["real_commit"][:7]


def test_r0021_verifies_v34_and_preserves_later_registered_s09_items() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert {key for key, item in issues.items() if item["status"] == "registered"} == {
        "V-33",
        "V-35",
        "V-36",
    }
    assert not {key for key, item in issues.items() if item["status"] == "claimed_fixed"}
    assert issues["V-33"]["target_release"] == "0.3"
    assert "S-10" in issues["V-33"]["notes"]
    for issue_id in ("V-35", "V-36"):
        assert "S-09" in issues[issue_id]["notes"]
        assert issues[issue_id]["source_report"] == (
            "PSSolver_sync/results/R-0022.result.json"
        )
    assert issues["V-34"]["status"] == "verified"
    assert issues["V-34"]["fixed_in"] == (
        "dea728f1a926284420e6b1a17edb56eb56ea3db1"
    )
    assert issues["V-34"]["verified_by"] == "R-0021"
    assert issues["V-34"]["target_release"] == "0.3"
    assert "R-0021" in issues["V-34"]["notes"]


def test_correction_record_authorizes_no_external_or_runtime_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
