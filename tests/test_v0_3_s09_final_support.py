"""Bindings for the final local S-09 support candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_36_s09_final_support.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_support_binds_fixture_payloads_and_unchanged_runtime() -> None:
    record = _read(RECORD)
    fixture = ROOT / record["fixture"]["root"]
    assert record["git_identity"]["provider_runtime_commit"] == (
        "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    )
    assert record["fixture"]["runtime_changed"] is False
    for name, digest in record["fixture"]["payloads"].items():
        assert _sha256(fixture / name) == digest
    assert record["fixture"]["all_manifest_files_git_tracked"] is True


def test_final_support_binds_plan_job_and_excluded_draft() -> None:
    draft = _read(RECORD)["h100_draft"]
    assert _sha256(ROOT / "notes/PSSolver_v0_3_0_s09_h100_plan.json") == resolve_s03_successor(
        "notes/PSSolver_v0_3_0_s09_h100_plan.json", draft["plan_sha256"]
    )
    assert _sha256(ROOT / "jobs/v0_3_s09_single_h100.sbatch") == resolve_s03_successor(
        "jobs/v0_3_s09_single_h100.sbatch", draft["job_script_sha256"]
    )
    assert _sha256(
        ROOT / "notes/PSSolver_v0_3_0_s09_h100_submission_draft_zh.md"
    ) == resolve_s03_successor(
        "notes/PSSolver_v0_3_0_s09_h100_submission_draft_zh.md",
        draft["draft_sha256"],
    )
    assert all(
        draft[key] is True
        for key in (
            "control_logs_created_before_sbatch",
            "complete_sbatch_command_recorded",
            "all_created_directories_and_dispositions_recorded",
            "output_root_must_initially_not_exist",
        )
    )


def test_final_support_preserves_the_review_boundary_and_authorizes_nothing() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["registry_boundary"] == {
        "V-35": "claimed_fixed",
        "V-36": "claimed_fixed",
        "V-37": "registered",
        "A3-18": "coverage_gap",
        "NEW-14": "open",
    }
    current = {key: issues[key]["status"] for key in record["registry_boundary"]}
    assert current == {
        "V-35": "verified",
        "V-36": "verified",
        "V-37": "verified",
        "A3-18": "verified",
        "NEW-14": "verified",
    }
    assert issues["V-37"]["fixed_in"] == (
        "87cf34cf4e17096a701f5cc5bc81076d14a9d237"
    )
    assert issues["V-37"]["status"] == "verified"
    assert all(
        issues[key]["verified_by"] == "R-0024"
        for key in ("V-35", "V-36", "V-37")
    )
    assert not any(record["scope"].values())
    assert record["next_request"] == "R-0024"
