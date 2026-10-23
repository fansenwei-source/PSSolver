"""Bindings for the final local S-09 R-0023 review candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s09_r0023_candidate.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0023_candidate_binds_final_runtime_and_cpu_gates() -> None:
    record = _read(RECORD)
    assert record["classification"] == "READY_R0023_S09_FINAL_LOCAL_REVIEW"
    assert record["git_identity"] == {
        "provider_qualification_support_commit": (
            "9dd83cf2bec1a01bde5fc4e7b5e3a4afbd1ad3ad"
        ),
        "provider_runtime_commit": "85ba3aef75548d549d9db453ca4a5ab604b17a9d",
        "pushed": False,
        "record_commit_is_qualified_source": False,
        "record_parent": "90ca4269ff949a6e030b9fd097aa4d46fe342dfb",
    }
    assert record["cpu_gates"]["conda"] == {
        "deselected": 7, "failed": 0, "passed": 4696, "skipped": 0
    }
    assert record["cpu_gates"]["pip"] == {
        "deselected": 7, "failed": 0, "passed": 4695, "skipped": 1
    }
    assert record["cpu_gates"]["provider_installed"]["passed"] == 7
    assert record["cpu_gates"]["consumer_installed"]["passed"] == 351


def test_r0023_candidate_keeps_review_items_registered() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    for issue_id in ("V-35", "V-36"):
        assert issues[issue_id]["status"] == "registered"
        assert "fixed_in" not in issues[issue_id]
        assert _read(RECORD)["rework"][issue_id]["registry_status"] == "registered"
    assert issues["A3-18"]["status"] == "coverage_gap"
    assert issues["NEW-14"]["status"] == "open"


def test_r0023_candidate_binds_frozen_plan_job_and_draft() -> None:
    record = _read(RECORD)["h100_draft"]
    assert _sha256(ROOT / "notes/PSSolver_v0_3_0_s09_h100_plan.json") == record[
        "plan_sha256"
    ]
    assert _sha256(ROOT / "jobs/v0_3_s09_single_h100.sbatch") == record[
        "job_script_sha256"
    ]
    assert _sha256(
        ROOT / "notes/PSSolver_v0_3_0_s09_h100_submission_draft_zh.md"
    ) == record["draft_sha256"]
    assert all(
        record[key] is True
        for key in (
            "performance_rules_match_analyzer",
            "all_S09_environment_values_bound",
            "output_root_must_initially_not_exist",
            "api_1_1_v0_2_x_incompatibility_documented",
            "review_before_user_authorization_required",
        )
    )


def test_r0023_candidate_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert not any(record["authorization"].values())
    assert record["next_request"] == "R-0023"
