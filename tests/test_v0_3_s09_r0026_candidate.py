"""Bindings for the local S-09 R-0026 recovery candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_40_s09_r0026_candidate.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0026_candidate_binds_only_qualification_support() -> None:
    record = _read(RECORD)
    assert record["classification"] == "READY_FOR_R0026_S09_RECOVERY_REVIEW"
    assert record["git_identity"]["provider_runtime_commit"] == (
        "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    )
    assert record["git_identity"]["v39_v40_fix_commit"] == (
        "72286694c43ee15e5859c0658c989431404fe505"
    )
    assert record["record_scope"] == {
        "evidence_only": True,
        "runtime_changed": False,
        "qualification_support_changed": True,
        "external_action_authorized": False,
    }


def test_r0026_candidate_claims_exactly_v39_and_v40() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert set(record["registry_claims"]["claimed_fixed"]) == {"V-39", "V-40"}
    for issue_id, fixed_in in record["registry_claims"]["claimed_fixed"].items():
        assert issues[issue_id]["status"] == "claimed_fixed"
        assert issues[issue_id]["fixed_in"] == fixed_in
        assert "verified_by" not in issues[issue_id]
    assert issues["NEW-14"]["status"] == "open"


def test_r0026_candidate_binds_final_archive_plan_job_and_rehearsal() -> None:
    record = _read(RECORD)
    rehearsal = record["cpu_rehearsal"]
    assert rehearsal["classification"] == "PASS_S09_INSTALLED_WHEEL_CPU_REHEARSAL"
    assert _sha256(Path(rehearsal["path"])) == rehearsal["sha256"]
    frozen = record["frozen_inputs"]
    assert _sha256(ROOT / "notes/PSSolver_v0_3_0_s09_h100_plan.json") == frozen["plan_sha256"]
    assert _sha256(ROOT / "jobs/v0_3_s09_single_h100.sbatch") == frozen["job_script_sha256"]
    assert _sha256(ROOT / "notes/PSSolver_v0_3_0_s09_h100_submission_draft_zh.md") == frozen["submission_draft_sha256"]
    archive = Path("/home/fansenwei/.pssolver_s09_r0026_local/provider_support_719ae76.tar")
    assert _sha256(archive) == frozen["provider_support_archive_sha256"]


def test_r0026_candidate_successors_are_forward_only() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == (
            successor["current_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0026_candidate_authorizes_no_external_action() -> None:
    authorization = _read(RECORD)["authorization"]
    assert authorization["local_rework"] is True
    assert all(value is False for key, value in authorization.items() if key != "local_rework")
