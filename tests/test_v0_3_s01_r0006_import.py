"""Evidence contract for importing the independent R-0006 S-01 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s01_r0006_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0006_import_binds_the_pass_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_R0006_S01_REGISTRY_MIGRATION_REVIEW_IMPORT"
    )
    assert record["plan_step"] == "S-01"
    assert record["decision"] == "D-0010"
    assert record["git_identity"]["sync_result_commit"] == (
        "4be9c56c3ee46d44eda3bb5388e5c458a575f54f"
    )
    result = record["sync_result"]
    assert result["id"] == "R-0006"
    assert result["result_sha256"] == (
        "b323748c7b2c1ed6d215df066904764a92f19856958774cae192fced8bc6edc4"
    )
    assert result["classification"] == "PASS"
    assert result["release_blockers"] == []
    assert result["registry_updates"] == []
    assert result["new_findings"] == []
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 1922
    assert archive["checksums_sha256"] == (
        "fbaecc57045fb4cbabeb91d91ebdbd5044a587b55c09478f592f2559f42a0598"
    )
    assert archive["complete_present"] is True


def test_r0006_import_preserves_all_s01_verdicts() -> None:
    review = _read(RECORD)["review"]
    assert review["verdicts"] == {
        "S01-REGISTRY-1": "PASS",
        "S01-REGISTRY-2": "PASS",
        "S01-REGISTRY-3": "PASS",
        "S01-REGISTRY-4": "PASS",
        "S01-REGISTRY-5": "PASS",
    }
    assert review["eligible_for_non_force_push"] is True
    assert review["full_cpu_suite"] == {
        "passed": 4252,
        "failed": 0,
        "skipped": 0,
        "deselected": 7,
        "subtests_passed": 8,
    }
    assert review["registered_issue_count"] == 20
    assert review["new_14_target"] == "0.3 GPU verification"
    assert review["b30_plan_step"] == "S-07"
    assert review["v17_preserved"] is True


def test_r0006_import_does_not_change_the_reviewed_registry() -> None:
    registry = _read(RECORD)["registry"]
    assert registry["sha256"] == (
        "18a769b757a9c4325d69e8a9e86751c19642e66804141cda678dde408fed9eb3"
    )
    assert _sha256(REGISTRY) == resolve_s03_successor(
        "notes/pssolver_v0_2_x_open_issues.json", registry["sha256"]
    )
    assert registry["issue_count"] == 28
    assert registry["modified_by_import"] is False
    assert registry["updates_imported"] == 0
    assert registry["new_findings_imported"] == 0


def test_r0006_import_records_the_d0008_environment_contract() -> None:
    environment = _read(RECORD)["test_environment"]
    assert environment["policy"] == "D-0008 baseline-only hard-link clone"
    assert environment["baseline_venv_source"] == (
        "/home/fansenwei/pssolver_v0_2_1_claude_acceptance_20261020_v1/"
        "a2/venv_wheel"
    )
    assert environment["clone_method"] == "cp -al"
    assert environment["python_executable"].startswith(environment["clone_path"])
    assert environment["sys_prefix"] == environment["clone_path"]
    assert environment["sys_path"] == [
        "",
        "/home/fansenwei/anaconda3/envs/Nematics3D/lib/python312.zip",
        "/home/fansenwei/anaconda3/envs/Nematics3D/lib/python3.12",
        "/home/fansenwei/anaconda3/envs/Nematics3D/lib/python3.12/lib-dynload",
        f"{environment['clone_path']}/lib/python3.12/site-packages",
    ]
    contract = environment["future_environment_contract"]
    assert contract["allowed_sources"] == (
        "only the baseline venvs enumerated by D-0008"
    )
    assert contract["required_environment_fields"] == [
        "baseline_venv_source",
        "sys_path",
    ]
    assert contract["hard_link_clone_required"] is True
    assert contract["prune_after_use"] is True
    assert environment["clone_pruned_after_tests"] is True


def test_r0006_import_records_the_prior_environment_note_without_overriding_pass() -> None:
    observation = _read(RECORD)["prior_environment_observation"]
    assert observation["source_was_not_a_D_0008_baseline"] is True
    assert observation["clone_was_not_fully_isolated"] is True
    assert "PASS" in observation["independent_review_effect"]


def test_r0006_import_is_evidence_only_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert all(value is False for value in record["authorization"].values())
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["git_identity"]["pushed"] is False
    assert record["roadmap_next"] == "S-02"
    assert record["next_sync_id"] == "R-0007"
