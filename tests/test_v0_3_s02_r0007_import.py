"""Evidence contract for importing the independent R-0007 S-02 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s02_r0007_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
CONTRACT = ROOT / "notes/PSSolver_v0_3_0_roadmap.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0007_import_binds_the_pass_result_and_archive() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_R0007_S02_COMPLETION_CONTRACT_REVIEW_IMPORT"
    )
    assert record["plan_step"] == "S-02"
    assert record["decision"] == "D-0011"
    assert record["git_identity"]["sync_result_commit"] == (
        "c89f78662f1b1e8b3c83ad4385a9efb64d5e2f9b"
    )
    result = record["sync_result"]
    assert result["id"] == "R-0007"
    assert result["result_sha256"] == (
        "e7aa88aea24cb1d3c5622b14d6e32c9e5ecc001c4b9aa76356eafe7c31da11e4"
    )
    assert result["classification"] == "PASS"
    assert result["release_blockers"] == []
    assert result["registry_updates"] == []
    assert result["new_findings"] == []
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 1925
    assert archive["checksums_sha256"] == (
        "d04f8c287f15a447f0f4cd02a31b40fe667d07d1a29cc145ca618846d7d58141"
    )
    assert archive["complete_present"] is True


def test_r0007_import_preserves_all_contract_verdicts() -> None:
    review = _read(RECORD)["review"]
    assert review["verdicts"] == {
        "S02-CONTRACT-1": "PASS",
        "S02-CONTRACT-2": "PASS",
        "S02-CONTRACT-3": "PASS",
        "S02-CONTRACT-4": "PASS",
        "S02-CONTRACT-5": "PASS",
    }
    assert review["eligible_for_non_force_push"] is True
    assert review["full_cpu_suite"] == {
        "passed": 4265,
        "failed": 0,
        "skipped": 0,
        "deselected": 7,
    }
    assert review["all_registry_ids_accounted_for"] == 28
    assert review["completion_gate_count"] == 9


def test_r0007_import_does_not_rewrite_registry_or_contract() -> None:
    record = _read(RECORD)
    registry = record["registry"]
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
    assert record["future_plan_updates"]["qualification_source_rewritten"] is False
    assert _read(CONTRACT)["status"] == "confirmed_pending_independent_review"


def test_r0007_import_records_the_required_future_g04_update() -> None:
    updates = _read(RECORD)["future_plan_updates"]
    assert updates["applied_by_this_import"] is False
    assert updates["required_on_next_v0_3_0_plan_update"] == [
        {
            "target": "G-04",
            "text": (
                "标准 Plane/Periodic/Channel 算例与 v0.2.2 的数值不退化比较："
                "代码路径未变处逐位相同，变化处用预先登记的容差。"
            ),
            "source": (
                "user instruction after R-0007 PASS; consistent with the "
                "R-0007 reviewer suggestion"
            ),
        }
    ]
    notes = " ".join(updates["reviewer_alignment_notes_for_future_roadmap_update"])
    assert "S-10" in notes and "R1-R9" in notes
    assert "rc1" in notes


def test_r0007_import_records_the_d0008_environment_contract() -> None:
    environment = _read(RECORD)["test_environment"]
    assert environment["policy"] == "D-0008 baseline-only hard-link clone"
    assert environment["baseline_venv_source"] == (
        "/home/fansenwei/pssolver_v0_2_1_claude_acceptance_20261020_v1/"
        "a2/venv_wheel"
    )
    assert environment["clone_method"] == "cp -al"
    assert environment["python_executable"].startswith(environment["clone_path"])
    assert environment["sys_prefix"] == environment["clone_path"]
    assert environment["sys_path"][-1] == (
        f"{environment['clone_path']}/lib/python3.12/site-packages"
    )
    assert environment["clone_pruned_after_tests"] is True


def test_r0007_import_is_evidence_only_and_authorizes_nothing() -> None:
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
    assert record["roadmap_next_after_push"] == "S-03"
    assert record["next_sync_id"] == "R-0008"
