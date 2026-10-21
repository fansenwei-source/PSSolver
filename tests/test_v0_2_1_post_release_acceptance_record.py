"""Evidence-only registration of the independent v0.2.1 acceptance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "pssolver_v0_2_1_post_release_acceptance.json"
RELEASE_RECORD = ROOT / "notes" / "pssolver_v0_2_1_release.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_acceptance_binds_the_immutable_release_identity():
    record = _read(RECORD)
    assert record["classification"] == (
        "PASS_V0_2_1_INDEPENDENT_POST_RELEASE_ACCEPTANCE_WITH_DOCUMENTED_LIMITATIONS"
    )
    identity = record["release_identity"]
    assert identity["tag_object"] == "daf1270b1f046ac767045cbbe536fd185599d83f"
    assert identity["tag_target"] == "ca9c6729670f07639e7877881c777d1653f4a592"
    assert identity["github_release_id"] == 418261343
    assert identity["runtime_delta_from_v0_2_0"] == [
        "pssolver/io/checkpoint.py",
        "pssolver/_version.py",
    ]


def test_acceptance_binds_the_complete_independent_archive():
    archive = _read(RECORD)["independent_archive"]
    assert archive["complete_present"] is True
    assert archive["checksum_entry_count"] == 10102
    assert archive["checksums_sha256"] == (
        "332543391e608dd9dc778d3b163d025e0d42c36ed2b2290b5097b73cc3190b70"
    )
    assert archive["checksum_verification"] == {"passed": 10102, "failed": 0}
    assert archive["excluded_file_count"] == 1
    assert archive["excluded_file"].endswith(
        "rename_lost_reply_ETIMEDOUT_readback_unreadable/ckpt/payload.bin"
    )
    assert "deliberately unreadable" in archive["exclusion_reason"]


def test_acceptance_hash_binds_all_four_external_reports():
    reports = _read(RECORD)["external_reports"]
    assert reports["files"] == {
        "PSSolver_v0_2_1_post_release_acceptance_claude.json": (
            "8aa5a1549b1071ad7d3756a8e429f657c5988f95d3b4c2c0da88ba566204e3eb"
        ),
        "PSSolver_v0_2_1_post_release_acceptance_claude_zh.md": (
            "eb9dbc5997c720ba8ebd8fad2f2ffef142ced2d53cfe3f27ed7e73e88a092781"
        ),
        "PSSolver_v0_2_1_post_release_acceptance_claude_zh.pdf": (
            "1d0a1e685cfa0f0d5a75bf46b7f551f959f8574e2574bbf4c7877e2171a33dde"
        ),
        "PSSolver_v0_2_1_post_release_acceptance_claude_zh.tex": (
            "36b5199bd576cf35888bbfacc5e4e4d507a6c957942e780906c1a63537d8ff6b"
        ),
    }
    assert reports["json_semantically_matches_archive_summary"] is True
    assert reports["markdown_byte_identical_to_archive_final_report"] is True
    assert reports["external_files_modified"] is False


def test_acceptance_records_cpu_nfs_h100_and_control_results():
    results = _read(RECORD)["acceptance_results"]
    assert results["blockers"] == 0
    assert results["regressions_relative_to_v0_2_0"] == 0
    assert results["cpu_suite"] == {
        "passed": 4135,
        "failed": 0,
        "skipped": 0,
        "deselected": 7,
        "subtests_passed": 8,
    }
    assert results["local_n1_contract"] == {
        "finding_count": 23,
        "passed": 17,
        "documented_limitations": 5,
        "completed_by_real_machine_evidence": 1,
    }
    assert all(results["hpcc_real_nfs"].values())
    assert results["codex_archive_crosscheck"]["jobs_completed"] == [10902631, 10902665]
    assert results["codex_archive_crosscheck"]["new_h100_job_submitted"] is False
    assert results["control_recheck"]["provider_results_bitwise_identical"] is True
    assert results["control_recheck"]["identity_matches_pin"] is True


def test_acceptance_is_a_non_requalifying_evidence_only_successor():
    record = _read(RECORD)
    scope = record["record_scope"]
    for key in (
        "record_commit_is_release_source",
        "requires_h100_requalification",
        "runtime_changed",
        "tag_or_release_modified",
        "release_assets_replaced",
        "pypi_modified",
        "pssolver_control_modified",
        "nematics3d_modified",
        "historical_records_rewritten",
    ):
        assert scope[key] is False
    successor = record["successor_records"]["v0_2_1_release_record"]
    assert successor["rewritten"] is False
    assert _sha256(RELEASE_RECORD) == successor["sha256"]
    authorization = record["authorization"]
    assert authorization["local_evidence_registration_authorized"] is True
    assert all(
        authorization[key] is False
        for key in (
            "push_authorized",
            "main_fast_forward_authorized",
            "maint_fast_forward_authorized",
            "dev_cherry_pick_authorized",
            "runtime_fix_authorized",
            "h100_authorized",
        )
    )


def test_acceptance_registers_exactly_the_six_new_limitations():
    record = _read(RECORD)
    assert record["documented_limitations"] == [
        "V-17",
        "V-16",
        "V-13",
        "V-14",
        "V-21",
        "OBS-4",
    ]
    validation = record["registration_validation"]
    assert validation["targeted_tests"] == {
        "passed": 19,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert validation["complete_cpu_suite"] == {
        "passed": 4160,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
        "warnings": 6,
    }
    assert len(validation["cuda_only_deselects"]) == 7
    assert len(set(validation["cuda_only_deselects"])) == 7
    assert validation["environment"]["nematics3d_installed"] is True
    assert validation["git_diff_check_passed"] is True
    assert validation["pssolver_runtime_changed_file_count"] == 0
