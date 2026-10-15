from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc4_post_release_findings.json"
RC5_W1 = ROOT / "notes" / "PSSolver_v0_2_0rc5_w1_documentation.json"
RC5_R51 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
RC5_R52 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R53 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
RC5_R55 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
RC5_R56 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC5_R57 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RC5_R58 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
RC5_R59 = ROOT / "notes" / "PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
RC5_R510 = ROOT / "notes" / (
    "PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
RC5_R511 = ROOT / "notes" / (
    "PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
RC5_R512 = ROOT / "notes" / (
    "PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_record_registers_the_rc4_post_release_failure_without_mutating_rc4():
    record = _record()
    assert record["schema_version"] == 1
    assert record["classification"] == "FAIL"
    assert record["registration_only"] is True
    assert record["release_identity"] == {
        "tag": "v0.2.0rc4",
        "tag_object_sha": "8822ab370e252d76549c6365dc5f5023ec3231c4",
        "qualified_source_commit": "658869456d762c917ca4409b091539c3733df2fb",
        "tag_or_release_mutated": False,
    }
    assert record["scope"] == {
        "pssolver_runtime_modified": False,
        "tests_relaxed": False,
        "thresholds_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "functional_api_version_changed": False,
        "pssolver_control_modified": False,
        "nematics3d_modified": False,
        "h100_job_submitted": False,
        "historical_records_rewritten": False,
    }


def test_record_binds_all_four_post_release_audits_by_digest():
    evidence = _record()["evidence"]
    assert evidence["external_manifest_verified"] is True
    assert evidence["post_release_verification"] == {
        "path": "PSSolver/notes/PSSolver_v0_2_0rc4_post_release_verification.json",
        "sha256": "24f7e1a1129b72894c634071d8f272dfbd3f1a91017650427550e1f35f5da773",
        "classification": "FAIL_NEW_REGRESSION",
    }
    assert evidence["codex_draft"] == {
        "path": "notes/PSSolver_v0_2_0rc4_post_release_verification_codex_draft.json",
        "sha256": "54e779aee7f622b78580aad1e03d26afea0eeaf6be8bcb07788bc8b53bb2e77b",
        "classification": "SUPERSEDED_PASS_WITH_DOCUMENTED_LIMITATIONS",
    }
    assert evidence["claude_blind_review"] == {
        "path": "notes/PSSolver_v0_2_0rc4_post_release_verification_claude_review.json",
        "sha256": "e29501ab68088d18ba567058554406e5742cb5c923a8161d928c7e7b7862ac66",
        "classification": "FAIL",
    }
    assert evidence["consolidated_report"] == {
        "path": "PSSolver/notes/PSSolver_v0_2_0rc4_consolidated_report.json",
        "sha256": "db3e644bd97bc28ca046d0632aa2eae1b142ad6a208ad10b9dcb82cc42e6e6c6",
        "classification": "FAIL",
    }
    assert _sha256(
        ROOT / "notes" / "PSSolver_v0_2_0rc4_post_release_verification_codex_draft.json"
    ) == evidence["codex_draft"]["sha256"]
    assert _sha256(
        ROOT / "notes" / "PSSolver_v0_2_0rc4_post_release_verification_claude_review.json"
    ) == evidence["claude_blind_review"]["sha256"]


def test_b10_is_registered_as_a_fail_closed_six_case_regression():
    regression = _record()["decisive_regression"]
    assert regression["item"] == "B10"
    assert regression["state"] == "regressed_after_closure"
    assert regression["published_generations"] == ["rc1", "rc2", "rc3"]
    assert regression["functional_families"] == ["periodic", "channel"]
    assert regression["fixture_cases"] == 6
    assert regression["direct_import_rejected"] == 6
    assert regression["explicit_migration_rejected"] == 6
    assert regression["wrong_data_accepted"] is False
    assert regression["failure_mode"] == "fail_closed"
    assert regression["historical_closure_record_rewritten"] is False


def test_current_matrix_transition_and_new_findings_are_exact():
    transition = _record()["known_issue_transition"]
    assert transition["historical_item_count"] == 50
    assert {
        key: transition[key]
        for key in (
            "closed_verified",
            "partial_residual",
            "regressed_after_closure",
            "accepted_limitation",
            "deferred_architecture",
        )
    } == {
        "closed_verified": 28,
        "partial_residual": 9,
        "regressed_after_closure": 1,
        "accepted_limitation": 3,
        "deferred_architecture": 9,
    }
    assert transition["regressed_items"] == ["B10"]
    assert transition["partial_residual_items"] == [
        "B7",
        "B11",
        "B13",
        "B17",
        "B20",
        "B23",
        "B24",
        "N3",
        "N6",
    ]
    assert transition["post_release_findings"] == [
        f"NEW-{index}" for index in range(1, 16)
    ]
    assert transition["a3_target"] == "rc3"
    assert transition["b8_physical_lifting_repaired"] is False
    assert transition["b30_documentation_complete"] is False


def test_rc5_work_order_is_frozen_before_implementation():
    assert _record()["rc5_work_order"] == [
        "rc5_documentation",
        "rc5_must",
        "rc5_should",
        "rc5_optional",
        "rc5_release_process",
        "rc5_qualification",
        "post_0_2_architecture",
    ]


def test_historical_successor_sources_match_the_current_files_exactly():
    successors = _record()["historical_successor_sources"]
    rc5_w1_successors = json.loads(RC5_W1.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r51_successors = json.loads(RC5_R51.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r52_successors = json.loads(RC5_R52.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r53_successors = json.loads(RC5_R53.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r55_successors = json.loads(RC5_R55.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r56_successors = json.loads(RC5_R56.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r57_successors = json.loads(RC5_R57.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r58_successors = json.loads(RC5_R58.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r59_successors = json.loads(RC5_R59.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r510_successors = json.loads(RC5_R510.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r511_successors = json.loads(RC5_R511.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    rc5_r512_successors = json.loads(RC5_R512.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    assert set(successors) == {
        "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json",
        "notes/PSSolver_v0_2_0rc3_known_issues_matrix_zh.md",
        "tests/test_rc3_known_issues_matrix.py",
        "tests/test_rc451_cross_runtime_adjudication.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc458_h100_closure_record.py",
        "tests/test_rc4_cross_runtime_robustness_plan.py",
        "tests/test_release_v0_2.py",
    }
    for relative_path, entry in successors.items():
        assert entry["historical_sha256"] != entry["current_sha256"]
        resolved = entry["current_sha256"]
        successor = rc5_w1_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r51_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r52_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r53_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r55_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r56_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r57_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r58_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r59_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r510_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r511_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r512_successors.get(relative_path)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert _sha256(ROOT / relative_path) == resolved
        assert entry["reason"]


def test_w0_authorizes_only_the_documentation_successor():
    assert _record()["verification"] == {
        "targeted_cpu": {
            "passed": 101,
            "failed": 0,
            "skipped": 0,
            "xfailed": 0,
        },
        "full_cpu_cuda_hidden": {
            "passed": 3415,
            "failed": 0,
            "skipped": 7,
            "subtests_passed": 8,
            "baseline_skip_count_unchanged": True,
            "new_passing_tests": 15,
        },
        "git_diff_check_passed": True,
        "strict_json_parse_passed": True,
        "cache_residue_count": 0,
    }
    assert _record()["authorization"] == {
        "w0_complete": True,
        "eligible_for_w1_documentation": True,
        "rc5_runtime_repair_complete": False,
        "eligible_for_rc5_release": False,
        "h100_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
    }
