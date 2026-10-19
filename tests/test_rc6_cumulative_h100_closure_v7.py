"""Immutable repository record for the successful RC6 H100 closure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_cumulative_h100_closure_v7.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc6_closure_binds_runtime_support_and_record_roles():
    record = _record()
    assert record["classification"] == "PASS_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION"
    assert record["source_identity"] == {
        "branch": "fix/v0.2.0rc6",
        "qualified_runtime_candidate_commit": "ea801ec08f5b2465661334e1c126133e55b68e96",
        "qualification_support_commit": "3228d3d723393818ee6149e9cf56021ffaaed411",
        "remote_tip_at_execution": "3228d3d723393818ee6149e9cf56021ffaaed411",
        "record_commit_is_qualified_source": False,
        "qualification_support_is_runtime_candidate": False,
        "runtime_candidate_is_ancestor_of_qualification_support": True,
    }


def test_rc6_closure_binds_the_only_v7_job_without_retry():
    slurm = _record()["slurm"]
    assert slurm["job_id"] == "10901236"
    assert slurm["state"] == "COMPLETED"
    assert slurm["exit_code"] == "0:0"
    assert slurm["node"] == "gpu-h100-4-0"
    assert slurm["recovery_v7_submission_count"] == 1
    assert slurm["automatic_retry"] is False
    assert slurm["second_recovery_v7_job_submitted"] is False


def test_rc6_closure_binds_all_cumulative_gate_counts():
    gates = _record()["cumulative_gates"]
    assert gates["completed_stages"] == gates["planned_stages"] == 11
    assert gates["cuda_only_passed"] == 7
    assert gates["cuda_only_failed"] == 0
    assert gates["current_checkpoint_cells"] == 8
    assert gates["released_checkpoint_primary_cases"] == 13
    assert gates["released_checkpoint_tests_passed"] == 20
    assert gates["numerical_runtime_families"] == 8
    assert gates["numerical_byte_identical_families"] == 8
    assert gates["profile_count"] == gates["finite_profile_count"] == 192
    assert gates["profile_tf32_pass_count"] == 192
    assert gates["profile_child_tf32_bootstrap_count"] == 192
    assert gates["profile_child_tf32_bootstrap_passed"] == 192
    assert gates["performance_memory_groups"] == 16
    assert gates["performance_memory_groups_passed"] == 16
    assert gates["consumer_passed"] == 344
    assert gates["consumer_skipped"] == 1
    assert gates["consumer_failed"] == gates["consumer_xfailed"] == 0


def test_rc6_closure_requires_explicit_per_child_tf32_evidence():
    contract = _record()["profile_contract"]
    assert contract["required_tf32_state"] == {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    for name in (
        "one_atomic_bootstrap_per_profile",
        "bootstrap_persisted_before_profile_start",
        "bootstrap_binds_profile_implementation_sha256",
        "all_profile_states_match_required_tf32_state",
        "all_profiles_finite",
        "graph_breaks_zero",
        "compile_fallback_false",
        "runtime_fallback_false",
    ):
        assert contract[name] is True


def test_rc6_closure_binds_complete_control_and_formal_archives():
    archives = _record()["archive_identities"]
    control = archives["control"]
    assert control["complete_present"] is True
    assert control["checksum_entries"] == control["checksum_passed"] == 24
    assert control["checksum_failed"] == 0
    assert control["checksums_sha256"] == (
        "26c99535fc92b1c33d0a8fe6adc5c2fab05d0332034eb7464657a4e047ecbb40"
    )
    formal = archives["formal"]
    assert formal["complete_present"] is True
    assert formal["checksum_entries"] == formal["checksum_passed"] == 4763
    assert formal["checksum_failed"] == 0
    assert formal["checksums_sha256"] == (
        "160a4960da3de7ea518ff2187393a6d7bb6e563a9611be1057ecb6c437ad894c"
    )
    assert control["complete_sha256"] == formal["complete_sha256"]


def test_rc6_closure_binds_exact_qualification_sources():
    record = _record()
    for relative, expected in record["qualification_source_sha256"].items():
        assert _sha256(relative) == expected
    reused = record["reused_v6_evidence"]
    assert reused["classification_preserved"] == (
        "FAIL_RC6_RECOVERY_V6_PROFILE_CHILD_TF32_BOOTSTRAP_CONTRACT"
    )
    assert reused["control_checksum_entries"] == 23
    assert reused["formal_checksum_entries"] == 4567
    assert reused["historical_archives_modified"] is False


def test_rc6_closure_authorizes_review_but_no_release_action():
    record = _record()
    assert all(value is False for value in record["scope"].values())
    authorization = record["authorization"]
    assert authorization["qualification_complete"] is True
    assert authorization["rc6_complete"] is True
    assert authorization["eligible_for_release_integration_review"] is True
    assert authorization["independent_post_release_verification_required"] is True
    assert authorization["qualification_support_push_executed"] is True
    for name in (
        "eligible_for_automatic_merge",
        "eligible_for_default_promotion",
        "closure_record_push_executed",
        "tag_authorized",
        "tag_created",
        "release_authorized",
        "release_created",
        "pypi_publication_authorized",
        "pypi_published",
        "production_default_changed",
    ):
        assert authorization[name] is False


def test_rc6_closure_records_the_repository_registration_gate():
    validation = _record()["repository_validation"]
    assert validation == {
        "json_parse": "PASS",
        "focused_tests": "28 passed, 0 failed, 0 skipped, 0 xfailed",
        "full_cpu_suite": (
            "4062 passed, 7 deselected, 8 subtests passed, "
            "0 failed, 0 skipped, 0 xfailed"
        ),
        "deselected_scope": "the seven pre-existing CUDA-only node IDs",
        "git_diff_check": "PASS",
    }
