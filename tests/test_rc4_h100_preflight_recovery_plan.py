from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.capture_rc4_h100_preflight import (
    FAIL_CLASSIFICATION,
    PASS_CLASSIFICATION,
    SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc412_preflight_recovery_plan.json"
)


def _plan():
    return json.loads(PLAN.read_text(encoding="utf-8"))


def test_recovery_plan_binds_parent_runtime_and_versioned_helper():
    plan = _plan()
    helper = plan["versioned_helper"]
    helper_path = ROOT / helper["path"]

    assert plan["parent_qualification_commit"] == (
        "ac9f5dfe1d270581f20494ab96968f2a972db63f"
    )
    assert plan["runtime_identity"]["candidate"] == (
        "d358ed66fe75da68a3e17fed402a956f21f1dd37"
    )
    assert plan["runtime_identity"]["runtime_source_changed_by_this_recovery"] is False
    assert hashlib.sha256(helper_path.read_bytes()).hexdigest() == helper["sha256"]
    assert helper["schema"] == SCHEMA
    assert helper["pass_classification"] == PASS_CLASSIFICATION
    assert helper["failure_classification"] == FAIL_CLASSIFICATION


def test_recovery_plan_freezes_all_eight_predicates_and_tf32_assignments():
    plan = _plan()

    assert [item["id"] for item in plan["predicates"]] == [
        f"P{index:02d}" for index in range(1, 9)
    ]
    assert plan["tf32_policy"]["assignments"] == [
        "torch.backends.cuda.matmul.allow_tf32 = False",
        "torch.backends.cudnn.allow_tf32 = False",
    ]
    assert plan["tf32_policy"]["capture_before_assignment"] is True
    assert plan["tf32_policy"]["capture_after_assignment"] is True
    assert plan["versioned_helper"]["atomic_checkpoint_after_each_predicate"] is True
    assert plan["versioned_helper"]["enforcement_after_complete_capture"] is True
    assert plan["versioned_helper"]["naked_assertions"] is False


def test_recovery_plan_preserves_prior_failures_and_nonclaims():
    plan = _plan()
    evidence = plan["prior_evidence"]

    assert evidence["opaque_preflight_archive"]["job_id"] == "10859008"
    assert evidence["opaque_preflight_archive"]["formal_h100_submission_count"] == 1
    assert evidence["opaque_preflight_archive"]["scientific_commands_started"] is False
    assert evidence["analysis_archive"]["classification"] == (
        "UNRESOLVED_REQUIRES_DECOMPOSED_H100_PREFLIGHT"
    )
    assert plan["scope"] == {
        "pssolver_runtime_modified": False,
        "pssolver_control_modified": False,
        "nematics3d_modified": False,
        "scientific_thresholds_modified": False,
        "profile_contract_modified": False,
    }


def test_recovery_plan_does_not_authorize_another_h100_job_or_b5_closure():
    plan = _plan()
    authorization = plan["authorization"]

    assert plan["status"] == "LOCAL_CPU_GATE_PASS_H100_NOT_AUTHORIZED"
    assert plan["local_tests"] == {
        "synthetic_preflight_tests": 5,
        "targeted_passed": 76,
        "full_suite_passed": 2790,
        "full_suite_deselected_cuda_only": 7,
        "full_suite_subtests_passed": 8,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "full_suite_elapsed_seconds": 203.69,
    }
    assert authorization["helper_implementation_complete"] is True
    assert authorization["new_h100_submission_authorized"] is False
    assert authorization["automatic_retry"] is False
    assert authorization["b5_complete"] is False
    assert authorization["eligible_for_rc4_2_planning"] is False
    assert authorization["eligible_for_default_promotion"] is False
    assert authorization["eligible_for_release"] is False
