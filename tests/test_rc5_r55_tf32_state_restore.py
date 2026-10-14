"""Static closure contract for the R5-5 TF32 caller-state repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
R5_6_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _r56_successor(relative: str, historical: str) -> str:
    registration = json.loads(R5_6_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if (
        registration is not None
        and registration["historical_sha256"] == historical
    ):
        return registration["current_sha256"]
    return historical


def test_r55_record_binds_issue_and_reachable_state_contract():
    record = _record()
    assert record["status"] == "implementation_complete_cpu_verified_h100_deferred"
    assert record["classification"] == "PASS_R5_5_TF32_STATE_RESTORE_CPU_CANDIDATE"
    assert record["issue"] == {
        "id": "B11",
        "related_issue": "NEW-14",
        "work_item": "R5-5",
        "source": "notes/PSSolver_v0_2_0rc4_consolidated_report.json",
    }
    assert record["contract"] == {
        "restore_order": [
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
            "torch_float32_matmul_precision",
        ],
        "reachable_precision_cuda_pairs": [
            ["highest", False],
            ["high", True],
            ["medium", True],
        ],
        "cudnn_states": [False, True],
        "caller_state_count": 6,
        "scope_policy_count": 2,
        "exit_path_count": 2,
        "cpu_state_matrix_case_count": 24,
        "h100_runtime_family_case_count": 48,
        "success_restores_exact_state": True,
        "exception_restores_exact_state": True,
        "nested_scope_restores_lifo": True,
        "invalid_policy_does_not_mutate": True,
    }


def test_r55_record_binds_exact_implementation_and_regression_sources():
    record = _record()
    for section in ("implementation_files", "qualification_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == _r56_successor(relative, expected)


def test_r55_historical_successors_are_exact_and_forward_only():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "pssolver/configuration/execution_device.py",
        "benchmarks/qualify_rc458_tf32_scope.py",
        "benchmarks/plan_rc458_tf32_h100.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_plan_rc458_tf32_h100.py",
        "tests/test_rc458_qualification_support.py",
        "tests/test_rc458_transform_counter_recovery.py",
        "tests/test_rc458_reserved_memory_recovery.py",
        "tests/test_rc458_h100_closure_record.py",
        "tests/test_rc451_cross_runtime_adjudication.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc4_cross_runtime_robustness_plan.py",
        "tests/test_rc4_post_release_findings.py",
        "tests/test_rc5_legacy_functional_checkpoint_record.py",
        "tests/test_rc5_r52_finite_workflow_record.py",
        "tests/test_rc5_r53_channel_checkpoint_record.py",
        "tests/test_rc5_w1_documentation.py",
        "tests/test_release_v0_2.py",
    }
    for relative, registration in successors.items():
        assert len(registration["historical_sha256"]) == 64
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(relative) == _r56_successor(
            relative, registration["current_sha256"]
        )
        assert registration["reason"]


def test_r55_validation_records_prefixed_failure_and_cpu_closure():
    validation = _record()["validation"]
    assert validation["pre_fix_reproducer"] == {
        "torch": "2.6.0",
        "case_count": 24,
        "mismatch_count": 8,
        "all_mismatches_precision": "medium",
    }
    assert validation["post_fix_reproducer"] == {
        "torch": "2.6.0",
        "case_count": 24,
        "mismatch_count": 0,
    }
    assert validation["targeted_cpu"].endswith("0 failed")
    assert validation["source_full_cpu"].endswith("0 failed")
    assert validation["installed_wheel_targeted_cpu"].endswith("0 failed")
    assert validation["installed_wheel_applicable_full_cpu"].endswith("0 failed")


def test_r55_scope_defers_h100_without_changing_scientific_contracts():
    record = _record()
    assert record["scope"] == {
        "scientific_equations_changed": False,
        "numerical_algorithms_changed": False,
        "checkpoint_schema_changed": False,
        "public_api_changed": False,
        "production_default_changed": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "H100_required": True,
        "H100_executed": False,
        "push_executed": False,
    }
    assert record["authorization"] == {
        "r5_5_cpu_candidate_complete": True,
        "h100_tf32_caller_state_matrix": "deferred_to_single_rc5_cumulative_qualification",
        "eligible_for_r5_6_implementation": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
