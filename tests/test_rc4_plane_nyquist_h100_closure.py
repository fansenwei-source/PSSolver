from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes"
CLOSURE = NOTES / "PSSolver_v0_2_0rc4_rc412_h100_closure.json"


def _record() -> dict[str, object]:
    return json.loads(CLOSURE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc412_closure_binds_runtime_and_qualification_identities():
    record = _record()
    identity = record["git_identity"]

    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_1_2_PLANE_NYQUIST_H100_NON_REGRESSION_WITH_"
        "AUTHORIZED_HOST_COMPILER_RECOVERY_V5"
    )
    assert identity["candidate_commit"] == (
        "d358ed66fe75da68a3e17fed402a956f21f1dd37"
    )
    assert identity["qualification_commit"] == (
        "c39dabba1478e7b47b94b8422096f50860ecc118"
    )
    assert identity["runtime_source_changed_after_candidate"] is False
    assert _sha256(ROOT / identity["implementation_source"]) == identity[
        "implementation_source_sha256"
    ]


def test_rc412_closure_binds_local_plans_and_external_archives():
    record = _record()
    for item in record["local_records"].values():
        if isinstance(item, dict):
            assert _sha256(ROOT / item["path"]) == item["sha256"]

    v4 = record["external_evidence"]["recovery_v4"]
    v5 = record["external_evidence"]["recovery_v5"]
    assert v4["classification_preserved"] is True
    assert (v4["checksum_passed"], v4["checksum_failed"]) == (30, 0)
    assert v4["partial_profile_reused"] is False
    assert v5["checksums_sha256"] == (
        "0f891b57d7b665bae6a0d0dd29839fbf1955853f96a9cdccb29d3fab77514717"
    )
    assert (v5["checksum_passed"], v5["checksum_failed"]) == (50, 0)
    assert v5["complete_marker_present"] is True


def test_rc412_closure_records_successful_h100_and_compiler_recovery():
    record = _record()
    job = record["h100_job"]
    compiler = record["host_compiler_recovery"]

    assert job == {
        "job_id": "10859206",
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:04:01",
        "node": "gpu-h100-4-0",
        "partition": "hagan-gpu",
        "account": "hagan-lab",
        "qos": "medium",
        "requeue": 0,
        "restarts": 0,
        "submission_attempt_count_total": 4,
        "formal_h100_submission_count_total": 3,
        "new_h100_submission_count": 1,
        "automatic_retry": False,
    }
    assert compiler["cc"].endswith("/gcc/7.3.0/bin/gcc")
    assert compiler["version"] == "7.3.0"
    for key in (
        "stdatomic_header_present",
        "stdatomic_pic_compile",
        "cuda_driver_link_and_load",
        "fresh_triton_cuda_utils_compile",
        "fresh_torch_compile_cuda",
    ):
        assert compiler[key] is True
    assert compiler["compile_fallback"] is False
    assert compiler["runtime_fallback"] is False
    assert compiler["tf32_matmul"] is False
    assert compiler["tf32_cudnn"] is False


def test_rc412_closure_freezes_correctness_and_profile_thresholds():
    record = _record()
    correctness = record["correctness"]
    profiles = record["profiles"]
    thresholds = profiles["thresholds"]

    assert correctness["decomposed_preflight_predicates_passed"] == 8
    assert correctness["cuda_only_tests_passed"] == 7
    assert correctness["storage_diagnostic_case_count"] == 5
    assert correctness["maximum_storage_relative_l2_bound"] == 1e-12
    assert correctness["forward_transforms_per_step"] == 7
    assert correctness["inverse_transforms_per_step"] == 32
    assert correctness["graph_breaks"] == 0
    assert profiles["expected"] == profiles["completed"] == 12
    assert profiles["all_new_in_recovery_v5"] is True
    assert profiles["independent_cache_directory_count"] == 24

    for grid in ("R128", "R320"):
        result = profiles[grid]
        assert result["mean_candidate_over_baseline"] <= thresholds[
            "mean_timestep_ratio_max"
        ]
        assert result["median_candidate_over_baseline"] <= thresholds[
            "median_timestep_ratio_max"
        ]
        assert max(result["paired_candidate_over_baseline"]) <= thresholds[
            "paired_timestep_ratio_max"
        ]
        assert result["peak_allocated_candidate_over_baseline"] <= thresholds[
            "peak_memory_ratio_max"
        ]
        assert result["peak_reserved_candidate_over_baseline"] <= thresholds[
            "peak_memory_ratio_max"
        ]


def test_rc412_closure_closes_only_b5_and_authorizes_only_rc42_planning():
    record = _record()
    assert record["known_issue"] == {
        "id": "B5",
        "state_before": "partial_residual",
        "state_after": "closed_verified",
        "batch": "rc4.1_plane_nyquist",
    }
    assert record["analyzer"] == {
        "classification": (
            "PASS_RC4_1_2_PLANE_NYQUIST_SINGLE_H100_NON_REGRESSION"
        ),
        "qualification_complete": True,
    }
    assert record["local_closeout"] == {
        "targeted_passed": 33,
        "full_suite_passed": 2796,
        "full_suite_deselected_cuda_only": 7,
        "full_suite_subtests_passed": 8,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "full_suite_elapsed_seconds": 198.34,
    }
    assert record["authorization"] == {
        "complete": True,
        "b5_complete": True,
        "eligible_for_rc4_2_planning": True,
        "rc4_2_implementation_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "eligible_for_release": False,
    }
    assert not any(record["scope"].values())
