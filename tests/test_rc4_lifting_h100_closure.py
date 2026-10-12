from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "notes/PSSolver_v0_2_0rc4_rc437_h100_closure.json"


def _record() -> dict[str, object]:
    return json.loads(CLOSURE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc437_closure_binds_candidate_job_and_external_manifest():
    record = _record()
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_3_7_INTERNAL_LIFTING_SINGLE_H100_CLOSURE"
    )
    assert record["git_identity"]["qualified_candidate_commit"] == (
        "e1dea73c5d0f00e097599c4ff2c94ed7dceaa915"
    )
    assert record["git_identity"]["profile_serialization_qualification_commit"] == (
        "4d176b24e13ecbe209dc4ca98c79b17bbfecbd39"
    )
    assert record["external_evidence"]["manifest_entries"] == 1105
    assert record["external_evidence"]["manifest_sha256"] == (
        "204c1b46cd9b4212a75743d17df1efb290a3cd6c805cb943df464665471cea65"
    )
    assert record["h100_job"] == {
        "job_id": 10875343,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:03:48",
        "node": "gpu-h100-4-0",
        "partition": "hagan-gpu",
        "account": "hagan-lab",
        "qos": "medium",
        "requeue": 0,
        "restarts": 0,
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "second_job_submitted": False,
    }


def test_rc437_closure_binds_every_versioned_local_record():
    for item in _record()["local_records"].values():
        assert _sha256(ROOT / item["path"]) == item["sha256"]


def test_rc437_closure_freezes_correct_layer_cuda_restart_and_profile_matrix():
    record = _record()
    q = record["qualification"]
    assert q["local_targeted_cpu_passed"] == 157
    assert q["local_full_cpu_passed"] == 3045
    assert q["hpcc_qualification_support_passed"] == 20
    assert q["decomposed_preflight_passed"] == 8
    assert q["cuda_only_tests_passed"] == 7
    assert q["profile_count_expected"] == q["profile_count_passed"] == 12
    assert q["profile_completed_steps"] == 60
    assert q["forward_transforms_per_step"] == 7.0
    assert q["inverse_transforms_per_step"] == 32.0
    assert q["graph_breaks"] == 0
    assert q["runtime_fallback_used"] is q["compile_fallback_used"] is False
    assert q["tf32_matmul"] is q["tf32_cudnn"] is False
    assert q["correct_layer"] == (
        "direct_conditioning_operator_output_before_spectral_projection"
    )
    assert q["interior_byte_identical"] is True
    assert q["transition_region_changed"] is True
    assert q["cpu_cuda_relative_l2"] == q["wall_residual_linf"] == 0.0
    assert q["restart_byte_identical"] is q["finite"] is True

    thresholds = record["performance"]["thresholds"]
    for grid in ("R128", "R320"):
        result = record["performance"][grid]
        assert result["mean_candidate_over_baseline"] <= thresholds[
            "mean_timestep_ratio_max"
        ]
        assert result["median_candidate_over_baseline"] <= thresholds[
            "median_timestep_ratio_max"
        ]
        assert result["peak_allocated_candidate_over_baseline"] <= thresholds[
            "peak_allocated_ratio_max"
        ]
        assert result["peak_active_candidate_over_baseline"] <= thresholds[
            "peak_active_ratio_max"
        ]


def test_rc437_closure_closes_exact_rc43_items_and_only_authorizes_planning():
    record = _record()
    assert {item["id"] for item in record["known_issues"]} == {"B8", "B20", "N4"}
    assert all(
        item["state_before"] == "partial_residual"
        and item["state_after"] == "closed_verified"
        for item in record["known_issues"]
    )
    assert record["authorization"] == {
        "qualification_complete": True,
        "rc4_3_7_complete": True,
        "rc4_3_complete": True,
        "eligible_for_rc4_4_planning": True,
        "rc4_4_implementation_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "eligible_for_release": False,
    }
    assert not any(record["scope"].values())
    assert record["claim_boundary"]["public_lifted_plane_reopened"] is False
    assert record["claim_boundary"]["localized_policy_publicly_qualified"] is False
