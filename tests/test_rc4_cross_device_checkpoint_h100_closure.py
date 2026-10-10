from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "notes/PSSolver_v0_2_0rc4_rc426_h100_closure.json"


def _record() -> dict[str, object]:
    return json.loads(CLOSURE.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc426_closure_binds_candidate_job_and_external_manifests():
    record = _record()
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_2_6_SINGLE_H100_CROSS_DEVICE_CHECKPOINT_PORTABILITY"
    )
    assert record["git_identity"]["qualified_candidate_commit"] == (
        "5e3e821458f3ff451c206a8be7cd9765f6e634db"
    )
    assert record["external_evidence"]["control_manifest_entries"] == 1428
    assert record["external_evidence"]["control_manifest_sha256"] == (
        "5c4053ce0807bd63c3b2016b9fc6bf69fd4410a08ed57af6b97f528cfef9d147"
    )
    assert record["external_evidence"]["scratch_manifest_entries"] == 673
    assert record["external_evidence"]["scratch_manifest_sha256"] == (
        "3ead63c184979cff688fca9c5b2ae3a08ee8782f7a7a1434341b0730269469d4"
    )
    assert record["h100_job"] == {
        "job_id": 10874642,
        "state": "COMPLETED",
        "exit_code": "0:0",
        "elapsed": "00:00:35",
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


def test_rc426_closure_binds_every_versioned_local_record():
    for item in _record()["local_records"].values():
        assert _sha256(ROOT / item["path"]) == item["sha256"]


def test_rc426_closure_freezes_bidirectional_matrix_and_negative_gates():
    q = _record()["qualification"]
    assert q["installed_wheel_harness_passed"] == 26
    assert q["decomposed_preflight_passed"] == 8
    assert q["cuda_only_tests_passed"] == 7
    for key in (
        "matrix_cells_expected",
        "matrix_cells_passed",
        "compatibility_identity_equal",
        "serialized_payload_equal",
        "progress_exact",
        "persistent_backend_state_equal",
        "restored_finite",
        "post_restore_step_finite",
        "integrity_negative_gate_passed",
        "source_tree_unchanged",
    ):
        assert q[key] == 14
    assert q["cpu_to_cuda_passed"] == q["cuda_to_cpu_passed"] == 7
    assert q["runtime_fallback_used"] == 0
    assert q["tf32_matmul"] is q["tf32_cudnn"] is False


def test_rc426_closure_closes_exact_rc42_items_and_only_authorizes_planning():
    record = _record()
    assert {item["id"] for item in record["known_issues"]} == {
        "B10", "B22", "N3", "N5", "N6"
    }
    assert all(
        item["state_before"] == "partial_residual"
        and item["state_after"] == "closed_verified"
        for item in record["known_issues"]
    )
    assert record["authorization"] == {
        "qualification_complete": True,
        "rc4_2_6_complete": True,
        "rc4_2_complete": True,
        "eligible_for_rc4_3_planning": True,
        "rc4_3_implementation_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "eligible_for_release": False,
    }
    assert not any(record["scope"].values())
