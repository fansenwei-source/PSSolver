"""Bind the RC6 recovery-v7 profile-child TF32 bootstrap record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_profile_child_tf32_recovery.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_preserves_v6_failure_and_completed_scientific_evidence():
    failure = _record()["failure_evidence"]
    assert failure["job_id"] == "10901220"
    assert failure["completed_stage_count"] == 10
    assert failure["planned_stage_count"] == 11
    assert failure["profile_count"] == 192
    assert failure["consumer_passed_count"] == 344
    assert failure["failed_stage"] == "analyze"
    assert failure["qualification_complete"] is False
    assert failure["scientific_failure"] is False
    assert failure["historical_classification_rewritten"] is False


def test_record_binds_exact_recovery_sources_and_forward_successors():
    record = _record()
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    successors = record["historical_successor_sources"]
    assert set(successors) == {
        "benchmarks/run_rc6_profile_matrix.py",
        "tests/test_rc6_qualification_support.py",
        "tests/test_rc6_qualification_support_record.py",
    }
    for relative, registration in successors.items():
        assert registration["successor_sha256"] == _sha256(ROOT / relative)
        assert registration["predecessor_sha256"] != registration["successor_sha256"]


def test_record_requires_atomic_per_child_tf32_evidence_before_profile_start():
    correction = _record()["correction"]
    assert correction["profile_child_count"] == 192
    assert correction["python_tf32_policy"] == {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    assert correction["one_bootstrap_report_per_profile_child"] is True
    assert correction["bootstrap_report_persisted_before_profile_start"] is True
    assert correction["matrix_runner_validates_all_bootstrap_reports"] is True
    assert correction["profile_implementation_modified"] is False


def test_recovery_job_retains_node_local_preflights_and_v7_log_identity():
    job = ROOT / "jobs/rc6_profile_tf32_recovery_v7.sbatch"
    text = job.read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue" in text
    assert "20261019_recovery_v7/logs/slurm_%j.out" in text
    assert "20261019_recovery_v7/logs/slurm_%j.err" in text
    checkpoint = text.index("run_rc5_g1_checkpoint_tmp_preflight.py")
    toolkit = text.index("run_rc6_cuda_toolkit_preflight.py")
    stages = text.index("run_rc6_single_h100_stage_plan.py")
    assert checkpoint < toolkit < stages
    assert 'export TMPDIR="${pytest_tmp}"' in text
    assert "module load cuda/12.1" in text


def test_record_is_qualification_only_and_does_not_authorize_push_or_h100():
    record = _record()
    assert record["classification"] == (
        "READY_RC6_RECOVERY_V7_PROFILE_CHILD_TF32_BOOTSTRAP_H100_NOT_AUTHORIZED"
    )
    scope = record["scope"]
    assert scope["qualification_support_only"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key != "qualification_support_only"
    )
    assert all(value is False for value in record["authorization"].values())
