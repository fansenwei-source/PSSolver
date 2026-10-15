"""Evidence contract for the first rc5 G-1 H100 attempt."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_h100_failure.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_g1_h100_failure_records_the_single_nonretried_job():
    record = _record()
    assert record["classification"] == (
        "FAIL_RC5_G1_H100_CUDA_ONLY_CHECKPOINT_FILESYSTEM_PORTABILITY"
    )
    assert record["qualification_complete"] is False
    assert record["eligible_for_rc5_release"] is False
    job = record["job"]
    assert job["id"] == "10885517"
    assert job["state"] == "FAILED"
    assert job["formal_h100_submission_count"] == 1
    assert job["automatic_retry"] is False
    assert job["second_job_submitted"] is False


def test_g1_h100_failure_is_the_glibc_renameat2_portability_gate():
    failure = _record()["failure"]
    assert failure["nodeid"].endswith(
        "test_cuda_continuous_split_and_file_restart_are_exact"
    )
    assert failure["exception_type"] == "pssolver.io.checkpoint.CheckpointIOError"
    assert failure["operation"] == "finite_q_checkpoint_export"
    assert failure["hpcc_glibc"] == "2.17"
    assert failure["renameat2_symbol_present"] is False
    assert failure["failure_domain"] == "provider_checkpoint_filesystem_portability"
    assert not any(_record()["failure_nonclaims"].values())


def test_g1_h100_failure_preserves_passed_and_unrun_stage_boundaries():
    stages = _record()["stage_results"]
    assert stages["environment"] == "PASS"
    assert stages["cuda_only"] == "FAIL_6_PASSED_1_FAILED"
    assert set(stages.values()) == {"PASS", "FAIL_6_PASSED_1_FAILED", "NOT_RUN"}
    assert all(stages[name] == "NOT_RUN" for name in list(stages)[2:])


def test_g1_h100_failure_archive_is_fail_closed():
    record = _record()
    archive = record["archive"]
    assert archive["checksum_entries"] == 20
    assert archive["checksum_passed"] == 20
    assert archive["checksum_failed"] == 0
    assert archive["complete_present"] is False
    assert not any(record["scope"].values())
    assert record["next_gate"].startswith("implement_and_cpu_validate")
