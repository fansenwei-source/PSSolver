"""Bind the RC6 Consumer execution-rebind correction and v4 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_consumer_execution_rebind_correction.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_binds_v4_without_reclassifying_the_failed_job():
    evidence = _record()["failure_evidence"]
    assert evidence["job_id"] == "10901131"
    assert evidence["state"] == "FAILED"
    assert evidence["exit_code"] == "1:0"
    assert evidence["completed_stage_count"] == 8
    assert evidence["failed_stage"] == "consumer"
    assert evidence["profile_count"] == 192
    assert evidence["checkpoint_work_root_correction_passed"] is True
    assert evidence["qualification_complete"] is False
    assert evidence["automatic_retry"] is False
    assert evidence["second_job_submitted"] is False


def test_record_is_qualification_only_and_has_no_push_or_h100_authority():
    record = _record()
    assert record["classification"] == (
        "READY_RC6_CONSUMER_EXECUTION_REBIND_CORRECTION_H100_NOT_AUTHORIZED"
    )
    scope = record["scope"]
    assert scope["qualification_support_only"] is True
    assert all(
        value is False for key, value in scope.items() if key != "qualification_support_only"
    )
    assert all(value is False for value in record["authorization"].values())
