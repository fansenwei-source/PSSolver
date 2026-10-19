"""Bind the RC6 checkpoint work-root correction and its support-only scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_checkpoint_work_root_correction.json"
CONSUMER_REBIND_SUCCESSOR_RECORD = (
    ROOT / "notes/PSSolver_v0_2_0rc6_consumer_execution_rebind_correction.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _current_sha256(relative: str, expected: str) -> str:
    registration = json.loads(
        CONSUMER_REBIND_SUCCESSOR_RECORD.read_text(encoding="utf-8")
    )["historical_successor_sources"].get(relative)
    if registration is None:
        return expected
    assert registration["predecessor_sha256"] == expected
    return registration["successor_sha256"]


def test_record_binds_the_failed_h100_attempt_without_reclassifying_it():
    evidence = _record()["failure_evidence"]
    assert evidence["job_id"] == "10901114"
    assert evidence["state"] == "FAILED"
    assert evidence["exit_code"] == "1:0"
    assert evidence["completed_stage_count"] == 4
    assert evidence["failed_stage"] == "current_checkpoints"
    assert evidence["pressure_smoke_contract_correction_passed"] is True
    assert evidence["qualification_complete"] is False
    assert evidence["automatic_retry"] is False
    assert evidence["second_job_submitted"] is False


def test_record_registers_exact_forward_successors_and_preserves_history():
    record = _record()
    for relative, value in record["historical_records"].items():
        assert value["rewritten"] is False
        assert _sha256(ROOT / relative) == value["sha256"]
    successors = record["historical_successor_sources"]
    assert set(successors) == {
        "benchmarks/run_rc6_current_checkpoints.py",
        "benchmarks/build_rc6_single_h100_stage_plan.py",
        "tests/test_rc6_h100_stage_producers.py",
        "tests/test_rc6_h100_stage_producer_record.py",
    }
    for relative, registration in successors.items():
        assert _current_sha256(
            relative,
            registration["successor_sha256"],
        ) == _sha256(ROOT / relative)
        assert registration["predecessor_sha256"] != registration["successor_sha256"]


def test_record_binds_sources_and_the_node_local_policy():
    record = _record()
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == _current_sha256(relative, expected)
    correction = record["correction"]
    assert correction["checkpoint_work_root_policy"] == "node_local_tmpdir"
    assert correction["scratch_contains_transient_checkpoint_cells"] is False
    assert correction["scratch_contains_final_json_report"] is True
    assert correction["temporary_root_removed_after_stage"] is True
    helper = (ROOT / "benchmarks/run_rc6_current_checkpoints.py").read_text(
        encoding="utf-8",
    )
    assert 'CHECKPOINT_WORK_ROOT_POLICY = "node_local_tmpdir"' in helper
    assert "tempfile.TemporaryDirectory" in helper


def test_record_is_qualification_only_and_has_no_push_or_h100_authority():
    record = _record()
    assert record["classification"] == (
        "READY_RC6_CHECKPOINT_WORK_ROOT_CORRECTION_H100_NOT_AUTHORIZED"
    )
    scope = record["scope"]
    assert scope["qualification_support_only"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key != "qualification_support_only"
    )
    assert all(value is False for value in record["authorization"].values())
