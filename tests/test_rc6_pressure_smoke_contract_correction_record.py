"""Bind the RC6 pressure-smoke correction and its qualification-only scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_pressure_smoke_contract_correction.json"
CHECKPOINT_SUCCESSOR_RECORD = (
    ROOT / "notes/PSSolver_v0_2_0rc6_checkpoint_work_root_correction.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_binds_the_completed_diagnostic_without_claiming_qualification():
    evidence = _record()["diagnostic_evidence"]
    assert evidence["job_id"] == "10901090"
    assert evidence["state"] == "COMPLETED"
    assert evidence["exit_code"] == "0:0"
    assert evidence["analysis_complete"] is True
    assert evidence["qualification_complete"] is False
    assert evidence["formal_h100_submission_count"] == 1
    assert evidence["automatic_retry"] is False


def test_record_preserves_history_and_registers_the_exact_helper_successor():
    record = _record()
    for relative, value in record["historical_records"].items():
        assert value["rewritten"] is False
        assert _sha256(ROOT / relative) == value["sha256"]
    successors = record["historical_successor_sources"]
    assert set(successors) == {
        "benchmarks/run_rc6_contract_smokes.py",
        "tests/test_rc6_h100_stage_producer_record.py",
    }
    successor = successors["benchmarks/run_rc6_contract_smokes.py"]
    assert successor["predecessor_sha256"] == (
        json.loads(
            (ROOT / "notes/PSSolver_v0_2_0rc6_h100_stage_producers.json").read_text(
                encoding="utf-8",
            )
        )["source_sha256"]["benchmarks/run_rc6_contract_smokes.py"]
    )
    assert successor["successor_sha256"] == _sha256(
        ROOT / "benchmarks/run_rc6_contract_smokes.py"
    )


def test_record_binds_exact_sources_and_authoritative_label():
    record = _record()
    successors = json.loads(
        CHECKPOINT_SUCCESSOR_RECORD.read_text(encoding="utf-8")
    )["historical_successor_sources"]
    for relative, expected in record["source_sha256"].items():
        actual = _sha256(ROOT / relative)
        if actual != expected:
            registration = successors.get(relative)
            assert registration is not None
            assert registration["predecessor_sha256"] == expected
            assert registration["successor_sha256"] == actual
    correction = record["correction"]
    assert correction["old_value"] == "fixed_iterations_completed"
    assert correction["new_value"] == "fixed_work_completed"
    source = (ROOT / correction["file"]).read_text(encoding="utf-8")
    assert 'FIXED_WORK_TERMINATION_REASON = "fixed_work_completed"' in source
    assert "fixed_iterations_completed" not in source


def test_record_is_support_only_and_has_no_push_or_h100_authority():
    record = _record()
    assert record["classification"] == (
        "READY_RC6_PRESSURE_SMOKE_CONTRACT_CORRECTION_H100_NOT_AUTHORIZED"
    )
    scope = record["scope"]
    assert scope["qualification_support_only"] is True
    assert all(
        value is False
        for key, value in scope.items()
        if key != "qualification_support_only"
    )
    assert all(value is False for value in record["authorization"].values())
