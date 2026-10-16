"""Evidence contract for the G-1 corrected-window lifecycle recovery v10."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_measurement_window_lifecycle_recovery_v10.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_lifecycle_record_binds_the_v9_failure_without_reclassifying_it():
    record = _record()
    failure = record["source_failure"]
    assert failure["job_id"] == "10885954"
    assert failure["classification"] == (
        "FAIL_RC5_G1_MEASUREMENT_WINDOW_RECOVERY_V9_QUALIFICATION_HARNESS_LIFECYCLE"
    )
    assert failure["exception_type"] == "UnboundLocalError"
    assert failure["profile_json_count"] == 0
    assert failure["scientific_failure"] is False
    assert failure["solver_failure"] is False
    predecessor = record["predecessor"]
    assert _sha256(ROOT / predecessor["implementation_record"]) == predecessor[
        "implementation_record_sha256"
    ]
    assert predecessor["record_rewritten"] is False


def test_lifecycle_record_registers_exact_forward_only_successors():
    record = _record()
    successors = record["historical_successor_sources"]
    assert set(successors) == {
        "benchmarks/profile_rc458_runtime_family.py",
        "tests/test_rc5_g1_measurement_window_implementation.py",
        "tests/test_rc5_g1_measurement_window_implementation_record.py",
        "tests/test_rc458_qualification_support.py",
    }
    for relative, registration in successors.items():
        assert registration["accepted_predecessor_sha256"]
        assert _sha256(ROOT / relative) == registration["current_sha256"]
        assert registration["production_runtime_changed"] is False
        assert registration["scientific_threshold_changed"] is False


def test_lifecycle_record_preserves_the_h100_authorization_boundary():
    record = _record()
    assert record["classification"] == "READY_FOR_CPU_PREFLIGHT_NO_H100"
    assert record["fix"] == {
        "scope": "qualification_only_profiler_object_lifecycle",
        "configuration_identity_snapshotted_before_case_deletion": True,
        "formal_runtime_still_released_before_diagnostic_replay": True,
        "result_uses_persisted_configuration_identity": True,
        "production_solver_modified": False,
        "production_runtime_modified": False,
        "scientific_equations_modified": False,
        "thresholds_modified": False,
        "measurement_window_modified": False,
    }
    assert record["cpu_verification"]["h100_executed"] is False
    assert record["formal_state"]["qualification_complete"] is False
    assert record["formal_state"]["g1_complete"] is False
    assert record["authorization"]["h100_submission_authorized"] is False
    assert record["authorization"]["automatic_retry_authorized"] is False
