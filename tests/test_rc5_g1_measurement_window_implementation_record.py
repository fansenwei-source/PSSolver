"""Evidence contract for the rc5 G-1 measurement-window implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_measurement_window_implementation_v9.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_record_binds_every_versioned_implementation_source():
    record = _record()
    assert record["classification"] == "READY_FOR_CPU_PREFLIGHT_NO_H100"
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected


def test_record_registers_exact_forward_only_historical_successors():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "benchmarks/profile_rc458_runtime_family.py",
        "benchmarks/analyze_rc5_g1_focused_performance_h100.py",
        "benchmarks/analyze_rc5_g1_timing_variability.py",
        "benchmarks/run_rc5_g1_focused_profile_matrix.py",
        "tests/test_rc5_g1_timing_variability_analysis.py",
        "tests/test_rc458_qualification_support.py",
    }
    for relative, registration in successors.items():
        assert registration["accepted_predecessor_sha256"]
        assert _sha256(ROOT / relative) == registration["current_sha256"]
        assert registration["production_runtime_changed"] is False
        assert registration["scientific_threshold_changed"] is False


def test_record_preserves_pre_registration_and_h100_boundary():
    record = _record()
    preregistration = ROOT / record["pre_registration"]["record"]
    assert _sha256(preregistration) == record["pre_registration"]["record_sha256"]
    assert record["pre_registration"]["contract_changed"] is False
    assert record["pre_registration"]["threshold_changed"] is False
    assert record["implementation"]["historical_profiler_default_post_flush_rewarm_steps"] == 0
    assert record["implementation"]["corrected_plan_post_flush_rewarm_steps"] == 3
    assert record["implementation"]["diagnostic_replay_used_for_adjudication"] is False
    assert record["formal_state"]["qualification_complete"] is False
    assert record["formal_state"]["g1_complete"] is False
    assert record["authorization"]["h100_submission_authorized"] is False
    assert record["authorization"]["automatic_retry_authorized"] is False
