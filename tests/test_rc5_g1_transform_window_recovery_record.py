"""Evidence contract for the rc5 G-1 transform-window recovery v6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_transform_window_recovery_v6.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_transform_window_record_binds_exact_raw_refresh_steady_contract():
    record = _record()
    assert record["classification"] == (
        "READY_RC5_G1_TRANSFORM_MEASUREMENT_WINDOW_ANALYSIS_ONLY_RECOVERY_V6"
    )
    contract = record["measurement_window_contract"]
    assert contract["profile_count"] == 96
    assert contract["scheduled_refresh_families"] == [
        "channel_compiled_v2",
        "channel_legacy",
    ]
    assert contract["profiles_with_scheduled_refreshes"] == 24
    assert contract["channel_raw_forward_calls"] == 663
    assert contract["channel_steady_forward_calls"] == 660
    assert contract["channel_steady_forward_per_step"] == 11.0
    assert contract["channel_raw_inverse_calls"] == 2160
    assert contract["source_profiles_are_immutable"] is True


def test_transform_window_record_registers_exact_forward_only_sources():
    record = _record()
    successors = record["historical_successor_sources"]
    assert set(successors) == {
        "benchmarks/analyze_rc5_g1_h100.py",
        "benchmarks/plan_rc5_g1_h100.py",
        "benchmarks/run_rc5_g1_stage_plan.py",
        "tests/test_analyze_rc5_g1_h100.py",
        "tests/test_plan_rc5_g1_h100.py",
        "tests/test_rc5_g1_consumer_recovery_record.py",
        "tests/test_rc5_g1_hpcc_plan_freeze_record.py",
        "tests/test_rc5_g1_qualification_support_record.py",
    }
    for relative, value in successors.items():
        assert _sha256(ROOT / relative) == value["current_sha256"]
        assert value["accepted_predecessor_sha256"]
        assert value["production_runtime_changed"] is False
        assert value["scientific_threshold_changed"] is False


def test_transform_window_record_preserves_analysis_only_authorization():
    record = _record()
    scope = record["execution_scope"]
    assert scope["analysis_only"] is True
    assert scope["h100_submission_count"] == 0
    assert scope["h100_submission_authorized"] is False
    assert scope["source_profile_mutation_authorized"] is False
    assert record["qualification_complete"] is False
    assert not any(record["authorization"].values())
