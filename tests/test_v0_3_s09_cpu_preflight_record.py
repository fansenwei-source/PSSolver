from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s09_cpu_preflight.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s09_cpu_preflight_uses_the_frozen_r128_contract():
    record = _record()
    assert record["plan_step"] == "S-09"
    assert record["baseline"]["commit"] == (
        "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
    )
    assert record["candidate"]["runtime_commit"] == (
        "f94b9be22fd767fee432caf30f0e5c47aad8e9c3"
    )
    contract = record["contract"]
    assert contract["shape"] == [128, 20, 20]
    assert contract["lengths"] == [40.0, 6.0, 6.0]
    assert contract["paired_trials"] == 5
    assert contract["measured_steps_per_trial"] == 30
    assert contract["threshold_relaxation_allowed"] is False


def test_s09_cpu_preflight_records_every_family_and_mode():
    record = _record()
    results = record["results"]
    assert len(results) == 8
    assert {(item["family"], item["mode"]) for item in results} == {
        (family, mode)
        for family in record["contract"]["families"]
        for mode in record["contract"]["modes"]
    }
    assert all(len(item["paired_time_ratios"]) == 5 for item in results)


def test_s09_cpu_preflight_fails_closed_before_h100():
    record = _record()
    assert record["classification"] == (
        "STOPPED_S09_AT_S06_CPU_PERFORMANCE_PREFLIGHT"
    )
    assert record["qualification_complete"] is False
    assert record["eligible_for_r0022"] is False
    assert record["eligible_for_h100_authorization"] is False
    assert record["verdict"] == {
        "timing_passed": False,
        "memory_passed": True,
        "stop_required": True,
        "thresholds_relaxed": False,
        "h100_preparation_continued": False,
    }
    assert record["authorization"]["h100_submitted"] is False


def test_s09_cpu_preflight_binds_the_versioned_runner():
    assert _sha256(ROOT / "benchmarks/benchmark_v0_3_s09_channel_cpu.py") == (
        _record()["evidence"]["runner_sha256"]
    )
