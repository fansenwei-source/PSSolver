from __future__ import annotations

import json
import hashlib
from pathlib import Path

from benchmarks.diagnose_v0_3_s09_functional_transient import analyze


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s09_transient_telemetry_preregistration.json"


def _profile(role: str, trial: int, *, wall: float, cpu: float, nivcsw: int = 0):
    sample = {
        "wall_seconds": wall,
        "process_cpu_seconds": cpu,
        "frequency_before_khz": 5_000_000,
        "frequency_after_khz": 5_000_000,
    }
    return {
        "config": {"role": role, "trial": trial},
        "samples": [sample] * 60,
        "summary": {
            "wall_center_seconds": wall,
            "cpu_center_seconds": cpu,
            "wall_block_means_seconds": [wall] * 6,
            "cpu_block_means_seconds": [cpu] * 6,
            "voluntary_context_switches": 0,
            "involuntary_context_switches": nivcsw,
            "minor_faults": 0,
            "major_faults": 0,
        },
    }


def _matrix(candidate_wall: float, candidate_cpu: float, *, nivcsw: int = 0):
    values = []
    for trial in range(1, 7):
        values.append(_profile("baseline", trial, wall=1.0, cpu=0.99))
        values.append(
            _profile(
                "candidate",
                trial,
                wall=candidate_wall if trial == 1 else 1.01,
                cpu=candidate_cpu if trial == 1 else 1.0,
                nivcsw=nivcsw if trial == 1 else 0,
            )
        )
    return values


def test_preregistration_freezes_one_shot_scope_and_h100_contract():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["decision"] == {
        "id": "Q-S09-PERF-03",
        "selected": "A",
        "one_execution_only": True,
    }
    assert record["scope"]["cpu_affinity"] == [8]
    assert record["scope"]["all_samples_retained"] is True
    assert record["idle_gate"]["control_research_computation_processes"] == 0
    assert record["formal_h100_contract"] == {
        "median_time_ratio_max": 1.03,
        "paired_time_ratio_max": 1.05,
        "all_samples_retained": True,
        "changed": False,
    }
    runner = record["runner"]
    assert hashlib.sha256((ROOT / runner["path"]).read_bytes()).hexdigest() == runner["sha256"]


def test_wall_only_slowdown_is_environmental_interference():
    result = analyze(_matrix(1.08, 1.01), 1.016)
    assert result["classification"] == "ENVIRONMENTAL_INTERFERENCE_SUPPORTED"
    assert result["cpu_preflight_reference_estimate_passed"] is True
    assert result["candidate_fix_required_before_h100"] is False


def test_unexplained_cpu_slowdown_stops_before_h100():
    result = analyze(_matrix(1.08, 1.08), 1.016)
    assert result["classification"] == "CAUSE_UNRESOLVED"
    assert result["cpu_preflight_reference_estimate_passed"] is False
    assert result["candidate_fix_required_before_h100"] is True


def test_no_paired_limit_exceedance_is_reference_pass():
    result = analyze(_matrix(1.02, 1.01), 1.016)
    assert result["classification"] == "NO_PAIRED_LIMIT_EXCEEDANCE"
    assert result["cpu_preflight_reference_estimate_passed"] is True
