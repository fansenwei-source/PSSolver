"""Bindings for the stopped S-09 CPU diagnosis and optimization candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes/PSSolver_v0_3_0_s03_successor_28_s09_cpu_diagnosis.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s09_cpu_diagnosis_stops_before_r0022_and_h100():
    record = _record()
    assert record["roadmap"]["plan_step"] == "S-09"
    assert record["classification"] == (
        "STOPPED_S09_AFTER_OPTIMIZATION_AT_BOUND_FIXED_WORK_TIMING_SAMPLE"
    )
    assert record["roadmap"]["request_created"] is False
    assert record["roadmap"]["h100_preparation_allowed"] is False
    assert record["authorization"]["r0022_created"] is False
    assert record["authorization"]["h100_submitted"] is False


def test_s09_cpu_noise_rule_was_not_silently_changed():
    record = _record()
    controls = record["noise_controls"]
    assert controls["v0_2_2_self_control_runs"] == 2
    assert controls["single_ratio_above_1_05"] is False
    assert controls["six_trial_sixty_step_contract_activated"] is False
    assert record["measurement_contract"]["paired_trials"] == 5
    assert record["measurement_contract"]["measured_steps"] == 30
    assert record["measurement_contract"]["thresholds_relaxed"] is False


def test_s09_bound_failure_is_preserved_without_sample_deletion():
    record = _record()
    final = record["final_bound_matrix"]
    assert final["passed"] is False
    assert final["failed_cells"] == 1
    assert final["failed_family"] == "channel_legacy"
    assert final["failed_mode"] == "fixed_work"
    assert final["failed_paired_ratio"] > 1.05
    assert final["optimization_path_active_for_failure"] is False
    assert record["measurement_contract"]["all_samples_retained"] is True


def test_s09_optimization_sources_are_content_addressed():
    record = _record()
    for relative, expected in record["runtime_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative,
            expected,
        )
    for relative, successor in record[
        "historical_successor_sources"
    ].items():
        assert successor["current_sha256"] != successor["historical_sha256"]
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative,
            successor["historical_sha256"],
        )


def test_s09_evidence_binds_every_measured_matrix():
    evidence = _record()["evidence"]
    archive = Path(evidence["archive"])
    assert (archive / "COMPLETE").is_file()
    assert evidence["runner_sha256"] == _sha256(
        ROOT / "benchmarks/benchmark_v0_3_s09_channel_cpu.py"
    )
    names = {
        "self_control_first_matrix_sha256": "s09_diagnosis_v022_self_5x30",
        "self_control_second_matrix_sha256": (
            "s09_diagnosis_v022_self_repeat_5x30"
        ),
        "s06_before_after_matrix_sha256": (
            "s09_diagnosis_s06_pre_post_5x30"
        ),
        "causal_matrix_sha256": "s09_diagnosis_c105_cost_5x30",
        "optimized_exploratory_matrix_sha256": (
            "s09_diagnosis_optimized_final_5x30"
        ),
        "optimized_bound_matrix_sha256": (
            "s09_diagnosis_optimized_bound_2573dec_5x30"
        ),
    }
    for field, directory in names.items():
        assert evidence[field] == _sha256(
            archive / "results" / directory / "matrix.json"
        )
