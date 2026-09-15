"""Synthetic tests for bounded Stage P operator/kernel diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from pssolver.experimental.stage_p_diagnostics import (
    analyze_stage_p_diagnostics,
    summarize_operator_kernel_events,
)


CLOSURE_SHA256 = "c" * 64


@dataclass
class _Event:
    name: str
    device_type: str
    count: int = 1
    device_time_total: float = 0.0


@dataclass
class _Average:
    key: str
    count: int
    self_cpu_time_total: float
    device_time_total: float
    self_device_time_total: float
    device_type: str = "cpu"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_operator_kernel_summary_is_bounded_complete_and_per_step():
    report = summarize_operator_kernel_events(
        (
            _Event("fft_kernel", "cuda", device_time_total=90.0),
            _Event("fft_kernel", "cuda", device_time_total=60.0),
            _Event("pointwise_kernel", "cuda", device_time_total=30.0),
        ),
        (
            _Average("aten::_fft", 6, 12.0, 150.0, 150.0),
            _Average("cudaLaunchKernel", 9, 4.0, 0.0, 0.0),
            _Average("Torch-Compiled Region", 3, 2.0, 30.0, 30.0),
        ),
        steps=3,
    )
    assert report["bounded_aggregate_only"] is True
    assert report["raw_trace_retained"] is False
    assert report["operators"]["distinct_names"] == 3
    assert report["kernels"]["distinct_names"] == 2
    assert report["kernels"]["total_launches_per_step"] == pytest.approx(1.0)
    assert report["kernels"]["total_device_microseconds_per_step"] == (
        pytest.approx(60.0)
    )
    assert report["cuda_runtime"]["calls_per_step"] == pytest.approx(3.0)
    assert report["compiler_markers"]["calls_per_step"] == pytest.approx(1.0)


def test_operator_kernel_summary_rejects_unbounded_or_invalid_inputs():
    with pytest.raises(ValueError, match="positive integer"):
        summarize_operator_kernel_events((), (), steps=0)
    events = tuple(_Event(f"kernel_{index}", "cuda") for index in range(3))
    with pytest.raises(RuntimeError, match="bounded event limit"):
        summarize_operator_kernel_events(
            events,
            (),
            steps=1,
            maximum_distinct_events=2,
        )


def _profile(role: str, *, trial: int) -> dict[str, object]:
    is_canary = role == "separated_canary"
    timestep = (0.050 if not is_canary else 0.070) + trial * 0.0001
    launches = 30.0 if not is_canary else 45.0
    runtime_calls = 40.0 if not is_canary else 58.0
    compiled_calls = 3.0 if not is_canary else 5.0
    transform_ms = 18.0 if not is_canary else 25.0
    kernel_time = 1200.0 if not is_canary else 1900.0
    operator_time = 600.0 if not is_canary else 950.0
    return {
        "schema_version": 1,
        "qualification_stage": "P",
        "classification": "OPERATOR_KERNEL_PROFILE_COMPLETE",
        "runtime_role": role,
        "stage_o_closure_sha256": CLOSURE_SHA256,
        "production_metadata_sha256": "a" * 64,
        "production_initial_q_sha256": "b" * 64,
        "configuration": {
            "shape": [320, 320, 80],
            "lengths": [100.0, 100.0, 20.0],
            "dtype": "float64",
            "dt": 0.005,
            "dealias_rule": "cubic_half",
            "projected_transform_execution": "truncated",
            "spectral_storage": "hermitian_half",
            "transform_execution_order": "real_first",
            "pointwise_execution": "compile",
            "spectral_refresh_interval": 2,
            "warmup_steps": 10,
            "throughput_steps": 20,
            "operator_audit_steps": 3,
            "semantic_steps": 3,
        },
        "throughput": {"mean_timestep_seconds": timestep},
        "operator_kernel_audit": {
            "bounded_aggregate_only": True,
            "operator_entries_complete": True,
            "kernel_entries_complete": True,
            "raw_trace_retained": False,
            "operators": {
                "entries": [
                    {
                        "name": "aten::example",
                        "category": "other",
                        "calls_per_step": launches,
                        "device_microseconds_per_step": operator_time,
                    }
                ]
            },
            "kernels": {
                "total_launches_per_step": launches,
                "entries": [
                    {
                        "name": "example_kernel",
                        "category": "other",
                        "calls_per_step": launches,
                        "device_microseconds_per_step": kernel_time,
                    }
                ],
            },
            "cuda_runtime": {"calls_per_step": runtime_calls},
            "compiler_markers": {"calls_per_step": compiled_calls},
        },
        "dynamo_delta": {"graph_breaks": 0},
        "matched_semantic_regions": {
            "transform_total_milliseconds_per_step": transform_ms,
        },
        "measurement_contract": {
            "throughput_operator_and_semantic_windows_are_separate": True,
            "throughput_uses_internal_semantic_instrumentation": False,
            "raw_trace_retained": False,
            "changes_equations": False,
            "changes_runtime_implementation": False,
            "changes_production_default": False,
        },
        "finite": True,
        "eligible_for_stage_q_optimization_candidate": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
    }


def _profiles(tmp_path: Path) -> tuple[list[Path], list[Path]]:
    result = []
    for role in ("legacy_production", "separated_canary"):
        paths = []
        for trial in range(1, 4):
            path = tmp_path / f"{role}_{trial}.json"
            _write_json(path, _profile(role, trial=trial))
            paths.append(path)
        result.append(paths)
    return result[0], result[1]


def test_analysis_attributes_gap_without_selecting_an_optimization(tmp_path):
    production, canary = _profiles(tmp_path)
    report = analyze_stage_p_diagnostics(
        production,
        canary,
        expected_stage_o_closure_sha256=CLOSURE_SHA256,
    )
    assert report["classification"] == "DIAGNOSTIC_COMPLETE"
    assert report["throughput"]["noninstrumented_window"] is True
    assert report["throughput"]["canary_over_production_ratio"] > 1.0
    assert report["gap_signals"]["kernel_launches"]["delta_per_step"] == (
        pytest.approx(15.0)
    )
    assert report["gap_signals"]["cuda_runtime_calls"]["delta_per_step"] == (
        pytest.approx(18.0)
    )
    assert report["positive_operator_device_time_deltas"][0]["name"] == (
        "aten::example"
    )
    assert report["positive_kernel_device_time_deltas"][0]["name"] == (
        "example_kernel"
    )
    assert report["eligible_for_stage_q_target_review"] is True
    assert report["eligible_for_stage_q_optimization_candidate"] is False
    assert report["eligible_for_production_promotion"] is False
    assert report["production_default_changed"] is False
    assert len(report["inputs"]) == 6


def test_analysis_rejects_duplicate_paths_or_identity_drift(tmp_path):
    production, canary = _profiles(tmp_path)
    with pytest.raises(ValueError, match="six distinct"):
        analyze_stage_p_diagnostics(
            [production[0]] * 3,
            canary,
            expected_stage_o_closure_sha256=CLOSURE_SHA256,
        )
    changed = json.loads(canary[0].read_text(encoding="utf-8"))
    changed["production_initial_q_sha256"] = "d" * 64
    _write_json(canary[0], changed)
    with pytest.raises(ValueError, match="identities differ"):
        analyze_stage_p_diagnostics(
            production,
            canary,
            expected_stage_o_closure_sha256=CLOSURE_SHA256,
        )


def test_analysis_rejects_graph_breaks_and_nonfinite_metrics(tmp_path):
    production, canary = _profiles(tmp_path)
    changed = json.loads(canary[0].read_text(encoding="utf-8"))
    changed["dynamo_delta"]["graph_breaks"] = 1
    _write_json(canary[0], changed)
    with pytest.raises(ValueError, match="graph break"):
        analyze_stage_p_diagnostics(
            production,
            canary,
            expected_stage_o_closure_sha256=CLOSURE_SHA256,
        )
