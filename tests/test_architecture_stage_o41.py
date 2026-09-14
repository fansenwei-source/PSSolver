"""CPU tests for the prospective Stage O.4.1 qualification contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from pssolver.diagnostics import cuda_memory_snapshot, cuda_memory_window
from pssolver.experimental import (
    analyze_stage_o41_qualification,
    build_stage_o41_h100_plan,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _o4_report() -> dict[str, object]:
    return {
        "qualification_stage": "O.4",
        "classification": "B_neutral",
        "eligible_for_stage_o5_decision": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "gate_failures": [
            "transform_count_identity",
            "memory_non_regression",
        ],
        "gates": {
            "six_step_numerical": True,
            "hundred_step_numerical": True,
            "legacy_same_backend_restart_exact": True,
            "canary_same_backend_restart_exact": True,
            "initial_q_identity": True,
            "transform_count_identity": False,
            "canary_lifecycle_consistency": True,
            "performance_non_regression": True,
            "memory_non_regression": False,
        },
        "trajectory": {
            "maximum_gate_relative_l2": 3.2e-15,
            "relative_l2_tolerance": 1.0e-10,
        },
    }


def _window(
    *,
    start: int,
    peak: int,
    end: int | None = None,
) -> dict[str, int]:
    if end is None:
        end = start
    return {
        "start_allocated_bytes": start,
        "start_reserved_bytes": 2 * start,
        "end_allocated_bytes": end,
        "end_reserved_bytes": 2 * start,
        "peak_allocated_bytes": peak,
        "peak_reserved_bytes": 2 * peak,
        "retained_allocated_growth_bytes": max(0, end - start),
        "retained_reserved_growth_bytes": 0,
    }


def _attribution(*, base: int, peak: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "allocator_scope": "current_process_cuda_allocator",
        "peak_reset_after_warmup": True,
        "observation_peak_reset_after_timestep_window": True,
        "phases": {},
        "windows": {
            "timestep": _window(start=base, peak=peak),
            "observation": _window(start=base, peak=peak),
        },
    }


def _configuration(shape: tuple[int, int, int]) -> dict[str, object]:
    return {
        "shape": list(shape),
        "lengths": [100.0, 100.0, 20.0],
        "device": "cuda",
        "dtype": "float64",
        "dt": 0.005,
        "dealias_rule": "cubic_half",
        "projected_transform_execution": "truncated",
        "spectral_storage": "hermitian_half",
        "spectral_refresh_interval": 2,
        "warmup_steps": 10,
        "profile_steps": 20,
        "timing_scope": "whole_timestep",
    }


def _environment() -> dict[str, object]:
    return {
        "cuda_available": True,
        "device_name": "NVIDIA H100 PCIe",
        "cuda_matmul_allow_tf32": False,
    }


def _materialization() -> dict[str, int]:
    return {
        "physical_materializations": 29,
        "on_demand_physical_materializations": 0,
        "physical_materialization_batches": 4,
        "batched_physical_components": 29,
        "singleton_materialization_batches": 0,
        "maximum_materialization_batch_size": 15,
        "physical_island_prefetches": 7,
        "physical_island_requested_components": 54,
        "unmaterialized_published_components": 25,
    }


def _legacy_profile(
    shape: tuple[int, int, int],
    *,
    mean: float,
    base: int,
    peak: int,
    q_sha256: str,
) -> dict[str, object]:
    return {
        "config": _configuration(shape),
        "environment": _environment(),
        "throughput": {"mean_timestep_seconds": mean},
        "memory_attribution": _attribution(base=base, peak=peak),
        "timings": {
            "transform_forward": {"calls": 150},
            "transform_inverse": {"calls": 650},
        },
        "profile_input": {"initial_q_sha256": q_sha256},
    }


def _canary_profile(
    shape: tuple[int, int, int],
    *,
    mean: float,
    base: int,
    peak: int,
    q_sha256: str,
    forward: float = 6.5,
    inverse: float = 11.5,
) -> dict[str, object]:
    return {
        "qualification_stage": "N.4",
        "classification": "PROFILE_COMPLETE",
        "mode": "candidate",
        "finite": True,
        "configuration": _configuration(shape),
        "configuration_authority": "unified_execution_policy",
        "algebraic_execution_policy": {"mode": "batched_physical_islands"},
        "environment": _environment(),
        "throughput": {"mean_timestep_seconds": mean},
        "memory_attribution": _attribution(base=base, peak=peak),
        "production_initial_q_sha256": q_sha256,
        "transform_call_audit": {
            "forward_calls_per_step": forward,
            "inverse_calls_per_step": inverse,
            "representation_reuse": {"retained_pairs_after_generation": 0},
            "physical_materialization": _materialization(),
        },
    }


def _inputs(tmp_path: Path, *, r320_canary_peak: int = 1020, forward=6.5):
    o4 = _write_json(tmp_path / "o4.json", _o4_report())
    result = {"o4": o4}
    for label, shape, legacy_base, legacy_peak, canary_base, canary_peak in (
        ("r128", (128, 128, 32), 1000, 1200, 1500, 1800),
        ("r320", (320, 320, 80), 900, 1000, 910, r320_canary_peak),
    ):
        q_sha256 = ("a" if label == "r128" else "b") * 64
        result[f"legacy_{label}"] = [
            _write_json(
                tmp_path / f"legacy_{label}_{trial}.json",
                _legacy_profile(
                    shape,
                    mean=0.0100,
                    base=legacy_base,
                    peak=legacy_peak,
                    q_sha256=q_sha256,
                ),
            )
            for trial in range(1, 4)
        ]
        result[f"canary_{label}"] = [
            _write_json(
                tmp_path / f"canary_{label}_{trial}.json",
                _canary_profile(
                    shape,
                    mean=0.0101,
                    base=canary_base,
                    peak=canary_peak,
                    q_sha256=q_sha256,
                    forward=forward,
                ),
            )
            for trial in range(1, 4)
        ]
    return result


def _analyze(inputs):
    return analyze_stage_o41_qualification(
        inputs["o4"],
        inputs["legacy_r128"],
        inputs["canary_r128"],
        inputs["legacy_r320"],
        inputs["canary_r320"],
        expected_stage_o4_sha256=_sha256(inputs["o4"]),
    )


def test_cuda_memory_helpers_have_a_complete_cpu_schema():
    snapshot = cuda_memory_snapshot("cpu")
    assert snapshot == {
        "allocated_bytes": None,
        "reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    assert all(
        value is None
        for value in cuda_memory_window(snapshot, snapshot, snapshot).values()
    )


def test_cuda_memory_window_rejects_an_impossible_peak():
    start = {
        "allocated_bytes": 10,
        "reserved_bytes": 20,
    }
    end = {"allocated_bytes": 12, "reserved_bytes": 20}
    peak = {"peak_allocated_bytes": 11, "peak_reserved_bytes": 20}
    with pytest.raises(ValueError, match="peak allocated"):
        cuda_memory_window(start, end, peak)


def test_o41_accepts_architecture_aware_counts_and_r320_memory(tmp_path):
    report = _analyze(_inputs(tmp_path))

    assert report["classification"] == "A_recommended"
    assert report["eligible_for_stage_o5_decision"] is True
    assert report["stage_o4_original_classification_unchanged"] is True
    assert report["r128_diagnostic"]["comparison"][
        "memory_ratios_canary_over_legacy"
    ]["timestep_peak_allocated"] == 1.5
    assert report["r320_decision"]["gates"][
        "transform_nonincrease_with_strict_reduction"
    ] is True
    assert report["gates"]["r320_phase_aware_memory_non_regression"] is True


def test_o41_rejects_r320_memory_regression(tmp_path):
    report = _analyze(_inputs(tmp_path, r320_canary_peak=1040))

    assert report["classification"] == "B_neutral"
    assert report["eligible_for_stage_o5_decision"] is False
    assert report["gate_failures"] == [
        "r320_phase_aware_memory_non_regression"
    ]


def test_o41_rejects_canary_transform_increase(tmp_path):
    report = _analyze(_inputs(tmp_path, forward=8.0))

    assert report["classification"] == "B_neutral"
    assert report["gates"][
        "architecture_aware_transform_and_lifecycle"
    ] is False


def test_o41_does_not_reinterpret_unexpected_o4_failures(tmp_path):
    inputs = _inputs(tmp_path)
    value = json.loads(inputs["o4"].read_text())
    value["gates"]["performance_non_regression"] = False
    inputs["o4"].write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen Stage O.4"):
        _analyze(inputs)


def test_o41_planner_is_read_only_and_balances_two_scales(tmp_path):
    o4 = _write_json(tmp_path / "o4.json", _o4_report())
    control = tmp_path / "new-control"
    scratch = tmp_path / "new-scratch"
    plan = build_stage_o41_h100_plan(
        project_root=PROJECT_ROOT,
        control_root=control,
        scratch_root=scratch,
        python=sys.executable,
        expected_commit="c" * 40,
        stage_o4_report=o4,
        expected_stage_o4_sha256=_sha256(o4),
    )

    assert plan["qualification_stage"] == "O.4.1"
    assert plan["runtime_default_may_change"] is False
    assert plan["fixed_gates"]["diagnostic_shape"] == [128, 128, 32]
    assert plan["fixed_gates"]["decision_shape"] == [320, 320, 80]
    assert len(plan["commands"]["references"]) == 2
    assert len(plan["commands"]["profiles_balanced"]) == 12
    assert plan["commands"]["analysis"][-1].endswith(
        "stage_o41_qualification.json"
    )
    assert not control.exists()
    assert not scratch.exists()
