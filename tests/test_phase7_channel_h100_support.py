"""Local and synthetic gates for the P7.6 H100 qualification support."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.profile_channel_runtime_timestep import (
    ChannelRuntimeProfileConfig,
    _scheduled_refreshes,
    run_profile,
)
from benchmarks.run_channel_runtime_trajectory import main as run_trajectory
from scripts_channel.analyze_phase7_channel_qualification import analyze_profiles
from scripts_channel.compare_channel_runtime_outputs import compare_channel_outputs


COMMIT = "1" * 40


@pytest.mark.parametrize(
    ("start", "interval", "steps", "expected"),
    ((10, 20, 50, 3), (0, 20, 20, 1), (19, 20, 1, 1), (7, None, 50, 0)),
)
def test_scheduled_refresh_count_is_phase_aware(start, interval, steps, expected):
    assert _scheduled_refreshes(
        start_step_count=start,
        interval=interval,
        steps=steps,
    ) == expected


def test_cpu_profiler_preserves_state_pcg_and_transform_identity():
    results = []
    for runtime in ("legacy_channel", "compiled_channel_v2"):
        results.append(run_profile(ChannelRuntimeProfileConfig(
            runtime_path=runtime, shape=(8, 6, 5), lengths=(8.0, 6.0, 5.0),
            warmup_steps=1, profile_steps=2,
        )))
    left, right = results
    assert left["initial_q_sha256"] == right["initial_q_sha256"]
    assert left["final_state_sha256"] == right["final_state_sha256"]
    assert left["transform_calls"] == right["transform_calls"] == {
        "forward_total": 22,
        "inverse_total": 72,
        "forward_per_step": 11.0,
        "inverse_per_step": 36.0,
        "base_forward_per_step": 11.0,
        "scheduled_refresh_forward_calls": 0,
    }
    assert left["spectral_refresh"] == right["spectral_refresh"]
    assert left["pressure_iterations"] == right["pressure_iterations"]
    assert all(result["finite"] for result in results)


def _profile(path: Path, shape, trial, runtime, *, ratio=1.0):
    lengths = [32.0, 5.0, 5.0] if shape[0] == 128 else [128.0, 10.0, 10.0]
    value = {
        "config": {"runtime_path": runtime, "trial": trial, "shape": list(shape), "lengths": lengths, "device": "cuda", "dt": 0.01, "activity": 5.0, "warmup_steps": 10, "profile_steps": 50, "seed": 24},
        "runtime_identity": {"requested": runtime, "effective": runtime, "fallback_used": False},
        "environment": {"git": {"head": COMMIT, "dirty": False}, "cuda_available": True, "device_name": "NVIDIA H100 PCIe", "cuda_matmul_allow_tf32": False},
        "finite": True,
        "completed_steps": 60,
        "transform_calls": {
            "forward_total": 553,
            "inverse_total": 1800,
            "forward_per_step": 11.06,
            "inverse_per_step": 36.0,
            "base_forward_per_step": 11.0,
            "scheduled_refresh_forward_calls": 3,
        },
        "spectral_refresh": {
            "interval": 20,
            "step_count_before": 10,
            "step_count_after": 0,
            "count_before": 0,
            "count_after": 3,
            "expected_in_window": 3,
            "observed_in_window": 3,
            "dynamic_transform_group_count": 1,
        },
        "pressure_iterations": {"mean": 9.0, "maximum": 10, "values": [9] * 50},
        "throughput": {"mean_timestep_seconds": 0.1 * ratio},
        "memory": {"peak_allocated_bytes": int(1000 * ratio), "peak_reserved_bytes": int(2000 * ratio)},
        "initial_q_sha256": "a" * 64,
        "final_state_sha256": f"{shape[0]:064x}"[-64:],
    }
    path.write_text(json.dumps(value))
    return path


def _matrix(tmp_path, *, candidate_ratio=1.0):
    paths = []
    for shape in ((128, 20, 20), (512, 40, 40)):
        for trial in (1, 2, 3):
            paths.append(_profile(tmp_path / f"{shape[0]}_{trial}_legacy.json", shape, trial, "legacy_channel"))
            paths.append(_profile(tmp_path / f"{shape[0]}_{trial}_compiled.json", shape, trial, "compiled_channel_v2", ratio=candidate_ratio))
    return paths


def test_analyzer_accepts_complete_non_regressing_matrix(tmp_path):
    result = analyze_profiles(_matrix(tmp_path), expected_commit=COMMIT)
    assert result["classification"] == "PASS_P7_6_CHANNEL_H100_PROFILE_MATRIX"
    assert result["profile_count"] == 12
    assert all(all(summary["gates"].values()) for summary in result["grids"].values())
    assert all(
        summary["transform_contract"]["base_forward_per_step"] == 11.0
        and summary["transform_contract"]["raw_forward_per_step"] == 11.06
        and summary["transform_contract"]["scheduled_refreshes_in_window"] == 3
        for summary in result["grids"].values()
    )


@pytest.mark.parametrize("mutation", ("timing", "state", "refresh", "missing"))
def test_analyzer_rejects_failed_or_incomplete_evidence(tmp_path, mutation):
    paths = _matrix(tmp_path, candidate_ratio=1.0)
    if mutation == "missing":
        paths.pop()
    else:
        path = paths[1]
        value = json.loads(path.read_text())
        if mutation == "timing":
            value["throughput"]["mean_timestep_seconds"] = 0.2
        elif mutation == "refresh":
            value["spectral_refresh"]["observed_in_window"] = 2
        else:
            value["final_state_sha256"] = "f" * 64
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        analyze_profiles(paths, expected_commit=COMMIT)


def test_trajectory_cli_and_comparator_execute_both_paths(tmp_path):
    directories = []
    for runtime in ("legacy_channel", "compiled_channel_v2"):
        output = tmp_path / runtime
        assert run_trajectory([
            "--runtime-path", runtime, "--shape", "8,6,5",
            "--lengths", "8,6,5", "--steps", "1",
            "--save-interval", "1", "--diagnostic-interval", "1",
            "--device", "cpu", "--output", str(output),
        ]) == 0
        directories.append(output)
    result = compare_channel_outputs(directories[0], directories[1], step=1, role="R8_cross_1")
    assert result["classification"] == "PASS"
    assert result["array_count"] == 3


def test_frozen_p76_plan_binds_the_qualified_support_sources():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    plan = json.loads(
        (root / "notes/architecture_v0_2/phase_7_p76_h100_qualification_plan.json").read_text()
    )
    paths = {
        "profiler": "benchmarks/profile_channel_runtime_timestep.py",
        "trajectory_runner": "benchmarks/run_channel_runtime_trajectory.py",
        "analyzer": "scripts_channel/analyze_phase7_channel_qualification.py",
        "comparator": "scripts_channel/compare_channel_runtime_outputs.py",
    }
    assert plan["status"] == "READY_FOR_H100_RECOVERY_EXECUTION"
    for name, relative in paths.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == plan["support"][f"{name}_sha256"]
    assert plan["authorization"]["production_default_changed"] is False
