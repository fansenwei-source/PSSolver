"""Synthetic gates for the implemented rc5 G-1 measurement window."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

import pytest

from benchmarks.analyze_rc5_g1_measurement_window_h100 import (
    PASS_CLASSIFICATION,
    analyze,
)
from benchmarks.plan_rc5_g1_measurement_window_recovery import build_plan
from benchmarks.profile_rc458_runtime_family import (
    RC458ProfileConfig,
    _normalize_public_diagnostics,
    _validate,
)
from benchmarks.run_rc5_g1_focused_profile_matrix import (
    build_command_plan as build_historical_command_plan,
)
from benchmarks.run_rc5_g1_measurement_window_profile_matrix import (
    SCHEMA as COMMAND_SCHEMA,
    build_command_plan,
)


OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _memory(value: int) -> dict[str, int]:
    return {
        "allocated_bytes": value,
        "reserved_bytes": value + 10,
        "active_bytes": value,
        "peak_allocated_bytes": value + 1,
        "peak_reserved_bytes": value + 11,
        "peak_active_bytes": value + 1,
    }


def _formal_memory(value: int) -> dict[str, int]:
    return {
        "allocated_bytes": value,
        "reserved_bytes": value,
        "active_bytes": value,
        "peak_allocated_bytes": value,
        "peak_reserved_bytes": value,
        "peak_active_bytes": value,
    }


def _profile(entry: dict[str, object], plan: dict[str, object]) -> dict[str, object]:
    role = str(entry["provider_role"])
    group = f"{entry['family']}:{entry['grid']}:trial{entry['trial']}"
    mean = (1.0 + 0.001 * int(entry["trial"])) * (
        1.01 if role == "candidate" else 1.0
    )
    samples = [mean] * 60
    initial = _hash("initial:" + group)
    final = _hash("final:" + group)
    if role == "candidate" and entry["family"] == "periodic_functional":
        final = _hash("repaired:" + group)
    memory_value = 101 if role == "candidate" else 100
    records = [
        {
            "step_index": index,
            "runtime_completed_steps": 13 + index,
            "diagnostics": {
                "applicable": True,
                "observed_fields": [
                    "pressure_iteration_count",
                    "pressure_relative_residual",
                ],
                "source_keys": {
                    "pressure_iteration_count": "last_pressure_iterations",
                    "pressure_relative_residual": (
                        "last_pressure_relative_residual"
                    ),
                },
                "pressure_iteration_count": 12,
                "pressure_relative_residual": 1.0e-11,
                "pressure_relative_residual_finite": True,
            },
        }
        for index in range(1, 61)
    ]
    return {
        "schema": "pssolver.rc4_5_8.runtime_family_profile.v1",
        "config": {
            "family": entry["family"],
            "provider_role": role,
            "provider_commit": entry["provider_commit"],
            "provider_wheel_sha256": entry["provider_wheel_sha256"],
            "trial": entry["trial"],
            "shape": plan["grid_contracts"][entry["grid"]]["shape"],
            "lengths": plan["grid_contracts"][entry["grid"]]["lengths"],
            "device": "cuda",
            "dtype": entry["dtype"],
            "warmup_steps": 10,
            "profile_steps": 60,
            "post_flush_rewarm_steps": 3,
        },
        "provider_identity": {
            "role": role,
            "commit": entry["provider_commit"],
            "wheel_sha256": entry["provider_wheel_sha256"],
        },
        "runtime_identity": {
            "requested": entry["runtime_path"],
            "effective": entry["runtime_path"],
            "fallback_used": False,
        },
        "environment": {
            "allocated_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_caller_before": OFF,
            "tf32_caller_after": OFF,
        },
        "timing": {
            "samples_seconds": samples,
            "mean_timestep_seconds": mean,
            "median_timestep_seconds": mean,
            "sample_standard_deviation_seconds": 0.0,
            "coefficient_of_variation": 0.0,
        },
        "memory": {
            "peak_allocated_bytes": memory_value,
            "peak_active_bytes": memory_value,
            "peak_reserved_bytes": memory_value,
        },
        "transform_calls": {
            "forward": entry["expected_forward_raw_calls"],
            "inverse": entry["expected_inverse_raw_calls"],
            "forward_per_step": entry["expected_forward_raw_calls"] / 60,
            "inverse_per_step": entry["expected_inverse_raw_calls"] / 60,
        },
        "compiler": {
            "graph_breaks": 0,
            "compile_fallback": False,
            "during_post_flush_rewarm": {"graph_breaks": 0},
            "during_profile": {"graph_breaks": 0},
        },
        "measurement_window": {
            "warmup_steps": 10,
            "post_flush_rewarm_steps": 3,
            "timed_steps": 60,
            "steps_after_warmup": 10,
            "measurement_start_completed_steps": 13,
            "rewarm_timed": False,
            "rewarm_in_formal_transform_counters": False,
            "rewarm_in_formal_compiler_counters": False,
            "rewarm_in_formal_memory_peaks": False,
            "formal_memory_semantics": "steady_state_after_allocation_recovery",
            "memory_snapshots": {
                "after_warmup": _memory(120),
                "after_empty_cache": _memory(90),
                "after_post_flush_rewarm": _memory(100),
                "after_formal_timed_window": _formal_memory(memory_value),
            },
            "cold_recovery_memory": {
                "allocated_delta_bytes": 10,
                "reserved_delta_bytes": 10,
                "active_delta_bytes": 10,
            },
            "device_telemetry_after_formal_window": {
                "applicable": True,
                "reason": None,
                "command": ["nvidia-smi"],
                "rows": [
                    {
                        "pstate": "P0",
                        "sm_clock_mhz": 1500.0,
                        "memory_clock_mhz": 1593.0,
                        "power_draw_watts": 250.0,
                        "temperature_celsius": 45.0,
                    }
                ],
            },
        },
        "diagnostic_replay": {
            "separate_fresh_runtime": True,
            "used_for_performance_or_memory_adjudication": False,
            "initial_state_sha256": initial,
            "final_state_sha256": final,
            "completed_steps": 73,
            "step_records": records,
            "finite": True,
        },
        "initial_state_sha256": initial,
        "final_state_sha256": final,
        "completed_profile_steps": 60,
        "runtime_completed_steps": 73,
        "finite": True,
    }


def _write_profiles(root: Path) -> None:
    plan = build_plan()
    root.mkdir()
    for entry in plan["profile_matrix"]:
        (root / f"{entry['key']}.json").write_text(
            json.dumps(_profile(entry, plan), allow_nan=False),
            encoding="utf-8",
        )


def test_profiler_config_preserves_zero_default_and_rejects_negative_rewarm():
    common = {
        "provider_role": "baseline",
        "provider_commit": "a" * 40,
        "provider_wheel_sha256": "b" * 64,
        "trial": 1,
        "shape": (8, 8, 4),
        "lengths": (2.0, 2.0, 1.0),
    }
    historical = RC458ProfileConfig(family="plane_compiled_v2", **common)
    assert historical.post_flush_rewarm_steps == 0
    _validate(historical)
    _validate(
        RC458ProfileConfig(
            family="plane_compiled_v2",
            post_flush_rewarm_steps=3,
            **common,
        )
    )
    with pytest.raises(ValueError, match="step counts"):
        _validate(
            RC458ProfileConfig(
                family="plane_compiled_v2",
                post_flush_rewarm_steps=-1,
                **common,
            )
        )


def test_public_diagnostics_are_normalized_without_inventing_missing_values():
    normalized = _normalize_public_diagnostics(
        lambda: {
            "flow": {
                "last_pressure_iterations": 12,
                "last_pressure_relative_residual": 1.0e-11,
            }
        }
    )
    assert normalized["applicable"] is True
    assert normalized["pressure_iteration_count"] == 12
    assert normalized["pressure_relative_residual"] == 1.0e-11
    missing = _normalize_public_diagnostics(lambda: {"unrelated": 1})
    assert missing == {
        "applicable": False,
        "observed_fields": [],
        "source_keys": {},
        "reason": "no_supported_pressure_field_was_exposed",
    }


def test_new_command_plan_adds_rewarm_without_changing_historical_command(tmp_path):
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    support = tmp_path / "support"
    (support / "benchmarks").mkdir(parents=True)
    (support / "benchmarks/profile_rc5_g1_runtime_family.py").write_text(
        "", encoding="utf-8"
    )
    arguments = {
        "baseline_python": python,
        "candidate_python": python,
        "support_root": support,
        "output_root": tmp_path / "new-output",
    }
    current = build_command_plan(**arguments)
    historical = build_historical_command_plan(
        **{**arguments, "output_root": tmp_path / "old-output"}
    )
    assert current["schema"] == COMMAND_SCHEMA
    assert current["profile_count"] == historical["profile_count"] == 48
    assert all(
        row["command"].count("--post-flush-rewarm-steps") == 1
        and row["command"][
            row["command"].index("--post-flush-rewarm-steps") + 1
        ]
        == "3"
        for row in current["rows"]
    )
    assert all("--post-flush-rewarm-steps" not in row["command"] for row in historical["rows"])


def test_measurement_analyzer_accepts_complete_corrected_evidence(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    result = analyze(root)
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["profile_count"] == 48
    assert result["measurement_window_corrected"] is True
    assert result["diagnostic_replay_is_non_adjudicating"] is True
    assert result["v7_failure_reclassified"] is False
    assert result["g1_complete"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("diagnostic_replay"), "diagnostic replay is missing"),
        (
            lambda value: value["measurement_window"].__setitem__(
                "rewarm_timed", True
            ),
            "rewarm_timed must be false",
        ),
        (
            lambda value: value["diagnostic_replay"]["step_records"].pop(),
            "step-record count differs",
        ),
    ],
)
def test_measurement_analyzer_rejects_incomplete_or_contaminated_windows(
    tmp_path, mutation, message
):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "baseline_plane_compiled_v2_plane_R128_trial1.json"
    value = json.loads(target.read_text(encoding="utf-8"))
    mutation(value)
    target.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        analyze(root)


def test_measurement_analyzer_still_uses_all_sixty_samples_for_cv(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "candidate_periodic_functional_periodic_R128_trial1.json"
    value = json.loads(target.read_text(encoding="utf-8"))
    samples = [1.0] * 59 + [2.0]
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    value["timing"] = {
        "samples_seconds": samples,
        "mean_timestep_seconds": mean,
        "median_timestep_seconds": statistics.median(samples),
        "sample_standard_deviation_seconds": stdev,
        "coefficient_of_variation": stdev / mean,
    }
    target.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match="timing validity failed"):
        analyze(root)
