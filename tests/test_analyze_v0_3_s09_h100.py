"""Synthetic fail-closed tests for the S-09 cumulative analyzer."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.analyze_v0_3_s09_h100 import (
    AUXILIARY_SCHEMA,
    PASS_CLASSIFICATION,
    PROFILE_SCHEMA,
    _validate_auxiliary,
    analyze,
)
from benchmarks.plan_v0_3_s09_h100 import (
    BASELINE_WHEEL_SHA256,
    CANDIDATE_COMMIT,
    CANDIDATE_WHEEL_SHA256,
    CONSUMER_RUNTIME_COMMIT,
    CONSUMER_WHEEL_SHA256,
    build_plan,
)


TF32_OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _auxiliary() -> dict[str, object]:
    plan = build_plan()
    shared = {
        "passed": True,
        "provider_candidate_commit": CANDIDATE_COMMIT,
        "candidate_wheel_sha256": CANDIDATE_WHEEL_SHA256,
    }
    rows = [
        {
            "name": "node_local_tmp_environment_and_host_compiler",
            **shared,
            "baseline_wheel_sha256": BASELINE_WHEEL_SHA256,
            "consumer_wheel_sha256": CONSUMER_WHEEL_SHA256,
            "cuda_available": True,
            "single_visible_h100": True,
            "allocated_cuda_0": True,
            "installed_wheels_only": True,
            "source_shadow_import_false": True,
            "pip_check": True,
            "node_local_tmp": True,
            "host_compiler": True,
            "empty_compile_caches": True,
        },
        {"name": "installed_wheel_cuda_only_tests", **shared},
        {
            "name": "cuda_long_horizon_hermitian",
            **shared,
            "cases": plan["long_horizon"]["cases"],
            "all_finite": True,
            "all_within_bound": True,
        },
        {
            "name": "tf32_forward_backward_scope",
            **shared,
            "caller_state_count": 6,
            "functional_case_count": 12,
            "forward_observed_policy": True,
            "backward_observed_policy": True,
            "success_restored": True,
            "exception_restored": True,
            "nested_lifo": True,
            "outside_scope_preserved": True,
        },
        {
            "name": "checkpoint_restart_and_released_fixture_matrix",
            **shared,
            "current_matrix": True,
            "released_fixtures": True,
            "restart_exact": True,
            "negative_gates": True,
            "sources_unchanged": True,
        },
        {
            "name": "numerical_gradient_taylor_and_replay",
            **shared,
            "standard_regression": True,
            "gradient": True,
            "finite_difference": True,
            "taylor": True,
            "replay": True,
            "gradient_taylor_evidence_stage": "independent_control_consumer",
        },
        {
            "name": "independent_control_consumer",
            **shared,
            "consumer_source_commit": CONSUMER_RUNTIME_COMMIT,
            "consumer_wheel_sha256": CONSUMER_WHEEL_SHA256,
            "installed_wheels_only": True,
            "functional_api_1_1": True,
            "provider_identity_migrated": True,
            "periodic_gradient": True,
            "channel_gradient": True,
            "checkpointed_adjoint": True,
            "negative_controls": True,
            "finite_cuda_smokes": True,
        },
    ]
    return {
        "schema": AUXILIARY_SCHEMA,
        "passed": True,
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "second_job_submitted": False,
        "stages": rows,
    }


def _write_profiles(root: Path, candidate_scale: float = 1.01) -> None:
    plan = build_plan()
    root.mkdir()
    for entry in plan["profile_matrix"]:
        scale = 1.0 if entry["provider_role"] == "baseline" else candidate_scale
        samples = [0.01 * scale] * 60
        profile = {
            "schema": PROFILE_SCHEMA,
            "config": {
                "family": entry["family"],
                "trial": entry["trial"],
                "shape": plan["grid_contracts"][entry["grid"]]["shape"],
                "lengths": plan["grid_contracts"][entry["grid"]]["lengths"],
                "device": "cuda",
                "dtype": entry["dtype"],
            },
            "provider_identity": {
                "role": entry["provider_role"],
                "commit": entry["provider_commit"],
                "wheel_sha256": entry["provider_wheel_sha256"],
            },
            "runtime_identity": {
                "requested": entry["family"],
                "effective": entry["family"],
                "fallback_used": False,
            },
            "environment": {
                "allocated_device": "cuda:0",
                "device_name": "NVIDIA H100 PCIe",
                "tf32_caller_before": TF32_OFF,
                "tf32_caller_after": TF32_OFF,
            },
            "timing": {
                "samples_seconds": samples,
                "mean_timestep_seconds": sum(samples) / len(samples),
                "median_timestep_seconds": samples[0],
            },
            "memory": {
                "peak_allocated_bytes": int(1_000_000 * scale),
                "peak_reserved_bytes": int(2_000_000 * scale),
            },
            "transform_calls": {
                "forward": entry["expected_forward_raw_calls"],
                "inverse": entry["expected_inverse_raw_calls"],
                "forward_per_step": entry["expected_forward_raw_calls"] / 60,
                "inverse_per_step": entry["expected_inverse_raw_calls"] / 60,
            },
            "compiler": {"graph_breaks": 0, "compile_fallback": False},
            "finite": True,
            "completed_profile_steps": 60,
            "runtime_completed_steps": 70,
            "initial_state_sha256": "0" * 64,
            "final_state_sha256": ("1" if entry["provider_role"] == "baseline" else "2") * 64,
        }
        (root / f"{entry['key']}.json").write_text(
            json.dumps(profile, allow_nan=False), encoding="utf-8"
        )


def test_s09_analyzer_accepts_the_complete_synthetic_contract(tmp_path):
    profiles = tmp_path / "profiles"
    _write_profiles(profiles)
    result = analyze(profiles, _auxiliary())
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["qualification_complete"] is True
    assert result["profile_count"] == 192
    assert result["group_count"] == 16


def test_s09_analyzer_rejects_a_long_horizon_failure():
    report = _auxiliary()
    report["stages"][2]["all_within_bound"] = False
    with pytest.raises(ValueError, match="Hermitian bound"):
        _validate_auxiliary(report, build_plan())


def test_s09_analyzer_rejects_three_profiles_over_the_frozen_reference(tmp_path):
    profiles = tmp_path / "profiles"
    _write_profiles(profiles)
    for trial in (1, 2, 3):
        path = profiles / f"candidate_plane_legacy_production_plane_R128_trial{trial}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["timing"]["samples_seconds"] = [0.011] * 60
        value["timing"]["mean_timestep_seconds"] = 0.011
        value["timing"]["median_timestep_seconds"] = 0.011
        path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="performance or memory gate"):
        analyze(profiles, _auxiliary())


def test_s09_analyzer_rejects_missing_profile(tmp_path):
    profiles = tmp_path / "profiles"
    _write_profiles(profiles)
    next(profiles.glob("*.json")).unlink()
    with pytest.raises(FileNotFoundError):
        analyze(profiles, _auxiliary())


def test_s09_auxiliary_rejects_reordered_control_and_performance_contract():
    report = _auxiliary()
    report["stages"][5], report["stages"][6] = report["stages"][6], report["stages"][5]
    with pytest.raises(ValueError, match="stage order"):
        _validate_auxiliary(report, build_plan())
