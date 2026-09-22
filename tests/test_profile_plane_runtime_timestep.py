"""Tests for the Phase 6 production-runtime profiler."""

from __future__ import annotations

import json

import pytest

from benchmarks.profile_plane_runtime_timestep import (
    RuntimeProfileConfig,
    main,
    run_profile,
)


def _config(runtime_path: str, **overrides) -> RuntimeProfileConfig:
    values = {
        "runtime_path": runtime_path,
        "trial": 1,
        "shape": (6, 6, 6),
        "lengths": (8.0, 9.0, 20.0),
        "device": "cpu",
        "dtype": "float64",
        "pointwise_execution": "eager",
        "warmup_steps": 1,
        "profile_steps": 2,
        "spectral_refresh_interval": None,
        "seed": 24,
    }
    values.update(overrides)
    return RuntimeProfileConfig(**values)


def test_legacy_and_compiled_profiles_use_distinct_paths_with_exact_state():
    legacy = run_profile(_config("legacy_production"))
    compiled = run_profile(_config("compiled_v2"))

    assert legacy["runtime_identity"]["effective"] == "legacy_production"
    assert compiled["runtime_identity"]["effective"] == "compiled_v2"
    assert legacy["runtime_identity"]["fallback_used"] is False
    assert compiled["runtime_identity"]["fallback_used"] is False
    assert legacy["initial_q_sha256"] == compiled["initial_q_sha256"]
    assert legacy["final_state_sha256"] == compiled["final_state_sha256"]
    assert legacy["completed_steps"] == compiled["completed_steps"] == 3
    assert legacy["finite"] is compiled["finite"] is True
    assert legacy["transform_calls"]["forward_per_step"] == 7
    assert compiled["transform_calls"]["forward_per_step"] == 7
    assert legacy["transform_calls"]["inverse_per_step"] == 32
    assert compiled["transform_calls"]["inverse_per_step"] == 32


@pytest.mark.parametrize(
    ("overrides", "exception", "message"),
    (
        ({"runtime_path": "separated_canary"}, ValueError, "runtime_path"),
        ({"trial": 0}, ValueError, "trial"),
        ({"profile_steps": 0}, ValueError, "profile_steps"),
        ({"warmup_steps": 0, "pointwise_execution": "compile"}, ValueError, "warmup"),
        ({"spectral_refresh_interval": 0}, ValueError, "spectral_refresh"),
    ),
)
def test_profile_contract_rejects_invalid_inputs(overrides, exception, message):
    values = dict(overrides)
    runtime_path = values.pop("runtime_path", "legacy_production")
    with pytest.raises(exception, match=message):
        run_profile(_config(runtime_path, **values))


def test_cli_writes_refuse_overwrite_json(tmp_path):
    output = tmp_path / "profile.json"
    arguments = [
        "--runtime-path",
        "compiled_v2",
        "--trial",
        "1",
        "--shape",
        "6,6,6",
        "--lengths",
        "8,9,20",
        "--pointwise-execution",
        "eager",
        "--warmup-steps",
        "1",
        "--profile-steps",
        "1",
        "--output",
        str(output),
    ]

    assert main(arguments) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["runtime_identity"]["effective"] == "compiled_v2"
    with pytest.raises(SystemExit):
        main(arguments)
