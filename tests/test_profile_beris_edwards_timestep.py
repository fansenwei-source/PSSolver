"""Tests for the complete Beris--Edwards timestep profiler."""

from __future__ import annotations

import json

import pytest

from benchmarks.profile_beris_edwards_timestep import ProfileConfig, run_profile


def _small_config(**overrides):
    values = {
        "shape": (6, 6, 5),
        "lengths": (3.0, 3.0, 2.0),
        "device": "cpu",
        "dtype": "float64",
        "warmup_steps": 1,
        "profile_steps": 2,
        "seed": 17,
    }
    values.update(overrides)
    return ProfileConfig(**values)


def test_complete_timestep_profile_has_expected_regions_and_provenance():
    result = run_profile(_small_config())

    assert result["schema_version"] == 1
    assert result["config"]["dealias_rule"] == "cubic_half"
    assert len(result["final_state_sha256"]) == 64
    assert result["throughput"]["timesteps_per_second"] > 0.0
    assert result["memory"] == {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }

    for name in (
        "whole_timestep",
        "static_fields",
        "q_nonlinear",
        "imex_and_dealias",
        "dynamic_inverse",
        "spectral_refresh",
        "nematic_force",
        "stokes_solve",
        "transform_forward",
        "transform_inverse",
    ):
        assert result["timings"][name]["calls"] >= 2
        assert result["timings"][name]["total_seconds"] >= 0.0

    json.dumps(result)


def test_profile_is_state_deterministic_for_fixed_seed():
    first = run_profile(_small_config(profile_steps=1))
    second = run_profile(_small_config(profile_steps=1))
    assert first["final_state_sha256"] == second["final_state_sha256"]


def test_snapshot_profile_writes_requested_fields(tmp_path):
    snapshot_directory = tmp_path / "snapshots"
    result = run_profile(
        _small_config(
            warmup_steps=0,
            profile_steps=1,
            snapshot_interval=1,
            snapshot_directory=str(snapshot_directory),
            save_hydrodynamics=True,
        )
    )

    assert (snapshot_directory / "Q_1.npy").is_file()
    assert (snapshot_directory / "u_1.npy").is_file()
    assert (snapshot_directory / "p_1.npy").is_file()
    assert result["timings"]["snapshot_cpu_transfer"]["calls"] == 1
    assert result["timings"]["snapshot_disk_write"]["calls"] == 1


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"shape": (4, 4, 0)}, "shape"),
        ({"dt": 0.0}, "dt"),
        ({"profile_steps": 0}, "profile_steps"),
        ({"snapshot_interval": 2}, "enabled together"),
        ({"snapshot_directory": "/tmp/unused"}, "enabled together"),
    ),
)
def test_invalid_profile_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        run_profile(_small_config(**overrides))
