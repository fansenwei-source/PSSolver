"""Tests for the complete Beris--Edwards timestep profiler."""

from __future__ import annotations

import json

import pytest
import torch

from benchmarks.profile_beris_edwards_timestep import (
    ProfileConfig,
    RegionTimer,
    _build_solver,
    run_profile,
)


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


def test_q_gradient_reuse_preserves_state_and_removes_repeated_transforms():
    cached = run_profile(_small_config(profile_steps=1, reuse_q_gradients=True))
    uncached = run_profile(_small_config(profile_steps=1, reuse_q_gradients=False))

    assert cached["final_state_sha256"] == uncached["final_state_sha256"]
    assert (
        uncached["timings"]["transform_inverse"]["calls"]
        - cached["timings"]["transform_inverse"]["calls"]
        == 15
    )


def test_callback_q_mutation_invalidates_cache_and_matches_uncached_path():
    cached_config = _small_config(reuse_q_gradients=True)
    uncached_config = _small_config(reuse_q_gradients=False)
    cached = _build_solver(cached_config, RegionTimer(torch.device("cpu")))
    uncached = _build_solver(uncached_config, RegionTimer(torch.device("cpu")))

    def mutate_q(solver):
        fields = solver.model.fields
        fields["Qxx"] = fields["Qxx"] + 1.0e-4

    cached.integrator.step(pre_update_callback=lambda: mutate_q(cached))
    uncached.integrator.step(pre_update_callback=lambda: mutate_q(uncached))

    torch.testing.assert_close(
        cached.model.fields.spatial,
        uncached.model.fields.spatial,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        cached.model.fields.spectral,
        uncached.model.fields.spectral,
        rtol=0.0,
        atol=0.0,
    )


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
        ({"transform_execution_order": "unknown"}, "execution_order"),
        ({"snapshot_interval": 2}, "enabled together"),
        ({"snapshot_directory": "/tmp/unused"}, "enabled together"),
    ),
)
def test_invalid_profile_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        run_profile(_small_config(**overrides))
