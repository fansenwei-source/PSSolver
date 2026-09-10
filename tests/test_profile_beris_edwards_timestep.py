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
        "pointwise_execution": "eager",
        "seed": 17,
    }
    values.update(overrides)
    return ProfileConfig(**values)


def test_profiler_defaults_to_production_numerics_and_accepts_legacy_control():
    assert ProfileConfig().transform_execution_order == "real_first"
    assert ProfileConfig().spectral_storage == "full_complex"
    assert ProfileConfig().molecular_field_linear_space == "spectral"
    assert ProfileConfig().stress_divergence_sum_space == "spectral"
    assert ProfileConfig().pointwise_execution == "compile"
    legacy = _small_config(transform_execution_order="legacy")
    assert legacy.transform_execution_order == "legacy"


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
    assert result["pointwise_kernels"]["requested"] == "eager"
    assert result["pointwise_kernels"]["effective"] == "eager"
    assert result["pointwise_kernels"]["compile"]["enabled"] is False
    assert result["pointwise_kernels"]["build_wall_seconds"] >= 0.0
    assert result["pointwise_kernels"]["warmup_steps"] == 1
    assert result["pointwise_kernels"]["warmup_wall_seconds"] >= 0.0
    assert result["pointwise_kernels"]["preprofile_wall_seconds"] >= 0.0
    assert "dynamo_during_build" in result["pointwise_kernels"]
    assert result["transforms"] == {
        "execution_order": "real_first",
        "spectral_storage": "full_complex",
        "physical_shape": [6, 6, 5],
        "spectral_shape": [6, 6, 5],
        "hermitian_axis": None,
        "basis_and_normalization_changed": False,
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


def test_q_and_stokes_models_share_one_pointwise_execution_policy():
    solver = _build_solver(
        _small_config(),
        RegionTimer(torch.device("cpu")),
    )
    assert solver.model.nlmodel.pointwise_kernels is solver.pointwise_kernels
    assert (
        solver.model.static_model.pointwise_kernels
        is solver.pointwise_kernels
    )


def test_hermitian_half_timestep_matches_full_complex_physical_state():
    full = _build_solver(
        _small_config(spectral_storage="full_complex"),
        RegionTimer(torch.device("cpu")),
    )
    half = _build_solver(
        _small_config(spectral_storage="hermitian_half"),
        RegionTimer(torch.device("cpu")),
    )

    for _ in range(3):
        full.integrator.step()
        half.integrator.step()

    torch.testing.assert_close(
        half.model.fields.spatial,
        full.model.fields.spatial,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    torch.testing.assert_close(
        half.model.fields.spectral,
        full.model.fields.spectral[..., :4, :],
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    assert half.model.fields.spectral.shape[-3:] == (6, 4, 5)
    assert half.model.fields.L_hat.shape[-3:] == (6, 4, 5)


def test_static_model_rejects_unknown_molecular_field_linear_space():
    config = _small_config(molecular_field_linear_space="unknown")
    with pytest.raises(ValueError, match="molecular_field_linear_space"):
        _build_solver(config, RegionTimer(torch.device("cpu")))


def test_spectral_linear_molecular_field_matches_physical_and_saves_inverses():
    physical_config = _small_config(
        warmup_steps=0,
        profile_steps=1,
        molecular_field_linear_space="physical",
    )
    spectral_config = _small_config(
        warmup_steps=0,
        profile_steps=1,
        molecular_field_linear_space="spectral",
    )
    physical_solver = _build_solver(
        physical_config,
        RegionTimer(torch.device("cpu")),
    )
    spectral_solver = _build_solver(
        spectral_config,
        RegionTimer(torch.device("cpu")),
    )

    physical_solver.integrator.step()
    spectral_solver.integrator.step()

    torch.testing.assert_close(
        spectral_solver.model.fields.spatial,
        physical_solver.model.fields.spatial,
        rtol=5.0e-12,
        atol=5.0e-12,
    )
    torch.testing.assert_close(
        spectral_solver.model.fields.spectral,
        physical_solver.model.fields.spectral,
        rtol=5.0e-12,
        atol=5.0e-12,
    )

    physical_profile = run_profile(physical_config)
    spectral_profile = run_profile(spectral_config)
    assert (
        physical_profile["timings"]["transform_inverse"]["calls"]
        - spectral_profile["timings"]["transform_inverse"]["calls"]
        == 5
    )


def test_spectral_stress_divergence_sum_matches_physical_and_saves_inverses():
    physical_config = _small_config(
        warmup_steps=0,
        profile_steps=1,
        stress_divergence_sum_space="physical",
    )
    spectral_config = _small_config(
        warmup_steps=0,
        profile_steps=1,
        stress_divergence_sum_space="spectral",
    )
    physical_solver = _build_solver(
        physical_config,
        RegionTimer(torch.device("cpu")),
    )
    spectral_solver = _build_solver(
        spectral_config,
        RegionTimer(torch.device("cpu")),
    )

    physical_solver.integrator.step()
    spectral_solver.integrator.step()

    torch.testing.assert_close(
        spectral_solver.model.fields.spatial,
        physical_solver.model.fields.spatial,
        rtol=5.0e-12,
        atol=5.0e-12,
    )
    torch.testing.assert_close(
        spectral_solver.model.fields.spectral,
        physical_solver.model.fields.spectral,
        rtol=5.0e-12,
        atol=5.0e-12,
    )

    physical_profile = run_profile(physical_config)
    spectral_profile = run_profile(spectral_config)
    assert (
        physical_profile["timings"]["transform_inverse"]["calls"]
        - spectral_profile["timings"]["transform_inverse"]["calls"]
        == 5
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
        ({"spectral_storage": "unknown"}, "spectral_storage"),
        (
            {
                "spectral_storage": "hermitian_half",
                "transform_execution_order": "legacy",
            },
            "requires real_first",
        ),
        ({"molecular_field_linear_space": "unknown"}, "linear_space"),
        ({"stress_divergence_sum_space": "unknown"}, "sum_space"),
        ({"pointwise_execution": "unknown"}, "pointwise_execution"),
        (
            {"pointwise_execution": "compile", "warmup_steps": 0},
            "at least one warmup",
        ),
        ({"snapshot_interval": 2}, "enabled together"),
        ({"snapshot_directory": "/tmp/unused"}, "enabled together"),
    ),
)
def test_invalid_profile_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        run_profile(_small_config(**overrides))
