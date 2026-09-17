"""Tests for the complete Beris--Edwards timestep profiler."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from pssolver import QualifiedBoundedTransformPolicy
from benchmarks.profile_beris_edwards_timestep import (
    ProfileConfig,
    RegionTimer,
    _build_solver,
    parse_args,
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
        "transform_attribution": True,
    }
    values.update(overrides)
    return ProfileConfig(**values)


def test_profiler_defaults_to_production_numerics_and_accepts_legacy_control():
    assert ProfileConfig().transform_execution_order == "real_first"
    assert ProfileConfig().spectral_storage == "hermitian_half"
    assert ProfileConfig().molecular_field_linear_space == "spectral"
    assert ProfileConfig().stress_divergence_sum_space == "spectral"
    assert ProfileConfig().pointwise_execution == "compile"
    assert ProfileConfig().projected_transform_execution == "truncated"
    assert ProfileConfig().bounded_transform_algorithm == "dense"
    assert ProfileConfig().bounded_transform_policy_path is None
    assert ProfileConfig().transform_attribution is False
    legacy = _small_config(
        transform_execution_order="legacy",
        spectral_storage="full_complex",
    )
    assert legacy.transform_execution_order == "legacy"


def test_profiler_cli_accepts_two_thirds_rule(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_beris_edwards_timestep.py", "--dealias-rule", "two_thirds"],
    )

    assert parse_args().dealias_rule == "two_thirds"


def test_profiler_cli_enables_transform_attribution_explicitly(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["profile_beris_edwards_timestep.py", "--transform-attribution"],
    )

    assert parse_args().transform_attribution is True


def test_profiler_cli_accepts_fft_bounded_transform_candidate(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_beris_edwards_timestep.py",
            "--bounded-transform-algorithm",
            "fft",
        ],
    )

    assert parse_args().bounded_transform_algorithm == "fft"


def test_profiler_auto_policy_falls_back_to_byte_identical_dense(tmp_path):
    policy = QualifiedBoundedTransformPolicy(
        name="empty_safe_fallback",
        cells=(),
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"policy": policy.to_metadata()}))

    dense = run_profile(_small_config(profile_steps=1))
    automatic = run_profile(
        _small_config(
            profile_steps=1,
            bounded_transform_algorithm="auto",
            bounded_transform_policy_path=str(policy_path),
        )
    )

    assert automatic["final_state_sha256"] == dense["final_state_sha256"]
    selection = automatic["transforms"]["bounded_transform_selection"]
    assert selection["requested_algorithm"] == "auto"
    assert selection["geometry"] == "plane"
    assert selection["effective_algorithms"] == ["dense"]
    assert selection["observed_decisions"]
    assert all(
        entry["selection"]["reason"]
        == "no_matching_qualification_cell"
        for entry in selection["observed_decisions"]
    )


def test_transform_attribution_is_absent_from_default_timing_path():
    result = run_profile(
        _small_config(profile_steps=1, transform_attribution=False)
    )

    assert result["timing_notes"]["transform_attribution"]["enabled"] is False
    assert result["timings"]["transform_forward"]["calls"] > 0
    assert result["timings"]["transform_inverse"]["calls"] > 0
    assert "periodic_fft_forward" not in result["timings"]
    assert "bounded_dct_forward" not in result["timings"]
    assert "bounded_dst_forward" not in result["timings"]


def test_whole_timestep_scope_counts_nested_regions_without_timing_them():
    result = run_profile(
        _small_config(profile_steps=1, timing_scope="whole_timestep")
    )

    assert result["timing_notes"]["timing_scope"] == "whole_timestep"
    assert result["timings"]["whole_timestep"]["timed"] is True
    assert result["timings"]["transform_forward"]["calls"] > 0
    assert result["timings"]["transform_forward"]["timed"] is False
    assert result["timings"]["transform_forward"]["total_seconds"] == 0.0
    assert result["timings"]["bounded_dct_forward"]["calls"] > 0
    assert result["timings"]["bounded_dct_forward"]["timed"] is False
    assert result["timings"]["periodic_fft_forward"]["calls"] > 0


def test_full_complex_profile_attributes_axis_fft_and_full_real_bases():
    result = run_profile(
        _small_config(
            warmup_steps=0,
            profile_steps=1,
            transform_execution_order="legacy",
            spectral_storage="full_complex",
            projected_transform_execution="full",
        )
    )

    for name in (
        "periodic_fft_forward_axis",
        "periodic_fft_inverse_axis",
        "bounded_dct_forward_full",
        "bounded_dct_inverse_full",
        "bounded_dst_forward_full",
        "bounded_dst_inverse_full",
    ):
        assert result["timings"][name]["calls"] > 0
        assert result["timings"][name]["timed"] is True


def test_profiler_accepts_production_layout_initial_q(tmp_path):
    path = tmp_path / "Q_0.npy"
    values = np.zeros((6, 6, 5, 5), dtype=np.float64)
    values[..., 0] = 0.2
    values[..., 3] = -0.1
    np.save(path, values, allow_pickle=False)

    result = run_profile(_small_config(initial_q_path=str(path)))

    assert result["config"]["initial_q_path"] == str(path)
    assert result["model"]["initial_condition"] == "production-layout Q_0.npy"


def test_profiler_rejects_incompatible_initial_q(tmp_path):
    path = tmp_path / "Q_0.npy"
    np.save(path, np.zeros((6, 6, 5, 4), dtype=np.float64))

    with pytest.raises(ValueError, match="initial Q shape"):
        run_profile(_small_config(initial_q_path=str(path)))


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
    assert result["memory_attribution"]["schema_version"] == 1
    assert all(
        value is None
        for window in result["memory_attribution"]["windows"].values()
        for value in window.values()
    )
    assert result["pointwise_kernels"]["requested"] == "eager"
    assert result["pointwise_kernels"]["effective"] == "eager"
    assert result["pointwise_kernels"]["compile"]["enabled"] is False
    assert result["projected_transforms"]["requested"] == "truncated"
    assert result["projected_transforms"]["effective"] == "truncated"
    assert result["projected_transforms"]["fallback_allowed"] is False
    assert result["projected_transforms"]["retained_axis_counts"] == {
        "q": [3, 2, 3],
        "normal_velocity": [3, 2, 2],
    }
    assert result["projected_transforms"]["computed_axis_sizes"] == {
        "q": [6, 4, 3],
        "normal_velocity": [6, 4, 2],
    }
    assert result["pointwise_kernels"]["build_wall_seconds"] >= 0.0
    assert result["pointwise_kernels"]["warmup_steps"] == 1
    assert result["pointwise_kernels"]["warmup_wall_seconds"] >= 0.0
    assert result["pointwise_kernels"]["preprofile_wall_seconds"] >= 0.0
    assert "dynamo_during_build" in result["pointwise_kernels"]
    assert result["transforms"] == {
        "bounded_transform_algorithm": "dense",
        "bounded_transform_selection": {
            "schema_version": 1,
            "requested_algorithm": "dense",
            "geometry": "plane",
            "decision_mode": "forced",
            "effective_algorithms": ["dense"],
            "policy": None,
            "observed_decisions": [],
        },
        "execution_order": "real_first",
        "spectral_storage": "hermitian_half",
        "physical_shape": [6, 6, 5],
        "spectral_shape": [6, 4, 5],
        "hermitian_axis": 1,
        "basis_and_normalization_changed": False,
    }
    assert result["timing_notes"]["transform_attribution"] == {
        "enabled": True,
        "aggregate_regions": [
            "periodic_fft_forward",
            "periodic_fft_inverse",
            "bounded_dct_forward",
            "bounded_dct_inverse",
            "bounded_dst_forward",
            "bounded_dst_inverse",
        ],
        "execution_detail_suffixes": [
            "axis",
            "nd",
            "full",
            "truncated",
        ],
        "regions_are_nested_inside_transform_forward_inverse": True,
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
        "periodic_fft_forward",
        "periodic_fft_forward_nd",
        "periodic_fft_inverse",
        "periodic_fft_inverse_nd",
        "bounded_dct_forward",
        "bounded_dct_forward_truncated",
        "bounded_dct_inverse",
        "bounded_dct_inverse_truncated",
        "bounded_dst_forward",
        "bounded_dst_forward_truncated",
        "bounded_dst_inverse",
        "bounded_dst_inverse_truncated",
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


@pytest.mark.parametrize("dealias_rule", ("two_thirds", "cubic_half"))
def test_truncated_projected_transforms_preserve_complete_cpu_timestep(
    dealias_rule,
):
    full = _build_solver(
        _small_config(
            dealias_rule=dealias_rule,
            projected_transform_execution="full",
        ),
        RegionTimer(torch.device("cpu")),
    )
    truncated = _build_solver(
        _small_config(
            dealias_rule=dealias_rule,
            projected_transform_execution="truncated",
        ),
        RegionTimer(torch.device("cpu")),
    )
    for _ in range(2):
        full.integrator.step()
        truncated.integrator.step()

    torch.testing.assert_close(
        truncated.model.fields.spatial,
        full.model.fields.spatial,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        truncated.model.fields.spectral,
        full.model.fields.spectral,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    metadata = truncated.model.spectral_projector.execution_metadata()
    assert metadata["effective"] == "truncated"
    assert metadata["truncated_real_basis_axes"]
    assert truncated.model.spectral_projector.computed_axis_sizes(
        ("periodic", "periodic", "neumann")
    ) == (6, 4, 3 if dealias_rule == "cubic_half" else 4)


@pytest.mark.parametrize(
    "projected_transform_execution",
    ("full", "truncated"),
)
def test_hermitian_half_timestep_matches_full_complex_physical_state(
    projected_transform_execution,
):
    full = _build_solver(
        _small_config(
            spectral_storage="full_complex",
            projected_transform_execution=projected_transform_execution,
        ),
        RegionTimer(torch.device("cpu")),
    )
    half = _build_solver(
        _small_config(
            spectral_storage="hermitian_half",
            projected_transform_execution=projected_transform_execution,
        ),
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


@pytest.mark.parametrize("spectral_storage", ("full_complex", "hermitian_half"))
def test_full_and_truncated_projected_execution_agree_for_each_storage(
    spectral_storage,
):
    full = _build_solver(
        _small_config(
            spectral_storage=spectral_storage,
            projected_transform_execution="full",
        ),
        RegionTimer(torch.device("cpu")),
    )
    truncated = _build_solver(
        _small_config(
            spectral_storage=spectral_storage,
            projected_transform_execution="truncated",
        ),
        RegionTimer(torch.device("cpu")),
    )

    for _ in range(2):
        full.integrator.step()
        truncated.integrator.step()

    torch.testing.assert_close(
        truncated.model.fields.spatial,
        full.model.fields.spatial,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        truncated.model.fields.spectral,
        full.model.fields.spectral,
        rtol=2.0e-12,
        atol=2.0e-12,
    )


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
        ({"bounded_transform_algorithm": "unknown"}, "must be one of"),
        (
            {"bounded_transform_algorithm": "auto"},
            "require a policy path",
        ),
        (
            {"bounded_transform_policy_path": "/tmp/unused"},
            "only valid with auto",
        ),
        (
            {"projected_transform_execution": "unknown"},
            "projected_transform_execution",
        ),
        (
            {
                "projected_transform_execution": "truncated",
                "dealias_rule": "none",
            },
            "enabled dealiasing",
        ),
        (
            {"pointwise_execution": "compile", "warmup_steps": 0},
            "at least one warmup",
        ),
        ({"snapshot_interval": 2}, "enabled together"),
        ({"snapshot_directory": "/tmp/unused"}, "enabled together"),
        ({"transform_attribution": "yes"}, "must be a boolean"),
    ),
)
def test_invalid_profile_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        run_profile(_small_config(**overrides))
