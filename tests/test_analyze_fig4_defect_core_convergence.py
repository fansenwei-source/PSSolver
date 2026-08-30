import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts_plane.analyze_fig4_defect_core_convergence as core


def compact_uniaxial_q(S, theta=0.0, nz_component=0.0):
    S_values, theta_values = np.broadcast_arrays(
        np.asarray(S, dtype=np.float64), np.asarray(theta, dtype=np.float64)
    )
    nx = np.cos(theta_values) * math.sqrt(1.0 - nz_component**2)
    ny = np.sin(theta_values) * math.sqrt(1.0 - nz_component**2)
    nz = np.full_like(nx, nz_component)
    q = np.empty(S_values.shape + (5,), dtype=np.float64)
    q[..., 0] = 1.5 * S_values * (nx * nx - 1.0 / 3.0)
    q[..., 1] = 1.5 * S_values * nx * ny
    q[..., 2] = 1.5 * S_values * nx * nz
    q[..., 3] = 1.5 * S_values * (ny * ny - 1.0 / 3.0)
    q[..., 4] = 1.5 * S_values * ny * nz
    return q


def periodic_core_plane(
    shape,
    lengths,
    center,
    *,
    core_radius=1.5,
    S_bulk=1.0 / 3.0,
    charge=None,
):
    nx, ny = shape
    dx, dy = lengths[0] / nx, lengths[1] / ny
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    delta_x = core.minimum_image_delta(xx - center[0], (lengths[0],))
    delta_y = core.minimum_image_delta(yy - center[1], (lengths[1],))
    radius = np.hypot(delta_x, delta_y)
    S = S_bulk * np.tanh(radius / core_radius)
    theta = 0.0 if charge is None else charge * np.arctan2(delta_y, delta_x)
    return compact_uniaxial_q(S, theta)


def candidate(x, y, charge):
    return core.CoreCandidate(
        x=x,
        y=y,
        S_min=0.0,
        charge=charge,
        charge_reason="test",
        q_windings=(),
        director_windings=(),
    )


def artifact(tmp_path, label, *, shape=(16, 16, 8), dt=0.005, steps=200, metadata=None):
    directory = tmp_path / label
    if metadata is None:
        metadata = {
            "schema_version": 1,
            "script": "Plane_fig4_benchmark.py",
            "status": "complete",
            "completed_steps": steps,
            "solver": {
                "shape": list(shape),
                "lengths": [100.0, 100.0, 20.0],
                "dt": dt,
                "steps": steps,
                "save_interval": 200,
            },
            "model": {
                "name": "active_nematics",
                "Q_convention": core.Q_convention_metadata(),
                "parameters": {"S_initial": 1 / 3, "S_bulk": 1 / 3, "fric": 0.0},
            },
            "initial_condition": {
                "seed": 24,
                "raw_q_sha256": f"{label}-raw",
                "projected_q_sha256": f"{label}-projected",
            },
            "zero_mode_policy": "zero_mean",
            "friction": 0.0,
        }
    return core.RunArtifact(
        label=label,
        directory=directory,
        metadata=metadata,
        shape=shape,
        lengths=(100.0, 100.0, 20.0),
        dt=dt,
        steps=steps,
        final_time=dt * steps,
        q_path=directory / f"Q_{steps}.npy",
        u_path=directory / f"u_{steps}.npy",
        p_path=directory / f"p_{steps}.npy",
        diagnostics_path=directory / "diagnostics.npy",
    )


def core_run(tmp_path, label, *, shape=(16, 16, 8), dt=0.005, steps=200, metadata=None):
    run_artifact = artifact(
        tmp_path, label, shape=shape, dt=dt, steps=steps, metadata=metadata
    )
    return core.CoreRun(
        artifact=run_artifact,
        q0_path=run_artifact.directory / "Q_0.npy",
        q2d_path=run_artifact.directory / "Q2D_initial.npy",
        defects_path=run_artifact.directory / "Q2D_defects.csv",
        defect_positions=np.asarray(((1.0, 2.0), (8.0, 7.0))),
        defect_charges=np.asarray((0.5, -0.5)),
    )


def beris_contract_metadata(
    shape,
    *,
    dt=0.005,
    steps=200,
    config_character="a",
    initial_character="b",
):
    ldg_l1 = 0.024691358024691357
    gamma = 2.94
    nx, ny, nz = shape
    return {
        "schema_version": 1,
        "script": core.BERIS_EDWARDS_BENCHMARK_SCRIPT,
        "validation_config_sha256": config_character * 64,
        "implementation_provenance": {
            "schema_version": 1,
            "files": {
                name: "1" * 64
                for name in core.REQUIRED_BERIS_EDWARDS_IMPLEMENTATION_FILES
            },
        },
        "runtime_environment": {
            "device_type": "cuda",
            "resolved_device": "cuda:0",
            "cuda_available": True,
            "cuda_device_name": "NVIDIA test GPU",
            "cuda_total_memory_bytes": 80 * 1024**3,
        },
        "status": "complete",
        "completed_steps": steps,
        "solver": {
            "shape": list(shape),
            "lengths": [100.0, 100.0, 20.0],
            "dt": dt,
            "steps": steps,
            "save_interval": steps,
            "real_dtype": "float64",
            "spectral_dtype": "complex128",
        },
        "model": {
            "name": "active_nematics",
            "variant": core.BERIS_EDWARDS_MODEL_VARIANT,
            "stage": "reusable_beris_edwards_complete_stress_stokes",
            "Q_convention": core.Q_convention_metadata(),
            "q_dynamics": {
                "implementation": (
                    "pssolver.models.active_nematics.BerisEdwardsQNonlinearModel"
                ),
                "flow_alignment_form": "full_beris_edwards",
                "raw_coefficients": {
                    "A": 0.0,
                    "B": -0.3,
                    "C": 0.3,
                    "L1": ldg_l1,
                    "rotational_viscosity_gamma": gamma,
                    "flow_alignment_lambda": 0.3,
                },
                "evolution_coefficients": {
                    "A_over_gamma": 0.0,
                    "B_over_gamma": -0.3 / gamma,
                    "C_over_gamma": 0.3 / gamma,
                    "L1_over_gamma": ldg_l1 / gamma,
                },
            },
            "flow_dynamics": {
                "nematic_stress": (
                    "complete_one_constant_beris_edwards_up_to_isotropic_pressure"
                ),
            },
            "parameters": {
                "activity_number": 18.0,
                "zeta": 0.01,
                "frank_K": 0.012345679012345678,
                "ldg_A": 0.0,
                "ldg_B": -0.3,
                "ldg_C": 0.3,
                "ldg_L1": ldg_l1,
                "rotational_viscosity_gamma": gamma,
                "flow_alignment_lambda": 0.3,
                "alpha": 0.01,
                "beta": -1.0,
                "S_initial": 1.0 / 3.0,
                "S_bulk": 1.0 / 3.0,
                "fric": 0.0,
                "eta": 2.0 / 3.0,
            },
        },
        "boundary_conditions": {
            "Q": ["periodic", "periodic", "neumann"],
            "velocity_tangential": ["periodic", "periodic", "neumann"],
            "velocity_normal": ["periodic", "periodic", "dirichlet"],
            "pressure": ["periodic", "periodic", "neumann"],
            "velocity_wall_model_note": (
                "homogeneous kinematic free-slip; not zero total nematic traction"
            ),
        },
        "numerics": {
            "precision": {
                "real_dtype": "float64",
                "spectral_dtype": "complex128",
                "tf32_requested": "off",
                "tf32_effective": False,
                "float32_matmul_precision": "highest",
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
            },
            "spectral_refresh": {
                "mode": "disabled",
                "requested_interval_time": None,
                "requested_interval_steps": None,
                "effective_interval_steps": None,
                "effective_interval_time": None,
                "phase_origin_step": 0,
                "actual_count": 0,
            },
        },
        "parameterization": "paper-window",
        "activity_number": 18.0,
        "height": 20.0,
        "domain": [100.0, 100.0, 20.0],
        "shape": list(shape),
        "dt": dt,
        "steps": steps,
        "zeta": 0.01,
        "frank_k": 0.012345679012345678,
        "ldg_l1": ldg_l1,
        "q_elastic_relaxation": ldg_l1 / gamma,
        "ldg_coefficients": {"A": 0.0, "B": -0.3, "C": 0.3},
        "gamma": gamma,
        "flow_alignment": 0.3,
        "eta": 2.0 / 3.0,
        "friction": 0.0,
        "active_stress_beta": -1.0,
        "S_initial": 1.0 / 3.0,
        "S_bulk": 1.0 / 3.0,
        "device": "cuda:0",
        "dtype": "float64",
        "tf32": "off",
        "zero_mode_policy": "zero_mean",
        "dealias_rule": "cubic_half",
        "dealias_fraction": 0.5,
        "save_start_step": 0,
        "save_interval": steps,
        "diagnostic_interval": min(100, steps),
        "save_hydrodynamics": True,
        "seed": 24,
        "initial_condition": {
            "name": "extruded_analytic_periodic_defect_gas_2d",
            "seed": 24,
            "S_initial": 1.0 / 3.0,
            "twist_amplitude": 0.01,
            "twist_modes": [1, 2, 3],
            "raw_q_sha256": initial_character * 64,
            "projected_q_sha256": initial_character * 64,
        },
        "initial_defect_gas": {
            "num_pairs": 6,
            "minimum_separation": 10.0,
            "core_radius": 1.5,
            "S_initial": 1.0 / 3.0,
            "background_angle": 0.0,
        },
        "initial_neumann_twist": {
            "rms_amplitude_radians": 0.01,
            "dct_modes": [1, 2, 3],
        },
        "retained_q_modes": [nx / 2 - 1, ny / 2 - 1, nz / 2],
        "retained_normal_velocity_modes": [
            nx / 2 - 1,
            ny / 2 - 1,
            nz / 2 - 1,
        ],
    }


def materialize_contract_core_run(tmp_path, label, metadata):
    shape = tuple(metadata["shape"])
    steps = int(metadata["steps"])
    directory = tmp_path / label
    directory.mkdir()
    for name in (
        "Q2D_initial.npy",
        "Q2D_defects.csv",
        "Q_0.npy",
        "u_0.npy",
        "p_0.npy",
        f"Q_{steps}.npy",
        f"u_{steps}.npy",
        f"p_{steps}.npy",
        "diagnostics.npy",
        "diagnostics.csv",
        "COMPLETE",
    ):
        (directory / name).touch()
    run_artifact = core.RunArtifact(
        label=label,
        directory=directory,
        metadata=metadata,
        shape=shape,
        lengths=(100.0, 100.0, 20.0),
        dt=float(metadata["dt"]),
        steps=steps,
        final_time=float(metadata["dt"]) * steps,
        q_path=directory / f"Q_{steps}.npy",
        u_path=directory / f"u_{steps}.npy",
        p_path=directory / f"p_{steps}.npy",
        diagnostics_path=directory / "diagnostics.npy",
    )
    positions = np.column_stack(
        (np.arange(12, dtype=np.float64), np.arange(12, dtype=np.float64))
    )
    charges = np.asarray([0.5] * 6 + [-0.5] * 6)
    return core.CoreRun(
        artifact=run_artifact,
        q0_path=directory / "Q_0.npy",
        q2d_path=directory / "Q2D_initial.npy",
        defects_path=directory / "Q2D_defects.csv",
        defect_positions=positions,
        defect_charges=charges,
    )


def test_continuous_reference_uses_all_defect_tanh_factors():
    positions = np.asarray(((2.0, 3.0), (7.0, 3.0), (2.0, 8.0)))
    observed = core.continuous_multidefect_S(
        np.asarray([3.0]),
        np.asarray([3.0]),
        positions=positions,
        lengths=(10.0, 10.0),
        core_radius=1.5,
        S_initial=1.0 / 3.0,
    )[0]
    distances = (1.0, 4.0, math.sqrt(26.0))
    expected = (1.0 / 3.0) * np.prod(np.tanh(np.asarray(distances) / 1.5))
    assert observed == pytest.approx(expected, rel=0, abs=2e-15)
    assert observed < (1.0 / 3.0) * math.tanh(1.0 / 1.5)


def test_periodic_minimum_image_handles_boundary_crossing():
    delta = core.minimum_image_delta(np.asarray((9.8, -9.7)), (10.0, 10.0))
    np.testing.assert_allclose(delta, (-0.2, 0.3), atol=1e-14)


def test_periodic_high_order_interpolation_recovers_fourier_modes_and_seam():
    shape = (96, 80)
    lengths = (12.0, 10.0)
    x = (np.arange(shape[0]) + 0.5) * lengths[0] / shape[0]
    y = (np.arange(shape[1]) + 0.5) * lengths[1] / shape[1]
    xx, yy = np.meshgrid(x, y, indexing="ij")
    S = 0.25 + 0.03 * np.cos(4 * np.pi * xx / lengths[0]) + 0.02 * np.sin(
        6 * np.pi * yy / lengths[1]
    )
    sampler = core.PeriodicQSampler(compact_uniaxial_q(S), lengths, order=5)
    query_x = np.asarray((-1e-8, lengths[0] - 1e-8, 2.123, 14.123))
    query_y = np.asarray((3.4, 3.4, -0.17, 9.83))
    expected = 0.25 + 0.03 * np.cos(4 * np.pi * query_x / lengths[0]) + 0.02 * np.sin(
        6 * np.pi * query_y / lengths[1]
    )
    np.testing.assert_allclose(sampler.sample_S(query_x, query_y), expected, atol=2e-10)
    assert sampler.sample_S(query_x[0], query_y[0]) == pytest.approx(
        sampler.sample_S(query_x[1], query_y[1]), abs=2e-12
    )


def test_subgrid_center_recovery_for_periodic_boundary_core():
    lengths = (16.0, 16.0)
    center = np.asarray((15.91, 0.13))
    q = periodic_core_plane((128, 128), lengths, center)
    sampler = core.PeriodicQSampler(q, lengths, order=5)
    recovered, _ = core.refine_core_center(
        sampler, (15.94, 0.06), search_half_width=0.35
    )
    error = np.linalg.norm(core.minimum_image_delta(recovered - center, lengths))
    assert error < 0.025


def test_r50_approaches_isolated_tanh_value_under_grid_refinement():
    lengths = (24.0, 24.0)
    center = np.asarray((11.93, 12.07))
    exact = 1.5 * np.arctanh(0.5)
    errors = []
    for n in (32, 64, 128):
        q = periodic_core_plane((n, n), lengths, center)
        sampler = core.PeriodicQSampler(q, lengths, order=5)
        metrics, _ = core.measure_core_profile(
            sampler,
            center,
            all_centers=center[None, :],
            center_index=0,
            S_bulk=1.0 / 3.0,
            r_max=4.0,
            dr=0.025,
            angular_samples=128,
        )
        errors.append(abs(metrics["r50_absolute"] - exact))
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.01


def test_cell_centered_dct_modal_evaluation_recovers_cosine_polynomial():
    for nz in (64, 80, 128):
        z = (np.arange(nz) + 0.5) / nz
        field = 0.7 + 0.3 * np.cos(np.pi * z) - 0.2 * np.cos(3 * np.pi * z)
        target = np.asarray((0.1, 0.5, 0.9))
        observed = core.dct_modal_evaluate_chunk(field, target)
        expected = 0.7 + 0.3 * np.cos(np.pi * target) - 0.2 * np.cos(3 * np.pi * target)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=8e-15)


def test_dct_basis_reconstructs_cell_centred_nodes_exactly():
    rng = np.random.default_rng(4)
    nz = 23
    field = rng.normal(size=nz)
    nodes = (np.arange(nz) + 0.5) / nz
    reconstructed = core.dct_modal_evaluate_chunk(field, nodes)
    np.testing.assert_allclose(reconstructed, field, rtol=0, atol=3e-14)


@pytest.mark.parametrize(("charge", "expected"), ((0.5, 0.5), (-0.5, -0.5)))
def test_q_and_director_winding_recover_half_charge(charge, expected):
    lengths = (24.0, 24.0)
    center = (12.13, 11.87)
    q = periodic_core_plane((192, 192), lengths, center, charge=charge)
    sampler = core.PeriodicQSampler(q, lengths, order=5)
    detected, reason, q_winding, director_winding = core.classify_candidate_charge(
        sampler,
        center,
        S_bulk=1.0 / 3.0,
        other_centers=np.asarray(((5.0, 5.0),)),
    )
    assert detected == expected
    assert reason == "reliable_double_winding"
    np.testing.assert_allclose(q_winding, (2 * charge, 2 * charge), atol=2e-3)
    np.testing.assert_allclose(director_winding, (2 * charge, 2 * charge), atol=2e-3)


def test_low_s_minimum_without_winding_is_ambiguous_not_inherited():
    lengths = (24.0, 24.0)
    center = (12.0, 12.0)
    q = periodic_core_plane((128, 128), lengths, center, charge=None)
    sampler = core.PeriodicQSampler(q, lengths, order=5)
    detected, reason, _, _ = core.classify_candidate_charge(
        sampler, center, S_bulk=1.0 / 3.0
    )
    assert detected is None
    assert reason == "inconsistent_or_non_half_winding"


def test_winding_rejects_ring_with_insufficient_S_even_if_phase_winds():
    lengths = (24.0, 24.0)
    center = (12.0, 12.0)
    q = periodic_core_plane(
        (128, 128), lengths, center, charge=0.5, S_bulk=1.0e-3
    )
    sampler = core.PeriodicQSampler(q, lengths, order=5)
    detected, reason, _, _ = core.classify_candidate_charge(
        sampler, center, S_bulk=1.0 / 3.0
    )
    assert detected is None
    assert reason == "ring_S_below_reliable_winding_threshold"


def test_winding_rejects_periodically_nearby_other_core():
    lengths = (24.0, 24.0)
    center = (0.2, 12.0)
    q = periodic_core_plane((192, 192), lengths, center, charge=0.5)
    sampler = core.PeriodicQSampler(q, lengths, order=5)
    detected, reason, q_winding, director_winding = core.classify_candidate_charge(
        sampler,
        center,
        S_bulk=1.0 / 3.0,
        other_centers=np.asarray(((23.8, 12.0),)),
    )
    assert detected is None
    assert reason == "insufficient_winding_ring_clearance_from_other_core"
    assert q_winding == director_winding == ()


def test_charge_aware_hungarian_never_crosses_charge_groups():
    references = np.asarray(((1.0, 1.0), (9.0, 1.0)))
    charges = np.asarray((0.5, -0.5))
    candidates = (candidate(8.9, 1.0, 0.5), candidate(1.1, 1.0, -0.5))
    matches, unmatched = core.charge_aware_hungarian(
        candidates, references, charges, (10.0, 10.0), maximum_distance=20.0
    )
    assert matches == {0: 0, 1: 1}
    assert not unmatched


def test_hungarian_matching_uses_periodic_minimum_image_distance():
    references = np.asarray(((0.1, 5.0), (5.0, 5.0)))
    charges = np.asarray((0.5, 0.5))
    candidates = (candidate(9.9, 5.0, 0.5), candidate(5.2, 5.0, 0.5))
    matches, unmatched = core.charge_aware_hungarian(
        candidates, references, charges, (10.0, 10.0), maximum_distance=1.0
    )
    assert matches == {0: 0, 1: 1}
    assert not unmatched


def test_hungarian_dummy_leaves_far_candidate_unmatched():
    matches, unmatched = core.charge_aware_hungarian(
        (candidate(8.0, 8.0, 0.5),),
        np.asarray(((1.0, 1.0),)),
        np.asarray((0.5,)),
        (20.0, 20.0),
        maximum_distance=4.0,
    )
    assert matches == {}
    assert unmatched == {0}


def test_absolute_and_local_radius_crossing_definitions_are_distinct():
    radii = np.asarray((0.0, 1.0, 2.0, 3.0))
    profile = np.asarray((0.02, 0.12, 0.20, 0.24))
    absolute = core.first_outward_crossing(radii, profile, 0.5 / 3.0)
    local = core.first_outward_crossing(radii, profile, 0.13)
    assert absolute.reason == local.reason == "ok"
    assert absolute.value == pytest.approx(1.5833333333333333)
    assert local.value == pytest.approx(1.125)


@pytest.mark.parametrize(
    ("profile", "reason"),
    (
        ((0.6, 0.7, 0.8), "center_at_or_above_threshold"),
        ((0.0, 0.1, 0.2), "threshold_not_reached_by_rmax"),
    ),
)
def test_missing_threshold_crossing_is_nan_with_reason(profile, reason):
    result = core.first_outward_crossing(np.asarray((0.0, 1.0, 2.0)), profile, 0.5)
    assert math.isnan(result.value)
    assert result.reason == reason


@pytest.mark.parametrize(
    ("profile", "reason"),
    (
        (
            (0.0, math.nan, 0.6),
            "threshold_crossing_bracket_interrupted_by_invalid_profile_gap",
        ),
        ((0.0, math.nan, 0.2, 0.6), "profile_gap_before_threshold_crossing"),
        ((0.0, math.nan, 0.2, 0.3), "insufficient_contiguous_profile_samples"),
    ),
)
def test_threshold_crossing_never_bridges_or_skips_profile_gap(profile, reason):
    radii = np.arange(len(profile), dtype=np.float64)
    result = core.first_outward_crossing(radii, np.asarray(profile), 0.5)
    assert math.isnan(result.value)
    assert result.reason == reason


def test_periodic_voronoi_ownership_prevents_double_counting():
    centers = np.asarray(((0.2, 5.0), (9.8, 5.0)))
    x = np.linspace(-1.0, 11.0, 401)
    y = np.full_like(x, 5.0)
    first = core._owned_mask(x, y, centers, 0, (10.0, 10.0))
    second = core._owned_mask(x, y, centers, 1, (10.0, 10.0))
    np.testing.assert_array_equal(first.astype(int) + second.astype(int), np.ones_like(x, dtype=int))


def test_biaxiality_is_zero_for_known_uniaxial_q():
    q = compact_uniaxial_q(np.asarray((0.1, 0.2, 1.0 / 3.0)), np.asarray((0.1, 0.7, 1.3)))
    biaxiality, valid = core.biaxiality_from_Q(q, S_scale=1.0 / 3.0)
    assert np.all(valid)
    np.testing.assert_allclose(biaxiality, 0.0, atol=2e-14)


def test_biaxiality_masks_q_near_zero_instead_of_forming_zero_over_zero():
    q = np.zeros((3, 5), dtype=np.float64)
    biaxiality, valid = core.biaxiality_from_Q(q, S_scale=1.0 / 3.0)
    assert not np.any(valid)
    assert np.all(np.isnan(biaxiality))


def test_mmap_loader_requests_read_only_memory_mapping(tmp_path, monkeypatch):
    observed = {}

    class FakeArray:
        shape = (2, 3, 5)
        dtype = np.dtype(np.float64)

    def fake_load(path, **kwargs):
        observed.update(kwargs)
        return FakeArray()

    monkeypatch.setattr(core.np, "load", fake_load)
    path = tmp_path / "Q.npy"
    path.touch()
    assert core._mmap_float_array(path, (2, 3, 5)).shape == (2, 3, 5)
    assert observed == {"mmap_mode": "r", "allow_pickle": False}


def test_dct_plane_evaluation_reads_only_bounded_x_chunks(monkeypatch):
    underlying = np.ones((11, 7, 5, 5), dtype=np.float64)

    class SliceSpy:
        shape = underlying.shape
        dtype = underlying.dtype

        def __init__(self):
            self.sections = []

        def __getitem__(self, key):
            self.sections.append(key[0])
            return underlying[key]

    spy = SliceSpy()
    monkeypatch.setattr(core, "_mmap_float_array", lambda *args, **kwargs: spy)
    result = core.evaluate_q_dct_planes(
        Path("unused.npy"), (11, 7, 5), (0.1, 0.5, 0.9), chunk_size=3
    )
    assert result.shape == (11, 7, 3, 5)
    assert len(spy.sections) == 4
    assert all(section.stop - section.start <= 3 for section in spy.sections)


def test_projected_snapshot_hash_matches_benchmark_component_hash(tmp_path):
    rng = np.random.default_rng(11)
    q = rng.normal(size=(5, 4, 3, 5)).astype(np.float64)
    path = tmp_path / "Q_0.npy"
    np.save(path, q)
    digest = hashlib.sha256()
    for component in range(5):
        values = np.ascontiguousarray(q[..., component][None, ...])
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    assert core.projected_snapshot_sha256(path, (5, 4, 3), chunk_size=2) == digest.hexdigest()


def test_profile_ownership_in_plane_analysis_uses_only_matched_cores(monkeypatch, tmp_path):
    run = core_run(tmp_path, "R256")
    reliable = candidate(1.1, 2.0, 0.5)
    ambiguous = candidate(8.1, 7.0, None)

    class FakeSampler:
        pass

    monkeypatch.setattr(
        core,
        "detect_core_candidates",
        lambda *args, **kwargs: ([reliable, ambiguous], FakeSampler()),
    )
    observed = {}

    def fake_measure(*args, **kwargs):
        observed["centers"] = kwargs["all_centers"]
        observed["index"] = kwargs["center_index"]
        profile = {
            "radii": np.asarray((0.0, 1.0)),
            "median": np.asarray((0.0, 1.0)),
            "mean": np.asarray((0.0, 1.0)),
            "std": np.asarray((0.0, 0.0)),
            "count": np.asarray((128, 128)),
        }
        return {}, profile

    monkeypatch.setattr(core, "measure_core_profile", fake_measure)
    core.analyze_q_plane(
        np.zeros((16, 16, 5)),
        run,
        state="projected_initial",
        step=0,
        z_fraction=0.5,
        S_bulk=1 / 3,
        r_max=4.0,
        dr=0.1,
        angular_samples=128,
        match_radius=4.0,
        interpolation_order=5,
    )
    np.testing.assert_allclose(observed["centers"], ((1.1, 2.0),))
    assert observed["index"] == 0


def test_space_comparison_allows_only_grid_dependent_hashes(tmp_path):
    coarse = artifact(tmp_path, "R256")
    fine_metadata = copy.deepcopy(coarse.metadata)
    fine_metadata["solver"]["shape"] = [32, 32, 16]
    fine_metadata["initial_condition"]["raw_q_sha256"] = "fine-raw"
    fine_metadata["initial_condition"]["projected_q_sha256"] = "fine-projected"
    fine = artifact(tmp_path, "R512", shape=(32, 32, 16), metadata=fine_metadata)
    core.validate_comparability((coarse, fine), "space")


def test_space_comparison_rejects_seed_change(tmp_path):
    coarse = artifact(tmp_path, "R256")
    fine_metadata = copy.deepcopy(coarse.metadata)
    fine_metadata["solver"]["shape"] = [32, 32, 16]
    fine_metadata["initial_condition"]["raw_q_sha256"] = "fine-raw"
    fine_metadata["initial_condition"]["projected_q_sha256"] = "fine-projected"
    fine_metadata["initial_condition"]["seed"] = 25
    fine = artifact(tmp_path, "R512", shape=(32, 32, 16), metadata=fine_metadata)
    with pytest.raises(ValueError, match="outside the fields allowed"):
        core.validate_comparability((coarse, fine), "space")


def test_space_comparison_rejects_q_convention_change(tmp_path):
    coarse = artifact(tmp_path, "R256")
    fine_metadata = copy.deepcopy(coarse.metadata)
    fine_metadata["solver"]["shape"] = [32, 32, 16]
    fine_metadata["model"]["Q_convention"] = {"id": "wrong"}
    fine = artifact(tmp_path, "R512", shape=(32, 32, 16), metadata=fine_metadata)
    with pytest.raises(ValueError, match="outside the fields allowed"):
        core.validate_comparability((coarse, fine), "space")


def test_core_run_validation_rejects_changed_defect_csv(monkeypatch, tmp_path):
    first = core_run(tmp_path, "R256")
    second = core_run(tmp_path, "R320", shape=(20, 20, 10))
    second = core.CoreRun(
        artifact=second.artifact,
        q0_path=second.q0_path,
        q2d_path=second.q2d_path,
        defects_path=second.defects_path,
        defect_positions=second.defect_positions + np.asarray(((0.0, 0.0), (1e-3, 0.0))),
        defect_charges=second.defect_charges,
    )
    monkeypatch.setattr(core, "validate_comparability", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="continuous coordinates differ"):
        core.validate_core_runs((first, second), "space")


def test_load_core_run_requires_q0_projected_initial(monkeypatch, tmp_path):
    run_artifact = artifact(tmp_path, "R256")
    run_artifact.metadata["save_start_step"] = 5
    monkeypatch.setattr(core, "load_run", lambda directory: run_artifact)
    monkeypatch.setattr(core, "_mmap_float_array", lambda *args, **kwargs: np.empty(args[1]))
    monkeypatch.setattr(
        core,
        "load_defects_csv",
        lambda path: (np.asarray(((1.0, 1.0), (2.0, 2.0))), np.asarray((0.5, -0.5))),
    )
    with pytest.raises(ValueError, match="save_start_step=0"):
        core.load_core_run(tmp_path / "R256")


def test_first_order_time_richardson_estimate_is_twice_raw_difference(tmp_path):
    coarse = core_run(tmp_path, "dt005", dt=0.005, steps=200)
    fine = core_run(tmp_path, "dt0025", dt=0.0025, steps=400)
    summaries = []
    for run, value in ((coarse, 1.1), (fine, 1.0)):
        row = {
            "state": "final",
            "resolution": run.label,
            "step": run.artifact.steps,
            "z_fraction": 0.5,
            "charge": 0.5,
        }
        for metric in core.CORE_METRICS:
            row[f"{metric}_median"] = value
        summaries.append(row)
    changes = core.pairwise_metric_changes(summaries, (coarse, fine), mode="time")
    r50 = next(row for row in changes if row["metric"] == "r50_absolute")
    assert r50["first_order_richardson_time_error"] == pytest.approx(0.2)


def test_summaries_include_all_charge_mad_and_outlier_ids():
    rows = []
    for defect_id, charge, value in ((0, 0.5, 1.0), (1, 0.5, 1.01), (2, -0.5, 2.0)):
        row = {
            "state": "final",
            "resolution": "R512",
            "step": 200,
            "z_fraction": 0.5,
            "initial_charge": charge,
            "defect_id": defect_id,
            "match_status": "matched",
        }
        for metric in core.CORE_METRICS:
            row[metric] = value
        rows.append(row)
    summaries = core.summarize_groups(rows)
    all_charge = next(row for row in summaries if row["charge"] == "all")
    assert all_charge["r50_absolute_count"] == 3
    assert all_charge["r50_absolute_mad"] == pytest.approx(0.01)
    assert all_charge["r50_absolute_outlier_defect_ids"] == [2]


def test_profile_pairwise_change_reports_weighted_norm_and_time_scaling(tmp_path):
    coarse = core_run(tmp_path, "dt005", dt=0.005, steps=200)
    fine = core_run(tmp_path, "dt0025", dt=0.0025, steps=400)
    radii = np.asarray((0.0, 1.0, 2.0))
    profiles = []
    for run, values in ((coarse, (0.0, 0.5, 1.0)), (fine, (0.0, 0.4, 0.8))):
        profiles.append(
            {
                "state": "final",
                "resolution": run.label,
                "z_fraction": 0.5,
                "defect_id": 0,
                "charge": 0.5,
                "radii": radii,
                "median": np.asarray(values),
            }
        )
    changes = core.profile_pairwise_changes(profiles, (coarse, fine), mode="time")
    charge_row = next(row for row in changes if row["charge"] == 0.5)
    assert charge_row["weighted_relative_l2"] == pytest.approx(0.25)
    assert charge_row["first_order_richardson_weighted_relative_l2"] == pytest.approx(
        2.0 / 3.0
    )
    np.testing.assert_allclose(
        charge_row["richardson_extrapolated_profile_median"], (0.0, 0.3, 0.6)
    )


def test_csv_writer_refuses_overwrite_and_preserves_reason(tmp_path):
    path = tmp_path / "metrics.csv"
    rows = [{"r50_absolute": math.nan, "r50_absolute_reason": "threshold_not_reached_by_rmax"}]
    core.write_csv(path, rows)
    text = path.read_text()
    assert "threshold_not_reached_by_rmax" in text
    with pytest.raises(FileExistsError):
        core.write_csv(path, rows)


def test_analysis_output_directory_must_be_wholly_new(tmp_path):
    existing = tmp_path / "analysis_core"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="wholly new"):
        core.reserve_outputs(existing)
    paths = core.reserve_outputs(tmp_path / "fresh_analysis_core")
    assert paths["projected_initial_npz"].name == "projected_initial_core_profiles.npz"
    assert paths["projected_initial_plot"].name == "projected_initial_profiles_by_resolution.png"


def test_unequal_grid_three_level_fit_recovers_exact_order_on_nonnequal_ratios():
    h = np.asarray((100.0 / 256.0, 100.0 / 320.0, 100.0 / 512.0))
    values = 1.0 + 2.0 * h**2
    fit = core.unequal_grid_three_level_fit(h, values)
    assert fit["accepted"] is True
    assert fit["status"] == "accepted_auxiliary_diagnostic"
    assert fit["observed_order"] == pytest.approx(2.0, abs=3e-12)
    assert fit["m_infinity"] == pytest.approx(1.0, abs=3e-14)
    assert fit["coefficient_C"] == pytest.approx(2.0, abs=5e-13)


def test_unequal_grid_three_level_fit_rejects_nonmonotone_sequence():
    fit = core.unequal_grid_three_level_fit(
        (100.0 / 256.0, 100.0 / 320.0, 100.0 / 512.0),
        (3.0, 2.0, 2.5),
    )
    assert fit["accepted"] is False
    assert fit["status"] == "non_monotone"
    assert fit["rejection_reasons"] == ["non_monotone_or_sign_change"]


def test_unequal_grid_three_level_fit_rejects_when_no_positive_order_root_exists():
    fit = core.unequal_grid_three_level_fit(
        (100.0 / 256.0, 100.0 / 320.0, 100.0 / 512.0),
        (3.0, 2.0, -0.5),
    )
    assert fit["accepted"] is False
    assert fit["status"] == "no_positive_order_root"
    assert fit["rejection_reasons"] == ["no_positive_order_root"]


def test_unequal_grid_three_level_fit_rejects_perturbation_unstable_tiny_signal():
    fit = core.unequal_grid_three_level_fit(
        (100.0 / 256.0, 100.0 / 320.0, 100.0 / 512.0),
        (1.0, 1.0 + 1.0e-12, 1.0 + 1.5e-12),
    )
    assert fit["accepted"] is False
    assert fit["status"] == "unstable_fit_rejected"
    assert "perturbation_destroyed_positive_order_fit" in fit["rejection_reasons"]
    assert math.isnan(fit["observed_order"])


def test_generalized_spatial_fit_rejects_changed_matched_count(tmp_path):
    runs = (
        core_run(tmp_path, "R256", shape=(16, 16, 8)),
        core_run(tmp_path, "R320", shape=(20, 20, 10)),
        core_run(tmp_path, "R512", shape=(32, 32, 16)),
    )
    summaries = []
    for run, count, value in zip(runs, (2, 2, 1), (1.2, 1.1, 1.05)):
        row = {
            "state": "final",
            "resolution": run.label,
            "z_fraction": 0.5,
            "charge": 0.5,
        }
        for metric in core.CORE_METRICS:
            row[f"{metric}_count"] = count
            row[f"{metric}_median"] = value
        summaries.append(row)
    fits = core.generalized_spatial_fits(summaries, runs)
    r50 = next(row for row in fits if row["metric"] == "r50_absolute")
    assert r50["accepted"] is False
    assert r50["status"] == "matched_count_mismatch"
    assert r50["finite_counts"] == [2, 2, 1]


def test_profile_pairing_uses_same_defect_and_reports_distribution_outlier(tmp_path):
    coarse = core_run(tmp_path, "R256", shape=(16, 16, 8))
    fine = core_run(tmp_path, "R512", shape=(32, 32, 16))
    radii = np.asarray((0.0, 1.0, 2.0))
    base = np.asarray((0.0, 0.5, 1.0))
    profiles = []
    for defect_id, amplitude in enumerate((1.0, 2.0, 3.0, 4.0)):
        profiles.append(
            {
                "state": "final",
                "resolution": coarse.label,
                "z_fraction": 0.5,
                "defect_id": defect_id,
                "charge": 0.5,
                "radii": radii,
                "median": amplitude * base,
            }
        )
    for defect_id in reversed(range(4)):
        scale = 0.5 if defect_id == 3 else 0.99
        profiles.append(
            {
                "state": "final",
                "resolution": fine.label,
                "z_fraction": 0.5,
                "defect_id": defect_id,
                "charge": 0.5,
                "radii": radii,
                "median": scale * (defect_id + 1.0) * base,
            }
        )
    changes = core.profile_pairwise_changes(profiles, (coarse, fine), mode="space")
    individual = [row for row in changes if row["aggregation"] == "individual"]
    assert [row["defect_id"] for row in individual] == [0, 1, 2, 3]
    id_zero = next(row for row in individual if row["defect_id"] == 0)
    id_three = next(row for row in individual if row["defect_id"] == 3)
    assert id_zero["weighted_relative_l2"] == pytest.approx(0.01 / 0.99)
    assert id_three["weighted_relative_l2"] == pytest.approx(1.0)
    distribution = next(
        row
        for row in changes
        if row["aggregation"] == "paired_distribution" and row["charge"] == 0.5
    )
    assert distribution["individual_count"] == 4
    assert distribution["weighted_relative_l2_count"] == 4
    assert distribution["weighted_relative_l2_p90"] > distribution[
        "weighted_relative_l2_median"
    ]
    assert distribution["weighted_relative_l2_max"] == pytest.approx(1.0)
    assert distribution["weighted_relative_l2_outlier_defect_ids"] == [3]


def test_paired_scalar_changes_use_same_id_and_suppress_near_zero_percentages(tmp_path):
    coarse = core_run(tmp_path, "R256", shape=(16, 16, 8))
    fine = core_run(tmp_path, "R512", shape=(32, 32, 16))

    def row(run, defect_id, radius_value, s_min):
        result = {
            "state": "final",
            "resolution": run.label,
            "z_fraction": 0.5,
            "defect_id": defect_id,
            "initial_charge": 0.5,
            "detected_charge": 0.5,
            "match_status": "matched",
        }
        for metric in core.CORE_METRICS:
            result[metric] = radius_value
        result["S_min"] = s_min
        for metric in core.RADIUS_METRICS:
            result[f"{metric}_reason"] = "ok"
            result[f"{metric}_crossing_count"] = 1
        return result

    rows = (
        row(coarse, 0, 1.0, 1.0e-12),
        row(coarse, 1, 10.0, 3.0e-12),
        row(fine, 1, 20.0, 4.0e-12),
        row(fine, 0, 1.1, 2.0e-12),
    )
    changes = core.paired_individual_metric_changes(rows, (coarse, fine), mode="space")
    r50_id_zero = next(
        item
        for item in changes
        if item["aggregation"] == "individual"
        and item["metric"] == "r50_absolute"
        and item["defect_id"] == 0
    )
    assert r50_id_zero["coarse_value"] == 1.0
    assert r50_id_zero["fine_value"] == 1.1
    assert r50_id_zero["absolute_difference"] == pytest.approx(0.1)
    s_min_id_zero = next(
        item
        for item in changes
        if item["aggregation"] == "individual"
        and item["metric"] == "S_min"
        and item["defect_id"] == 0
    )
    assert math.isnan(s_min_id_zero["relative_change"])
    assert s_min_id_zero["relative_change_reason"] == "not_reported_for_near_zero_metric"
    assert s_min_id_zero["within_5_percent"] is False


def test_multiple_outward_crossings_are_nan_with_explicit_reason():
    crossing = core.first_outward_crossing(
        np.asarray((0.0, 1.0, 2.0, 3.0)),
        np.asarray((0.0, 0.6, 0.4, 0.7)),
        0.5,
    )
    assert math.isnan(crossing.value)
    assert crossing.reason == "multiple_outward_crossings_unstable_branch"
    assert crossing.crossing_count == 2


def test_S_far_is_invalidated_when_voronoi_ownership_leaves_too_little_ring():
    class ConstantSampler:
        lengths = np.asarray((20.0, 20.0))

        def sample_q(self, x, y):
            shape = np.broadcast(np.asarray(x), np.asarray(y)).shape
            return compact_uniaxial_q(np.full(shape, 1.0 / 3.0))

        def sample_S(self, x, y):
            shape = np.broadcast(np.asarray(x), np.asarray(y)).shape
            return np.full(shape, 1.0 / 3.0)

    centers = np.asarray(((10.0, 10.0), (10.2, 10.0), (10.0, 10.2)))
    metrics, _ = core.measure_core_profile(
        ConstantSampler(),
        centers[0],
        all_centers=centers,
        center_index=0,
        S_bulk=1.0 / 3.0,
        r_max=4.0,
        dr=0.1,
        angular_samples=128,
    )
    assert 0.0 < metrics["S_far_valid_fraction"] < core.S_FAR_MINIMUM_VALID_FRACTION
    assert math.isnan(metrics["S_far"])
    assert metrics["S_far_reason"] == "insufficient_far_ring_valid_fraction"


def test_small_local_reference_span_invalidates_all_local_metrics():
    class WeakContrastSampler:
        lengths = np.asarray((20.0, 20.0))
        center = np.asarray((10.0, 10.0))

        def _S(self, x, y):
            x_values, y_values = np.broadcast_arrays(
                np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
            )
            radius = np.hypot(x_values - self.center[0], y_values - self.center[1])
            return 0.30 + 1.0e-4 * (1.0 - np.exp(-(radius**2)))

        def sample_q(self, x, y):
            return compact_uniaxial_q(self._S(x, y))

        def sample_S(self, x, y):
            return self._S(x, y)

    sampler = WeakContrastSampler()
    metrics, _ = core.measure_core_profile(
        sampler,
        sampler.center,
        all_centers=sampler.center[None, :],
        center_index=0,
        S_bulk=1.0 / 3.0,
        r_max=4.0,
        dr=0.05,
        angular_samples=128,
    )
    assert metrics["local_reference_well_conditioned"] is False
    assert metrics["local_reference_quality_reason"] == (
        "local_reference_span_below_minimum_fraction_of_S_bulk"
    )
    assert metrics["local_reference_span"] < metrics["local_reference_minimum_span"]
    for metric in ("r50_local", "r90_local", "core_deficit_local", "low_s_area_local"):
        assert math.isnan(metrics[metric])
    assert metrics["r50_local_reason"] == metrics["local_reference_quality_reason"]
    assert metrics["r90_local_reason"] == metrics["local_reference_quality_reason"]
    assert metrics["core_deficit_local_reason"] == metrics["local_reference_quality_reason"]
    assert metrics["low_s_area_local_reason"] == metrics["local_reference_quality_reason"]


def test_status_counts_report_missing_initial_and_ambiguous_extra_together():
    common = {
        "state": "final",
        "resolution": "R512",
        "step": 200,
        "z_fraction": 0.5,
    }
    rows = (
        {
            **common,
            "defect_id": 0,
            "initial_charge": 0.5,
            "detected_charge": 0.5,
            "match_status": "matched",
        },
        {
            **common,
            "defect_id": 1,
            "initial_charge": -0.5,
            "detected_charge": math.nan,
            "match_status": "missing",
        },
        {
            **common,
            "defect_id": "",
            "initial_charge": math.nan,
            "detected_charge": math.nan,
            "match_status": "ambiguous",
            "center_x": 3.0,
            "center_y": 4.0,
        },
    )
    counts = core.summarize_status_counts(rows)
    all_charges = next(row for row in counts if row["charge"] == "all")
    assert all_charges["expected_initial_count"] == 2
    assert all_charges["matched_initial_count"] == 1
    assert all_charges["missing_initial_count"] == 1
    assert all_charges["missing_initial_ids"] == [1]
    assert all_charges["ambiguous_candidate_count"] == 1
    assert all_charges["all_expected_initial_reliably_matched"] is False


def test_chunked_dct_plane_evaluation_matches_full_array_evaluation(tmp_path):
    rng = np.random.default_rng(2026)
    shape = (13, 9, 17)
    q = rng.normal(size=(*shape, 5)).astype(np.float64)
    path = tmp_path / "Q.npy"
    np.save(path, q)
    fractions = (0.1, 0.5, 0.9)
    chunked = core.evaluate_q_dct_planes(path, shape, fractions, chunk_size=4)
    full = core.dct_modal_evaluate_chunk(np.moveaxis(q, 2, -1), fractions)
    full = np.moveaxis(full, -1, 2)
    np.testing.assert_allclose(chunked, full, rtol=3e-14, atol=3e-14)


def test_multidefect_sampled_r50_approaches_continuous_multidefect_reference():
    lengths = (24.0, 24.0)
    positions = np.asarray(((8.23, 12.07), (15.91, 12.41)))
    reference, _ = core.continuous_reference_profile(
        0,
        positions,
        lengths,
        core_radius=1.5,
        S_initial=1.0 / 3.0,
        S_bulk=1.0 / 3.0,
        r_max=4.0,
        dr=0.025,
        angular_samples=128,
    )
    errors = []
    for n in (32, 64, 128):
        x = (np.arange(n) + 0.5) * lengths[0] / n
        y = (np.arange(n) + 0.5) * lengths[1] / n
        xx, yy = np.meshgrid(x, y, indexing="ij")
        sampled_S = core.continuous_multidefect_S(
            xx,
            yy,
            positions=positions,
            lengths=lengths,
            core_radius=1.5,
            S_initial=1.0 / 3.0,
        )
        sampler = core.PeriodicQSampler(
            compact_uniaxial_q(sampled_S), lengths, order=5
        )
        metrics, _ = core.measure_core_profile(
            sampler,
            positions[0],
            all_centers=positions,
            center_index=0,
            S_bulk=1.0 / 3.0,
            r_max=4.0,
            dr=0.025,
            angular_samples=128,
        )
        assert metrics["r50_absolute_reason"] == "ok"
        errors.append(abs(metrics["r50_absolute"] - reference["r50_absolute"]))
    assert errors[-1] < errors[0]
    assert errors[-1] < 1.0e-3


def test_space_comparison_rejects_physical_parameter_change(tmp_path):
    coarse = artifact(tmp_path, "R256")
    fine_metadata = copy.deepcopy(coarse.metadata)
    fine_metadata["solver"]["shape"] = [32, 32, 16]
    fine_metadata["initial_condition"]["raw_q_sha256"] = "fine-raw"
    fine_metadata["initial_condition"]["projected_q_sha256"] = "fine-projected"
    fine_metadata["model"]["parameters"]["S_bulk"] = 0.34
    fine = artifact(tmp_path, "R512", shape=(32, 32, 16), metadata=fine_metadata)
    with pytest.raises(ValueError, match="outside the fields allowed"):
        core.validate_comparability((coarse, fine), "space")


def test_mmap_loader_rejects_float32_snapshot(tmp_path):
    path = tmp_path / "Q_float32.npy"
    np.save(path, np.zeros((2, 3, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="must contain float64"):
        core._mmap_float_array(path, (2, 3, 5))


def test_generalized_fit_rejects_different_finite_metric_id_sets(tmp_path):
    runs = (
        core_run(tmp_path, "R256", shape=(16, 16, 8)),
        core_run(tmp_path, "R320", shape=(20, 20, 10)),
        core_run(tmp_path, "R512", shape=(32, 32, 16)),
    )
    summaries = []
    for run, ids, value in zip(runs, ((0, 1), (0, 2), (0, 1)), (1.2, 1.1, 1.05)):
        row = {
            "state": "final",
            "resolution": run.label,
            "z_fraction": 0.5,
            "charge": 0.5,
        }
        for metric in core.CORE_METRICS:
            row[f"{metric}_count"] = 2
            row[f"{metric}_defect_ids"] = list(ids)
            row[f"{metric}_median"] = value
        summaries.append(row)
    fits = core.generalized_spatial_fits(summaries, runs)
    r50 = next(row for row in fits if row["metric"] == "r50_absolute")
    assert r50["accepted"] is False
    assert r50["status"] == "paired_defect_id_mismatch"


def test_near_zero_space_difference_makes_time_gate_indeterminate(
    tmp_path, monkeypatch
):
    time_runs = (
        core_run(tmp_path, "dt005", shape=(32, 32, 16), dt=0.005, steps=200),
        core_run(tmp_path, "dt0025", shape=(32, 32, 16), dt=0.0025, steps=400),
    )
    identity = {
        "resolution": "dt005",
        "directory": str(time_runs[0].artifact.directory),
        "shape": [512, 512, 128],
        "lengths": [100.0, 100.0, 20.0],
        "dt": 0.005,
        "steps": 200,
        "raw_q_sha256": "a" * 64,
        "projected_q_sha256": "b" * 64,
        "defect_table_sha256": "c" * 64,
    }
    monkeypatch.setattr(core, "input_run_manifest", lambda runs: [identity])

    def status_rows(labels):
        return [
            {
                "state": "final",
                "resolution": label,
                "z_fraction": z_fraction,
                "charge": "all",
                "expected_initial_count": 12,
                "matched_initial_count": 12,
                "missing_initial_count": 0,
                "ambiguous_candidate_count": 0,
                "unmatched_reliable_candidate_count": 0,
            }
            for label in labels
            for z_fraction in core.DEFAULT_Z_FRACTIONS
        ]

    space_report = {
        "mode": "space",
        "physical_scales_and_retained_modes": {
            "grids": [
                {"resolution": "R320", "shape": [320, 320, 80]},
                {"resolution": "R512", "shape": [512, 512, 128]},
            ]
        },
        "input_run_manifest": [{**identity, "resolution": "R512"}],
        "status_counts": status_rows(("R320", "R512")),
        "paired_individual_metric_changes_and_distributions": [
            {
                "aggregation": "individual",
                "state": "final",
                "z_fraction": 0.5,
                "defect_id": 0,
                "metric": "r50_absolute",
                "coarse": "R320",
                "fine": "R512",
                "absolute_difference": 0.0,
            }
        ],
        "charge_conditioned_profile_pairwise_changes": [],
    }
    path = tmp_path / "space.json"
    path.write_text(json.dumps(space_report))
    time_metric_changes = [
        {
            "aggregation": "individual",
            "state": "final",
            "z_fraction": 0.5,
            "charge": 0.5,
            "defect_id": 0,
            "metric": "r50_absolute",
            "richardson_model_valid": True,
            "first_order_richardson_time_error": 1.0e-5,
        }
    ]
    result = core.time_vs_space_error_gate(
        path,
        time_metric_changes,
        (),
        time_runs,
        status_rows(("dt005", "dt0025")),
    )
    gate = result["scalar_metric_gates"][0]
    assert gate["passed"] is None
    assert gate["absolute_time_tolerance_satisfied"] is True
    assert result["all_available_scalar_gates_pass"] is False


def test_beris_task_contract_accepts_space_matrix(monkeypatch, tmp_path):
    shapes = ((256, 256, 64), (320, 320, 80), (512, 512, 128))
    runs = tuple(
        materialize_contract_core_run(
            tmp_path,
            f"R{shape[0]}",
            beris_contract_metadata(
                shape,
                config_character=character,
                initial_character=character,
            ),
        )
        for shape, character in zip(shapes, "abc")
    )
    monkeypatch.setattr(core, "_mmap_float_array", lambda *args, **kwargs: None)

    core.validate_core_runs(runs, "space")
    core.validate_task_contract(runs, "space")


def test_beris_task_contract_accepts_r512_time_pair(monkeypatch, tmp_path):
    shape = (512, 512, 128)
    runs = (
        materialize_contract_core_run(
            tmp_path,
            "dt005",
            beris_contract_metadata(
                shape, dt=0.005, steps=200, config_character="a"
            ),
        ),
        materialize_contract_core_run(
            tmp_path,
            "dt0025",
            beris_contract_metadata(
                shape, dt=0.0025, steps=400, config_character="c"
            ),
        ),
    )
    monkeypatch.setattr(core, "_mmap_float_array", lambda *args, **kwargs: None)

    core.validate_core_runs(runs, "time")
    core.validate_task_contract(runs, "time")


def test_beris_physical_scale_uses_raw_ldg_l1(tmp_path):
    run = materialize_contract_core_run(
        tmp_path,
        "R512",
        beris_contract_metadata((512, 512, 128)),
    )

    scales = core.physical_scale_summary((run,))

    assert scales["mu_S"] == pytest.approx(0.05)
    assert scales["xi_S"] == pytest.approx(
        math.sqrt(0.024691358024691357 / 0.05)
    )
    assert scales["active_length"] == pytest.approx(
        math.sqrt(0.012345679012345678 / 0.01)
    )


def _set_nested_value(mapping, path, value):
    target = mapping
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "message"),
    (
        (("ldg_l1",), "ldg_l1"),
        (("q_elastic_relaxation",), "q_elastic_relaxation"),
        (("model", "parameters", "frank_K"), "frank_K"),
        (("model", "parameters", "ldg_L1"), "ldg_L1"),
        (("model", "q_dynamics", "raw_coefficients", "L1"), "raw_coefficients.L1"),
        (
            ("model", "q_dynamics", "evolution_coefficients", "L1_over_gamma"),
            "L1_over_gamma",
        ),
    ),
)
def test_beris_parameter_mapping_is_strict(
    monkeypatch, tmp_path, path, message
):
    shapes = ((256, 256, 64), (320, 320, 80), (512, 512, 128))
    runs = []
    for index, shape in enumerate(shapes):
        metadata = beris_contract_metadata(
            shape,
            config_character="abc"[index],
            initial_character="abc"[index],
        )
        if index == 0:
            _set_nested_value(metadata, path, 0.123456789)
        runs.append(
            materialize_contract_core_run(tmp_path, f"R{shape[0]}", metadata)
        )
    monkeypatch.setattr(core, "_mmap_float_array", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match=message):
        core.validate_task_contract(tuple(runs), "space")


def test_core_validation_rejects_cross_model_comparison(tmp_path):
    legacy = core_run(tmp_path, "legacy", shape=(256, 256, 64))
    beris = materialize_contract_core_run(
        tmp_path,
        "beris",
        beris_contract_metadata((320, 320, 80)),
    )

    with pytest.raises(ValueError, match="outside the fields allowed"):
        core.validate_core_runs((legacy, beris), "space")


def test_core_validation_rejects_implementation_provenance_change(tmp_path):
    first = materialize_contract_core_run(
        tmp_path,
        "R256",
        beris_contract_metadata((256, 256, 64), initial_character="a"),
    )
    changed = beris_contract_metadata(
        (320, 320, 80), config_character="b", initial_character="b"
    )
    changed["implementation_provenance"]["files"][
        "pssolver/solver.py"
    ] = "2" * 64
    second = materialize_contract_core_run(tmp_path, "R320", changed)

    with pytest.raises(ValueError, match="outside the fields allowed"):
        core.validate_core_runs((first, second), "space")


def test_defect_core_api_explicitly_rejects_dealias_mode():
    with pytest.raises(ValueError, match="unsupported defect-core validation mode"):
        core.validate_task_contract((), "dealias")


def test_beris_input_manifest_and_readme_record_correct_model(tmp_path):
    run = materialize_contract_core_run(
        tmp_path,
        "R512",
        beris_contract_metadata((512, 512, 128)),
    )
    manifest = core.input_run_manifest((run,))[0]
    assert manifest["script"] == core.BERIS_EDWARDS_BENCHMARK_SCRIPT
    assert manifest["model_variant"] == core.BERIS_EDWARDS_MODEL_VARIANT
    assert len(manifest["metadata_sha256"]) == 64
    assert manifest["implementation_provenance"]["schema_version"] == 1

    readme = tmp_path / "README.md"
    args = SimpleNamespace(
        mode="space",
        z_fractions=(0.1, 0.5, 0.9),
        r_max=4.0,
        dr=0.05,
        angular_samples=128,
        match_radius=4.0,
        chunk_size=8,
        interpolation_order=5,
    )
    core.write_readme(
        readme,
        args=args,
        runs=(run,),
        output_paths={"json": tmp_path / "summary.json"},
        provenance={
            "analysis_runtime": {
                "package_versions": {},
                "command": "python analyzer.py",
                "executable": "python",
                "prefix": "test",
                "loaded_modules": "",
            },
            "simulation_provenance": {
                "provenance_directory": "test-provenance",
                "jobs_directory": "test-jobs",
            },
        },
        scales=core.physical_scale_summary((run,)),
        continuous_reference_dr=0.005,
        continuous_reference_angular_samples=512,
    )
    text = readme.read_text(encoding="utf-8")
    assert "complete one-constant Beris--Edwards" in text
    assert "omits some passive elastic/reactive nematic stresses" not in text
    assert core.BERIS_EDWARDS_BENCHMARK_SCRIPT in text

    args.mode = "time"
    time_readme = tmp_path / "README-time.md"
    core.write_readme(
        time_readme,
        args=args,
        runs=(run,),
        output_paths={"json": tmp_path / "summary-time.json"},
        provenance={
            "analysis_runtime": {
                "package_versions": {},
                "command": "python analyzer.py",
                "executable": "python",
                "prefix": "test",
                "loaded_modules": "",
            },
            "simulation_provenance": {
                "provenance_directory": "test-provenance",
                "jobs_directory": "test-jobs",
            },
        },
        scales=core.physical_scale_summary((run,)),
        continuous_reference_dr=0.005,
        continuous_reference_angular_samples=512,
    )
    time_text = time_readme.read_text(encoding="utf-8")
    assert "Defect-core R512 time-step sensitivity precheck" in time_text
    assert "does not establish a temporal convergence order" in time_text
