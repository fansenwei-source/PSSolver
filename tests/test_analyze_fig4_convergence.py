import copy

import numpy as np
import pytest

from scripts_plane.analyze_fig4_convergence import (
    RunArtifact,
    compact_q_frobenius_squared,
    demeaned_relative_l2,
    observed_order,
    q_relative_errors,
    q_scalar_metrics,
    select_geometric_dt_triplet,
    validate_comparability,
    velocity_scalar_metrics,
)


def test_compact_q_frobenius_squared_counts_qzz_and_off_diagonals():
    # Qzz=-(Qxx+Qyy)=-3, so Q:Q=1+4+9+2*(9+16+25)=114.
    q = np.asarray([1.0, 3.0, 4.0, 2.0, 5.0])
    assert compact_q_frobenius_squared(q) == pytest.approx(114.0)


def test_q_relative_errors_use_full_tensor_frobenius_norm(tmp_path):
    reference = np.zeros((2, 1, 1, 5), dtype=np.float32)
    reference[0, 0, 0] = (1.0, 3.0, 4.0, 2.0, 5.0)
    reference[1, 0, 0] = (0.5, 0.0, 0.0, -0.25, 0.0)
    candidate = 2.0 * reference
    reference_path = tmp_path / "reference.npy"
    candidate_path = tmp_path / "candidate.npy"
    np.save(reference_path, reference)
    np.save(candidate_path, candidate)

    relative_l2, relative_linf = q_relative_errors(
        candidate_path,
        reference_path,
        shape=(2, 1, 1),
        chunk_size=1,
    )

    assert relative_l2 == pytest.approx(1.0)
    assert relative_linf == pytest.approx(1.0)


def test_pressure_relative_error_is_invariant_to_constant_gauge():
    reference = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert demeaned_relative_l2(reference + 123.0, reference) == pytest.approx(0.0)


def test_velocity_metrics_report_vector_and_component_rms(tmp_path):
    velocity = np.zeros((2, 1, 1, 3), dtype=np.float32)
    velocity[..., 0] = 3.0
    velocity[..., 1] = 4.0
    path = tmp_path / "u.npy"
    np.save(path, velocity)

    metrics = velocity_scalar_metrics(path, (2, 1, 1), chunk_size=1)

    assert metrics["u_rms"] == pytest.approx(5.0)
    assert metrics["ux_rms"] == pytest.approx(3.0)
    assert metrics["uy_rms"] == pytest.approx(4.0)
    assert metrics["uz_rms"] == pytest.approx(0.0)
    assert metrics["ux_mean"] == pytest.approx(3.0)
    assert metrics["uy_mean"] == pytest.approx(4.0)


def test_q_scalar_metrics_preserve_float64_order_parameter_precision(tmp_path):
    expected = np.asarray((1.0 / 3.0, 1.0 / 3.0 + 2.0e-10))
    q = np.zeros((2, 1, 1, 5), dtype=np.float64)
    q[..., 0] = expected.reshape(2, 1, 1)
    q[..., 3] = -0.5 * expected.reshape(2, 1, 1)
    path = tmp_path / "Q.npy"
    np.save(path, q)

    metrics = q_scalar_metrics(path, (2, 1, 1), chunk_size=1)

    assert metrics["S_mean"] == pytest.approx(np.mean(expected), abs=1.0e-14)


def test_observed_order_for_halved_first_order_error():
    assert observed_order(0.2, 0.1, 2.0) == pytest.approx(1.0)
    assert observed_order(0.0, 0.0, 2.0) is None


def test_geometric_order_triplet_is_selected_from_four_dt_levels():
    selected = select_geometric_dt_triplet((0.01, 0.005, 0.0025, 0.001))

    assert selected == pytest.approx((0.01, 0.005, 0.0025))
    assert select_geometric_dt_triplet((0.01, 0.006, 0.002)) is None


def test_automatic_order_triplet_never_skips_irregular_dt_levels():
    # A combinatorial search would incorrectly find (0.1, 0.01, 0.001).
    levels = (0.1, 0.07, 0.03, 0.01, 0.007, 0.003, 0.001, 0.0007, 0.0003)

    assert select_geometric_dt_triplet(levels) is None


def test_explicit_order_triplet_may_select_nonadjacent_supplied_levels():
    levels = (0.1, 0.07, 0.03, 0.01, 0.007, 0.003, 0.001)

    assert select_geometric_dt_triplet(
        levels, requested=(0.1, 0.01, 0.001)
    ) == pytest.approx((0.1, 0.01, 0.001))


def test_explicit_order_triplet_must_be_available_ordered_and_geometric():
    levels = (0.01, 0.005, 0.0025, 0.001)

    with pytest.raises(ValueError, match="absent"):
        select_geometric_dt_triplet(levels, requested=(0.02, 0.01, 0.005))
    with pytest.raises(ValueError, match="COARSE > MEDIUM > FINE"):
        select_geometric_dt_triplet(levels, requested=(0.0025, 0.005, 0.01))
    with pytest.raises(ValueError, match="geometrically refined"):
        select_geometric_dt_triplet(levels, requested=(0.01, 0.005, 0.001))


def _spectral_refresh_metadata(
    *,
    dt,
    steps,
    mode="physical_time",
    requested_time=0.2,
    requested_steps=None,
    effective_steps=20,
    effective_time=0.2,
):
    return {
        "schema_version": 1,
        "script": "Plane_fig4_benchmark.py",
        "status": "complete",
        "completed_steps": steps,
        "solver": {
            "shape": [8, 8, 4],
            "lengths": [10.0, 10.0, 4.0],
            "dt": dt,
            "steps": steps,
            "save_interval": steps,
        },
        "model": {"activity": 0.018},
        "numerics": {
            "dealias": "cubic_half",
            "spectral_refresh": {
                "mode": mode,
                "requested_interval_time": requested_time,
                "requested_interval_steps": requested_steps,
                "effective_interval_steps": effective_steps,
                "effective_interval_time": effective_time,
                "phase_origin_step": 0,
            },
        },
    }


def _artifact(tmp_path, label, metadata):
    solver = metadata["solver"]
    shape = tuple(solver["shape"])
    lengths = tuple(solver["lengths"])
    directory = tmp_path / label
    return RunArtifact(
        label=label,
        directory=directory,
        metadata=metadata,
        shape=shape,
        lengths=lengths,
        dt=solver["dt"],
        steps=solver["steps"],
        final_time=solver["dt"] * solver["steps"],
        q_path=directory / "Q.npy",
        u_path=directory / "u.npy",
        p_path=directory / "p.npy",
        diagnostics_path=directory / "diagnostics.npy",
    )


def test_time_comparison_accepts_physical_refresh_with_different_step_intervals(
    tmp_path,
):
    coarse = _artifact(
        tmp_path,
        "coarse",
        _spectral_refresh_metadata(dt=0.01, steps=100, effective_steps=20),
    )
    fine = _artifact(
        tmp_path,
        "fine",
        _spectral_refresh_metadata(dt=0.005, steps=200, effective_steps=40),
    )

    validate_comparability((coarse, fine), "time")


@pytest.mark.parametrize(
    ("field", "different_value"),
    (
        ("requested_interval_time", 0.1),
        ("effective_interval_time", 0.1),
        ("phase_origin_step", 1),
    ),
)
def test_time_comparison_rejects_other_physical_refresh_differences(
    tmp_path, field, different_value
):
    coarse_metadata = _spectral_refresh_metadata(
        dt=0.01, steps=100, effective_steps=20
    )
    fine_metadata = _spectral_refresh_metadata(
        dt=0.005, steps=200, effective_steps=40
    )
    fine_metadata["numerics"]["spectral_refresh"][field] = different_value

    with pytest.raises(ValueError):
        validate_comparability(
            (
                _artifact(tmp_path, "coarse", coarse_metadata),
                _artifact(tmp_path, "fine", fine_metadata),
            ),
            "time",
        )


def test_time_comparison_keeps_step_refresh_interval_strict(tmp_path):
    coarse_metadata = _spectral_refresh_metadata(
        dt=0.01,
        steps=100,
        mode="steps",
        requested_time=None,
        requested_steps=20,
        effective_steps=20,
    )
    fine_metadata = _spectral_refresh_metadata(
        dt=0.005,
        steps=200,
        mode="steps",
        requested_time=None,
        requested_steps=40,
        effective_steps=40,
    )

    with pytest.raises(ValueError, match="outside the fields allowed"):
        validate_comparability(
            (
                _artifact(tmp_path, "coarse", coarse_metadata),
                _artifact(tmp_path, "fine", fine_metadata),
            ),
            "time",
        )


def test_time_comparison_keeps_unrelated_provenance_strict(tmp_path):
    coarse_metadata = _spectral_refresh_metadata(
        dt=0.01, steps=100, effective_steps=20
    )
    fine_metadata = copy.deepcopy(
        _spectral_refresh_metadata(dt=0.005, steps=200, effective_steps=40)
    )
    fine_metadata["model"]["activity"] = 0.019

    with pytest.raises(ValueError, match="outside the fields allowed"):
        validate_comparability(
            (
                _artifact(tmp_path, "coarse", coarse_metadata),
                _artifact(tmp_path, "fine", fine_metadata),
            ),
            "time",
        )


def test_space_comparison_ignores_only_grid_dependent_initial_hashes(tmp_path):
    coarse_metadata = _spectral_refresh_metadata(dt=0.005, steps=200)
    coarse_metadata["initial_condition"] = {
        "name": "analytic",
        "seed": 24,
        "raw_q_sha256": "coarse-raw",
        "projected_q_sha256": "coarse-projected",
    }
    fine_metadata = copy.deepcopy(coarse_metadata)
    fine_metadata["solver"]["shape"] = [16, 16, 8]
    fine_metadata["initial_condition"]["raw_q_sha256"] = "fine-raw"
    fine_metadata["initial_condition"]["projected_q_sha256"] = "fine-projected"

    validate_comparability(
        (
            _artifact(tmp_path, "coarse", coarse_metadata),
            _artifact(tmp_path, "fine", fine_metadata),
        ),
        "space",
    )

    fine_metadata["initial_condition"]["seed"] = 25
    with pytest.raises(ValueError, match="outside the fields allowed"):
        validate_comparability(
            (
                _artifact(tmp_path, "coarse", coarse_metadata),
                _artifact(tmp_path, "fine", fine_metadata),
            ),
            "space",
        )
