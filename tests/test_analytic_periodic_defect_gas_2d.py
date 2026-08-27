from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    analytic_periodic_defect_gas_2d,
    create_initial_condition,
    eigenframe_from_Q,
    sample_periodic_neutral_defects_2d,
)
from pssolver.models.active_nematics.initial_conditions import _periodic_nematic_phase_2d


def periodic_pairwise_distances(positions, lengths):
    positions = np.asarray(positions)
    lengths = np.asarray(lengths)
    delta = positions[:, None, :] - positions[None, :, :]
    delta -= lengths * np.round(delta / lengths)
    distance = np.linalg.norm(delta, axis=-1)
    distance[np.diag_indices_from(distance)] = np.inf
    return distance


def test_periodic_defect_sampling_is_reproducible_neutral_and_separated():
    kwargs = dict(
        lengths=(64.0, 48.0),
        num_defect_pairs=5,
        min_separation=7.5,
        seed=23,
    )
    positions, charges = sample_periodic_neutral_defects_2d(**kwargs)
    repeated_positions, repeated_charges = sample_periodic_neutral_defects_2d(**kwargs)

    np.testing.assert_array_equal(positions, repeated_positions)
    np.testing.assert_array_equal(charges, repeated_charges)
    assert positions.shape == (10, 2)
    assert np.count_nonzero(charges == 0.5) == 5
    assert np.count_nonzero(charges == -0.5) == 5
    assert charges.sum() == 0.0
    assert periodic_pairwise_distances(positions, kwargs["lengths"]).min() >= 7.5


@pytest.mark.parametrize("lengths", [(32.0, 48.0), (48.0, 32.0), (40.0, 40.0)])
def test_jacobi_theta_phase_is_seamless_on_rectangular_torus(lengths):
    positions, charges = sample_periodic_neutral_defects_2d(
        lengths=lengths,
        num_defect_pairs=3,
        min_separation=5.0,
        seed=9,
    )
    x = np.linspace(0.13, lengths[0] - 0.27, 13)[:, None]
    y = np.linspace(0.19, lengths[1] - 0.31, 15)[None, :]
    phase = _periodic_nematic_phase_2d(
        x,
        y,
        lengths=lengths,
        positions=positions,
        charges=charges,
    )
    phase_after_x_period = _periodic_nematic_phase_2d(
        x + lengths[0],
        y,
        lengths=lengths,
        positions=positions,
        charges=charges,
    )
    phase_after_y_period = _periodic_nematic_phase_2d(
        x,
        y + lengths[1],
        lengths=lengths,
        positions=positions,
        charges=charges,
    )

    np.testing.assert_allclose(phase_after_x_period, phase, rtol=0, atol=2e-13)
    np.testing.assert_allclose(phase_after_y_period, phase, rtol=0, atol=2e-13)
    np.testing.assert_allclose(np.abs(phase), 1.0, rtol=0, atol=2e-15)


def test_analytic_defect_gas_returns_pssolver_normalized_reproducible_q():
    kwargs = dict(
        lengths=(48.0, 40.0),
        num_defect_pairs=3,
        min_separation=7.0,
        core_radius=1.2,
        seed=31,
        S_initial=1.0 / 3.0,
    )
    fields = create_initial_condition(
        "analytic_periodic_defect_gas_2d",
        (96, 80),
        **kwargs,
    )
    repeated = analytic_periodic_defect_gas_2d((96, 80), **kwargs)
    changed_seed = analytic_periodic_defect_gas_2d(
        (96, 80),
        **{**kwargs, "seed": 32},
    )

    assert set(fields) == set(Q_COMPONENTS)
    for name in Q_COMPONENTS:
        assert fields[name].shape == (96, 80)
        assert fields[name].dtype == torch.float32
        assert fields[name].device.type == "cpu"
        torch.testing.assert_close(fields[name], repeated[name], rtol=0, atol=0)
    assert not torch.allclose(fields["Qxy"], changed_seed["Qxy"])
    assert torch.count_nonzero(fields["Qxz"]) == 0
    assert torch.count_nonzero(fields["Qyz"]) == 0

    # For Q=(3S/2)(nn-I/3), the complex in-plane order has magnitude 3S/2.
    local_S = (2.0 / 3.0) * torch.sqrt(
        (fields["Qxx"] - fields["Qyy"]).square()
        + (2.0 * fields["Qxy"]).square()
    )
    assert float(local_S.max()) <= 1.0 / 3.0 + 2e-7
    assert float(local_S.max()) >= 0.3326
    assert float(local_S.min()) < 0.14
    Qzz = -(fields["Qxx"] + fields["Qyy"])
    torch.testing.assert_close(Qzz, -local_S / 2.0, rtol=2e-5, atol=2e-6)


def _import_nematics3d():
    project_root = Path(__file__).resolve().parents[1]
    for candidate in (
        project_root.parent / "Nematics3D" / "src",
        project_root / "Nematics3D" / "src",
    ):
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    return pytest.importorskip("nematics3d")


def _wrapped_phase_difference(value):
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _plaquette_integer_charge(complex_order, defect_index):
    nx, ny = complex_order.shape
    i = int(np.floor(defect_index[0])) % nx
    j = int(np.floor(defect_index[1])) % ny
    phase = np.angle(complex_order)
    corners = (
        phase[i, j],
        phase[(i + 1) % nx, j],
        phase[(i + 1) % nx, (j + 1) % ny],
        phase[i, (j + 1) % ny],
    )
    winding = sum(
        _wrapped_phase_difference(corners[(index + 1) % 4] - corners[index])
        for index in range(4)
    )
    return int(np.rint(winding / (2.0 * np.pi)))


def test_nematics3d_recovers_requested_defect_count_charges_and_positions():
    n3d = _import_nematics3d()
    shape = (128, 128)
    lengths = np.asarray((64.0, 64.0))
    kwargs = dict(
        lengths=tuple(lengths),
        num_defect_pairs=4,
        min_separation=10.0,
        core_radius=1.25,
        S_initial=1.0 / 3.0,
        seed=17,
    )
    fields = analytic_periodic_defect_gas_2d(shape, **kwargs)
    Q_tensor = np.stack([fields[name].numpy() for name in Q_COMPONENTS], axis=-1)
    eigenvectors = eigenframe_from_Q(Q_tensor)
    director = eigenvectors[..., 0]
    detected = n3d.defect_detect(
        director[:, :, None, :],
        threshold=0.0,
        is_boundary_periodic=(True, True, False),
        planes=(False, False, True),
    )[:, :2]

    expected_positions, expected_charges = sample_periodic_neutral_defects_2d(
        lengths=tuple(lengths),
        num_defect_pairs=4,
        min_separation=10.0,
        seed=17,
    )
    assert len(detected) == len(expected_positions) == 8

    # Nematics3D reports the plaquette center in grid-index coordinates. The
    # simulation nodes are cell centered, hence the additional half-cell here.
    detected_positions = (
        (detected + 0.5) * lengths / np.asarray(shape)
    )
    delta = detected_positions[:, None, :] - expected_positions[None, :, :]
    delta -= lengths * np.round(delta / lengths)
    distance = np.linalg.norm(delta, axis=-1)
    matching_expected_index = np.argmin(distance, axis=1)
    grid_diagonal = np.linalg.norm(lengths / np.asarray(shape))
    assert np.all(distance.min(axis=0) <= grid_diagonal)
    assert np.all(distance.min(axis=1) <= grid_diagonal)
    assert len(np.unique(matching_expected_index)) == len(expected_positions)

    complex_order = (
        Q_tensor[..., 0] - Q_tensor[..., 3] + 2j * Q_tensor[..., 1]
    )
    detected_integer_charges = np.asarray(
        [_plaquette_integer_charge(complex_order, index) for index in detected]
    )
    expected_integer_charges = np.rint(
        2.0 * expected_charges[matching_expected_index]
    ).astype(int)
    np.testing.assert_array_equal(detected_integer_charges, expected_integer_charges)
    assert np.count_nonzero(detected_integer_charges == 1) == 4
    assert np.count_nonzero(detected_integer_charges == -1) == 4
