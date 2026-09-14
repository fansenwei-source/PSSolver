import inspect
import math

import numpy as np
import pytest
import torch

from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    Q_components,
    Q_magnitude,
    S_from_Q,
    aligned_x_band_limited_noise_2d,
    aligned_x_smooth_noise,
    analytic_periodic_defect_gas_2d,
    available_initial_conditions,
    create_initial_condition,
    extruded_2d_unbiased_rotation,
    extruded_2d_twist,
    neumann_twist_profile,
    sample_periodic_neutral_defects_2d,
    uniaxial_Q,
)


def _sample_Q_2d(nx=9, ny=7):
    x = torch.linspace(-1.0, 1.0, nx).reshape(nx, 1)
    y = torch.linspace(-0.8, 0.8, ny).reshape(1, ny)
    theta = 0.25 * math.pi + 0.3 * x - 0.2 * y
    phi = 0.35 * x + 0.15 * y
    director = torch.stack(
        (
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta),
        ),
        dim=-1,
    )
    S = 0.7 + 0.1 * torch.cos(math.pi * x) * torch.cos(math.pi * y)
    return Q_components(uniaxial_Q(director, S))


def test_aligned_initializer_requires_S_initial_and_uses_canonical_eigenvalues():
    pytest.importorskip("scipy")
    kwargs = dict(
        shape=(4, 3, 2),
        boundary_conditions=("periodic", "periodic", "neumann"),
        noise_theta=0.0,
        noise_phi=0.0,
    )
    with pytest.raises(TypeError, match="S_initial"):
        aligned_x_smooth_noise(**kwargs)

    fields = aligned_x_smooth_noise(S_initial=0.4, **kwargs)
    assert tuple(fields) == Q_COMPONENTS
    torch.testing.assert_close(
        S_from_Q(fields),
        torch.full((4, 3, 2), 0.4),
        rtol=2e-6,
        atol=2e-7,
    )
    torch.testing.assert_close(
        Q_magnitude(fields),
        torch.full((4, 3, 2), 0.4),
        rtol=2e-6,
        atol=2e-7,
    )


def test_analytic_defect_gas_requires_S_initial_and_is_reproducible():
    common = dict(
        shape=(48, 40),
        lengths=(24.0, 20.0),
        num_defect_pairs=2,
        min_separation=4.0,
        core_radius=0.8,
        seed=17,
    )
    parameters = inspect.signature(analytic_periodic_defect_gas_2d).parameters
    assert "S_initial" in parameters
    assert "S_bulk" not in parameters
    with pytest.raises(TypeError, match="S_initial"):
        analytic_periodic_defect_gas_2d(**common)

    fields = create_initial_condition(
        "analytic_periodic_defect_gas_2d",
        S_initial=1.0 / 3.0,
        **common,
    )
    repeated = analytic_periodic_defect_gas_2d(S_initial=1.0 / 3.0, **common)

    assert tuple(fields) == Q_COMPONENTS
    for name in Q_COMPONENTS:
        assert fields[name].shape == (48, 40)
        assert fields[name].dtype == torch.float32
        assert fields[name].device.type == "cpu"
        torch.testing.assert_close(fields[name], repeated[name], rtol=0, atol=0)
    assert torch.count_nonzero(fields["Qxz"]) == 0
    assert torch.count_nonzero(fields["Qyz"]) == 0
    local_S = S_from_Q(fields)
    assert float(local_S.max()) <= 1.0 / 3.0 + 2e-7
    assert float(local_S.max()) >= 0.332
    assert float(local_S.min()) < 0.2


def test_periodic_defect_sampling_is_neutral_separated_and_reproducible():
    kwargs = dict(
        lengths=(32.0, 24.0),
        num_defect_pairs=3,
        min_separation=4.0,
        seed=23,
    )
    positions, charges = sample_periodic_neutral_defects_2d(**kwargs)
    repeated_positions, repeated_charges = sample_periodic_neutral_defects_2d(
        **kwargs
    )

    np.testing.assert_array_equal(positions, repeated_positions)
    np.testing.assert_array_equal(charges, repeated_charges)
    assert np.count_nonzero(charges == 0.5) == 3
    assert np.count_nonzero(charges == -0.5) == 3
    assert charges.sum() == 0.0
    lengths = np.asarray(kwargs["lengths"])
    delta = positions[:, None, :] - positions[None, :, :]
    delta -= lengths * np.round(delta / lengths)
    distances = np.linalg.norm(delta, axis=-1)
    distances[np.diag_indices_from(distances)] = np.inf
    assert distances.min() >= kwargs["min_separation"]


def test_zero_twist_is_exact_extrusion_and_Q_2d_is_the_only_input_name():
    source = _sample_Q_2d()
    fields = extruded_2d_twist(
        (9, 7, 12),
        Q_2d=source,
        boundary_conditions=("periodic", "periodic", "neumann"),
        twist_amplitude=0.0,
    )

    assert "Q_2d" in inspect.signature(extruded_2d_twist).parameters
    for name in Q_COMPONENTS:
        expected = source[name].to(torch.float32).unsqueeze(-1).expand(-1, -1, 12)
        torch.testing.assert_close(fields[name], expected, rtol=0, atol=0)


def test_v3_2d_seed_is_periodic_band_limited_reproducible_and_has_fixed_S():
    kwargs = dict(
        shape=(24, 20),
        S_initial=1.0 / 3.0,
        seed=31,
        angle_rms=0.012,
        max_mode_x=3,
        max_mode_y=2,
        dtype=torch.float64,
    )
    fields = aligned_x_band_limited_noise_2d(**kwargs)
    repeated = aligned_x_band_limited_noise_2d(**kwargs)

    assert tuple(fields) == Q_COMPONENTS
    for name in Q_COMPONENTS:
        assert fields[name].shape == (24, 20)
        assert fields[name].dtype == torch.float64
        torch.testing.assert_close(fields[name], repeated[name], rtol=0, atol=0)
    torch.testing.assert_close(
        S_from_Q(fields),
        torch.full((24, 20), 1.0 / 3.0, dtype=torch.float64),
        rtol=2e-13,
        atol=2e-14,
    )
    assert torch.count_nonzero(fields["Qxz"]) == 0
    assert torch.count_nonzero(fields["Qyz"]) == 0
    assert float(fields["Qxy"].std()) > 0.0


def test_v3_zero_rotation_is_exact_extrusion():
    source = _sample_Q_2d()
    fields = extruded_2d_unbiased_rotation(
        (9, 7, 8),
        Q_2d=source,
        boundary_conditions=("periodic", "periodic", "neumann"),
        rotation_rms=0.0,
        z_modes=(1, 2, 3),
        dtype=torch.float64,
    )

    for name in Q_COMPONENTS:
        expected = source[name].to(torch.float64).unsqueeze(-1).expand(-1, -1, 8)
        torch.testing.assert_close(fields[name], expected, rtol=0, atol=0)


def test_v3_unbiased_rotation_preserves_S_and_is_spatially_local():
    source = _sample_Q_2d(12, 10)
    fields = extruded_2d_unbiased_rotation(
        (12, 10, 9),
        Q_2d=source,
        boundary_conditions=("periodic", "periodic", "neumann"),
        rotation_rms=1.0e-3,
        max_mode_x=2,
        max_mode_y=2,
        z_modes=(1, 2, 3),
        seed=47,
        dtype=torch.float64,
    )

    source_float64 = {name: values.to(torch.float64) for name, values in source.items()}
    expected_S = S_from_Q(source_float64).unsqueeze(-1).expand(-1, -1, 9)
    torch.testing.assert_close(
        S_from_Q(fields),
        expected_S,
        rtol=5e-13,
        atol=5e-14,
    )
    assert float(fields["Qxz"].std()) > 0.0
    assert float(fields["Qyz"].std()) > 0.0
    # A local perturbation must vary within an xy plane, unlike V1's coherent
    # layer rotation, and positive z modes must vary across the channel.
    assert float(fields["Qxz"][..., 0].std()) > 0.0
    assert not torch.allclose(fields["Qxz"][..., 0], fields["Qxz"][..., -1])


def test_twist_preserves_pointwise_Q_magnitude():
    source = _sample_Q_2d()
    fields = extruded_2d_twist(
        (9, 7, 13),
        Q_2d=source,
        boundary_conditions=("periodic", "periodic", "neumann"),
        twist_amplitude=0.02,
        twist_modes=(1, 2, 3),
        seed=19,
    )

    expected = Q_magnitude(source).unsqueeze(-1).expand(-1, -1, 13)
    torch.testing.assert_close(Q_magnitude(fields), expected, rtol=2e-6, atol=2e-7)
    assert not torch.allclose(fields["Qxy"][..., 0], fields["Qxy"][..., -1])


def test_registry_exposes_only_model_initializers_and_accepts_stacked_Q_2d():
    assert available_initial_conditions() == (
        "aligned_x_band_limited_noise_2d",
        "analytic_periodic_defect_gas_2d",
        "aligned_x_smooth_noise",
        "extruded_2d_unbiased_rotation",
        "extruded_2d_twist",
    )
    source = _sample_Q_2d(6, 5)
    stacked = np.stack([source[name].numpy() for name in Q_COMPONENTS], axis=-1)
    fields = create_initial_condition(
        "extruded_2d_twist",
        (6, 5, 8),
        Q_2d=stacked[:, :, None, :],
        boundary_conditions=("periodic", "periodic", "neumann"),
        twist_amplitude=0.01,
        seed=3,
    )
    assert tuple(fields) == Q_COMPONENTS

    with pytest.raises(ValueError, match="unknown active-nematic"):
        create_initial_condition("missing", (2, 2, 2))


def test_neumann_twist_profile_is_reproducible_and_has_requested_rms():
    first = neumann_twist_profile(16, amplitude=0.012, modes=(1, 2, 4), seed=7)
    second = neumann_twist_profile(16, amplitude=0.012, modes=(1, 2, 4), seed=7)
    other = neumann_twist_profile(16, amplitude=0.012, modes=(1, 2, 4), seed=8)

    torch.testing.assert_close(first, second)
    assert not torch.allclose(first, other)
    torch.testing.assert_close(
        torch.sqrt(torch.mean(first.square())),
        torch.tensor(0.012),
        rtol=2e-6,
        atol=1e-8,
    )
