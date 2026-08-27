import math

import numpy as np
import pytest
import torch

from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    available_initial_conditions,
    create_initial_condition,
    extruded_2d_twist,
    neumann_twist_profile,
)


def q_frobenius_squared(fields):
    qxx = fields["Qxx"]
    qxy = fields["Qxy"]
    qxz = fields["Qxz"]
    qyy = fields["Qyy"]
    qyz = fields["Qyz"]
    qzz = -(qxx + qyy)
    return qxx.square() + qyy.square() + qzz.square() + 2.0 * (
        qxy.square() + qxz.square() + qyz.square()
    )


def sample_Q_2d(nx=9, ny=7):
    x = torch.linspace(-1.0, 1.0, nx).reshape(nx, 1)
    y = torch.linspace(-0.8, 0.8, ny).reshape(1, ny)
    theta = 0.25 * math.pi + 0.3 * x - 0.2 * y
    phi = 0.35 * x + 0.15 * y
    director_x = torch.sin(theta) * torch.cos(phi)
    director_y = torch.sin(theta) * torch.sin(phi)
    director_z = torch.cos(theta)
    S = 0.7 + 0.1 * torch.cos(math.pi * x) * torch.cos(math.pi * y)
    coefficient = 1.5 * S
    return {
        "Qxx": coefficient * (director_x.square() - 1.0 / 3.0),
        "Qxy": coefficient * director_x * director_y,
        "Qxz": coefficient * director_x * director_z,
        "Qyy": coefficient * (director_y.square() - 1.0 / 3.0),
        "Qyz": coefficient * director_y * director_z,
    }


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


def test_zero_twist_is_exact_vertical_extrusion():
    source = sample_Q_2d()
    fields = extruded_2d_twist(
        (9, 7, 12),
        Q_2d=source,
        boundary_conditions=("periodic", "periodic", "neumann"),
        twist_amplitude=0.0,
    )

    for name in Q_COMPONENTS:
        assert fields[name].shape == (9, 7, 12)
        expected = source[name].to(torch.float32).unsqueeze(-1).expand(-1, -1, 12)
        torch.testing.assert_close(fields[name], expected, rtol=0, atol=0)


def test_twist_preserves_pointwise_Q_eigenvalue_invariants_and_core_locations():
    source = sample_Q_2d()
    fields = extruded_2d_twist(
        (9, 7, 13),
        Q_2d=source,
        boundary_conditions=("periodic", "periodic", "neumann"),
        twist_amplitude=0.02,
        twist_modes=(1, 2, 3),
        seed=19,
    )

    source_norm = q_frobenius_squared(source).unsqueeze(-1).expand(-1, -1, 13)
    torch.testing.assert_close(
        q_frobenius_squared(fields),
        source_norm,
        rtol=2e-6,
        atol=2e-7,
    )
    # Pointwise tensor invariants, including order minima at defect cores, are
    # unchanged on every z plane even though the director texture is twisted.
    assert torch.equal(
        torch.argmin(q_frobenius_squared(fields)[..., 0]),
        torch.argmin(q_frobenius_squared(fields)[..., -1]),
    )
    assert not torch.allclose(fields["Qxy"][..., 0], fields["Qxy"][..., -1])


def test_registered_initializer_accepts_stacked_2d_snapshot_layout():
    source = sample_Q_2d(6, 5)
    stacked = np.stack([source[name].numpy() for name in Q_COMPONENTS], axis=-1)
    fields = create_initial_condition(
        "extruded_2d_twist",
        (6, 5, 8),
        Q_2d=stacked[:, :, None, :],
        boundary_conditions=("periodic", "periodic", "neumann"),
        twist_amplitude=0.01,
        seed=3,
    )

    assert "extruded_2d_twist" in available_initial_conditions()
    assert set(fields) == set(Q_COMPONENTS)
    assert all(field.dtype == torch.float32 for field in fields.values())
    assert all(field.device.type == "cpu" for field in fields.values())


def test_extruded_twist_rejects_non_neumann_wall_condition():
    with pytest.raises(ValueError, match="Neumann-compatible"):
        extruded_2d_twist(
            (9, 7, 12),
            Q_2d=sample_Q_2d(),
            boundary_conditions=("periodic", "periodic", "dirichlet"),
        )
