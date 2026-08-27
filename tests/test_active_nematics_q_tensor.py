import math

import numpy as np
import pytest
import torch

from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    Q_components,
    Q_convention_metadata,
    Q_magnitude,
    S_from_Q,
    positive_equilibrium_S,
    uniaxial_Q,
)


def test_numpy_uniaxial_Q_has_canonical_eigenvalues_and_round_trips_S():
    director = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
        ]
    )
    director[1] /= np.linalg.norm(director[1])
    S = np.asarray([0.4, 0.7])

    Q = uniaxial_Q(director, S)
    eigenvalues = np.linalg.eigvalsh(Q)

    np.testing.assert_allclose(eigenvalues[:, 2], S, rtol=1e-14, atol=1e-14)
    expected_transverse = np.broadcast_to(-S[:, None] / 2.0, (2, 2))
    np.testing.assert_allclose(eigenvalues[:, :2], expected_transverse)
    np.testing.assert_allclose(S_from_Q(Q), S)
    np.testing.assert_allclose(Q_magnitude(Q), S)


def test_torch_uniaxial_Q_preserves_backend_dtype_and_gradients():
    director = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]],
        dtype=torch.float64,
    )
    S = torch.tensor([0.25, 0.9], dtype=torch.float64, requires_grad=True)

    Q = uniaxial_Q(director, S)

    assert isinstance(Q, torch.Tensor)
    assert Q.dtype == torch.float64
    assert Q.shape == (2, 3, 3)
    torch.testing.assert_close(S_from_Q(Q), S)
    torch.testing.assert_close(Q_magnitude(Q), S)
    Q_magnitude(Q).sum().backward()
    torch.testing.assert_close(S.grad, torch.ones_like(S))


def test_Q_components_round_trip_full_compact_and_mapping_layouts():
    director = np.asarray([2.0, -1.0, 2.0]) / 3.0
    full = uniaxial_Q(director, 0.6)
    components = Q_components(full)
    compact = np.stack([components[name] for name in Q_COMPONENTS], axis=-1)

    assert tuple(components) == Q_COMPONENTS
    np.testing.assert_allclose(S_from_Q(components), 0.6)
    np.testing.assert_allclose(S_from_Q(compact), 0.6)
    np.testing.assert_allclose(Q_magnitude(components), 0.6)
    reconstructed = Q_components(compact)
    for name in Q_COMPONENTS:
        np.testing.assert_allclose(reconstructed[name], components[name])


def test_Q_magnitude_is_distinct_from_principal_S_for_biaxial_Q():
    Q = np.diag([0.6, -0.1, -0.5])

    principal_S = S_from_Q(Q)
    magnitude = Q_magnitude(Q)

    np.testing.assert_allclose(principal_S, 0.6)
    np.testing.assert_allclose(magnitude, math.sqrt((2.0 / 3.0) * 0.62))
    assert not np.isclose(magnitude, principal_S)


@pytest.mark.parametrize("backend", [np.asarray, torch.as_tensor])
@pytest.mark.parametrize(
    ("invalid_Q", "message"),
    [
        (
            [[0.6, 0.2, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.5]],
            "symmetric",
        ),
        (
            [[0.6, 0.0, 0.0], [0.0, -0.1, 0.0], [0.0, 0.0, -0.4]],
            "traceless",
        ),
        (
            [[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "finite",
        ),
    ],
)
def test_explicit_full_Q_rejects_invalid_tensor_structure(
    backend,
    invalid_Q,
    message,
):
    Q = backend(invalid_Q)

    for operation in (Q_components, S_from_Q, Q_magnitude):
        with pytest.raises(ValueError, match=message):
            operation(Q)


def test_positive_equilibrium_S_uses_canonical_bulk_equation():
    np.testing.assert_allclose(positive_equilibrium_S(0.0, -0.3, 0.3), 1.0 / 3.0)
    np.testing.assert_allclose(
        positive_equilibrium_S(-1.0, -6.0, 6.0),
        0.5393446629166317,
    )

    with pytest.raises(ValueError, match="C must be positive"):
        positive_equilibrium_S(-1.0, -6.0, 0.0)
    with pytest.raises(ValueError, match="real uniaxial equilibrium"):
        positive_equilibrium_S(1.0, 0.0, 1.0)


def test_Q_convention_metadata_is_explicit_and_fresh():
    first = Q_convention_metadata()
    second = Q_convention_metadata()

    assert first == {
        "id": "de_gennes_S_lambda_max_v1",
        "definition": "Q=(3S/2)(nn-I/3)",
        "S_definition": "S=lambda_max(Q)",
        "uniaxial_eigenvalues": ["S", "-S/2", "-S/2"],
        "version": 1,
    }
    assert first is not second
    assert first["uniaxial_eigenvalues"] is not second["uniaxial_eigenvalues"]


def test_uniaxial_Q_rejects_nonunit_directors_and_negative_S():
    with pytest.raises(ValueError, match="unit length"):
        uniaxial_Q(np.asarray([2.0, 0.0, 0.0]), 0.4)
    with pytest.raises(ValueError, match="non-negative"):
        uniaxial_Q(np.asarray([1.0, 0.0, 0.0]), -0.4)
