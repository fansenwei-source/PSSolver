import numpy as np
import pytest

from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    Q_components,
    uniaxial_Q,
)
from pssolver.models.active_nematics import nematics3d_adapter as adapter


def _compact_Q(director, S=0.47):
    components = Q_components(uniaxial_Q(np.asarray(director, dtype=float), S))
    return np.stack([components[name] for name in Q_COMPONENTS], axis=-1)


def test_canonical_S_and_director_for_array_field():
    director = np.asarray([2.0, -1.0, 2.0]) / 3.0
    compact = _compact_Q(director)
    field = np.broadcast_to(compact, (2, 2, 2, 5)).copy()

    S, measured = adapter.S_and_director_from_Q(field)

    np.testing.assert_allclose(S, 0.47, atol=1.0e-14)
    alignment = np.abs(np.sum(measured * director, axis=-1))
    np.testing.assert_allclose(alignment, 1.0, atol=1.0e-12)


@pytest.mark.parametrize(
    "director",
    (
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
        np.asarray([1.0, 1.0, 0.0]) / np.sqrt(2.0),
    ),
)
def test_uniaxial_degenerate_spectrum_has_robust_principal_director(director):
    compact = _compact_Q(director)

    S, measured = adapter.S_and_director_from_Q(compact)

    np.testing.assert_allclose(S, 0.47, atol=1.0e-14)
    np.testing.assert_allclose(abs(np.dot(measured, director)), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(adapter.director_from_Q(compact), measured)


def test_full_tensor_input_and_descending_eigenframe():
    director = np.asarray([2.0, -1.0, 2.0]) / 3.0
    full = uniaxial_Q(director, 0.47)

    S, measured = adapter.S_and_director_from_Q(full)
    frame = adapter.eigenframe_from_Q(full)

    np.testing.assert_allclose(S, 0.47, atol=1.0e-14)
    np.testing.assert_allclose(abs(np.dot(measured, director)), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(abs(np.dot(frame[:, 0], director)), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1.0e-12)


def test_isotropic_director_is_finite_but_arbitrary():
    compact = np.zeros(5)

    S, director = adapter.S_and_director_from_Q(compact)
    frame = adapter.eigenframe_from_Q(compact)

    np.testing.assert_allclose(S, 0.0)
    assert np.isfinite(director).all()
    np.testing.assert_allclose(np.linalg.norm(director), 1.0)
    assert np.isfinite(frame).all()
    np.testing.assert_allclose(frame.T @ frame, np.eye(3))


@pytest.mark.parametrize(
    ("Q", "message"),
    (
        (np.full(5, np.nan), "finite"),
        (
            np.asarray([[1.0, 1.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, -0.5]]),
            "symmetric",
        ),
        (np.eye(3), "traceless"),
    ),
)
def test_adapter_rejects_invalid_Q(Q, message):
    with pytest.raises(ValueError, match=message):
        adapter.S_and_director_from_Q(Q)


def test_q_field_object_factory_scales_canonical_Q_for_nematics3d(monkeypatch):
    canonical = _compact_Q(np.asarray([1.0, 0.0, 0.0]), S=0.4)
    captured = {}

    class FakeInputQ:
        def __init__(self, *, Q, box_periodic_flag):
            captured["Q"] = Q
            captured["periodic"] = box_periodic_flag

    class FakeQFieldObject:
        def __init__(self, *, inputValue, name, **kwargs):
            self.inputValue = inputValue
            self.name = name
            self.kwargs = kwargs

    def fake_import_module(name):
        if name == "nematics3d":
            return type("FakeN3D", (), {"QFieldObject": FakeQFieldObject})
        if name == "nematics3d.classes.q_field_object":
            return type("FakeQModule", (), {"InputQ": FakeInputQ})
        raise AssertionError(name)

    monkeypatch.setattr(adapter, "import_module", fake_import_module)
    result = adapter.q_field_object_from_Q(
        canonical,
        box_periodic_flag=(True, False, False),
        name="canonical",
        is_detect_defects=False,
    )

    np.testing.assert_allclose(captured["Q"], (2.0 / 3.0) * canonical)
    assert captured["periodic"] == (True, False, False)
    assert result.name == "canonical"
    assert result.kwargs == {
        "is_detect_defects": False,
        "is_classify_lines": False,
    }
    np.testing.assert_allclose(result._raw_S, 0.4)
    np.testing.assert_allclose(abs(result._raw_n[0]), 1.0)
    np.testing.assert_allclose(result._raw_n[1:], 0.0)


def test_full_float32_canonical_Q_is_not_rejected_as_nontraceless():
    director = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    director /= np.linalg.norm(director)
    full = uniaxial_Q(director, np.float32(0.47))

    S, measured = adapter.S_and_director_from_Q(full)

    np.testing.assert_allclose(S, 0.47, rtol=1.0e-6)
    np.testing.assert_allclose(
        abs(np.dot(measured, director)),
        1.0,
        rtol=1.0e-6,
    )
