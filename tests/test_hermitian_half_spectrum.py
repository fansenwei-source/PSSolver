from types import SimpleNamespace

import pytest
import torch

from pssolver.transforms import (
    BasisAwareSpectralProjector,
    DEFAULT_SPECTRAL_STORAGE,
    TensorProductTransformBackend,
)


LENGTHS = (3.0, 4.0, 2.5)
PLANE_BOUNDARY_CONDITIONS = (
    ("periodic", "periodic", "neumann"),
    ("periodic", "periodic", "dirichlet"),
    ("periodic", "periodic", "periodic"),
)


def _backend(shape, *, storage):
    return TensorProductTransformBackend(
        shape,
        LENGTHS,
        device="cpu",
        dtype=torch.float64,
        execution_order="real_first",
        spectral_storage=storage,
        hermitian_axis=1,
    )


def _positive_y_slice(tensor, ny):
    index = [slice(None)] * tensor.ndim
    index[-2] = slice(0, ny // 2 + 1)
    return tensor[tuple(index)]


def test_full_complex_remains_the_default_storage():
    backend = TensorProductTransformBackend(
        (7, 6, 5),
        LENGTHS,
        device="cpu",
        dtype=torch.float64,
    )

    assert DEFAULT_SPECTRAL_STORAGE == "full_complex"
    assert backend.spectral_storage == "full_complex"
    assert backend.spectral_shape == backend.shape


def test_float32_hermitian_forward_and_roundtrip_match_full_spectrum():
    shape = (8, 10, 7)
    boundary_conditions = ("periodic", "periodic", "neumann")
    generator = torch.Generator().manual_seed(20260911)
    values = torch.randn(2, *shape, generator=generator, dtype=torch.float32)
    full = TensorProductTransformBackend(
        shape,
        LENGTHS,
        device="cpu",
        dtype=torch.float32,
        spectral_storage="full_complex",
    )
    half = TensorProductTransformBackend(
        shape,
        LENGTHS,
        device="cpu",
        dtype=torch.float32,
        spectral_storage="hermitian_half",
        hermitian_axis=1,
    )

    half_spectral = half.forward(values, boundary_conditions)
    torch.testing.assert_close(
        half_spectral,
        _positive_y_slice(
            full.forward(values, boundary_conditions),
            shape[1],
        ),
        rtol=4.0e-5,
        atol=4.0e-5,
    )
    torch.testing.assert_close(
        half.inverse(half_spectral, boundary_conditions),
        values,
        rtol=4.0e-5,
        atol=4.0e-5,
    )


@pytest.mark.parametrize("shape", ((7, 6, 5), (8, 7, 6), (8, 8, 8)))
@pytest.mark.parametrize("boundary_conditions", PLANE_BOUNDARY_CONDITIONS)
def test_hermitian_forward_is_positive_y_half_of_full_spectrum(
    shape,
    boundary_conditions,
):
    generator = torch.Generator().manual_seed(20260910)
    values = torch.randn(2, 3, *shape, generator=generator, dtype=torch.float64)
    full = _backend(shape, storage="full_complex")
    half = _backend(shape, storage="hermitian_half")

    full_spectral = full.forward(values, boundary_conditions)
    half_spectral = half.forward(values, boundary_conditions)

    assert half.spectral_shape == (shape[0], shape[1] // 2 + 1, shape[2])
    assert half_spectral.shape[-3:] == half.spectral_shape
    torch.testing.assert_close(
        half_spectral,
        _positive_y_slice(full_spectral, shape[1]),
        rtol=5.0e-13,
        atol=5.0e-13,
    )


@pytest.mark.parametrize("shape", ((7, 6, 5), (8, 7, 6), (8, 8, 8)))
@pytest.mark.parametrize("boundary_conditions", PLANE_BOUNDARY_CONDITIONS)
def test_hermitian_roundtrip_recovers_physical_values(shape, boundary_conditions):
    generator = torch.Generator().manual_seed(812)
    values = torch.randn(2, *shape, generator=generator, dtype=torch.float64)
    half = _backend(shape, storage="hermitian_half")

    observed = half.inverse(
        half.forward(values, boundary_conditions),
        boundary_conditions,
    )

    torch.testing.assert_close(observed, values, rtol=8.0e-13, atol=8.0e-13)


@pytest.mark.parametrize("shape", ((7, 6, 5), (8, 7, 6), (8, 8, 8)))
@pytest.mark.parametrize("axis", (0, 1, 2))
def test_hermitian_gradient_matches_full_physical_result(shape, axis):
    boundary_conditions = ("periodic", "periodic", "neumann")
    generator = torch.Generator().manual_seed(4100 + axis)
    values = torch.randn(2, *shape, generator=generator, dtype=torch.float64)
    full = _backend(shape, storage="full_complex")
    half = _backend(shape, storage="hermitian_half")

    full_gradient_hat, full_gradient_bcs = full.gradient_hat(
        full.forward(values, boundary_conditions),
        boundary_conditions,
        axis,
    )
    half_gradient_hat, half_gradient_bcs = half.gradient_hat(
        half.forward(values, boundary_conditions),
        boundary_conditions,
        axis,
    )

    assert half_gradient_bcs == full_gradient_bcs
    torch.testing.assert_close(
        half.inverse(half_gradient_hat, half_gradient_bcs),
        full.inverse(full_gradient_hat, full_gradient_bcs),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("shape", ((7, 6, 5), (8, 8, 8)))
def test_hermitian_laplacian_matches_full_physical_result(shape):
    boundary_conditions = ("periodic", "periodic", "neumann")
    generator = torch.Generator().manual_seed(7331)
    values = torch.randn(2, *shape, generator=generator, dtype=torch.float64)
    full = _backend(shape, storage="full_complex")
    half = _backend(shape, storage="hermitian_half")

    full_result = full.inverse(
        full.laplacian_hat(
            full.forward(values, boundary_conditions),
            boundary_conditions,
        ),
        boundary_conditions,
    )
    half_result = half.inverse(
        half.laplacian_hat(
            half.forward(values, boundary_conditions),
            boundary_conditions,
        ),
        boundary_conditions,
    )

    torch.testing.assert_close(
        half_result,
        full_result,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("rule", ("two_thirds", "cubic_half"))
def test_hermitian_projector_is_positive_y_half_of_full_mask(rule):
    shape = (8, 10, 7)
    boundary_conditions = ("periodic", "periodic", "neumann")
    full_backend = _backend(shape, storage="full_complex")
    half_backend = _backend(shape, storage="hermitian_half")
    full = BasisAwareSpectralProjector(
        SimpleNamespace(shape=shape, transform_backend=full_backend),
        rule=rule,
    )
    half = BasisAwareSpectralProjector(
        SimpleNamespace(shape=shape, transform_backend=half_backend),
        rule=rule,
    )

    torch.testing.assert_close(
        half.mask(boundary_conditions),
        _positive_y_slice(full.mask(boundary_conditions), shape[1]),
    )


def test_hermitian_storage_requires_real_first_and_periodic_packed_axis():
    with pytest.raises(TypeError, match="hermitian_axis"):
        TensorProductTransformBackend(
            (8, 8, 8),
            LENGTHS,
            device="cpu",
            dtype=torch.float64,
            spectral_storage="hermitian_half",
        )

    with pytest.raises(ValueError, match="requires real_first"):
        TensorProductTransformBackend(
            (8, 8, 8),
            LENGTHS,
            device="cpu",
            dtype=torch.float64,
            execution_order="legacy",
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        )

    backend = _backend((8, 8, 8), storage="hermitian_half")
    with pytest.raises(ValueError, match="periodic boundary condition on axis 1"):
        backend.get_metadata(("periodic", "neumann", "neumann"))


def test_hermitian_inverse_rejects_full_shape_coefficients():
    shape = (8, 8, 8)
    backend = _backend(shape, storage="hermitian_half")
    spectral = torch.zeros(shape, dtype=torch.complex128)

    with pytest.raises(ValueError, match="Expected trailing spectral shape"):
        backend.inverse(spectral, ("periodic", "periodic", "neumann"))


def test_hermitian_axis_can_be_a_nonterminal_tensor_axis():
    shape = (8, 7, 6)
    values = torch.randn(2, *shape, dtype=torch.float64)
    full = TensorProductTransformBackend(
        shape,
        LENGTHS,
        device="cpu",
        dtype=torch.float64,
        spectral_storage="full_complex",
    )
    half = TensorProductTransformBackend(
        shape,
        LENGTHS,
        device="cpu",
        dtype=torch.float64,
        spectral_storage="hermitian_half",
        hermitian_axis=0,
    )
    boundary_conditions = ("periodic", "periodic", "neumann")

    full_spectral = full.forward(values, boundary_conditions)
    half_spectral = half.forward(values, boundary_conditions)
    expected = full_spectral[..., : shape[0] // 2 + 1, :, :]

    assert half.spectral_shape == (shape[0] // 2 + 1, shape[1], shape[2])
    torch.testing.assert_close(half_spectral, expected, rtol=5.0e-13, atol=5.0e-13)
    torch.testing.assert_close(
        half.inverse(half_spectral, boundary_conditions),
        values,
        rtol=8.0e-13,
        atol=8.0e-13,
    )
