"""Qualification tests for the opt-in FFT DCT-II/DST-II candidate."""

from __future__ import annotations

import pytest
import torch

from pssolver import (
    BOUNDED_TRANSFORM_ALGORITHMS,
    DEFAULT_BOUNDED_TRANSFORM_ALGORITHM,
    SpectralSolver,
)
from pssolver.transforms import TensorProductTransformBackend


def _backend(
    shape,
    *,
    algorithm,
    dtype=torch.float64,
    spectral_storage="full_complex",
    hermitian_axis=None,
):
    return TensorProductTransformBackend(
        shape,
        tuple(float(size) for size in shape),
        device="cpu",
        dtype=dtype,
        execution_order="real_first",
        bounded_transform_algorithm=algorithm,
        spectral_storage=spectral_storage,
        hermitian_axis=hermitian_axis,
    )


def _tolerances(dtype):
    if dtype == torch.float64:
        return {"rtol": 6.0e-12, "atol": 6.0e-12}
    return {"rtol": 5.0e-5, "atol": 5.0e-5}


def _values(shape, *, dtype, complex_values, seed):
    generator = torch.Generator().manual_seed(seed)
    real = torch.randn(*shape, generator=generator, dtype=dtype)
    if not complex_values:
        return real
    imaginary = torch.randn(*shape, generator=generator, dtype=dtype)
    return torch.complex(real, imaginary)


def test_fft_bounded_transform_is_opt_in_and_validated():
    solver = SpectralSolver((5,), L=(2.0,), device="cpu")

    assert BOUNDED_TRANSFORM_ALGORITHMS == ("dense", "fft", "auto")
    assert DEFAULT_BOUNDED_TRANSFORM_ALGORITHM == "dense"
    assert solver.transform_backend.bounded_transform_algorithm == "dense"
    with pytest.raises(ValueError, match="bounded_transform_algorithm"):
        TensorProductTransformBackend(
            (5,),
            (2.0,),
            device="cpu",
            bounded_transform_algorithm="unknown",
        )


@pytest.mark.parametrize("size", (1, 2, 3, 5, 8, 17))
@pytest.mark.parametrize("kind", ("dct", "dst"))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("complex_values", (False, True))
def test_fft_full_axis_matches_dense_forward_and_inverse(
    size,
    kind,
    dtype,
    complex_values,
):
    dense = _backend((size,), algorithm="dense", dtype=dtype)
    candidate = _backend((size,), algorithm="fft", dtype=dtype)
    values = _values(
        (3, size),
        dtype=dtype,
        complex_values=complex_values,
        seed=1000 + size,
    )

    dense_coefficients = dense._apply_axis_transform(values, kind, -1)
    candidate_coefficients = candidate._apply_axis_transform(values, kind, -1)
    torch.testing.assert_close(
        candidate_coefficients,
        dense_coefficients,
        **_tolerances(dtype),
    )

    arbitrary_coefficients = _values(
        (3, size),
        dtype=dtype,
        complex_values=complex_values,
        seed=2000 + size,
    )
    dense_inverse = dense._apply_axis_transform(
        arbitrary_coefficients,
        kind,
        -1,
        inverse=True,
    )
    candidate_inverse = candidate._apply_axis_transform(
        arbitrary_coefficients,
        kind,
        -1,
        inverse=True,
    )
    torch.testing.assert_close(
        candidate_inverse,
        dense_inverse,
        **_tolerances(dtype),
    )


@pytest.mark.parametrize("size,retained", ((3, 1), (8, 4), (17, 8)))
@pytest.mark.parametrize("kind", ("dct", "dst"))
@pytest.mark.parametrize("complex_values", (False, True))
def test_fft_truncated_axis_matches_dense_forward_and_inverse(
    size,
    retained,
    kind,
    complex_values,
):
    dense = _backend((size,), algorithm="dense")
    candidate = _backend((size,), algorithm="fft")
    values = _values(
        (4, size),
        dtype=torch.float64,
        complex_values=complex_values,
        seed=3000 + size,
    )

    dense_coefficients = dense._apply_retained_real_axis_transform(
        values,
        kind,
        0,
        retained,
        inverse=False,
    )
    candidate_coefficients = candidate._apply_retained_real_axis_transform(
        values,
        kind,
        0,
        retained,
        inverse=False,
    )
    torch.testing.assert_close(
        candidate_coefficients,
        dense_coefficients,
        rtol=6.0e-12,
        atol=6.0e-12,
    )

    coefficients = _values(
        (4, retained),
        dtype=torch.float64,
        complex_values=complex_values,
        seed=4000 + size,
    )
    dense_inverse = dense._apply_retained_real_axis_transform(
        coefficients,
        kind,
        0,
        retained,
        inverse=True,
    )
    candidate_inverse = candidate._apply_retained_real_axis_transform(
        coefficients,
        kind,
        0,
        retained,
        inverse=True,
    )
    torch.testing.assert_close(
        candidate_inverse,
        dense_inverse,
        rtol=6.0e-12,
        atol=6.0e-12,
    )


@pytest.mark.parametrize(
    "boundary_conditions,spectral_storage,hermitian_axis",
    (
        (("periodic", "neumann", "dirichlet"), "full_complex", None),
        (("neumann", "dirichlet", "neumann"), "full_complex", None),
        (("periodic", "periodic", "neumann"), "hermitian_half", 1),
        (("periodic", "periodic", "dirichlet"), "hermitian_half", 1),
    ),
)
def test_fft_tensor_product_transform_matches_dense_and_roundtrips(
    boundary_conditions,
    spectral_storage,
    hermitian_axis,
):
    shape = (7, 8, 5)
    dense = _backend(
        shape,
        algorithm="dense",
        spectral_storage=spectral_storage,
        hermitian_axis=hermitian_axis,
    )
    candidate = _backend(
        shape,
        algorithm="fft",
        spectral_storage=spectral_storage,
        hermitian_axis=hermitian_axis,
    )
    values = _values(
        (2, *shape),
        dtype=torch.float64,
        complex_values=False,
        seed=5000,
    )

    dense_spectral = dense.forward(values, boundary_conditions)
    candidate_spectral = candidate.forward(values, boundary_conditions)
    torch.testing.assert_close(
        candidate_spectral,
        dense_spectral,
        rtol=8.0e-12,
        atol=8.0e-12,
    )
    torch.testing.assert_close(
        candidate.inverse(candidate_spectral, boundary_conditions),
        values,
        rtol=8.0e-12,
        atol=8.0e-12,
    )


@pytest.mark.parametrize("kind", ("dct", "dst"))
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("truncated", (False, True))
def test_fft_candidate_preserves_dense_autograd(kind, inverse, truncated):
    size = 9
    retained = 4
    input_size = retained if inverse and truncated else size
    dense = _backend((size,), algorithm="dense")
    candidate = _backend((size,), algorithm="fft")
    base = _values(
        (2, input_size),
        dtype=torch.float64,
        complex_values=False,
        seed=6000,
    )
    dense_input = base.clone().requires_grad_(True)
    candidate_input = base.clone().requires_grad_(True)

    if truncated:
        dense_output = dense._apply_retained_real_axis_transform(
            dense_input,
            kind,
            0,
            retained,
            inverse=inverse,
        )
        candidate_output = candidate._apply_retained_real_axis_transform(
            candidate_input,
            kind,
            0,
            retained,
            inverse=inverse,
        )
    else:
        dense_output = dense._apply_axis_transform(
            dense_input,
            kind,
            -1,
            inverse=inverse,
        )
        candidate_output = candidate._apply_axis_transform(
            candidate_input,
            kind,
            -1,
            inverse=inverse,
        )
    weights = torch.linspace(
        0.5,
        1.5,
        dense_output.numel(),
        dtype=torch.float64,
    ).reshape_as(dense_output)
    (dense_output * weights).sum().backward()
    (candidate_output * weights).sum().backward()

    torch.testing.assert_close(
        candidate_output,
        dense_output,
        rtol=8.0e-12,
        atol=8.0e-12,
    )
    torch.testing.assert_close(
        candidate_input.grad,
        dense_input.grad,
        rtol=8.0e-12,
        atol=8.0e-12,
    )


def test_fft_candidate_uses_phase_cache_without_dense_matrices():
    backend = _backend((16,), algorithm="fft")
    values = torch.randn(3, 16, dtype=torch.float64)

    backend._apply_axis_transform(values, "dct", -1)
    backend._apply_axis_transform(values, "dst", -1)

    assert backend._matrix_cache == {}
    assert len(backend._fft_r2r_cache) == 1
