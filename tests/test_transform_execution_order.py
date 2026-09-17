import pytest
import torch

from pssolver.transforms import (
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    TensorProductTransformBackend,
)


SHAPE = (7, 6, 5)
LENGTHS = (3.0, 4.0, 2.5)


def _backend(
    dtype,
    execution_order,
    periodic_transform_execution=None,
):
    kwargs = {}
    if periodic_transform_execution is not None:
        kwargs["periodic_transform_execution"] = periodic_transform_execution
    return TensorProductTransformBackend(
        SHAPE,
        LENGTHS,
        device="cpu",
        dtype=dtype,
        execution_order=execution_order,
        **kwargs,
    )


def _tolerances(dtype):
    if dtype == torch.float64:
        return {"rtol": 4.0e-12, "atol": 4.0e-12}
    return {"rtol": 3.0e-5, "atol": 3.0e-5}


def test_transform_backend_defaults_to_real_first_and_keeps_legacy_opt_in():
    default_backend = TensorProductTransformBackend(
        SHAPE,
        LENGTHS,
        device="cpu",
        dtype=torch.float64,
    )
    legacy_backend = _backend(torch.float64, "legacy")

    assert DEFAULT_TRANSFORM_EXECUTION_ORDER == "real_first"
    assert default_backend.execution_order == "real_first"
    assert legacy_backend.execution_order == "legacy"


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    "boundary_conditions",
    (
        ("periodic", "periodic", "neumann"),
        ("periodic", "periodic", "dirichlet"),
        ("neumann", "periodic", "dirichlet"),
        ("dirichlet", "neumann", "periodic"),
        ("neumann", "dirichlet", "neumann"),
        ("periodic", "periodic", "periodic"),
    ),
)
def test_real_first_forward_and_roundtrip_match_legacy(
    dtype,
    boundary_conditions,
):
    generator = torch.Generator().manual_seed(20260909)
    values = torch.randn(2, 3, *SHAPE, generator=generator, dtype=dtype)
    legacy = _backend(dtype, "legacy")
    real_first = _backend(dtype, "real_first")

    legacy_spectral = legacy.forward(values, boundary_conditions)
    real_first_spectral = real_first.forward(values, boundary_conditions)
    torch.testing.assert_close(
        real_first_spectral,
        legacy_spectral,
        **_tolerances(dtype),
    )
    torch.testing.assert_close(
        real_first.inverse(real_first_spectral, boundary_conditions),
        values,
        **_tolerances(dtype),
    )


@pytest.mark.parametrize(
    "boundary_conditions",
    (
        ("periodic", "periodic", "neumann"),
        ("neumann", "periodic", "dirichlet"),
        ("neumann", "dirichlet", "neumann"),
    ),
)
def test_real_first_inverse_matches_legacy_for_arbitrary_complex_coefficients(
    boundary_conditions,
):
    generator = torch.Generator().manual_seed(711)
    spectral = torch.complex(
        torch.randn(2, *SHAPE, generator=generator, dtype=torch.float64),
        torch.randn(2, *SHAPE, generator=generator, dtype=torch.float64),
    )
    legacy = _backend(torch.float64, "legacy")
    real_first = _backend(torch.float64, "real_first")

    torch.testing.assert_close(
        real_first.inverse(spectral, boundary_conditions),
        legacy.inverse(spectral, boundary_conditions),
        rtol=4.0e-12,
        atol=4.0e-12,
    )


@pytest.mark.parametrize("source_bc", ("neumann", "dirichlet"))
@pytest.mark.parametrize("axis", (0, 1, 2))
def test_real_first_preserves_mixed_basis_spectral_derivatives(source_bc, axis):
    boundary_conditions = ["periodic", "periodic", "periodic"]
    boundary_conditions[axis] = source_bc
    boundary_conditions = tuple(boundary_conditions)
    generator = torch.Generator().manual_seed(3100 + axis)
    values = torch.randn(2, *SHAPE, generator=generator, dtype=torch.float64)
    legacy = _backend(torch.float64, "legacy")
    real_first = _backend(torch.float64, "real_first")

    legacy_hat = legacy.forward(values, boundary_conditions)
    real_first_hat = real_first.forward(values, boundary_conditions)
    legacy_derivative_hat, derivative_bcs = legacy.gradient_hat(
        legacy_hat,
        boundary_conditions,
        axis,
    )
    real_first_derivative_hat, real_first_derivative_bcs = (
        real_first.gradient_hat(
            real_first_hat,
            boundary_conditions,
            axis,
        )
    )

    assert real_first_derivative_bcs == derivative_bcs
    expected = legacy.inverse(legacy_derivative_hat, derivative_bcs)
    observed = real_first.inverse(
        real_first_derivative_hat,
        real_first_derivative_bcs,
    )
    torch.testing.assert_close(observed, expected, rtol=5.0e-12, atol=5.0e-12)


def test_real_first_applies_dct_while_data_are_real():
    # This test observes individual axis calls, so explicitly exercise the
    # retained rollback implementation rather than the multidimensional
    # periodic default.
    backend = _backend(
        torch.float64,
        "real_first",
        periodic_transform_execution="axiswise",
    )
    calls = []
    original = backend._apply_axis_transform

    def recording_transform(tensor, kind, axis, inverse=False):
        calls.append((inverse, kind, tensor.is_complex()))
        return original(tensor, kind, axis, inverse=inverse)

    backend._apply_axis_transform = recording_transform
    values = torch.randn(2, *SHAPE, dtype=torch.float64)
    boundary_conditions = ("periodic", "periodic", "neumann")
    spectral = backend.forward(values, boundary_conditions)
    backend.inverse(spectral, boundary_conditions)

    assert calls[:3] == [
        (False, "dct", False),
        (False, "fft", False),
        (False, "fft", True),
    ]
    assert calls[3:] == [
        (True, "fft", True),
        (True, "fft", True),
        (True, "dct", False),
    ]


def test_transform_backend_rejects_unknown_execution_order():
    with pytest.raises(ValueError, match="execution_order"):
        _backend(torch.float64, "unknown")
