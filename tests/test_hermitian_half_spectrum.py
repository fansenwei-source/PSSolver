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


def _self_conjugate_plane_violation(
    spectral,
    physical_shape,
    boundary_conditions,
    *,
    hermitian_axis=1,
):
    """Return the largest packed-plane real-field constraint violation."""

    spatial_offset = spectral.ndim - len(physical_shape)
    tensor_axis = spatial_offset + hermitian_axis
    plane_indices = [0]
    if physical_shape[hermitian_axis] % 2 == 0:
        plane_indices.append(physical_shape[hermitian_axis] // 2)
    indices = torch.tensor(plane_indices, device=spectral.device)
    planes = spectral.index_select(tensor_axis, indices)
    reflected = planes
    for axis, boundary_condition in enumerate(boundary_conditions):
        if axis == hermitian_axis or boundary_condition != "periodic":
            continue
        reverse = torch.remainder(
            -torch.arange(physical_shape[axis], device=spectral.device),
            physical_shape[axis],
        )
        reflected = reflected.index_select(spatial_offset + axis, reverse)
    return (planes - reflected.conj()).abs().max()


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


@pytest.mark.parametrize("rule", ("two_thirds", "cubic_half"))
def test_truncated_projected_transform_combines_with_hermitian_storage(rule):
    shape = (8, 10, 7)
    boundary_conditions = ("periodic", "periodic", "neumann")
    generator = torch.Generator().manual_seed(20260912)
    values = torch.randn(3, *shape, generator=generator, dtype=torch.float64)
    backend = _backend(shape, storage="hermitian_half")
    solver = SimpleNamespace(shape=shape, transform_backend=backend)
    full = BasisAwareSpectralProjector(
        solver,
        rule=rule,
        transform_execution="full",
    )
    truncated = BasisAwareSpectralProjector(
        solver,
        rule=rule,
        transform_execution="truncated",
    )

    full_spectral = full.forward_transform(values, boundary_conditions)
    truncated_spectral = truncated.forward_transform(
        values,
        boundary_conditions,
    )

    assert full_spectral.shape[-3:] == backend.spectral_shape
    assert truncated_spectral.shape[-3:] == backend.spectral_shape
    torch.testing.assert_close(
        truncated_spectral,
        full_spectral,
        rtol=8.0e-13,
        atol=8.0e-13,
    )
    torch.testing.assert_close(
        truncated.inverse_transform(
            truncated_spectral,
            boundary_conditions,
        ),
        full.inverse_transform(full_spectral, boundary_conditions),
        rtol=8.0e-13,
        atol=8.0e-13,
    )
    assert truncated.computed_axis_sizes(boundary_conditions) == (
        shape[0],
        shape[1] // 2 + 1,
        5 if rule == "two_thirds" else 4,
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


@pytest.mark.parametrize("shape", ((8, 10, 7), (7, 9, 5)))
@pytest.mark.parametrize(
    "boundary_conditions",
    (
        ("periodic", "periodic", "neumann"),
        ("periodic", "periodic", "periodic"),
    ),
)
@pytest.mark.parametrize("rule", ("none", "cubic_half"))
def test_real_spectrum_projection_enforces_packed_self_conjugate_planes(
    shape,
    boundary_conditions,
    rule,
):
    generator = torch.Generator().manual_seed(20260930)
    backend = _backend(shape, storage="hermitian_half")
    projector = BasisAwareSpectralProjector(
        SimpleNamespace(shape=shape, transform_backend=backend),
        rule=rule,
    )
    spectral = torch.complex(
        torch.randn(
            2,
            3,
            *backend.spectral_shape,
            generator=generator,
            dtype=torch.float64,
        ),
        torch.randn(
            2,
            3,
            *backend.spectral_shape,
            generator=generator,
            dtype=torch.float64,
        ),
    )
    before = spectral.clone()

    projected = projector.project_real_spectrum(
        spectral,
        boundary_conditions,
    )

    assert torch.equal(spectral, before)
    assert projected is not spectral
    assert _self_conjugate_plane_violation(
        projected,
        shape,
        boundary_conditions,
    ).item() == 0.0
    torch.testing.assert_close(
        backend.inverse(projected, boundary_conditions),
        backend.inverse(
            spectral
            * (
                projector.mask(boundary_conditions)
                if projector.enabled
                else 1
            ),
            boundary_conditions,
        ),
        rtol=5.0e-13,
        atol=5.0e-13,
    )


def test_real_spectrum_projection_in_place_enforces_planes_and_reuses_storage():
    shape = (8, 10, 7)
    boundary_conditions = ("periodic", "periodic", "neumann")
    backend = _backend(shape, storage="hermitian_half")
    projector = BasisAwareSpectralProjector(
        SimpleNamespace(shape=shape, transform_backend=backend),
        rule="none",
    )
    spectral = torch.randn(
        2,
        *backend.spectral_shape,
        dtype=torch.complex128,
    )

    returned = projector.project_real_spectrum_(
        spectral,
        boundary_conditions,
    )

    assert returned is spectral
    assert _self_conjugate_plane_violation(
        spectral,
        shape,
        boundary_conditions,
    ).item() == 0.0


def test_plane_projection_supports_a_nonterminal_packed_axis():
    shape = (8, 7, 6)
    boundary_conditions = ("periodic", "periodic", "neumann")
    backend = TensorProductTransformBackend(
        shape,
        LENGTHS,
        device="cpu",
        dtype=torch.float64,
        spectral_storage="hermitian_half",
        hermitian_axis=0,
    )
    projector = BasisAwareSpectralProjector(
        SimpleNamespace(shape=shape, transform_backend=backend),
        rule="cubic_half",
    )
    spectral = torch.randn(
        2,
        *backend.spectral_shape,
        dtype=torch.complex128,
    )

    projected = projector.project_real_spectrum(
        spectral,
        boundary_conditions,
    )

    assert _self_conjugate_plane_violation(
        projected,
        shape,
        boundary_conditions,
        hermitian_axis=0,
    ).item() == 0.0


def test_hermitian_plane_projection_remains_autograd_compatible():
    shape = (8, 10, 7)
    boundary_conditions = ("periodic", "periodic", "periodic")
    backend = _backend(shape, storage="hermitian_half")
    projector = BasisAwareSpectralProjector(
        SimpleNamespace(shape=shape, transform_backend=backend),
        rule="cubic_half",
    )
    spectral = torch.randn(
        2,
        *backend.spectral_shape,
        dtype=torch.complex128,
        requires_grad=True,
    )

    physical = backend.inverse(
        projector.project_real_spectrum(spectral, boundary_conditions),
        boundary_conditions,
    )
    physical.square().sum().backward()

    assert spectral.grad is not None
    assert bool(torch.isfinite(spectral.grad).all())


def test_hermitian_projection_is_recorded_only_for_packed_storage():
    shape = (8, 10, 7)
    half = BasisAwareSpectralProjector(
        SimpleNamespace(
            shape=shape,
            transform_backend=_backend(shape, storage="hermitian_half"),
        ),
        rule="cubic_half",
    )
    full = BasisAwareSpectralProjector(
        SimpleNamespace(
            shape=shape,
            transform_backend=_backend(shape, storage="full_complex"),
        ),
        rule="cubic_half",
    )

    assert half.execution_metadata()["hermitian_state_projection"] == (
        "self_conjugate_planes"
    )
    assert "hermitian_state_projection" not in full.execution_metadata()
