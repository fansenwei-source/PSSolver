import math
from types import SimpleNamespace

import pytest
import torch

from pssolver.transforms import (
    BasisAwareSpectralProjector,
    FreeSlipModalStokesSolver,
    TensorProductTransformBackend,
)


TANGENTIAL_BC = ("periodic", "periodic", "neumann")
NORMAL_BC = ("periodic", "periodic", "dirichlet")
PRESSURE_BC = ("periodic", "periodic", "neumann")


def _backend(
    shape=(9, 8, 7),
    lengths=(2.0 * math.pi, 3.0 * math.pi, 2.5),
    *,
    execution_order="legacy",
    spectral_storage="full_complex",
):
    return TensorProductTransformBackend(
        shape,
        lengths,
        device="cpu",
        dtype=torch.float64,
        execution_order=execution_order,
        spectral_storage=spectral_storage,
        hermitian_axis=1,
    )


def _stokes(backend, *, friction, zero_mode_policy, viscosity=0.73):
    return FreeSlipModalStokesSolver(
        backend,
        tangential_boundary_conditions=TANGENTIAL_BC,
        normal_boundary_conditions=NORMAL_BC,
        pressure_boundary_conditions=PRESSURE_BC,
        friction=friction,
        viscosity=viscosity,
        zero_mode_policy=zero_mode_policy,
    )


@pytest.mark.parametrize(
    ("boundary_conditions", "mode_number", "expected_boundary_conditions"),
    (
        (TANGENTIAL_BC, 1, NORMAL_BC),
        (TANGENTIAL_BC, 6, NORMAL_BC),
        (NORMAL_BC, 1, PRESSURE_BC),
        (NORMAL_BC, 6, PRESSURE_BC),
    ),
)
def test_cell_centered_dct_dst_z_derivatives_match_analytic_modes(
    boundary_conditions,
    mode_number,
    expected_boundary_conditions,
):
    backend = _backend()
    x, y, z = backend.spatial_grids
    kx = 2.0
    ky = 2.0 / 3.0
    kz = mode_number * math.pi / backend.lengths[2]
    phase = kx * x + ky * y

    if boundary_conditions[2] == "neumann":
        field = torch.cos(phase) * torch.cos(kz * z)
        expected = -kz * torch.cos(phase) * torch.sin(kz * z)
    else:
        field = torch.cos(phase) * torch.sin(kz * z)
        expected = kz * torch.cos(phase) * torch.cos(kz * z)

    spectral = backend.forward(field, boundary_conditions)
    derivative_hat, derivative_bcs = backend.gradient_hat(
        spectral,
        boundary_conditions,
        axis=2,
    )
    observed = backend.inverse(derivative_hat, derivative_bcs)

    assert derivative_bcs == expected_boundary_conditions
    torch.testing.assert_close(observed, expected, rtol=3.0e-12, atol=3.0e-12)


def test_dct_dst_derivative_pair_is_negative_transpose():
    backend = TensorProductTransformBackend(
        (9,),
        (2.7,),
        device="cpu",
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(711)
    neumann_hat = torch.complex(
        torch.randn(3, 9, generator=generator, dtype=torch.float64),
        torch.randn(3, 9, generator=generator, dtype=torch.float64),
    )
    dirichlet_hat = torch.complex(
        torch.randn(3, 9, generator=generator, dtype=torch.float64),
        torch.randn(3, 9, generator=generator, dtype=torch.float64),
    )
    # The terminal DST mode maps to the unavailable DCT mode m=N and is not
    # part of the resolved derivative pair.
    dirichlet_hat[..., -1] = 0

    dn_hat, dn_bcs = backend.gradient_hat(
        neumann_hat,
        ("neumann",),
        axis=0,
    )
    dd_hat, dd_bcs = backend.gradient_hat(
        dirichlet_hat,
        ("dirichlet",),
        axis=0,
    )
    pairing = torch.vdot(dn_hat.reshape(-1), dirichlet_hat.reshape(-1))
    pairing += torch.vdot(neumann_hat.reshape(-1), dd_hat.reshape(-1))

    assert dn_bcs == ("dirichlet",)
    assert dd_bcs == ("neumann",)
    torch.testing.assert_close(
        pairing,
        torch.zeros((), dtype=torch.complex128),
        rtol=0,
        atol=2.0e-12,
    )


def test_terminal_dst_mode_has_zero_derivative_and_is_filtered():
    shape = (8, 10, 9)
    backend = _backend(shape=shape)
    terminal = torch.zeros((1, *shape), dtype=torch.complex128)
    terminal[0, 0, 0, -1] = 1.0 + 0.25j

    derivative_hat, derivative_bcs = backend.gradient_hat(
        terminal,
        NORMAL_BC,
        axis=2,
    )
    assert derivative_bcs == PRESSURE_BC
    torch.testing.assert_close(
        derivative_hat,
        torch.zeros_like(derivative_hat),
        rtol=0,
        atol=0,
    )

    solver_adapter = SimpleNamespace(shape=shape, transform_backend=backend)
    for rule in ("cubic_half", "two_thirds"):
        projector = BasisAwareSpectralProjector(solver_adapter, rule=rule)
        projected = projector.project(terminal, NORMAL_BC)
        torch.testing.assert_close(
            projected,
            torch.zeros_like(projected),
            rtol=0,
            atol=0,
        )

    unfiltered = BasisAwareSpectralProjector(solver_adapter, rule="none")
    torch.testing.assert_close(
        unfiltered.project(terminal, NORMAL_BC),
        terminal,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("real_dtype", (torch.float32, torch.float64))
def test_projector_bool_mask_matches_typed_mask_without_mutating_input(
    real_dtype,
):
    shape = (8, 10, 9)
    backend = _backend(shape=shape)
    solver_adapter = SimpleNamespace(shape=shape, transform_backend=backend)
    projector = BasisAwareSpectralProjector(solver_adapter, rule="cubic_half")
    generator = torch.Generator().manual_seed(1701)
    spectral = torch.complex(
        torch.randn((3, *shape), generator=generator, dtype=real_dtype),
        torch.randn((3, *shape), generator=generator, dtype=real_dtype),
    )
    original = spectral.clone()
    mask = projector.mask(TANGENTIAL_BC)
    expected = spectral * mask.to(dtype=spectral.dtype)

    observed = projector.project(spectral, TANGENTIAL_BC)

    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(spectral, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("execution_order", "spectral_storage"),
    (
        ("legacy", "full_complex"),
        ("real_first", "full_complex"),
        ("real_first", "hermitian_half"),
    ),
)
def test_float64_manufactured_free_slip_solution_recovers_u_p_and_gauge(
    execution_order,
    spectral_storage,
):
    shape = (10, 12, 9)
    lengths = (2.0 * math.pi, 3.0 * math.pi, 2.5)
    backend = _backend(
        shape=shape,
        lengths=lengths,
        execution_order=execution_order,
        spectral_storage=spectral_storage,
    )
    friction = 0.17
    viscosity = 0.73
    stokes = _stokes(
        backend,
        friction=friction,
        viscosity=viscosity,
        zero_mode_policy="friction",
    )

    x, y, z = backend.spatial_grids
    kx = 2.0 * math.pi / lengths[0]
    ky = 4.0 * math.pi / lengths[1]
    kz = 2.0 * math.pi / lengths[2]
    phase = kx * x + ky * y
    theta = kz * z
    u_amplitude = 0.7
    v_amplitude = -0.3
    p_amplitude = 0.4
    pressure_gauge = 1.7
    w_amplitude = (
        kx * u_amplitude + ky * v_amplitude
    ) / kz

    ux = u_amplitude * torch.cos(phase) * torch.cos(theta)
    uy = v_amplitude * torch.cos(phase) * torch.cos(theta)
    uz = w_amplitude * torch.sin(phase) * torch.sin(theta)
    pressure_zero_mean = (
        p_amplitude * torch.sin(phase) * torch.cos(theta)
    )
    pressure = pressure_gauge + pressure_zero_mean

    k_squared = kx * kx + ky * ky + kz * kz
    helmholtz = friction + viscosity * k_squared
    fx = (
        helmholtz * ux
        + p_amplitude * kx * torch.cos(phase) * torch.cos(theta)
    )
    fy = (
        helmholtz * uy
        + p_amplitude * ky * torch.cos(phase) * torch.cos(theta)
    )
    fz = (
        helmholtz * uz
        - p_amplitude * kz * torch.sin(phase) * torch.sin(theta)
    )

    fx_hat = backend.forward(fx.unsqueeze(0), TANGENTIAL_BC)
    fy_hat = backend.forward(fy.unsqueeze(0), TANGENTIAL_BC)
    fz_hat = backend.forward(fz.unsqueeze(0), NORMAL_BC)
    ux_hat, uy_hat, uz_hat, pressure_hat = stokes.solve_force_hats(
        fx_hat,
        fy_hat,
        fz_hat,
    )

    observed_ux = backend.inverse(ux_hat, TANGENTIAL_BC)
    observed_uy = backend.inverse(uy_hat, TANGENTIAL_BC)
    observed_uz = backend.inverse(uz_hat, NORMAL_BC)
    observed_pressure = backend.inverse(pressure_hat, PRESSURE_BC)
    torch.testing.assert_close(
        observed_ux, ux.unsqueeze(0), rtol=5.0e-12, atol=5.0e-12
    )
    torch.testing.assert_close(
        observed_uy, uy.unsqueeze(0), rtol=5.0e-12, atol=5.0e-12
    )
    torch.testing.assert_close(
        observed_uz, uz.unsqueeze(0), rtol=5.0e-12, atol=5.0e-12
    )
    torch.testing.assert_close(
        observed_pressure,
        pressure_zero_mean.unsqueeze(0),
        rtol=5.0e-12,
        atol=5.0e-12,
    )

    # The manufactured force depends on grad(p), so adding the arbitrary
    # constant above cannot affect the recovered velocity or effective pressure.
    exact_pressure_hat = backend.forward(pressure.unsqueeze(0), PRESSURE_BC)
    exact_pressure_hat = exact_pressure_hat.masked_fill(
        stokes.pressure_null_mask,
        0,
    )
    torch.testing.assert_close(
        pressure_hat,
        exact_pressure_hat,
        rtol=5.0e-12,
        atol=5.0e-12,
    )
    assert pressure_hat[0, 0, 0, 0].item() == 0j
    assert int(stokes.pressure_null_mask.sum().item()) == 1
    assert abs(observed_pressure.mean().item()) < 1.0e-13

    divergence_hat = stokes.divergence_hat(ux_hat, uy_hat, uz_hat)
    divergence_scale = max(
        torch.linalg.vector_norm(ux_hat.reshape(-1)).item(),
        torch.linalg.vector_norm(uy_hat.reshape(-1)).item(),
        torch.linalg.vector_norm(uz_hat.reshape(-1)).item(),
        1.0,
    )
    assert (
        torch.linalg.vector_norm(divergence_hat.reshape(-1)).item()
        / divergence_scale
        < 5.0e-12
    )
    assert stokes.last_pressure_iterations == 1
    assert stokes.last_pressure_relative_residual < 1.0e-12

    grad_px_hat, grad_py_hat, grad_pz_hat = stokes.pressure_gradient_hats(
        pressure_hat
    )
    residuals = (
        -grad_px_hat
        + viscosity * backend.laplacian_hat(ux_hat, TANGENTIAL_BC)
        - friction * ux_hat
        + fx_hat,
        -grad_py_hat
        + viscosity * backend.laplacian_hat(uy_hat, TANGENTIAL_BC)
        - friction * uy_hat
        + fy_hat,
        -grad_pz_hat
        + viscosity * backend.laplacian_hat(uz_hat, NORMAL_BC)
        - friction * uz_hat
        + fz_hat,
    )
    residual_norm = torch.sqrt(
        sum(torch.linalg.vector_norm(value.reshape(-1)).square() for value in residuals)
    ).item()
    force_norm = torch.sqrt(
        sum(
            torch.linalg.vector_norm(value.reshape(-1)).square()
            for value in (fx_hat, fy_hat, fz_hat)
        )
    ).item()
    assert residual_norm / force_norm < 5.0e-12


def test_one_mixed_mode_matches_independent_dense_kkt_oracle():
    shape = (6, 7, 5)
    lengths = (2.0 * math.pi, 3.0 * math.pi, 2.5)
    backend = _backend(shape=shape, lengths=lengths)
    friction = 0.19
    viscosity = 0.67
    stokes = _stokes(
        backend,
        friction=friction,
        viscosity=viscosity,
        zero_mode_policy="friction",
    )
    pressure_metadata = backend.get_metadata(PRESSURE_BC)
    ix, iy, m = 1, 2, 2
    kx = pressure_metadata.axis_modes[0][ix]
    ky = pressure_metadata.axis_modes[1][iy]
    kz = pressure_metadata.axis_modes[2][m]
    a_tangential = friction + viscosity * (kx.square() + ky.square() + kz.square())
    a_normal = a_tangential

    matrix = torch.zeros((4, 4), dtype=torch.complex128)
    matrix[0, 0] = a_tangential
    matrix[1, 1] = a_tangential
    matrix[2, 2] = a_normal
    matrix[0, 3] = 1j * kx
    matrix[1, 3] = 1j * ky
    matrix[2, 3] = -kz
    matrix[3, 0] = 1j * kx
    matrix[3, 1] = 1j * ky
    matrix[3, 2] = kz
    force = torch.tensor(
        (0.7 + 0.2j, -0.4 + 0.1j, 0.3 - 0.6j),
        dtype=torch.complex128,
    )
    rhs = torch.cat((force, torch.zeros(1, dtype=torch.complex128)))
    expected = torch.linalg.solve(matrix, rhs)

    fx_hat = torch.zeros((1, *shape), dtype=torch.complex128)
    fy_hat = torch.zeros_like(fx_hat)
    fz_hat = torch.zeros_like(fx_hat)
    fx_hat[0, ix, iy, m] = force[0]
    fy_hat[0, ix, iy, m] = force[1]
    # Pressure DCT mode m couples to normal DST slot r-1=m-1.
    fz_hat[0, ix, iy, m - 1] = force[2]
    observed_hats = stokes.solve_force_hats(fx_hat, fy_hat, fz_hat)
    observed = torch.stack(
        (
            observed_hats[0][0, ix, iy, m],
            observed_hats[1][0, ix, iy, m],
            observed_hats[2][0, ix, iy, m - 1],
            observed_hats[3][0, ix, iy, m],
        )
    )

    torch.testing.assert_close(observed, expected, rtol=3.0e-12, atol=3.0e-13)
    assert stokes.last_pressure_relative_residual < 1.0e-12


def test_constant_tangential_force_distinguishes_zero_mean_and_friction_modes():
    shape = (8, 10, 7)
    backend = _backend(shape=shape)
    force_x = 1.25
    force_y = -0.75
    fx = torch.full((1, *shape), force_x, dtype=torch.float64)
    fy = torch.full((1, *shape), force_y, dtype=torch.float64)
    fz = torch.zeros((1, *shape), dtype=torch.float64)
    fx_hat = backend.forward(fx, TANGENTIAL_BC)
    fy_hat = backend.forward(fy, TANGENTIAL_BC)
    fz_hat = backend.forward(fz, NORMAL_BC)

    zero_mean = _stokes(
        backend,
        friction=0.0,
        zero_mode_policy="zero_mean",
    )
    zero_hats = zero_mean.solve_force_hats(fx_hat, fy_hat, fz_hat)
    zero_ux = backend.inverse(zero_hats[0], TANGENTIAL_BC)
    zero_uy = backend.inverse(zero_hats[1], TANGENTIAL_BC)
    torch.testing.assert_close(
        zero_ux, torch.zeros_like(zero_ux), rtol=0, atol=1.0e-12
    )
    torch.testing.assert_close(
        zero_uy, torch.zeros_like(zero_uy), rtol=0, atol=1.0e-12
    )
    torch.testing.assert_close(
        zero_hats[3], torch.zeros_like(zero_hats[3]), rtol=0, atol=1.0e-12
    )
    assert zero_mean.has_tangential_null_mode is True

    friction = 0.4
    dragged = _stokes(
        backend,
        friction=friction,
        zero_mode_policy="friction",
    )
    dragged_hats = dragged.solve_force_hats(fx_hat, fy_hat, fz_hat)
    dragged_ux = backend.inverse(dragged_hats[0], TANGENTIAL_BC)
    dragged_uy = backend.inverse(dragged_hats[1], TANGENTIAL_BC)
    torch.testing.assert_close(
        dragged_ux,
        torch.full_like(dragged_ux, force_x / friction),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        dragged_uy,
        torch.full_like(dragged_uy, force_y / friction),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    torch.testing.assert_close(
        dragged_hats[3],
        torch.zeros_like(dragged_hats[3]),
        rtol=0,
        atol=1.0e-12,
    )
    assert dragged.has_tangential_null_mode is False
