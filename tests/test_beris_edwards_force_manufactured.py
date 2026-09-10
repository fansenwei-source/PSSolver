import math

import pytest
import torch

from pssolver import (
    BasisAwareSpectralProjector,
    SpectralSolver,
    projected_common_basis_stress_divergence,
    projected_distortion_stress_divergence,
)
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsPointwiseKernels,
    Q_COMPONENTS,
    beris_edwards_active_stress_components,
    beris_edwards_algebraic_stress_components,
    beris_edwards_distortion_stress_components,
    beris_edwards_molecular_field_components,
)


Q_BC = ("periodic", "periodic", "neumann")
DISTORTION_ODD_Z_BC = ("periodic", "periodic", "dirichlet")


def _solver_and_projector(shape, lengths):
    solver = SpectralSolver(
        shape,
        L=lengths,
        device="cpu",
        dtype=torch.float64,
    )
    projector = BasisAwareSpectralProjector(solver, rule="cubic_half")
    return solver, projector


def _compact(matrix):
    return (
        matrix[..., 0, 0],
        matrix[..., 0, 1],
        matrix[..., 0, 2],
        matrix[..., 1, 1],
        matrix[..., 1, 2],
    )


def _matrix(components):
    xx, xy, xz, yy, yz = components
    zz = -xx - yy
    return torch.stack(
        (
            torch.stack((xx, xy, xz), dim=-1),
            torch.stack((xy, yy, yz), dim=-1),
            torch.stack((xz, yz, zz), dim=-1),
        ),
        dim=-2,
    )


def _common_basis_mode(index, x, y, z):
    amplitude = 0.09 * (index + 1)
    kx = 1 + index % 2
    ky = 1 + (index // 2) % 2
    kz = 1 + index % 3
    phase = 0.11 * (index + 1)

    sx = torch.sin(kx * x + phase)
    cx = torch.cos(kx * x + phase)
    cy = torch.cos(ky * y - 0.5 * phase)
    sy = torch.sin(ky * y - 0.5 * phase)
    cz = torch.cos(kz * z)
    sz = torch.sin(kz * z)
    field = amplitude * sx * cy * cz
    gradients = (
        amplitude * kx * cx * cy * cz,
        -amplitude * ky * sx * sy * cz,
        -amplitude * kz * sx * cy * sz,
    )
    return field, gradients


@pytest.mark.parametrize("sum_space", ("physical", "spectral"))
def test_nine_distinct_row_major_stresses_have_row_wise_divergence(sum_space):
    solver, projector = _solver_and_projector(
        (20, 18, 16),
        (2.0 * math.pi, 2.0 * math.pi, math.pi),
    )
    x, y, z = solver.transform_backend.spatial_grids
    modes = tuple(_common_basis_mode(index, x, y, z) for index in range(9))
    stress = tuple(mode[0] for mode in modes)
    gradients = tuple(mode[1] for mode in modes)

    observed = projected_common_basis_stress_divergence(
        solver.transform_backend,
        stress,
        Q_BC,
        projector=projector,
        sum_space=sum_space,
    )
    expected = torch.stack(
        tuple(
            gradients[3 * row][0]
            + gradients[3 * row + 1][1]
            + gradients[3 * row + 2][2]
            for row in range(3)
        )
    )

    # The manufactured tensor is deliberately nonsymmetric and all nine entries
    # differ, so a column-wise partial_j Pi_ji implementation cannot pass.
    transposed_divergence = torch.stack(
        tuple(
            gradients[row][0]
            + gradients[3 + row][1]
            + gradients[6 + row][2]
            for row in range(3)
        )
    )
    assert torch.linalg.vector_norm(expected - transposed_divergence) > 1.0
    torch.testing.assert_close(observed, expected, rtol=2e-12, atol=2e-12)


def _parity_split_mode(index, x, y, z):
    amplitude = 0.07 * (index + 1)
    kx = 1 + (index + 1) % 2
    ky = 1 + (index // 3) % 2
    kz = 1 + (2 * index) % 3
    phase = 0.08 * (index + 1)
    sx = torch.sin(kx * x + phase)
    cx = torch.cos(kx * x + phase)
    cy = torch.cos(ky * y - phase)
    sy = torch.sin(ky * y - phase)
    odd_in_z = index in (2, 5, 6, 7)

    if odd_in_z:
        z_factor = torch.sin(kz * z)
        dz_factor = kz * torch.cos(kz * z)
    else:
        z_factor = torch.cos(kz * z)
        dz_factor = -kz * torch.sin(kz * z)

    field = amplitude * sx * cy * z_factor
    gradients = (
        amplitude * kx * cx * cy * z_factor,
        -amplitude * ky * sx * sy * z_factor,
        amplitude * sx * cy * dz_factor,
    )
    return field, gradients


@pytest.mark.parametrize("sum_space", ("physical", "spectral"))
def test_distortion_parity_split_differentiates_all_nine_components(sum_space):
    solver, projector = _solver_and_projector(
        (20, 18, 16),
        (2.0 * math.pi, 2.0 * math.pi, math.pi),
    )
    x, y, z = solver.transform_backend.spatial_grids
    modes = tuple(_parity_split_mode(index, x, y, z) for index in range(9))
    stress = tuple(mode[0] for mode in modes)
    gradients = tuple(mode[1] for mode in modes)

    observed = projected_distortion_stress_divergence(
        solver.transform_backend,
        stress,
        Q_BC,
        DISTORTION_ODD_Z_BC,
        projector=projector,
        sum_space=sum_space,
    )
    expected = torch.stack(
        tuple(
            gradients[3 * row][0]
            + gradients[3 * row + 1][1]
            + gradients[3 * row + 2][2]
            for row in range(3)
        )
    )

    torch.testing.assert_close(observed, expected, rtol=2e-12, atol=2e-12)


def test_stress_divergence_helpers_reject_unknown_sum_space():
    solver, projector = _solver_and_projector((6, 6, 5), (3.0, 3.0, 2.0))
    zeros = tuple(torch.zeros((6, 6, 5), dtype=torch.float64) for _ in range(9))

    with pytest.raises(ValueError, match="sum_space"):
        projected_common_basis_stress_divergence(
            solver.transform_backend,
            zeros,
            Q_BC,
            projector=projector,
            sum_space="unknown",
        )
    with pytest.raises(ValueError, match="sum_space"):
        projected_distortion_stress_divergence(
            solver.transform_backend,
            zeros,
            Q_BC,
            DISTORTION_ODD_Z_BC,
            projector=projector,
            sum_space="unknown",
        )


def test_neumann_qxz_active_stress_has_nonzero_mean_tangential_force():
    shape = (12, 10, 32)
    length_z = 3.0
    solver, projector = _solver_and_projector(
        shape,
        (2.0 * math.pi, 2.0 * math.pi, length_z),
    )
    x, _, z = solver.transform_backend.spatial_grids
    amplitude = 0.23
    active_prefactor = -0.17
    kz = math.pi / length_z
    zeros = torch.zeros_like(x + z)
    qxz = amplitude * torch.cos(kz * z) + zeros
    q_components = (zeros, zeros, qxz, zeros, zeros)

    active_stress = beris_edwards_active_stress_components(
        q_components,
        active_prefactor=active_prefactor,
    )
    observed = projected_common_basis_stress_divergence(
        solver.transform_backend,
        active_stress,
        Q_BC,
        projector=projector,
    )
    expected_fx = (
        -active_prefactor * amplitude * kz * torch.sin(kz * z) + zeros
    )
    expected = torch.stack((expected_fx, zeros, zeros))

    torch.testing.assert_close(observed, expected, rtol=2e-12, atol=2e-12)
    observed_mean = observed[0].mean()
    expected_discrete_mean = (
        -active_prefactor
        * amplitude
        * kz
        / (shape[2] * math.sin(math.pi / (2.0 * shape[2])))
    )
    torch.testing.assert_close(
        observed_mean,
        torch.as_tensor(expected_discrete_mean, dtype=torch.float64),
        rtol=2e-13,
        atol=2e-13,
    )
    assert observed_mean.abs() > 1e-2


def test_production_stokes_compute_force_preserves_qxz_active_mean():
    shape = (10, 8, 24)
    length_z = 2.5
    solver, projector = _solver_and_projector(
        shape,
        (2.0 * math.pi, 2.0 * math.pi, length_z),
    )
    x, _, z = solver.transform_backend.spatial_grids
    amplitude = 0.19
    alpha = torch.tensor(0.13, dtype=torch.float64)
    beta = -1.0
    active_prefactor = beta * alpha.item()
    kz = math.pi / length_z
    zeros = torch.zeros_like(x + z)
    qxz = amplitude * torch.cos(kz * z) + zeros
    initial_q = {
        "Qxx": zeros,
        "Qxy": zeros,
        "Qxz": qxz,
        "Qyy": zeros,
        "Qyz": zeros,
    }
    zero_operator = torch.zeros(shape, dtype=torch.float64)
    for name in Q_COMPONENTS:
        solver.model.add_dynamic_field(
            name,
            init=initial_q[name],
            L_hat=zero_operator,
            boundary_conditions=Q_BC,
        )
    solver.model.build()

    production_model = BerisEdwardsFreeSlipStokes(
        solver,
        projector,
        beta_value=beta,
        friction=0.0,
        viscosity=2.0 / 3.0,
        ldg_a=0.0,
        ldg_b=0.0,
        ldg_c=0.0,
        ldg_l1=0.17,
        flow_alignment=0.0,
        pointwise_kernels=BerisEdwardsPointwiseKernels("eager"),
        zero_mode_policy="zero_mean",
    )
    total_force, active_tangential_force = (
        production_model.compute_nematic_force(solver.fields, alpha)
    )
    expected_fx = (
        -active_prefactor * amplitude * kz * torch.sin(kz * z) + zeros
    )
    expected_tangential = torch.stack((expected_fx, zeros))

    # H is proportional to Q and lambda=0, while the z-only distortion stress
    # contributes only to f_z.  The complete production path therefore has an
    # exactly active tangential force for this manufactured Q field.
    torch.testing.assert_close(
        total_force[:2, 0], expected_tangential, rtol=3e-12, atol=3e-12
    )
    torch.testing.assert_close(
        active_tangential_force[:, 0],
        expected_tangential,
        rtol=3e-12,
        atol=3e-12,
    )
    observed_mean = total_force[0].mean()
    expected_discrete_mean = (
        -active_prefactor
        * amplitude
        * kz
        / (shape[2] * math.sin(math.pi / (2.0 * shape[2])))
    )
    torch.testing.assert_close(
        observed_mean,
        torch.as_tensor(expected_discrete_mean, dtype=torch.float64),
        rtol=3e-13,
        atol=3e-13,
    )
    assert observed_mean.abs() > 1e-2

    velocity_pressure_hat = production_model(
        solver.fields,
        {"alpha": alpha},
    )
    expected_mean = torch.stack(
        (
            torch.as_tensor(expected_discrete_mean, dtype=torch.float64),
            torch.zeros((), dtype=torch.float64),
        )
    ).unsqueeze(-1)
    torch.testing.assert_close(
        production_model.last_total_tangential_force_mean,
        expected_mean,
        rtol=3e-13,
        atol=3e-13,
    )
    torch.testing.assert_close(
        production_model.last_active_tangential_force_mean,
        expected_mean,
        rtol=3e-13,
        atol=3e-13,
    )
    torch.testing.assert_close(
        production_model.last_passive_tangential_force_mean,
        torch.zeros_like(expected_mean),
        rtol=0,
        atol=3e-13,
    )
    torch.testing.assert_close(
        production_model.last_removed_tangential_force_mean,
        expected_mean,
        rtol=3e-13,
        atol=3e-13,
    )
    ux = solver.transform_backend.inverse(
        velocity_pressure_hat[0],
        ("periodic", "periodic", "neumann"),
    )
    assert ux.mean().abs() < 3e-13


def _autodiff_row_divergence(stress, coordinates):
    rows = []
    for row in range(3):
        divergence = torch.zeros_like(coordinates[0])
        for column in range(3):
            derivative = torch.autograd.grad(
                stress[..., row, column].sum(),
                coordinates[column],
                retain_graph=True,
            )[0]
            divergence = divergence + derivative
        rows.append(divergence)
    return torch.stack(rows)


def test_complete_beris_edwards_force_matches_full_matrix_autodiff_oracle():
    solver, projector = _solver_and_projector(
        (16, 16, 16),
        (2.0 * math.pi, 2.0 * math.pi, math.pi),
    )
    coordinates = tuple(
        grid.detach().clone().requires_grad_(True)
        for grid in solver.transform_backend.spatial_grids
    )
    x, y, z = coordinates
    dtype = torch.float64
    t1 = torch.tensor(
        (
            (0.20, 0.11, -0.07),
            (0.11, -0.05, 0.09),
            (-0.07, 0.09, -0.15),
        ),
        dtype=dtype,
    )
    t2 = torch.tensor(
        (
            (-0.08, 0.04, 0.13),
            (0.04, 0.17, -0.06),
            (0.13, -0.06, -0.09),
        ),
        dtype=dtype,
    )
    amplitude = 0.08
    ldg_l1 = 0.19
    flow_alignment = 0.31
    active_prefactor = -0.14
    phi1 = torch.cos(x) * torch.cos(2.0 * z)
    phi2 = torch.sin(y) * torch.cos(z)
    q = amplitude * (
        phi1[..., None, None] * t1 + phi2[..., None, None] * t2
    )
    lap_q = -amplitude * (
        5.0 * phi1[..., None, None] * t1
        + 2.0 * phi2[..., None, None] * t2
    )
    expected_h = ldg_l1 * lap_q
    q_gradients = (
        -amplitude
        * torch.sin(x)[..., None, None]
        * torch.cos(2.0 * z)[..., None, None]
        * t1,
        amplitude
        * torch.cos(y)[..., None, None]
        * torch.cos(z)[..., None, None]
        * t2,
        amplitude
        * (
            -2.0
            * torch.cos(x)[..., None, None]
            * torch.sin(2.0 * z)[..., None, None]
            * t1
            - torch.sin(y)[..., None, None]
            * torch.sin(z)[..., None, None]
            * t2
        ),
    )

    q_components = _compact(q)
    h_components = beris_edwards_molecular_field_components(
        q_components,
        _compact(lap_q),
        ldg_a=0.0,
        ldg_b=0.0,
        ldg_c=0.0,
        ldg_l1=ldg_l1,
    )
    torch.testing.assert_close(
        _matrix(h_components), expected_h, rtol=2e-13, atol=2e-13
    )

    algebraic_components = beris_edwards_algebraic_stress_components(
        q_components,
        h_components,
        flow_alignment=flow_alignment,
        active_prefactor=active_prefactor,
    )
    distortion_components = beris_edwards_distortion_stress_components(
        tuple(_compact(gradient) for gradient in q_gradients),
        ldg_l1=ldg_l1,
    )
    observed_algebraic = projected_common_basis_stress_divergence(
        solver.transform_backend,
        algebraic_components,
        Q_BC,
        projector=projector,
    )
    observed_distortion = projected_distortion_stress_divergence(
        solver.transform_backend,
        distortion_components,
        Q_BC,
        DISTORTION_ODD_Z_BC,
        projector=projector,
    )

    identity = torch.eye(3, dtype=dtype)
    m = q + identity / 3.0
    q_dot_h = torch.einsum("...ij,...ij->...", q, expected_h)
    reactive = (
        2.0 * flow_alignment * m * q_dot_h[..., None, None]
        - flow_alignment * (expected_h @ m + m @ expected_h)
        + q @ expected_h
        - expected_h @ q
    )
    active = active_prefactor * q
    gradient_tensor = torch.stack(q_gradients, dim=-3)
    distortion = -ldg_l1 * torch.einsum(
        "...ikl,...jkl->...ij", gradient_tensor, gradient_tensor
    )
    assert torch.max(torch.abs(q @ expected_h - expected_h @ q)) > 1e-5

    expected_algebraic = _autodiff_row_divergence(
        reactive + active,
        coordinates,
    )
    expected_distortion = _autodiff_row_divergence(
        distortion,
        coordinates,
    )
    torch.testing.assert_close(
        observed_algebraic, expected_algebraic, rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(
        observed_distortion, expected_distortion, rtol=2e-11, atol=2e-11
    )
    torch.testing.assert_close(
        observed_algebraic + observed_distortion,
        expected_algebraic + expected_distortion,
        rtol=2e-11,
        atol=2e-11,
    )
