import math

import pytest
import torch

from pssolver import (
    BasisAwareSpectralProjector,
    FreeSlipModalStokesSolver,
    SpectralSolver,
    TensorProductTransformBackend,
)
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsPointwiseKernels,
    Q_COMPONENTS,
    beris_edwards_active_stress_components,
    beris_edwards_flow_alignment_components,
    beris_edwards_free_energy_density,
    beris_edwards_molecular_field_components,
    q_tensor_contraction,
)


Q_BC = ("periodic", "periodic", "neumann")
U_TANGENTIAL_BC = ("periodic", "periodic", "neumann")
U_NORMAL_BC = ("periodic", "periodic", "dirichlet")
PRESSURE_BC = ("periodic", "periodic", "neumann")


def _cell_volume(backend):
    return math.prod(
        length / size
        for length, size in zip(backend.lengths, backend.shape)
    )


def _gradient(backend, tensor, boundary_conditions, axis):
    tensor_hat = backend.forward(tensor, boundary_conditions)
    gradient_hat, gradient_bcs = backend.gradient_hat(
        tensor_hat,
        boundary_conditions,
        axis,
    )
    return backend.inverse(gradient_hat, gradient_bcs)


def _q_gradients(backend, q_components):
    q_tensor = torch.stack(tuple(q_components))
    q_hat = backend.forward(q_tensor, Q_BC)
    gradients = []
    for axis in range(3):
        gradient_hat, gradient_bcs = backend.gradient_hat(q_hat, Q_BC, axis)
        gradient_tensor = backend.inverse(gradient_hat, gradient_bcs)
        gradients.append(tuple(gradient_tensor[index] for index in range(5)))
    return tuple(gradients)


def _q_laplacian(backend, q_components):
    q_tensor = torch.stack(tuple(q_components))
    q_hat = backend.forward(q_tensor, Q_BC)
    laplacian_hat = backend.laplacian_hat(q_hat, Q_BC)
    laplacian = backend.inverse(laplacian_hat, Q_BC)
    return tuple(laplacian[index] for index in range(5))


def _free_energy(backend, q_components, coefficients):
    density = beris_edwards_free_energy_density(
        q_components,
        _q_gradients(backend, q_components),
        **coefficients,
    )
    return _cell_volume(backend) * density.sum()


def _manufactured_q(backend):
    x, y, z = backend.spatial_grids
    lx, ly, lz = backend.lengths
    kx = 2.0 * math.pi / lx
    ky = 2.0 * math.pi / ly
    kz = math.pi / lz
    return (
        0.04
        + 0.11 * torch.cos(kx * x) * torch.cos(2.0 * ky * y) * torch.cos(kz * z),
        0.07 * torch.sin(2.0 * kx * x) * torch.cos(ky * y) * torch.cos(2.0 * kz * z),
        0.05 * torch.cos(kx * x) * torch.sin(ky * y) * torch.cos(3.0 * kz * z),
        -0.02
        + 0.09 * torch.sin(kx * x) * torch.sin(2.0 * ky * y) * torch.cos(2.0 * kz * z),
        0.06 * torch.cos(2.0 * kx * x) * torch.sin(ky * y) * torch.cos(kz * z),
    )


def _manufactured_q_direction(backend):
    x, y, z = backend.spatial_grids
    lx, ly, lz = backend.lengths
    kx = 2.0 * math.pi / lx
    ky = 2.0 * math.pi / ly
    kz = math.pi / lz
    return (
        0.13 * torch.sin(kx * x) * torch.cos(ky * y) * torch.cos(2.0 * kz * z),
        -0.08 * torch.cos(2.0 * kx * x) * torch.sin(ky * y) * torch.cos(kz * z),
        0.04 * torch.sin(kx * x) * torch.sin(2.0 * ky * y) * torch.cos(3.0 * kz * z),
        0.10 * torch.cos(kx * x) * torch.cos(2.0 * ky * y) * torch.cos(kz * z),
        -0.05 * torch.sin(2.0 * kx * x) * torch.cos(ky * y) * torch.cos(2.0 * kz * z),
    )


def test_free_energy_directional_derivative_is_minus_raw_molecular_field_pairing():
    backend = TensorProductTransformBackend(
        shape=(18, 16, 14),
        lengths=(2.0 * math.pi, 3.0 * math.pi, 1.7),
        device="cpu",
        dtype=torch.float64,
    )
    coefficients = {
        "ldg_a": 0.17,
        "ldg_b": -0.31,
        "ldg_c": 0.43,
        "ldg_l1": 0.08,
    }
    q = _manufactured_q(backend)
    direction = _manufactured_q_direction(backend)

    epsilon = torch.zeros((), dtype=torch.float64, requires_grad=True)
    perturbed_q = tuple(
        component + epsilon * perturbation
        for component, perturbation in zip(q, direction)
    )
    directional_derivative = torch.autograd.grad(
        _free_energy(backend, perturbed_q, coefficients),
        epsilon,
    )[0]

    raw_h = beris_edwards_molecular_field_components(
        q,
        _q_laplacian(backend, q),
        **coefficients,
    )
    expected = -_cell_volume(backend) * q_tensor_contraction(
        raw_h,
        direction,
    ).sum()

    torch.testing.assert_close(
        directional_derivative,
        expected,
        rtol=2e-11,
        atol=2e-11,
    )


class _ProjectorOwner:
    def __init__(self, backend):
        self.shape = backend.shape
        self.transform_backend = backend


def test_resolved_passive_relaxation_dissipates_free_energy_as_h_squared_over_gamma():
    backend = TensorProductTransformBackend(
        shape=(20, 18, 16),
        lengths=(2.0 * math.pi, 3.0 * math.pi, 1.7),
        device="cpu",
        dtype=torch.float64,
    )
    coefficients = {
        "ldg_a": 0.17,
        "ldg_b": -0.31,
        "ldg_c": 0.43,
        "ldg_l1": 0.08,
    }
    gamma = 2.94
    q = _manufactured_q(backend)
    raw_h = beris_edwards_molecular_field_components(
        q,
        _q_laplacian(backend, q),
        **coefficients,
    )
    projector = BasisAwareSpectralProjector(
        _ProjectorOwner(backend),
        rule="cubic_half",
    )
    raw_h_tensor = torch.stack(raw_h)
    resolved_h_tensor = backend.inverse(
        projector.project(backend.forward(raw_h_tensor, Q_BC), Q_BC),
        Q_BC,
    )
    resolved_h = tuple(resolved_h_tensor[index] for index in range(5))
    q_dot = tuple(component / gamma for component in resolved_h)

    # Make sure this exercises the resolved field rather than an identity
    # projection of a deliberately low-order molecular field.
    assert torch.linalg.vector_norm(raw_h_tensor - resolved_h_tensor) > 1e-6

    epsilon = torch.zeros((), dtype=torch.float64, requires_grad=True)
    perturbed_q = tuple(
        component + epsilon * rate
        for component, rate in zip(q, q_dot)
    )
    observed_rate = torch.autograd.grad(
        _free_energy(backend, perturbed_q, coefficients),
        epsilon,
    )[0]
    dissipation = (
        _cell_volume(backend)
        * q_tensor_contraction(resolved_h, resolved_h).sum()
        / gamma
    )

    assert dissipation > 0
    torch.testing.assert_close(
        observed_rate,
        -dissipation,
        rtol=3e-11,
        atol=3e-11,
    )


def test_manufactured_stokes_power_balance_and_pressure_orthogonality():
    shape = (18, 14, 12)
    lengths = (2.0 * math.pi, 3.0 * math.pi, 1.5)
    backend = TensorProductTransformBackend(
        shape=shape,
        lengths=lengths,
        device="cpu",
        dtype=torch.float64,
    )
    friction = 0.23
    viscosity = 0.71
    stokes = FreeSlipModalStokesSolver(
        backend,
        tangential_boundary_conditions=U_TANGENTIAL_BC,
        normal_boundary_conditions=U_NORMAL_BC,
        pressure_boundary_conditions=PRESSURE_BC,
        friction=friction,
        viscosity=viscosity,
        zero_mode_policy="friction",
    )

    x, _, z = backend.spatial_grids
    lx, _, lz = lengths
    kx = 2.0 * (2.0 * math.pi / lx)
    kz = 2.0 * math.pi / lz
    stream_amplitude = 0.17
    ux = (
        stream_amplitude * kz * torch.sin(kx * x) * torch.cos(kz * z)
    ).expand(shape)
    uz = (
        -stream_amplitude * kx * torch.cos(kx * x) * torch.sin(kz * z)
    ).expand(shape)

    uy_kx = 2.0 * math.pi / lx
    uy_kz = math.pi / lz
    uy = (
        0.11 * torch.cos(uy_kx * x) * torch.cos(uy_kz * z)
    ).expand(shape)
    pressure_amplitude = 0.13
    pressure = (
        pressure_amplitude * torch.cos(kx * x) * torch.cos(kz * z)
    ).expand(shape)

    velocity_eigenvalue = friction + viscosity * (kx**2 + kz**2)
    uy_eigenvalue = friction + viscosity * (uy_kx**2 + uy_kz**2)
    pressure_x = -pressure_amplitude * kx * torch.sin(kx * x) * torch.cos(kz * z)
    pressure_z = -pressure_amplitude * kz * torch.cos(kx * x) * torch.sin(kz * z)
    fx = velocity_eigenvalue * ux + pressure_x
    fy = uy_eigenvalue * uy
    fz = velocity_eigenvalue * uz + pressure_z

    fx_hat = backend.forward(fx.unsqueeze(0), U_TANGENTIAL_BC)
    fy_hat = backend.forward(fy.unsqueeze(0), U_TANGENTIAL_BC)
    fz_hat = backend.forward(fz.unsqueeze(0), U_NORMAL_BC)
    ux_hat, uy_hat, uz_hat, pressure_hat = stokes.solve_force_hats(
        fx_hat,
        fy_hat,
        fz_hat,
    )
    observed_ux = backend.inverse(ux_hat, U_TANGENTIAL_BC)
    observed_uy = backend.inverse(uy_hat, U_TANGENTIAL_BC)
    observed_uz = backend.inverse(uz_hat, U_NORMAL_BC)
    observed_pressure = backend.inverse(pressure_hat, PRESSURE_BC)

    torch.testing.assert_close(observed_ux[0], ux, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(observed_uy[0], uy, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(observed_uz[0], uz, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        observed_pressure[0],
        pressure,
        rtol=2e-12,
        atol=2e-12,
    )

    divergence = backend.inverse(
        stokes.divergence_hat(ux_hat, uy_hat, uz_hat),
        PRESSURE_BC,
    )
    assert divergence.abs().max() < 2e-12
    assert stokes.last_pressure_relative_residual < 2e-12

    pressure_gradient_hats = stokes.pressure_gradient_hats(pressure_hat)
    pressure_gradients = (
        backend.inverse(pressure_gradient_hats[0], U_TANGENTIAL_BC),
        backend.inverse(pressure_gradient_hats[1], U_TANGENTIAL_BC),
        backend.inverse(pressure_gradient_hats[2], U_NORMAL_BC),
    )
    dv = _cell_volume(backend)
    pressure_power = dv * sum(
        (velocity * gradient).sum()
        for velocity, gradient in zip(
            (observed_ux, observed_uy, observed_uz),
            pressure_gradients,
        )
    )

    velocity_fields = (observed_ux, observed_uy, observed_uz)
    velocity_bcs = (U_TANGENTIAL_BC, U_TANGENTIAL_BC, U_NORMAL_BC)
    viscous_dissipation = viscosity * dv * sum(
        _gradient(backend, velocity, boundary_conditions, axis).square().sum()
        for velocity, boundary_conditions in zip(velocity_fields, velocity_bcs)
        for axis in range(3)
    )
    drag_dissipation = friction * dv * sum(
        velocity.square().sum()
        for velocity in velocity_fields
    )
    force_power = dv * (
        (observed_ux[0] * fx).sum()
        + (observed_uy[0] * fy).sum()
        + (observed_uz[0] * fz).sum()
    )

    torch.testing.assert_close(
        pressure_power,
        torch.zeros_like(pressure_power),
        rtol=0,
        atol=2e-11,
    )
    torch.testing.assert_close(
        force_power,
        viscous_dissipation + drag_dissipation,
        rtol=3e-12,
        atol=3e-11,
    )


def _wall_bilinear_error(nz):
    nx = ny = 4
    lx = ly = 2.0 * math.pi
    lz = 1.7
    backend = TensorProductTransformBackend(
        shape=(nx, ny, nz),
        lengths=(lx, ly, lz),
        device="cpu",
        dtype=torch.float64,
    )
    z = backend.spatial_grids[2]
    u0 = 0.8
    u1 = 0.23
    traction_amplitude = 0.6
    velocity = (
        u0 + u1 * torch.cos(2.0 * math.pi * z / lz)
    ).expand(nx, ny, nz)
    stress_xz = (
        traction_amplitude * torch.cos(math.pi * z / lz)
    ).expand(nx, ny, nz)
    dz_velocity = _gradient(backend, velocity, U_TANGENTIAL_BC, axis=2)
    force_x = _gradient(backend, stress_xz, Q_BC, axis=2)

    dv = _cell_volume(backend)
    force_power = dv * (velocity * force_x).sum()
    stress_power = dv * (stress_xz * dz_velocity).sum()
    discrete_wall_bilinear = force_power + stress_power
    nodes = torch.arange(nz, dtype=torch.float64)
    modes = torch.arange(nz, dtype=torch.float64)
    phase = math.pi * (nodes + 0.5) / nz
    dct = torch.cos(modes[:, None] * phase[None, :])
    dct[0] *= math.sqrt(1.0 / nz)
    dct[1:] *= math.sqrt(2.0 / nz)
    dst = math.sqrt(2.0 / nz) * torch.sin(
        (modes[:, None] + 1.0) * phase[None, :]
    )
    dst[-1] *= math.sqrt(0.5)
    modal_derivative = torch.zeros((nz, nz), dtype=torch.float64)
    paired_modes = torch.arange(1, nz, dtype=torch.float64)
    modal_derivative[
        torch.arange(nz - 1),
        torch.arange(1, nz),
    ] = -paired_modes * math.pi / lz
    physical_derivative = dst.transpose(0, 1) @ modal_derivative @ dct
    direct_boundary_bilinear = (
        lx
        * ly
        * (lz / nz)
        * velocity[0, 0]
        @ (physical_derivative + physical_derivative.transpose(0, 1))
        @ stress_xz[0, 0]
    )
    torch.testing.assert_close(
        discrete_wall_bilinear,
        direct_boundary_bilinear,
        rtol=0,
        atol=3e-13,
    )

    analytic_wall_power = (
        lx
        * ly
        * (-2.0 * traction_amplitude * (u0 + u1))
    )
    return abs(discrete_wall_bilinear.item() - analytic_wall_power)


def test_discrete_wall_bilinear_closes_and_converges_second_order_to_wall_traction():
    errors = [_wall_bilinear_error(nz) for nz in (16, 32, 64)]
    observed_orders = [
        math.log(errors[index] / errors[index + 1], 2.0)
        for index in range(len(errors) - 1)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert all(1.95 < order < 2.05 for order in observed_orders)


def test_active_power_sign_and_zero_mean_constraint_power():
    q = tuple(
        torch.tensor(value, dtype=torch.float64)
        for value in (0.20, 0.03, 0.00, -0.05, 0.00)
    )
    strain = tuple(
        torch.tensor(value, dtype=torch.float64)
        for value in (0.10, 0.02, 0.00, -0.03, 0.00)
    )
    zeta = 0.04
    active_stress = beris_edwards_active_stress_components(
        q,
        active_prefactor=-zeta,
    )
    exx, exy, exz, eyy, eyz = strain
    ezz = -exx - eyy
    strain_row_major = (
        exx, exy, exz,
        exy, eyy, eyz,
        exz, eyz, ezz,
    )
    active_power = -sum(
        stress_component * strain_component
        for stress_component, strain_component in zip(
            active_stress,
            strain_row_major,
        )
    )
    expected_active_power = zeta * q_tensor_contraction(q, strain)
    assert active_power > 0
    torch.testing.assert_close(
        active_power,
        expected_active_power,
        rtol=0,
        atol=2e-16,
    )

    shape = (8, 6, 6)
    lengths = (2.0 * math.pi, 3.0 * math.pi, 1.5)
    backend = TensorProductTransformBackend(
        shape=shape,
        lengths=lengths,
        device="cpu",
        dtype=torch.float64,
    )
    stokes = FreeSlipModalStokesSolver(
        backend,
        tangential_boundary_conditions=U_TANGENTIAL_BC,
        normal_boundary_conditions=U_NORMAL_BC,
        pressure_boundary_conditions=PRESSURE_BC,
        friction=0.0,
        viscosity=0.71,
        zero_mode_policy="zero_mean",
    )
    force_x = torch.full((1, *shape), 0.37, dtype=torch.float64)
    zeros = torch.zeros_like(force_x)
    ux_hat, uy_hat, uz_hat, pressure_hat = stokes.solve_force_hats(
        backend.forward(force_x, U_TANGENTIAL_BC),
        backend.forward(zeros, U_TANGENTIAL_BC),
        backend.forward(zeros, U_NORMAL_BC),
    )
    ux = backend.inverse(ux_hat, U_TANGENTIAL_BC)
    uy = backend.inverse(uy_hat, U_TANGENTIAL_BC)
    uz = backend.inverse(uz_hat, U_NORMAL_BC)
    reaction_x = -force_x.mean()
    constraint_power = math.prod(lengths) * (
        reaction_x * ux.mean()
    )

    assert ux.abs().max() < 2e-14
    assert uy.abs().max() < 2e-14
    assert uz.abs().max() < 2e-14
    assert pressure_hat.abs().max() < 2e-14
    torch.testing.assert_close(
        constraint_power,
        torch.zeros_like(constraint_power),
        rtol=0,
        atol=2e-14,
    )


@pytest.mark.parametrize("sum_space", ("physical", "spectral"))
def test_coupled_passive_semidiscrete_energy_budget_closes_without_wall_power(
    sum_space,
):
    shape = (16, 16, 6)
    lengths = (2.0 * math.pi, 2.0 * math.pi, 1.5)
    solver = SpectralSolver(
        shape,
        L=lengths,
        device="cpu",
        dtype=torch.float64,
    )
    backend = solver.transform_backend
    projector = BasisAwareSpectralProjector(solver, rule="cubic_half")
    x, y, z = backend.spatial_grids
    zeros = torch.zeros_like(x + y + z)
    q_values = (
        0.07 * torch.cos(x) + 0.025 * torch.sin(y) + zeros,
        0.045 * torch.sin(x) * torch.cos(y) + zeros,
        zeros,
        -0.035 * torch.cos(y) + 0.02 * torch.sin(x + y) + zeros,
        zeros,
    )
    zero_operator = torch.zeros(shape, dtype=torch.float64)
    for name, value in zip(Q_COMPONENTS, q_values):
        solver.model.add_dynamic_field(
            name,
            init=value,
            L_hat=zero_operator,
            boundary_conditions=Q_BC,
        )
    solver.model.build()

    coefficients = {
        "ldg_a": 0.17,
        "ldg_b": -0.31,
        "ldg_c": 0.43,
        "ldg_l1": 0.08,
    }
    gamma = 2.94
    flow_alignment = 0.31
    viscosity = 0.71
    friction = 0.23
    production = BerisEdwardsFreeSlipStokes(
        solver,
        projector,
        beta_value=-1.0,
        friction=friction,
        viscosity=viscosity,
        flow_alignment=flow_alignment,
        stress_divergence_sum_space=sum_space,
        pointwise_kernels=BerisEdwardsPointwiseKernels("eager"),
        zero_mode_policy="friction",
        **coefficients,
    )
    force, active_tangential_force = production.compute_nematic_force(
        solver.fields,
        torch.zeros((), dtype=torch.float64),
    )
    torch.testing.assert_close(
        active_tangential_force,
        torch.zeros_like(active_tangential_force),
        rtol=0,
        atol=0,
    )
    force_t_hat = projector.project(
        backend.forward(force[:2], U_TANGENTIAL_BC),
        U_TANGENTIAL_BC,
    )
    force_n_hat = projector.project(
        backend.forward(force[2], U_NORMAL_BC),
        U_NORMAL_BC,
    )
    velocity_hats = production.solve_force_hats(
        force_t_hat[0],
        force_t_hat[1],
        force_n_hat,
    )
    velocity = (
        backend.inverse(velocity_hats[0], U_TANGENTIAL_BC),
        backend.inverse(velocity_hats[1], U_TANGENTIAL_BC),
        backend.inverse(velocity_hats[2], U_NORMAL_BC),
    )

    q = tuple(solver.fields[name] for name in Q_COMPONENTS)
    q_gradients = _q_gradients(backend, q)
    raw_h_tensor = torch.stack(
        beris_edwards_molecular_field_components(
            q,
            _q_laplacian(backend, q),
            **coefficients,
        )
    )
    resolved_h_tensor = backend.inverse(
        projector.project(backend.forward(raw_h_tensor, Q_BC), Q_BC),
        Q_BC,
    )
    resolved_h = tuple(resolved_h_tensor[index] for index in range(5))

    velocity_gradients = tuple(
        tuple(
            _gradient(backend, velocity[component], (
                U_TANGENTIAL_BC,
                U_TANGENTIAL_BC,
                U_NORMAL_BC,
            )[component], axis)
            for component in range(3)
        )
        for axis in range(3)
    )
    alignment = beris_edwards_flow_alignment_components(
        q,
        velocity_gradients,
        flow_alignment=flow_alignment,
    )
    q_rate_unprojected = tuple(
        resolved_h[index] / gamma
        - sum(
            velocity[axis] * q_gradients[axis][index]
            for axis in range(3)
        )
        + alignment[index]
        for index in range(5)
    )
    q_rate_tensor = backend.inverse(
        projector.project(
            backend.forward(torch.stack(q_rate_unprojected), Q_BC),
            Q_BC,
        ),
        Q_BC,
    )
    q_rate = tuple(q_rate_tensor[index] for index in range(5))

    dv = _cell_volume(backend)
    free_energy_rate = -dv * q_tensor_contraction(
        resolved_h,
        q_rate,
    ).sum()
    rotational_dissipation = (
        dv
        * q_tensor_contraction(resolved_h, resolved_h).sum()
        / gamma
    )
    velocity_gradient_matrix = torch.stack(
        tuple(
            torch.stack(
                tuple(velocity_gradients[axis][component] for axis in range(3)),
                dim=-1,
            )
            for component in range(3)
        ),
        dim=-2,
    )
    strain = 0.5 * (
        velocity_gradient_matrix
        + velocity_gradient_matrix.transpose(-1, -2)
    )
    viscous_dissipation = 2.0 * viscosity * dv * strain.square().sum()
    drag_dissipation = friction * dv * sum(
        component.square().sum() for component in velocity
    )
    budget_residual = (
        free_energy_rate
        + rotational_dissipation
        + viscous_dissipation
        + drag_dissipation
    )
    budget_scale = (
        free_energy_rate.abs()
        + rotational_dissipation
        + viscous_dissipation
        + drag_dissipation
    )

    # Q and all stresses are z-extruded with Qxz=Qyz=0, so tangential
    # nematic traction and physical wall power vanish identically.
    assert force[2].abs().max() < 2e-13
    assert production.last_pressure_relative_residual < 2e-12
    assert budget_residual.abs() / budget_scale < 2e-11
