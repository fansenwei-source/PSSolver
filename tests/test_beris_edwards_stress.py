import torch

from pssolver.models.active_nematics import (
    BerisEdwardsPointwiseKernels,
    BerisEdwardsQNonlinearModel,
    Q_COMPONENTS,
    beris_edwards_active_stress_components,
    beris_edwards_algebraic_stress_components,
    beris_edwards_bulk_molecular_field_components,
    beris_edwards_distortion_stress_components,
    beris_edwards_flow_alignment_components,
    beris_edwards_linear_operator,
    beris_edwards_molecular_field_components,
    beris_edwards_q_nonlinear_components,
    beris_edwards_reactive_stress_components,
)


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


def _random_traceless_symmetric(generator, count=7):
    raw = torch.randn(count, 3, 3, dtype=torch.float64, generator=generator)
    symmetric = 0.5 * (raw + raw.transpose(-1, -2))
    trace = torch.diagonal(symmetric, dim1=-2, dim2=-1).sum(-1)
    identity = torch.eye(3, dtype=raw.dtype, device=raw.device)
    return symmetric - trace[..., None, None] * identity / 3.0


def _axis_first_velocity_gradients(velocity_gradient):
    return tuple(
        tuple(
            velocity_gradient[..., component, axis]
            for component in range(3)
        )
        for axis in range(3)
    )


def test_molecular_field_matches_full_matrix_oracle():
    generator = torch.Generator().manual_seed(31)
    q = _random_traceless_symmetric(generator)
    lap_q = _random_traceless_symmetric(generator)
    coefficients = dict(
        ldg_a=-0.07,
        ldg_b=-0.3,
        ldg_c=0.41,
        ldg_l1=0.019,
    )

    observed = _matrix(
        beris_edwards_molecular_field_components(
            _compact(q),
            _compact(lap_q),
            **coefficients,
        )
    )
    q2 = q @ q
    tr_q2 = torch.diagonal(q2, dim1=-2, dim2=-1).sum(-1)
    expected = (
        -coefficients["ldg_a"] * q
        - coefficients["ldg_b"]
        * (q2 - torch.eye(3, dtype=q.dtype) * tr_q2[..., None, None] / 3.0)
        - coefficients["ldg_c"] * tr_q2[..., None, None] * q
        + coefficients["ldg_l1"] * lap_q
    )

    torch.testing.assert_close(observed, expected, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        torch.diagonal(observed, dim1=-2, dim2=-1).sum(-1),
        torch.zeros(q.shape[0], dtype=q.dtype),
        rtol=0,
        atol=1e-14,
    )


def test_bulk_molecular_field_plus_laplacian_matches_complete_helper():
    generator = torch.Generator().manual_seed(310)
    q = _compact(_random_traceless_symmetric(generator))
    lap_q = _compact(_random_traceless_symmetric(generator))
    coefficients = dict(ldg_a=-0.07, ldg_b=-0.3, ldg_c=0.41)
    ldg_l1 = 0.019

    bulk = beris_edwards_bulk_molecular_field_components(
        q,
        **coefficients,
    )
    complete = beris_edwards_molecular_field_components(
        q,
        lap_q,
        **coefficients,
        ldg_l1=ldg_l1,
    )

    torch.testing.assert_close(
        torch.stack(tuple(
            bulk_component + ldg_l1 * laplacian_component
            for bulk_component, laplacian_component in zip(bulk, lap_q)
        )),
        torch.stack(complete),
        rtol=1.0e-13,
        atol=1.0e-13,
    )

def test_reactive_and_distortion_stresses_match_matrix_oracles():
    generator = torch.Generator().manual_seed(47)
    q = _random_traceless_symmetric(generator)
    h = _random_traceless_symmetric(generator)
    gradients = tuple(
        _random_traceless_symmetric(generator)
        for _ in range(3)
    )
    lam = 0.37
    ldg_l1 = 0.021

    reactive = torch.stack(
        beris_edwards_reactive_stress_components(
            _compact(q),
            _compact(h),
            flow_alignment=lam,
        ),
        dim=-1,
    ).reshape(q.shape[0], 3, 3)
    m = q + torch.eye(3, dtype=q.dtype) / 3.0
    q_dot_h = torch.einsum("...ij,...ij->...", q, h)
    expected_reactive = (
        2.0 * lam * m * q_dot_h[..., None, None]
        - lam * (h @ m + m @ h)
        + q @ h
        - h @ q
    )

    distortion = torch.stack(
        beris_edwards_distortion_stress_components(
            tuple(_compact(gradient) for gradient in gradients),
            ldg_l1=ldg_l1,
        ),
        dim=-1,
    ).reshape(q.shape[0], 3, 3)
    expected_distortion = torch.empty_like(distortion)
    for i in range(3):
        for j in range(3):
            expected_distortion[..., i, j] = -ldg_l1 * torch.einsum(
                "...kl,...kl->...",
                gradients[i],
                gradients[j],
            )

    torch.testing.assert_close(
        reactive,
        expected_reactive,
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        distortion,
        expected_distortion,
        rtol=1e-13,
        atol=1e-13,
    )


def test_uniform_shendruk_bulk_equilibrium_has_zero_molecular_field():
    q = torch.tensor(
        [1.0 / 3.0, 0.0, 0.0, -1.0 / 6.0, 0.0],
        dtype=torch.float64,
    )
    zeros = torch.zeros_like(q)
    h = beris_edwards_molecular_field_components(
        q,
        zeros,
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=0.02,
    )
    torch.testing.assert_close(
        torch.stack(h),
        torch.zeros_like(q),
        rtol=0,
        atol=1e-15,
    )


def test_reactive_stress_cancels_local_alignment_energy_exchange():
    generator = torch.Generator().manual_seed(71)
    q = _random_traceless_symmetric(generator, count=11)
    h = _random_traceless_symmetric(generator, count=11)
    velocity_gradient = torch.randn(
        11,
        3,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    trace = torch.diagonal(
        velocity_gradient,
        dim1=-2,
        dim2=-1,
    ).sum(-1)
    velocity_gradient = (
        velocity_gradient
        - trace[..., None, None] * torch.eye(3, dtype=velocity_gradient.dtype) / 3.0
    )
    strain = 0.5 * (
        velocity_gradient + velocity_gradient.transpose(-1, -2)
    )
    vorticity = 0.5 * (
        velocity_gradient - velocity_gradient.transpose(-1, -2)
    )
    lam = 0.29
    m = q + torch.eye(3, dtype=q.dtype) / 3.0
    q_dot_strain = torch.einsum("...ij,...ij->...", q, strain)
    alignment = (
        (lam * strain + vorticity) @ m
        + m @ (lam * strain - vorticity)
        - 2.0 * lam * m * q_dot_strain[..., None, None]
    )
    reactive = torch.stack(
        beris_edwards_reactive_stress_components(
            _compact(q),
            _compact(h),
            flow_alignment=lam,
        ),
        dim=-1,
    ).reshape(q.shape[0], 3, 3)
    exchange = (
        torch.einsum("...ij,...ij->...", h, alignment)
        + torch.einsum(
            "...ij,...ij->...",
            reactive,
            velocity_gradient,
        )
    )
    torch.testing.assert_close(
        exchange,
        torch.zeros_like(exchange),
        rtol=0,
        atol=2e-13,
    )

def _half_band_project(field):
    """Self-adjoint strict half-Nyquist projector along the sample axis."""
    sample_count = field.shape[0]
    spectrum = torch.fft.fft(field, dim=0)
    modes = torch.fft.fftfreq(
        sample_count,
        d=1.0 / sample_count,
        device=field.device,
    )
    keep = torch.abs(modes) < sample_count / 4
    mask_shape = (sample_count,) + (1,) * (field.ndim - 1)
    return torch.fft.ifft(
        spectrum * keep.reshape(mask_shape),
        dim=0,
    ).real


def test_complete_stress_projection_preserves_global_energy_exchange():
    generator = torch.Generator().manual_seed(83)
    sample_count = 32
    q = _half_band_project(
        _random_traceless_symmetric(generator, count=sample_count)
    )
    h = _half_band_project(
        _random_traceless_symmetric(generator, count=sample_count)
    )
    velocity_gradient = torch.randn(
        sample_count,
        3,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    trace = torch.diagonal(
        velocity_gradient,
        dim1=-2,
        dim2=-1,
    ).sum(-1)
    velocity_gradient = _half_band_project(
        velocity_gradient
        - trace[..., None, None]
        * torch.eye(3, dtype=velocity_gradient.dtype)
        / 3.0
    )
    strain = 0.5 * (
        velocity_gradient + velocity_gradient.transpose(-1, -2)
    )
    vorticity = 0.5 * (
        velocity_gradient - velocity_gradient.transpose(-1, -2)
    )
    lam = 0.29
    m = q + torch.eye(3, dtype=q.dtype) / 3.0
    q_dot_strain = torch.einsum("...ij,...ij->...", q, strain)
    alignment = _half_band_project(
        (lam * strain + vorticity) @ m
        + m @ (lam * strain - vorticity)
        - 2.0 * lam * m * q_dot_strain[..., None, None]
    )
    reactive = torch.stack(
        beris_edwards_reactive_stress_components(
            _compact(q),
            _compact(h),
            flow_alignment=lam,
        ),
        dim=-1,
    ).reshape(sample_count, 3, 3)
    reactive = _half_band_project(reactive)
    exchange = torch.mean(
        torch.einsum("...ij,...ij->...", h, alignment)
        + torch.einsum(
            "...ij,...ij->...",
            reactive,
            velocity_gradient,
        )
    )

    torch.testing.assert_close(
        exchange,
        torch.zeros_like(exchange),
        rtol=0,
        atol=2e-13,
    )

    projected_q_dot_h = _half_band_project(
        torch.einsum("...ij,...ij->...", q, h)
    )
    incorrectly_staged_reactive = torch.stack(
        beris_edwards_reactive_stress_components(
            _compact(q),
            _compact(h),
            flow_alignment=lam,
            q_dot_h=projected_q_dot_h,
        ),
        dim=-1,
    ).reshape(sample_count, 3, 3)
    incorrectly_staged_reactive = _half_band_project(
        incorrectly_staged_reactive
    )
    incorrect_exchange = torch.mean(
        torch.einsum("...ij,...ij->...", h, alignment)
        + torch.einsum(
            "...ij,...ij->...",
            incorrectly_staged_reactive,
            velocity_gradient,
        )
    )
    assert torch.abs(incorrect_exchange) > 1e-6

def test_flow_alignment_and_q_nonlinear_rhs_match_matrix_oracle():
    generator = torch.Generator().manual_seed(59)
    sample_count = 13
    q = _random_traceless_symmetric(generator, count=sample_count)
    q_gradients = tuple(
        _random_traceless_symmetric(generator, count=sample_count)
        for _ in range(3)
    )
    velocity = torch.randn(
        sample_count,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    velocity_gradient = torch.randn(
        sample_count,
        3,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    trace = torch.diagonal(
        velocity_gradient,
        dim1=-2,
        dim2=-1,
    ).sum(-1)
    velocity_gradient = (
        velocity_gradient
        - trace[..., None, None]
        * torch.eye(3, dtype=velocity_gradient.dtype)
        / 3.0
    )
    velocity_gradients = _axis_first_velocity_gradients(velocity_gradient)
    lam = 0.31
    b_over_gamma = -0.3 / 2.94
    c_over_gamma = 0.3 / 2.94

    observed_alignment = _matrix(
        beris_edwards_flow_alignment_components(
            _compact(q),
            velocity_gradients,
            flow_alignment=lam,
        )
    )
    strain = 0.5 * (
        velocity_gradient + velocity_gradient.transpose(-1, -2)
    )
    vorticity = 0.5 * (
        velocity_gradient - velocity_gradient.transpose(-1, -2)
    )
    m = q + torch.eye(3, dtype=q.dtype) / 3.0
    q_dot_strain = torch.einsum("...ij,...ij->...", q, strain)
    expected_alignment = (
        (lam * strain + vorticity) @ m
        + m @ (lam * strain - vorticity)
        - 2.0 * lam * m * q_dot_strain[..., None, None]
    )
    torch.testing.assert_close(
        observed_alignment,
        expected_alignment,
        rtol=1e-13,
        atol=1e-13,
    )

    observed_rhs = _matrix(
        beris_edwards_q_nonlinear_components(
            _compact(q),
            tuple(velocity[..., axis] for axis in range(3)),
            tuple(_compact(gradient) for gradient in q_gradients),
            velocity_gradients,
            ldg_b_over_gamma=b_over_gamma,
            ldg_c_over_gamma=c_over_gamma,
            flow_alignment=lam,
        )
    )
    q2 = q @ q
    tr_q2 = torch.diagonal(q2, dim1=-2, dim2=-1).sum(-1)
    nonlinear_bulk = (
        -b_over_gamma
        * (
            q2
            - torch.eye(3, dtype=q.dtype)
            * tr_q2[..., None, None]
            / 3.0
        )
        - c_over_gamma * tr_q2[..., None, None] * q
    )
    advection = sum(
        velocity[..., axis, None, None] * q_gradients[axis]
        for axis in range(3)
    )
    expected_rhs = nonlinear_bulk - advection + expected_alignment
    torch.testing.assert_close(
        observed_rhs,
        expected_rhs,
        rtol=1e-13,
        atol=1e-13,
    )


class _RecordingProjector:
    def __init__(self):
        self.boundary_conditions = None

    def project(self, spectral, boundary_conditions):
        self.boundary_conditions = tuple(boundary_conditions)
        return spectral - 0.125

    def forward_transform(self, tensor, boundary_conditions):
        return self.project(1.75 * tensor, boundary_conditions)


class _AdapterFields:
    def __init__(self, values, gradients):
        self.values = values
        self.gradients = gradients
        self.gradient_calls = []
        self.transformed_boundary_conditions = None

    def __getitem__(self, name):
        return self.values[name]

    def gradient(self, name, axis, projector=None):
        assert projector is not None
        self.gradient_calls.append((name, axis))
        return self.gradients[name, axis]

    def transform_tensor(self, tensor, boundary_conditions):
        self.transformed_boundary_conditions = tuple(boundary_conditions)
        return 1.75 * tensor


def test_q_nonlinear_model_adapter_wires_fields_transform_and_projection():
    generator = torch.Generator().manual_seed(67)
    sample_count = 5
    q = _random_traceless_symmetric(generator, count=sample_count)
    q_gradients = tuple(
        _random_traceless_symmetric(generator, count=sample_count)
        for _ in range(3)
    )
    velocity = torch.randn(
        sample_count,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    velocity_gradient = torch.randn(
        sample_count,
        3,
        3,
        dtype=torch.float64,
        generator=generator,
    )
    trace = torch.diagonal(
        velocity_gradient,
        dim1=-2,
        dim2=-1,
    ).sum(-1)
    velocity_gradient = (
        velocity_gradient
        - trace[..., None, None]
        * torch.eye(3, dtype=velocity_gradient.dtype)
        / 3.0
    )
    velocity_gradients = _axis_first_velocity_gradients(velocity_gradient)

    values = {
        name: component
        for name, component in zip(Q_COMPONENTS, _compact(q))
    }
    values.update(
        {
            name: velocity[..., index]
            for index, name in enumerate(("ux", "uy", "uz"))
        }
    )
    gradients = {}
    for axis in range(3):
        gradients.update(
            {
                (name, axis): component
                for name, component in zip(
                    Q_COMPONENTS,
                    _compact(q_gradients[axis]),
                )
            }
        )
        gradients.update(
            {
                (name, axis): velocity_gradients[axis][index]
                for index, name in enumerate(("ux", "uy", "uz"))
            }
        )

    q_boundary_conditions = ("periodic", "periodic", "neumann")
    projector = _RecordingProjector()
    fields = _AdapterFields(values, gradients)
    model = BerisEdwardsQNonlinearModel(
        projector,
        q_boundary_conditions,
        ldg_b=-0.3,
        ldg_c=0.3,
        rotational_viscosity=2.94,
        flow_alignment=0.31,
        pointwise_kernels=BerisEdwardsPointwiseKernels("eager"),
    )
    observed = model(fields, params={})
    expected_spatial = torch.stack(
        beris_edwards_q_nonlinear_components(
            _compact(q),
            tuple(velocity[..., axis] for axis in range(3)),
            tuple(_compact(gradient) for gradient in q_gradients),
            velocity_gradients,
            ldg_b_over_gamma=-0.3 / 2.94,
            ldg_c_over_gamma=0.3 / 2.94,
            flow_alignment=0.31,
        )
    )
    torch.testing.assert_close(
        observed,
        1.75 * expected_spatial - 0.125,
        rtol=1e-13,
        atol=1e-13,
    )
    assert observed.shape == (5, sample_count)
    assert observed.dtype == torch.float64
    assert fields.transformed_boundary_conditions is None
    assert projector.boundary_conditions == q_boundary_conditions
    assert fields.gradient_calls == [
        *(
            (name, axis)
            for axis in range(3)
            for name in Q_COMPONENTS
        ),
        *(
            (name, axis)
            for axis in range(3)
            for name in ("ux", "uy", "uz")
        ),
    ]


def test_linear_operator_and_algebraic_stress_composition():
    wavenumber_squared = torch.linspace(0.0, 3.0, 11, dtype=torch.float64)
    observed_linear = beris_edwards_linear_operator(
        wavenumber_squared,
        ldg_a=-0.02,
        ldg_l1=0.017,
        rotational_viscosity=2.94,
    )
    expected_linear = -(-0.02 + 0.017 * wavenumber_squared) / 2.94
    torch.testing.assert_close(
        observed_linear,
        expected_linear,
        rtol=0,
        atol=0,
    )

    generator = torch.Generator().manual_seed(61)
    q = _random_traceless_symmetric(generator, count=9)
    h = _random_traceless_symmetric(generator, count=9)
    lam = 0.27
    active_prefactor = -0.013
    active = torch.stack(
        beris_edwards_active_stress_components(
            _compact(q),
            active_prefactor=active_prefactor,
        ),
        dim=-1,
    ).reshape(q.shape[0], 3, 3)
    algebraic = torch.stack(
        beris_edwards_algebraic_stress_components(
            _compact(q),
            _compact(h),
            flow_alignment=lam,
            active_prefactor=active_prefactor,
        ),
        dim=-1,
    ).reshape(q.shape[0], 3, 3)

    m = q + torch.eye(3, dtype=q.dtype) / 3.0
    q_dot_h = torch.einsum("...ij,...ij->...", q, h)
    reactive = (
        2.0 * lam * m * q_dot_h[..., None, None]
        - lam * (h @ m + m @ h)
        + q @ h
        - h @ q
    )
    torch.testing.assert_close(
        active,
        active_prefactor * q,
        rtol=1e-13,
        atol=1e-13,
    )
    torch.testing.assert_close(
        algebraic,
        reactive + active_prefactor * q,
        rtol=1e-13,
        atol=1e-13,
    )
