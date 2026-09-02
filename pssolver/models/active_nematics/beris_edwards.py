"""Reusable five-component Beris--Edwards dynamics and stress algebra.

The canonical compact ordering is (Qxx, Qxy, Qxz, Qyy, Qyz) with
Qzz = -Qxx - Qyy. Pointwise constitutive helpers are kept independent of
geometry. BerisEdwardsQNonlinearModel is the thin PSSolver adapter that
collects spatial derivatives and applies the caller's basis-aware projector.
"""

import math

import torch

from .fields import Q_COMPONENTS

STRESS_COMPONENTS = (
    "xx",
    "xy",
    "xz",
    "yx",
    "yy",
    "yz",
    "zx",
    "zy",
    "zz",
)


def _five_components(values, name):
    values = tuple(values)
    if len(values) != 5:
        raise ValueError(f"{name} must contain five compact Q components.")
    return values


def _three_components(values, name):
    values = tuple(values)
    if len(values) != 3:
        raise ValueError(f"{name} must contain three Cartesian components.")
    return values


def _three_q_gradients(values, name):
    values = tuple(values)
    if len(values) != 3:
        raise ValueError(f"{name} must contain x, y, and z gradients.")
    return tuple(
        _five_components(gradient, f"{name}[{axis}]")
        for axis, gradient in enumerate(values)
    )


def _three_velocity_gradients(values, name):
    values = tuple(values)
    if len(values) != 3:
        raise ValueError(f"{name} must contain x, y, and z gradients.")
    return tuple(
        _three_components(gradient, f"{name}[{axis}]")
        for axis, gradient in enumerate(values)
    )


def _q_square_components(q_components):
    qxx, qxy, qxz, qyy, qyz = _five_components(
        q_components,
        "q_components",
    )
    qzz = -qxx - qyy
    tr_q2 = (
        qxx.square()
        + qyy.square()
        + qzz.square()
        + 2.0 * (qxy.square() + qxz.square() + qyz.square())
    )
    return tr_q2, (
        qxx.square() + qxy.square() + qxz.square(),
        qxy * (qxx + qyy) + qxz * qyz,
        qxy * qyz - qxz * qyy,
        qxy.square() + qyy.square() + qyz.square(),
        qxy * qxz - qyz * qxx,
    )


def q_tensor_contraction(left, right):
    """Return the full symmetric-tensor contraction left:right."""
    lxx, lxy, lxz, lyy, lyz = _five_components(left, "left")
    rxx, rxy, rxz, ryy, ryz = _five_components(right, "right")
    lzz = -lxx - lyy
    rzz = -rxx - ryy
    return (
        lxx * rxx
        + lyy * ryy
        + lzz * rzz
        + 2.0 * (lxy * rxy + lxz * rxz + lyz * ryz)
    )


def beris_edwards_free_energy_density(
    q_components,
    q_gradients,
    *,
    ldg_a,
    ldg_b,
    ldg_c,
    ldg_l1,
):
    """Return the one-constant Landau--de Gennes free-energy density.

    The convention is

    ``A/2 tr(Q^2) + B/3 tr(Q^3) + C/4 tr(Q^2)^2``
    ``+ L1/2 partial_k Q:partial_k Q``.

    The five compact components are expanded with their full symmetric,
    traceless contraction weights.  Keeping this helper geometry-independent
    lets equation-level tests compare its variational derivative with the
    production molecular field on any supported transform grid.
    """
    coefficients = (ldg_a, ldg_b, ldg_c, ldg_l1)
    if not all(math.isfinite(float(value)) for value in coefficients):
        raise ValueError("Landau--de Gennes coefficients must be finite.")
    if ldg_l1 < 0:
        raise ValueError("ldg_l1 must be non-negative.")

    qxx, qxy, qxz, qyy, qyz = _five_components(
        q_components,
        "q_components",
    )
    gradients = _three_q_gradients(q_gradients, "q_gradients")
    qzz = -qxx - qyy
    tr_q2, q2 = _q_square_components(q_components)
    q2_xx, q2_xy, q2_xz, q2_yy, q2_yz = q2
    q2_zz = qxz.square() + qyz.square() + qzz.square()
    tr_q3 = (
        qxx * q2_xx
        + qyy * q2_yy
        + qzz * q2_zz
        + 2.0 * (qxy * q2_xy + qxz * q2_xz + qyz * q2_yz)
    )
    gradient_sq = sum(
        q_tensor_contraction(gradient, gradient)
        for gradient in gradients
    )
    return (
        0.5 * ldg_a * tr_q2
        + (ldg_b / 3.0) * tr_q3
        + 0.25 * ldg_c * tr_q2.square()
        + 0.5 * ldg_l1 * gradient_sq
    )


def beris_edwards_linear_operator(
    wavenumber_squared,
    *,
    ldg_a,
    ldg_l1,
    rotational_viscosity,
):
    """Return the IMEX linear symbol -(A + L1*k_squared)/gamma."""
    coefficients = (ldg_a, ldg_l1, rotational_viscosity)
    if not all(math.isfinite(float(value)) for value in coefficients):
        raise ValueError("Beris--Edwards linear coefficients must be finite.")
    if rotational_viscosity <= 0:
        raise ValueError("rotational_viscosity must be positive.")
    if ldg_l1 < 0:
        raise ValueError("ldg_l1 must be non-negative.")
    return -(
        ldg_a + ldg_l1 * wavenumber_squared
    ) / rotational_viscosity


def beris_edwards_flow_alignment_components(
    q_components,
    velocity_gradients,
    *,
    flow_alignment,
):
    """Return the five components of the full Beris--Edwards S(W,Q).

    velocity_gradients is axis-first: entry j contains
    (partial_j ux, partial_j uy, partial_j uz). Equivalently the velocity
    gradient matrix is W_ij = partial_j u_i. The returned expression is

    (lambda E + Omega) M + M (lambda E - Omega) - 2 lambda M (Q:E),

    with M=Q+I/3, E=(W+W.T)/2, and Omega=(W-W.T)/2.

    The compact five-component return assumes an incompressible velocity
    field, as enforced by the coupled Stokes solver.
    """
    if not math.isfinite(float(flow_alignment)):
        raise ValueError("flow_alignment must be finite.")
    qxx, qxy, qxz, qyy, qyz = _five_components(
        q_components,
        "q_components",
    )
    gradient_x, gradient_y, gradient_z = _three_velocity_gradients(
        velocity_gradients,
        "velocity_gradients",
    )
    dux_dx, duy_dx, duz_dx = gradient_x
    dux_dy, duy_dy, duz_dy = gradient_y
    dux_dz, duy_dz, duz_dz = gradient_z

    exx = dux_dx
    exy = 0.5 * (dux_dy + duy_dx)
    exz = 0.5 * (dux_dz + duz_dx)
    eyy = duy_dy
    eyz = 0.5 * (duy_dz + duz_dy)
    ezz = duz_dz
    omega_xy = 0.5 * (dux_dy - duy_dx)
    omega_xz = 0.5 * (dux_dz - duz_dx)
    omega_yz = 0.5 * (duy_dz - duz_dy)

    qzz = -qxx - qyy
    mxx = qxx + 1.0 / 3.0
    myy = qyy + 1.0 / 3.0
    mzz = qzz + 1.0 / 3.0
    q_dot_e = (
        qxx * exx
        + qyy * eyy
        + qzz * ezz
        + 2.0 * (qxy * exy + qxz * exz + qyz * eyz)
    )

    lam = flow_alignment
    axx = lam * exx
    axy = lam * exy + omega_xy
    axz = lam * exz + omega_xz
    ayx = lam * exy - omega_xy
    ayy = lam * eyy
    ayz = lam * eyz + omega_yz
    azx = lam * exz - omega_xz
    azy = lam * eyz - omega_yz
    azz = lam * ezz

    return (
        2.0 * (axx * mxx + axy * qxy + axz * qxz)
        - 2.0 * lam * mxx * q_dot_e,
        (
            axx * qxy
            + axy * myy
            + axz * qyz
            + mxx * ayx
            + qxy * ayy
            + qxz * ayz
            - 2.0 * lam * qxy * q_dot_e
        ),
        (
            axx * qxz
            + axy * qyz
            + axz * mzz
            + mxx * azx
            + qxy * azy
            + qxz * azz
            - 2.0 * lam * qxz * q_dot_e
        ),
        2.0 * (ayx * qxy + ayy * myy + ayz * qyz)
        - 2.0 * lam * myy * q_dot_e,
        (
            ayx * qxz
            + ayy * qyz
            + ayz * mzz
            + qxy * azx
            + myy * azy
            + qyz * azz
            - 2.0 * lam * qyz * q_dot_e
        ),
    )


def beris_edwards_q_nonlinear_components(
    q_components,
    velocity_components,
    q_gradients,
    velocity_gradients,
    *,
    ldg_b_over_gamma,
    ldg_c_over_gamma,
    flow_alignment,
):
    """Return the nonlinear IMEX part of the Beris--Edwards Q equation.

    The -A Q/gamma and L1 lap(Q)/gamma terms are deliberately absent; they
    belong in beris_edwards_linear_operator. This function contains the B/C
    bulk molecular field, material advection, alignment, and co-rotation.
    """
    coefficients = (
        ldg_b_over_gamma,
        ldg_c_over_gamma,
        flow_alignment,
    )
    if not all(math.isfinite(float(value)) for value in coefficients):
        raise ValueError("Beris--Edwards nonlinear coefficients must be finite.")
    q_components = _five_components(q_components, "q_components")
    velocity_components = _three_components(
        velocity_components,
        "velocity_components",
    )
    gradient_x, gradient_y, gradient_z = _three_q_gradients(
        q_gradients,
        "q_gradients",
    )
    alignment = beris_edwards_flow_alignment_components(
        q_components,
        velocity_gradients,
        flow_alignment=flow_alignment,
    )
    tr_q2, q2 = _q_square_components(q_components)
    qxx, qxy, qxz, qyy, qyz = q_components
    isotropic_q2 = tr_q2 / 3.0
    nonlinear_bulk = (
        -ldg_b_over_gamma * (q2[0] - isotropic_q2)
        - ldg_c_over_gamma * tr_q2 * qxx,
        -ldg_b_over_gamma * q2[1]
        - ldg_c_over_gamma * tr_q2 * qxy,
        -ldg_b_over_gamma * q2[2]
        - ldg_c_over_gamma * tr_q2 * qxz,
        -ldg_b_over_gamma * (q2[3] - isotropic_q2)
        - ldg_c_over_gamma * tr_q2 * qyy,
        -ldg_b_over_gamma * q2[4]
        - ldg_c_over_gamma * tr_q2 * qyz,
    )
    ux, uy, uz = velocity_components
    return tuple(
        nonlinear_bulk[index]
        - (
            ux * gradient_x[index]
            + uy * gradient_y[index]
            + uz * gradient_z[index]
        )
        + alignment[index]
        for index in range(5)
    )


def beris_edwards_molecular_field_components(
    q_components,
    laplacian_components,
    *,
    ldg_a,
    ldg_b,
    ldg_c,
    ldg_l1,
):
    """Return the raw one-constant molecular field H.

    This evaluates

    H = -A Q - B (Q^2 - I tr(Q^2)/3) - C tr(Q^2) Q + L1 lap(Q).

    The returned field is intentionally not divided by the rotational
    viscosity. The stress law requires this raw thermodynamic field.
    """
    qxx, qxy, qxz, qyy, qyz = _five_components(q_components, "q_components")
    lxx, lxy, lxz, lyy, lyz = _five_components(
        laplacian_components,
        "laplacian_components",
    )
    qzz = -qxx - qyy
    tr_q2 = (
        qxx.square()
        + qyy.square()
        + qzz.square()
        + 2.0 * (qxy.square() + qxz.square() + qyz.square())
    )

    q2_xx = qxx.square() + qxy.square() + qxz.square()
    q2_xy = qxy * (qxx + qyy) + qxz * qyz
    q2_xz = qxy * qyz - qxz * qyy
    q2_yy = qxy.square() + qyy.square() + qyz.square()
    q2_yz = qxy * qxz - qyz * qxx
    isotropic_q2 = tr_q2 / 3.0

    hxx = (
        -ldg_a * qxx
        - ldg_b * (q2_xx - isotropic_q2)
        - ldg_c * tr_q2 * qxx
        + ldg_l1 * lxx
    )
    hxy = (
        -ldg_a * qxy
        - ldg_b * q2_xy
        - ldg_c * tr_q2 * qxy
        + ldg_l1 * lxy
    )
    hxz = (
        -ldg_a * qxz
        - ldg_b * q2_xz
        - ldg_c * tr_q2 * qxz
        + ldg_l1 * lxz
    )
    hyy = (
        -ldg_a * qyy
        - ldg_b * (q2_yy - isotropic_q2)
        - ldg_c * tr_q2 * qyy
        + ldg_l1 * lyy
    )
    hyz = (
        -ldg_a * qyz
        - ldg_b * q2_yz
        - ldg_c * tr_q2 * qyz
        + ldg_l1 * lyz
    )
    return hxx, hxy, hxz, hyy, hyz


def beris_edwards_reactive_stress_components(
    q_components,
    h_components,
    *,
    flow_alignment,
    q_dot_h=None,
):
    """Return the nine row-major reactive-stress components.

    With M = Q + I/3 the implemented convention is

    2 lambda M (Q:H) - lambda (H M + M H) + Q H - H Q.

    q_dot_h may be supplied by a caller that already evaluated the exact
    contraction; otherwise it is formed from the compact components. To retain
    the Beris--Edwards energy pairing under spectral filtering, project the
    complete reactive stress rather than q_dot_h alone.
    """
    qxx, qxy, qxz, qyy, qyz = _five_components(q_components, "q_components")
    hxx, hxy, hxz, hyy, hyz = _five_components(h_components, "h_components")
    qzz = -qxx - qyy
    hzz = -hxx - hyy
    if q_dot_h is None:
        q_dot_h = q_tensor_contraction(q_components, h_components)

    mxx = qxx + 1.0 / 3.0
    myy = qyy + 1.0 / 3.0
    mzz = qzz + 1.0 / 3.0
    m_components = (
        mxx,
        qxy,
        qxz,
        qxy,
        myy,
        qyz,
        qxz,
        qyz,
        mzz,
    )

    # Components of M @ H in row-major order. Since both M and H are
    # symmetric, H @ M is the transpose of this product.
    mh = (
        mxx * hxx + qxy * hxy + qxz * hxz,
        mxx * hxy + qxy * hyy + qxz * hyz,
        mxx * hxz + qxy * hyz + qxz * hzz,
        qxy * hxx + myy * hxy + qyz * hxz,
        qxy * hxy + myy * hyy + qyz * hyz,
        qxy * hxz + myy * hyz + qyz * hzz,
        qxz * hxx + qyz * hxy + mzz * hxz,
        qxz * hxy + qyz * hyy + mzz * hyz,
        qxz * hxz + qyz * hyz + mzz * hzz,
    )
    transpose = (0, 3, 6, 1, 4, 7, 2, 5, 8)
    lam = flow_alignment
    return tuple(
        2.0 * lam * m_components[index] * q_dot_h
        + (1.0 - lam) * mh[index]
        - (1.0 + lam) * mh[transpose[index]]
        for index in range(9)
    )


def beris_edwards_active_stress_components(
    q_components,
    *,
    active_prefactor,
):
    """Return the nine row-major components of active_prefactor * Q."""
    qxx, qxy, qxz, qyy, qyz = _five_components(
        q_components,
        "q_components",
    )
    return tuple(
        active_prefactor * component
        for component in (
            qxx,
            qxy,
            qxz,
            qxy,
            qyy,
            qyz,
            qxz,
            qyz,
            -qxx - qyy,
        )
    )


def beris_edwards_algebraic_stress_components(
    q_components,
    h_components,
    *,
    flow_alignment,
    active_prefactor,
    q_dot_h=None,
):
    """Return reactive plus active stress, which share Q's wall parity."""
    reactive = beris_edwards_reactive_stress_components(
        q_components,
        h_components,
        flow_alignment=flow_alignment,
        q_dot_h=q_dot_h,
    )
    active = beris_edwards_active_stress_components(
        q_components,
        active_prefactor=active_prefactor,
    )
    return tuple(
        reactive[index] + active[index]
        for index in range(len(STRESS_COMPONENTS))
    )


def beris_edwards_distortion_stress_components(
    q_gradients,
    *,
    ldg_l1,
):
    """Return -L1 partial_i Q_kl partial_j Q_kl in row-major order."""
    gradients = tuple(q_gradients)
    if len(gradients) != 3:
        raise ValueError("q_gradients must contain the x, y, and z gradients.")
    gx, gy, gz = (
        _five_components(values, f"q_gradients[{axis}]")
        for axis, values in enumerate(gradients)
    )

    gxx = q_tensor_contraction(gx, gx)
    gxy = q_tensor_contraction(gx, gy)
    gxz = q_tensor_contraction(gx, gz)
    gyy = q_tensor_contraction(gy, gy)
    gyz = q_tensor_contraction(gy, gz)
    gzz = q_tensor_contraction(gz, gz)
    return tuple(
        -ldg_l1 * component
        for component in (
            gxx,
            gxy,
            gxz,
            gxy,
            gyy,
            gyz,
            gxz,
            gyz,
            gzz,
        )
    )


class BerisEdwardsQNonlinearModel(torch.nn.Module):
    """PSSolver adapter for the nonlinear part of the full Q equation.

    A and L1 are intentionally handled by beris_edwards_linear_operator in the
    semi-implicit field registration. This model evaluates B/C bulk relaxation,
    material advection, full flow alignment, and co-rotation, then transforms
    and projects the five compact Q components in their native basis.
    """

    def __init__(
        self,
        spectral_projector,
        q_boundary_conditions,
        *,
        ldg_b,
        ldg_c,
        rotational_viscosity,
        flow_alignment,
    ):
        super().__init__()
        coefficients = (
            ldg_b,
            ldg_c,
            rotational_viscosity,
            flow_alignment,
        )
        if not all(math.isfinite(float(value)) for value in coefficients):
            raise ValueError("Beris--Edwards Q coefficients must be finite.")
        if rotational_viscosity <= 0:
            raise ValueError("rotational_viscosity must be positive.")
        q_boundary_conditions = tuple(q_boundary_conditions)
        if len(q_boundary_conditions) != 3:
            raise ValueError(
                "q_boundary_conditions must contain three axis conditions."
            )
        self.spectral_projector = spectral_projector
        self.q_boundary_conditions = q_boundary_conditions
        self.ldg_b_over_gamma = float(ldg_b) / float(rotational_viscosity)
        self.ldg_c_over_gamma = float(ldg_c) / float(rotational_viscosity)
        self.flow_alignment = float(flow_alignment)

    def forward(self, fields, params):
        del params
        q_components = tuple(fields[name] for name in Q_COMPONENTS)
        velocity_components = tuple(
            fields[name]
            for name in ("ux", "uy", "uz")
        )
        q_gradients = tuple(
            tuple(
                fields.gradient(name, axis=axis)
                for name in Q_COMPONENTS
            )
            for axis in range(3)
        )
        velocity_gradients = tuple(
            tuple(
                fields.gradient(name, axis=axis)
                for name in ("ux", "uy", "uz")
            )
            for axis in range(3)
        )
        nonlinear_components = beris_edwards_q_nonlinear_components(
            q_components,
            velocity_components,
            q_gradients,
            velocity_gradients,
            ldg_b_over_gamma=self.ldg_b_over_gamma,
            ldg_c_over_gamma=self.ldg_c_over_gamma,
            flow_alignment=self.flow_alignment,
        )
        nonlinear_hat = fields.transform_tensor(
            torch.stack(nonlinear_components),
            self.q_boundary_conditions,
        )
        return self.spectral_projector.project(
            nonlinear_hat,
            self.q_boundary_conditions,
        )


__all__ = [
    "BerisEdwardsQNonlinearModel",
    "STRESS_COMPONENTS",
    "beris_edwards_active_stress_components",
    "beris_edwards_algebraic_stress_components",
    "beris_edwards_distortion_stress_components",
    "beris_edwards_flow_alignment_components",
    "beris_edwards_free_energy_density",
    "beris_edwards_linear_operator",
    "beris_edwards_molecular_field_components",
    "beris_edwards_q_nonlinear_components",
    "beris_edwards_reactive_stress_components",
    "q_tensor_contraction",
]
