import csv
import os
import time

import nematics3d as n3d
import numpy as np
import torch
from pssolver import SpectralSolver, create_q_initial_condition
from tqdm import trange

Q_BC = ("periodic", "neumann", "neumann")
U_BC = ("periodic", "dirichlet", "dirichlet")
# Pressure is used as a modal Lagrange multiplier for incompressibility, not as
# an independently prescribed wall boundary condition. The DCT space supplies the
# pressure gauge/null mode and pairs with div(u) for the Schur complement.
PRESSURE_MODAL_BC = ("periodic", "neumann", "neumann")
ENABLE_DIAGNOSTICS = True
DIAGNOSTIC_INTERVAL = 10
SAVE_INTERVAL = 10
CONTROL_INTERVAL = 10
DEFECT_DETECTION_THRESHOLD = 0.0
BOX_DEFECT_OFF_THRESHOLD = 60
OFF_CONFIRMATIONS = 2
ZERO_CONFIRMATIONS = 3
MIN_OFF_CHECKS = 3
ALPHA_ON = 5.0
ALPHA_OFF = 0.0
ALPHA_BOX_CENTER = (128.0, 20.0, 20.0)
ALPHA_BOX_GRID_POINTS = (20, 32, 32)
ALPHA_BOX_TRANSITION_WIDTH = 3.0
OBSERVATION_BOX_MARGIN = 4
CHANNEL_PERIODIC_BOUNDARY = (True, False, False)
DEFECT_DETECTION_PLANES = (True, True, True)


def divergence_stats(fields):
    div_u = (
        fields.gradient('ux', axis=0)
        + fields.gradient('uy', axis=1)
        + fields.gradient('uz', axis=2)
    )
    div_abs = div_u.abs()
    div_max = div_abs.max().item()
    div_rms = torch.sqrt(torch.mean(div_abs.square())).item()

    gxux = fields.gradient('ux', axis=0)
    gyux = fields.gradient('ux', axis=1)
    gzux = fields.gradient('ux', axis=2)
    gxuy = fields.gradient('uy', axis=0)
    gyuy = fields.gradient('uy', axis=1)
    gzuy = fields.gradient('uy', axis=2)
    gxuz = fields.gradient('uz', axis=0)
    gyuz = fields.gradient('uz', axis=1)
    gzuz = fields.gradient('uz', axis=2)
    grad_u_sq = (
        gxux.abs().square() + gyux.abs().square() + gzux.abs().square()
        + gxuy.abs().square() + gyuy.abs().square() + gzuy.abs().square()
        + gxuz.abs().square() + gyuz.abs().square() + gzuz.abs().square()
    )
    grad_u_rms = torch.sqrt(torch.mean(grad_u_sq)).item()
    div_rel = div_rms / max(grad_u_rms, 1e-30)
    return div_max, div_rms, div_rel


def active_force(fields, alpha_field):
    """Return div(beta * alpha * Q) for a spatially varying activity field."""
    Qxx = fields['Qxx']
    Qxy = fields['Qxy']
    Qxz = fields['Qxz']
    Qyy = fields['Qyy']
    Qyz = fields['Qyz']
    Qzz = -Qxx - Qyy

    fx = beta * (
        fields.gradient('Qxx', axis=0, tensor=alpha_field * Qxx)
        + fields.gradient('Qxy', axis=1, tensor=alpha_field * Qxy)
        + fields.gradient('Qxz', axis=2, tensor=alpha_field * Qxz)
    )
    fy = beta * (
        fields.gradient('Qxy', axis=0, tensor=alpha_field * Qxy)
        + fields.gradient('Qyy', axis=1, tensor=alpha_field * Qyy)
        + fields.gradient('Qyz', axis=2, tensor=alpha_field * Qyz)
    )
    fz = beta * (
        fields.gradient('Qxz', axis=0, tensor=alpha_field * Qxz)
        + fields.gradient('Qyz', axis=1, tensor=alpha_field * Qyz)
        + fields.gradient('Qxx', axis=2, tensor=alpha_field * Qzz)
    )
    return torch.stack([fx, fy, fz])


def wall_normal_momentum_stats(fields, params):
    """Check normal momentum balance at the y and z walls for the modal pressure."""
    _, fy, fz = active_force(fields, params['alpha'])

    lap_uy = fields.laplacian('uy')
    lap_uz = fields.laplacian('uz')
    dyp = fields.gradient('p', axis=1)
    dzp = fields.gradient('p', axis=2)
    residual_y = dyp - (fy + eta * lap_uy - fric * fields['uy'])
    residual_z = dzp - (fz + eta * lap_uz - fric * fields['uz'])

    y_wall_residual = torch.stack([residual_y[:, :, 0, :], residual_y[:, :, -1, :]], dim=-1)
    z_wall_residual = torch.stack([residual_z[..., 0], residual_z[..., -1]], dim=-1)
    wall_residual = torch.cat([y_wall_residual.reshape(-1), z_wall_residual.reshape(-1)])
    wall_abs = wall_residual.abs()
    return wall_abs.max().item(), torch.sqrt(torch.mean(wall_abs.square())).item()

class NonlinearModel(torch.nn.Module):
    def __init__(self, solver):
        super().__init__()

    def forward(self, fields, params):
        Qxx = fields['Qxx']
        Qxy = fields['Qxy']
        Qxz = fields['Qxz']
        Qyy = fields['Qyy']
        Qyz = fields['Qyz']

        Qsq = Qxx ** 2 + 2 * Qxy ** 2 + 2 * Qxz ** 2 + (Qxx + Qyy) ** 2 + Qyy ** 2 + 2 * Qyz ** 2

        ux = fields['ux']
        uy = fields['uy']
        uz = fields['uz']

        gxux = fields.gradient('ux', axis=0)
        gyux = fields.gradient('ux', axis=1)
        gzux = fields.gradient('ux', axis=2)
        gxuy = fields.gradient('uy', axis=0)
        gyuy = fields.gradient('uy', axis=1)
        gzuy = fields.gradient('uy', axis=2)
        gxuz = fields.gradient('uz', axis=0)
        gyuz = fields.gradient('uz', axis=1)

        wxy = -0.5 * (gxuy - gyux)
        wxz = -0.5 * (gxuz - gzux)
        wyz = -0.5 * (gyuz - gzuy)

        Axx = gxux
        Axy = 0.5 * (gxuy + gyux)
        Axz = 0.5 * (gxuz + gzux)
        Ayy = gyuy
        Ayz = 0.5 * (gyuz + gzuy)

        trQA = Qxx*Axx + 2*Qxy*Axy + 2*Qxz*Axz + Qyy*Ayy + 2*Qyz*Ayz + (Qxx + Qyy)*(Axx + Ayy)

        gxQxx = fields.gradient('Qxx', axis=0)
        gyQxx = fields.gradient('Qxx', axis=1)
        gzQxx = fields.gradient('Qxx', axis=2)
        gxQxy = fields.gradient('Qxy', axis=0)
        gyQxy = fields.gradient('Qxy', axis=1)
        gzQxy = fields.gradient('Qxy', axis=2)
        gxQxz = fields.gradient('Qxz', axis=0)
        gyQxz = fields.gradient('Qxz', axis=1)
        gzQxz = fields.gradient('Qxz', axis=2)
        gxQyy = fields.gradient('Qyy', axis=0)
        gyQyy = fields.gradient('Qyy', axis=1)
        gzQyy = fields.gradient('Qyy', axis=2)
        gxQyz = fields.gradient('Qyz', axis=0)
        gyQyz = fields.gradient('Qyz', axis=1)
        gzQyz = fields.gradient('Qyz', axis=2)

        lam = 1.0
        out0 = (
            + bQ*(- (Qxx**2 + Qxy**2 + Qxz**2 - 2*Qxx*Qyy - 2*Qyy**2 - 2*Qyz**2)/3)
            - cQ*Qsq*Qxx
            - ux*gxQxx - uy*gyQxx - uz*gzQxx
            + lam*(2/3*Axx + 2*(Qxx*Axx + Qxy*Axy + Qxz*Axz) - 2/3*trQA)
            + 2*(wxy*Qxy + wxz*Qxz)
        )
        out1 = (
            - bQ*(Qxx*Qxy + Qxy*Qyy + Qxz*Qyz)
            - cQ*Qsq*Qxy
            - ux*gxQxy - uy*gyQxy - uz*gzQxy
            + lam*(2/3*Axy + (Qxx*Axy + Qxy*Ayy + Qxz*Ayz) + (Axx*Qxy + Axy*Qyy + Axz*Qyz))
            + wxy*(Qyy - Qxx) + wxz*Qyz + wyz*Qxz
        )
        out2 = (
            - bQ*(Qxy*Qyz - Qxz*Qyy)
            - cQ*Qsq*Qxz
            - ux*gxQxz - uy*gyQxz - uz*gzQxz
            + lam*(2/3*Axz + (Qxy*Ayz - Qxz*Ayy + Axy*Qyz - Axz*Qyy))
            - wxz*(Qyy + 2*Qxx) + wxy*Qyz - wyz*Qxy
        )
        out3 = (
            + bQ*(-( -2*Qxx**2 + Qxy**2 - 2*Qxz**2 - 2*Qxx*Qyy + Qyy**2 + Qyz**2 )/3)
            - cQ*Qsq*Qyy
            - ux*gxQyy - uy*gyQyy - uz*gzQyy
            + lam*(2/3*Ayy + 2*(Qxy*Axy + Qyy*Ayy + Qyz*Ayz) - 2/3*trQA)
            - 2*(wxy*Qxy - wyz*Qyz)
        )
        out4 = (
            - bQ*(Qxy*Qxz - Qyz*Qxx)
            - cQ*Qsq*Qyz
            - ux*gxQyz - uy*gyQyz - uz*gzQyz
            + lam*(2/3*Ayz + (Qxy*Axz - Qyz*Axx + Axy*Qxz - Ayz*Qxx))
            - wyz*(Qxx + 2*Qyy) - wxz*Qxy - wxy*Qxz
        )

        return fields.transform_tensor(torch.stack([out0, out1, out2, out3, out4]), Q_BC)

class ModalSaddleStokesCompute(torch.nn.Module):
    """
    Matrix-free Schur complement for the modal saddle-point Stokes/Brinkman solve:

        A_D u + G p = f
        D u       = 0

    u is represented in the Dirichlet/DST velocity space in y and z. p is
    represented in a DCT multiplier space on the same wall-normal axes so that
    D u and pressure test functions live in the same modal space. This should be
    interpreted as the pressure space of the saddle-point discretization, not as
    a physical homogeneous Neumann pressure wall condition.
    """

    def __init__(self, solver, pressure_rel_tol=1e-6, pressure_max_iter=80):
        super().__init__()
        backend = solver.transform_backend
        spectral_dtype = backend.spectral_dtype
        real_dtype = backend.real_dtype
        device = solver.qx.device

        velocity_metadata = backend.get_metadata(U_BC)
        pressure_metadata = backend.get_metadata(PRESSURE_MODAL_BC)

        qx = velocity_metadata.axis_modes[0]
        ky_dirichlet = velocity_metadata.axis_modes[1]
        kz_dirichlet = velocity_metadata.axis_modes[2]
        ky_neumann = pressure_metadata.axis_modes[1]
        kz_neumann = pressure_metadata.axis_modes[2]
        nx = qx.numel()
        ny = ky_dirichlet.numel()
        nz = kz_dirichlet.numel()

        wall_ops = {}
        for axis, n, k_dirichlet in (
            (1, ny, ky_dirichlet),
            (2, nz, kz_dirichlet),
        ):
            dirichlet_matrix = backend._get_matrix("dst", n).to(device=device, dtype=real_dtype)
            neumann_matrix = backend._get_matrix("dct", n).to(device=device, dtype=real_dtype)

            basis_neumann_to_dirichlet = neumann_matrix @ dirichlet_matrix.transpose(0, 1)
            basis_dirichlet_to_neumann = dirichlet_matrix @ neumann_matrix.transpose(0, 1)

            d_dirichlet_to_neumann = torch.zeros((n, n), device=device, dtype=real_dtype)
            if n > 1:
                idx = torch.arange(n - 1, device=device)
                d_dirichlet_to_neumann[idx, idx + 1] = k_dirichlet[:-1]
            d_neumann_to_dirichlet = -d_dirichlet_to_neumann.transpose(0, 1)

            wall_ops[axis] = {
                "basis_neumann_to_dirichlet": basis_neumann_to_dirichlet.to(dtype=spectral_dtype),
                "basis_dirichlet_to_neumann": basis_dirichlet_to_neumann.to(dtype=spectral_dtype),
                "d_neumann_to_dirichlet": d_neumann_to_dirichlet.to(dtype=spectral_dtype),
                "d_dirichlet_to_neumann": d_dirichlet_to_neumann.to(dtype=spectral_dtype),
            }

        a_diag = fric + eta * (
            qx.square().view(nx, 1, 1)
            + ky_dirichlet.square().view(1, ny, 1)
            + kz_dirichlet.square().view(1, 1, nz)
        )
        a_inv = 1.0 / a_diag

        pressure_q2 = (
            qx.square().view(nx, 1, 1)
            + ky_neumann.square().view(1, ny, 1)
            + kz_neumann.square().view(1, 1, nz)
        )
        schur_diag = pressure_q2 / (fric + eta * pressure_q2.clamp_min(torch.finfo(real_dtype).eps))

        pressure_null_mask = torch.zeros((1, nx, ny, nz), device=device, dtype=torch.bool)
        pressure_null_mask[:, 0, 0, 0] = True
        schur_diag_safe = schur_diag.unsqueeze(0).clone()
        schur_diag_safe[pressure_null_mask] = 1.0

        self.register_buffer("ikx", (1j * qx).view(1, nx, 1, 1).to(dtype=spectral_dtype))
        self.register_buffer("a_inv", a_inv.unsqueeze(0).to(dtype=real_dtype))
        self.register_buffer("basis_neumann_to_dirichlet_y", wall_ops[1]["basis_neumann_to_dirichlet"])
        self.register_buffer("basis_dirichlet_to_neumann_y", wall_ops[1]["basis_dirichlet_to_neumann"])
        self.register_buffer("d_neumann_to_dirichlet_y", wall_ops[1]["d_neumann_to_dirichlet"])
        self.register_buffer("d_dirichlet_to_neumann_y", wall_ops[1]["d_dirichlet_to_neumann"])
        self.register_buffer("basis_neumann_to_dirichlet_z", wall_ops[2]["basis_neumann_to_dirichlet"])
        self.register_buffer("basis_dirichlet_to_neumann_z", wall_ops[2]["basis_dirichlet_to_neumann"])
        self.register_buffer("d_neumann_to_dirichlet_z", wall_ops[2]["d_neumann_to_dirichlet"])
        self.register_buffer("d_dirichlet_to_neumann_z", wall_ops[2]["d_dirichlet_to_neumann"])
        self.register_buffer("schur_diag_safe", schur_diag_safe.to(dtype=real_dtype))
        self.register_buffer("pressure_null_mask", pressure_null_mask)

        self.pressure_rel_tol = pressure_rel_tol
        self.pressure_max_iter = pressure_max_iter
        self.pressure_guess = None
        self.last_pressure_hat = None
        self.last_pressure_iterations = 0
        self.last_pressure_residual = 0.0
        self.last_pressure_relative_residual = 0.0

    def _apply_axis_matrix(self, tensor, matrix, axis):
        spectral_axis = tensor.ndim - 3 + axis
        moved = tensor.movedim(spectral_axis, -1)
        transformed = torch.matmul(moved, matrix)
        return transformed.movedim(-1, spectral_axis)

    def _project_pressure_gauge(self, p_hat):
        return p_hat.masked_fill(self.pressure_null_mask, 0)

    def _dirichlet_helmholtz_inverse(self, rhs_hat):
        return rhs_hat * self.a_inv

    def _pressure_to_velocity(self, p_hat):
        spectral = self._apply_axis_matrix(p_hat, self.basis_neumann_to_dirichlet_y, axis=1)
        return self._apply_axis_matrix(spectral, self.basis_neumann_to_dirichlet_z, axis=2)

    def _velocity_to_pressure(self, spectral):
        out = self._apply_axis_matrix(spectral, self.basis_dirichlet_to_neumann_y, axis=1)
        return self._apply_axis_matrix(out, self.basis_dirichlet_to_neumann_z, axis=2)

    def _pressure_gradient(self, p_hat, axis):
        if axis == 0:
            return self.ikx * self._pressure_to_velocity(p_hat)
        if axis == 1:
            out = self._apply_axis_matrix(p_hat, self.d_neumann_to_dirichlet_y, axis=1)
            return self._apply_axis_matrix(out, self.basis_neumann_to_dirichlet_z, axis=2)
        if axis == 2:
            out = self._apply_axis_matrix(p_hat, self.basis_neumann_to_dirichlet_y, axis=1)
            return self._apply_axis_matrix(out, self.d_neumann_to_dirichlet_z, axis=2)
        raise IndexError(f"Unsupported pressure-gradient axis {axis}.")

    def _velocity_divergence_component(self, velocity_hat, axis):
        if axis == 0:
            return self._velocity_to_pressure(self.ikx * velocity_hat)
        if axis == 1:
            out = self._apply_axis_matrix(velocity_hat, self.d_dirichlet_to_neumann_y, axis=1)
            return self._apply_axis_matrix(out, self.basis_dirichlet_to_neumann_z, axis=2)
        if axis == 2:
            out = self._apply_axis_matrix(velocity_hat, self.basis_dirichlet_to_neumann_y, axis=1)
            return self._apply_axis_matrix(out, self.d_dirichlet_to_neumann_z, axis=2)
        raise IndexError(f"Unsupported velocity-divergence axis {axis}.")

    def _pressure_operator(self, p_hat):
        p_hat = self._project_pressure_gauge(p_hat)

        ux_hat = self._dirichlet_helmholtz_inverse(self._pressure_gradient(p_hat, axis=0))
        uy_hat = self._dirichlet_helmholtz_inverse(self._pressure_gradient(p_hat, axis=1))
        uz_hat = self._dirichlet_helmholtz_inverse(self._pressure_gradient(p_hat, axis=2))

        divergence_hat = (
            self._velocity_divergence_component(ux_hat, axis=0)
            + self._velocity_divergence_component(uy_hat, axis=1)
            + self._velocity_divergence_component(uz_hat, axis=2)
        )
        return self._project_pressure_gauge(-divergence_hat)

    def _pressure_preconditioner(self, rhs_hat):
        return rhs_hat / self.schur_diag_safe

    def _solve_pressure(self, rhs_hat):
        rhs_hat = self._project_pressure_gauge(rhs_hat)
        rhs_norm = torch.linalg.vector_norm(rhs_hat.reshape(-1)).item()
        if rhs_norm == 0.0:
            self.last_pressure_iterations = 0
            self.last_pressure_residual = 0.0
            self.last_pressure_relative_residual = 0.0
            return torch.zeros_like(rhs_hat)

        if self.pressure_guess is None or self.pressure_guess.shape != rhs_hat.shape:
            x_hat = torch.zeros_like(rhs_hat)
        else:
            x_hat = self.pressure_guess.to(device=rhs_hat.device, dtype=rhs_hat.dtype)
            x_hat = self._project_pressure_gauge(x_hat)

        residual = rhs_hat - self._pressure_operator(x_hat)
        z_vec = self._pressure_preconditioner(residual)
        search_dir = z_vec.clone()
        rz_old = torch.sum(torch.conj(residual) * z_vec).real

        tol = self.pressure_rel_tol * rhs_norm
        residual_norm = torch.linalg.vector_norm(residual.reshape(-1)).item()
        iterations = 0

        while iterations < self.pressure_max_iter and residual_norm > tol:
            operator_search = self._pressure_operator(search_dir)
            denom = torch.sum(torch.conj(search_dir) * operator_search).real
            if denom.abs().item() < 1e-30:
                break

            alpha = rz_old / denom
            x_hat = self._project_pressure_gauge(x_hat + alpha * search_dir)
            residual = self._project_pressure_gauge(residual - alpha * operator_search)
            residual_norm = torch.linalg.vector_norm(residual.reshape(-1)).item()
            iterations += 1
            if residual_norm <= tol:
                break

            z_vec = self._pressure_preconditioner(residual)
            rz_new = torch.sum(torch.conj(residual) * z_vec).real
            if rz_old.abs().item() < 1e-30:
                break
            beta_cg = rz_new / rz_old
            search_dir = z_vec + beta_cg * search_dir
            rz_old = rz_new

        self.pressure_guess = x_hat.detach()
        self.last_pressure_iterations = iterations
        self.last_pressure_residual = residual_norm
        self.last_pressure_relative_residual = residual_norm / rhs_norm
        return x_hat

    def forward(self, fields, params):
        force = active_force(fields, params['alpha'])
        force_hat = fields.transform_tensor(force, U_BC)
        fx_hat, fy_hat, fz_hat = force_hat[0], force_hat[1], force_hat[2]

        ux_hat_free = self._dirichlet_helmholtz_inverse(fx_hat)
        uy_hat_free = self._dirichlet_helmholtz_inverse(fy_hat)
        uz_hat_free = self._dirichlet_helmholtz_inverse(fz_hat)

        provisional_divergence = (
            self._velocity_divergence_component(ux_hat_free, axis=0)
            + self._velocity_divergence_component(uy_hat_free, axis=1)
            + self._velocity_divergence_component(uz_hat_free, axis=2)
        )
        pressure_rhs = self._project_pressure_gauge(-provisional_divergence)
        pressure_hat = self._solve_pressure(pressure_rhs)
        self.last_pressure_hat = pressure_hat.detach()

        ux_hat = ux_hat_free - self._dirichlet_helmholtz_inverse(self._pressure_gradient(pressure_hat, axis=0))
        uy_hat = uy_hat_free - self._dirichlet_helmholtz_inverse(self._pressure_gradient(pressure_hat, axis=1))
        uz_hat = uz_hat_free - self._dirichlet_helmholtz_inverse(self._pressure_gradient(pressure_hat, axis=2))

        return torch.stack([ux_hat, uy_hat, uz_hat, pressure_hat])

seed = 24
dt = 1e-3
steps = 10000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batchsize = 1

Nx, Ny, Nz = 256, 40, 40
Lx, Ly, Lz = 64.0, 10.0, 10.0

solver = SpectralSolver(shape=(Nx,Ny,Nz), L=(Lx,Ly,Lz), dt=dt, device=device, batchsize=batchsize)
q_initial_condition = create_q_initial_condition(
    "aligned_x_smooth_noise",
    shape=(Nx, Ny, Nz),
    boundary_conditions=Q_BC,
    seed=seed,
    noise_theta=0.01,
    noise_phi=0.01,
    sigma_x=1.0,
    sigma_y=1.0,
    sigma_z=1.0,
)
Qxx_0 = q_initial_condition["Qxx"]
Qxy_0 = q_initial_condition["Qxy"]
Qxz_0 = q_initial_condition["Qxz"]
Qyy_0 = q_initial_condition["Qyy"]
Qyz_0 = q_initial_condition["Qyz"]

# Nematic 参数
rho = 6
aQ = 1 - rho/3
bQ = -rho
cQ = rho
KQ = 1.0

# 流体/应力参数
beta = -1.0
fric = 0.0
eta  = 1.0

q2_Q = solver.get_q2(Q_BC)

# --- Add active fields ---
solver.model.add_dynamic_field(
    "Qxx",
    init = Qxx_0,
    L_hat = -( aQ + q2_Q * KQ),
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qxy",
    init =  Qxy_0,
    L_hat = -( aQ + q2_Q * KQ),
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qxz",
    init = Qxz_0,
    L_hat = -( aQ + q2_Q * KQ),
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qyy",
    init =  Qyy_0,
    L_hat = -( aQ + q2_Q * KQ),
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qyz",
    init =  Qyz_0,
    L_hat = -( aQ + q2_Q * KQ),
    boundary_conditions = Q_BC,
)

# --- Add static fields ---
# ModalSaddleStokesCompute solves the pure FFT/DST/DCT modal saddle-point
# discretization through its pressure Schur complement.
solver.model.add_static_field("ux", boundary_conditions=U_BC)
solver.model.add_static_field("uy", boundary_conditions=U_BC)
solver.model.add_static_field("uz", boundary_conditions=U_BC)
solver.model.add_static_field("p", boundary_conditions=PRESSURE_MODAL_BC)

solver.model.set_nonlinear_model(NonlinearModel(solver))
solver.model.set_static_compute_model(ModalSaddleStokesCompute(solver))


def validate_box_bounds(name, bounds, size):
    lower, upper = bounds
    if not (0 <= lower < upper <= size):
        raise ValueError(
            f"{name} must satisfy 0 <= lower < upper <= {size}, got {bounds}"
        )


def box_bounds_from_center_grid_points(center, grid_points, shape):
    if len(center) != 3 or len(grid_points) != 3:
        raise ValueError(
            "ALPHA_BOX_CENTER and ALPHA_BOX_GRID_POINTS must each contain x, y, z"
        )

    bounds = []
    for axis, (axis_center, axis_count, axis_size) in enumerate(
        zip(center, grid_points, shape)
    ):
        if not isinstance(axis_count, int) or isinstance(axis_count, bool):
            raise ValueError(
                f"ALPHA_BOX_GRID_POINTS[{axis}] must be an integer, got {axis_count}"
            )
        if axis_count <= 0:
            raise ValueError(
                f"ALPHA_BOX_GRID_POINTS[{axis}] must be positive, got {axis_count}"
            )
        lower_float = axis_center - 0.5 * axis_count
        upper_float = axis_center + 0.5 * axis_count
        lower = round(lower_float)
        upper = round(upper_float)
        if (
            abs(lower_float - lower) > 1e-9
            or abs(upper_float - upper) > 1e-9
        ):
            raise ValueError(
                "Each center +/- half the grid-point count must fall on an integer grid "
                f"boundary; axis {axis} gives ({lower_float}, {upper_float})"
            )
        axis_bounds = (int(lower), int(upper))
        validate_box_bounds(f"alpha box axis {axis}", axis_bounds, axis_size)
        bounds.append(axis_bounds)
    return tuple(bounds)


def smooth_box_axis(size, bounds, width, device):
    lower, upper = bounds
    coordinate = torch.arange(size, device=device, dtype=torch.float32)
    inside = (coordinate >= lower) & (coordinate < upper)
    left = ((coordinate - (lower - 1)) / width).clamp(0.0, 1.0)
    right = ((upper - coordinate) / width).clamp(0.0, 1.0)
    ramp = torch.minimum(left, right)
    smooth = ramp.square() * (3.0 - 2.0 * ramp)
    return torch.where(inside, smooth, torch.zeros_like(smooth))


def make_alpha_box_mask(shape, bounds, device):
    if ALPHA_BOX_TRANSITION_WIDTH <= 0:
        raise ValueError("ALPHA_BOX_TRANSITION_WIDTH must be positive")

    mask_x = smooth_box_axis(
        shape[0], bounds[0], ALPHA_BOX_TRANSITION_WIDTH, device
    ).view(1, shape[0], 1, 1)
    mask_y = smooth_box_axis(
        shape[1], bounds[1], ALPHA_BOX_TRANSITION_WIDTH, device
    ).view(1, 1, shape[1], 1)
    mask_z = smooth_box_axis(
        shape[2], bounds[2], ALPHA_BOX_TRANSITION_WIDTH, device
    ).view(1, 1, 1, shape[2])
    return mask_x * mask_y * mask_z


def expanded_observation_bounds(bounds, margin, size):
    return max(0, bounds[0] - margin), min(size, bounds[1] + margin)


if OBSERVATION_BOX_MARGIN < 0:
    raise ValueError("OBSERVATION_BOX_MARGIN must be non-negative")

alpha_box_bounds = box_bounds_from_center_grid_points(
    ALPHA_BOX_CENTER,
    ALPHA_BOX_GRID_POINTS,
    (Nx, Ny, Nz),
)
alpha_box_x, alpha_box_y, alpha_box_z = alpha_box_bounds
alpha_box_mask = make_alpha_box_mask(
    (Nx, Ny, Nz),
    alpha_box_bounds,
    device,
)
alpha = ALPHA_ON * alpha_box_mask.repeat(batchsize, 1, 1, 1)
observation_box_x = expanded_observation_bounds(
    alpha_box_x, OBSERVATION_BOX_MARGIN, Nx
)
observation_box_y = expanded_observation_bounds(
    alpha_box_y, OBSERVATION_BOX_MARGIN, Ny
)
observation_box_z = expanded_observation_bounds(
    alpha_box_z, OBSERVATION_BOX_MARGIN, Nz
)
solver.model.parameters.new_param('alpha', alpha)

solver.build()

diagnostic_history = []
output_dir = "data_control_box_Nx256_Lx64"
os.makedirs(output_dir, exist_ok=True)
np.save(
    os.path.join(output_dir, "alpha_box_mask.npy"),
    alpha_box_mask[0].detach().cpu().numpy(),
)
print(
    "[control] alpha_box "
    f"center={ALPHA_BOX_CENTER} grid_points={ALPHA_BOX_GRID_POINTS} "
    f"x={alpha_box_x} y={alpha_box_y} z={alpha_box_z} "
    f"observation_x={observation_box_x} "
    f"observation_y={observation_box_y} "
    f"observation_z={observation_box_z} "
    f"alpha_on={ALPHA_ON} transition_width={ALPHA_BOX_TRANSITION_WIDTH}",
    flush=True,
)
control_history_path = os.path.join(output_dir, "control_history.csv")
control_history_fields = (
    "step",
    "global_defect_points",
    "box_defect_points",
    "box_defect_density",
    "alpha_box_amplitude",
    "control_state",
    "above_threshold_count",
    "zero_count",
    "off_checks",
    "transition",
)
with open(control_history_path, "w", newline="") as control_history_file:
    csv.DictWriter(control_history_file, fieldnames=control_history_fields).writeheader()

control_state = {
    "active": True,
    "above_threshold_count": 0,
    "zero_count": 0,
    "off_checks": 0,
    "last_global_defect_points": 0,
    "last_box_defect_points": 0,
}
q_snapshot_cache = {
    "step": None,
    "value": None,
}
start = time.time()
pbar = trange(steps)


def q_snapshot_numpy(step):
    if q_snapshot_cache["step"] != step:
        snapshot = torch.stack([
            solver.model.fields[name].detach().cpu()
            for name in ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"]
        ])
        snapshot = snapshot.permute(1, 2, 3, 4, 0)
        q_snapshot_cache["step"] = step
        q_snapshot_cache["value"] = snapshot[0].numpy()
    return q_snapshot_cache["value"]


def defects_inside_observation_box(defects):
    if len(defects) == 0:
        return defects
    inside = (
        (defects[:, 0] >= observation_box_x[0])
        & (defects[:, 0] < observation_box_x[1])
        & (defects[:, 1] >= observation_box_y[0])
        & (defects[:, 1] < observation_box_y[1])
        & (defects[:, 2] >= observation_box_z[0])
        & (defects[:, 2] < observation_box_z[1])
    )
    return defects[inside]


def update_defect_control(step):
    q_snapshot = q_snapshot_numpy(step)
    _, director = n3d.Q_diagonalize(q_snapshot)
    defects = n3d.defect_detect(
        director,
        threshold=DEFECT_DETECTION_THRESHOLD,
        is_boundary_periodic=CHANNEL_PERIODIC_BOUNDARY,
        planes=DEFECT_DETECTION_PLANES,
    )
    box_defects = defects_inside_observation_box(defects)
    global_defect_points = len(defects)
    box_defect_points = len(box_defects)
    observation_volume = (
        (observation_box_x[1] - observation_box_x[0])
        * (observation_box_y[1] - observation_box_y[0])
        * (observation_box_z[1] - observation_box_z[0])
    )
    box_defect_density = box_defect_points / observation_volume
    control_state["last_global_defect_points"] = global_defect_points
    control_state["last_box_defect_points"] = box_defect_points
    transition = ""

    if control_state["active"]:
        control_state["zero_count"] = 0
        control_state["off_checks"] = 0
        if box_defect_points >= BOX_DEFECT_OFF_THRESHOLD:
            control_state["above_threshold_count"] += 1
        else:
            control_state["above_threshold_count"] = 0

        if control_state["above_threshold_count"] >= OFF_CONFIRMATIONS:
            alpha.zero_()
            control_state["active"] = False
            control_state["above_threshold_count"] = 0
            control_state["zero_count"] = 0
            control_state["off_checks"] = 0
            transition = "alpha_off"
    else:
        control_state["above_threshold_count"] = 0
        control_state["off_checks"] += 1
        if box_defect_points == 0:
            control_state["zero_count"] += 1
        else:
            control_state["zero_count"] = 0

        if (
            control_state["off_checks"] >= MIN_OFF_CHECKS
            and control_state["zero_count"] >= ZERO_CONFIRMATIONS
        ):
            alpha.copy_(ALPHA_ON * alpha_box_mask)
            control_state["active"] = True
            control_state["zero_count"] = 0
            control_state["off_checks"] = 0
            transition = "alpha_restored"

    row = {
        "step": step,
        "global_defect_points": global_defect_points,
        "box_defect_points": box_defect_points,
        "box_defect_density": box_defect_density,
        "alpha_box_amplitude": ALPHA_ON if control_state["active"] else ALPHA_OFF,
        "control_state": "on" if control_state["active"] else "off",
        "above_threshold_count": control_state["above_threshold_count"],
        "zero_count": control_state["zero_count"],
        "off_checks": control_state["off_checks"],
        "transition": transition,
    }
    with open(control_history_path, "a", newline="") as control_history_file:
        writer = csv.DictWriter(control_history_file, fieldnames=control_history_fields)
        writer.writerow(row)

    if transition:
        pbar.write(
            f"step={step}: {transition}, "
            f"box_defects={box_defect_points}, "
            f"global_defects={global_defect_points}, "
            f"alpha_box={ALPHA_ON if control_state['active'] else ALPHA_OFF:.1f}"
        )


def record_step_state(i):
    if ENABLE_DIAGNOSTICS and i % DIAGNOSTIC_INTERVAL == 0:
        div_max, div_rms, div_rel = divergence_stats(solver.model.fields)
        wall_mom_max, wall_mom_rms = wall_normal_momentum_stats(
            solver.model.fields,
            solver.model.parameters,
        )
        static_model = solver.model.static_model
        diagnostic_history.append((
            i,
            div_max,
            div_rms,
            div_rel,
            static_model.last_pressure_iterations,
            static_model.last_pressure_residual,
            static_model.last_pressure_relative_residual,
            wall_mom_max,
            wall_mom_rms,
        ))
        pbar.set_postfix(
            box_defects=control_state["last_box_defect_points"],
            global_defects=control_state["last_global_defect_points"],
            alpha_box=f"{ALPHA_ON if control_state['active'] else ALPHA_OFF:.1f}",
            div_max=f"{div_max:.2e}",
            div_rms=f"{div_rms:.2e}",
            div_rel=f"{div_rel:.2e}",
            schur_it=static_model.last_pressure_iterations,
            schur_rel=f"{static_model.last_pressure_relative_residual:.2e}",
            wall_n_rms=f"{wall_mom_rms:.2e}",
        )
    if i % SAVE_INTERVAL == 0:
        np.save(f"{output_dir}/Q_{i}.npy", q_snapshot_numpy(i))

        u_snapshot = torch.stack([
            solver.model.fields[name].detach().cpu()
            for name in ["ux", "uy", "uz"]
        ])  # shape -> (3, batch, N, N, N)
        u_snapshot = u_snapshot.permute(1, 2, 3, 4, 0)  # shape -> (batch, N, N, N, 3)
        np.save(f"{output_dir}/u_{i}.npy", u_snapshot[0].numpy())

        p_snapshot = solver.model.fields["p"].detach().cpu()
        np.save(f"{output_dir}/p_{i}.npy", p_snapshot[0].numpy())


for i in pbar:
    if i % CONTROL_INTERVAL == 0:
        update_defect_control(i)
    solver.run(1, pre_update_callback=lambda _solver, _step, i=i: record_step_state(i))


end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
if ENABLE_DIAGNOSTICS:
    solver.refresh_static_fields()
    final_div_max, final_div_rms, final_div_rel = divergence_stats(solver.model.fields)
    final_wall_mom_max, final_wall_mom_rms = wall_normal_momentum_stats(
        solver.model.fields,
        solver.model.parameters,
    )
    static_model = solver.model.static_model
    diagnostic_history.append((
        steps,
        final_div_max,
        final_div_rms,
        final_div_rel,
        static_model.last_pressure_iterations,
        static_model.last_pressure_residual,
        static_model.last_pressure_relative_residual,
        final_wall_mom_max,
        final_wall_mom_rms,
    ))
    diagnostic_array = np.array(
        diagnostic_history,
        dtype=[
            ("step", np.int64),
            ("div_max", np.float64),
            ("div_rms", np.float64),
            ("div_rel", np.float64),
            ("schur_iterations", np.float64),
            ("schur_abs_residual", np.float64),
            ("schur_rel_residual", np.float64),
            ("wall_normal_momentum_max", np.float64),
            ("wall_normal_momentum_rms", np.float64),
        ],
    )
    np.save(f"{output_dir}/diagnostics.npy", diagnostic_array)
    np.savetxt(
        f"{output_dir}/diagnostics.csv",
        diagnostic_array,
        delimiter=",",
        header="step,div_max,div_rms,div_rel,schur_iterations,schur_abs_residual,schur_rel_residual,wall_normal_momentum_max,wall_normal_momentum_rms",
        comments="",
    )
    print(
        "Final div(u) diagnostic: "
        f"max={final_div_max:.6e}, "
        f"rms={final_div_rms:.6e}, "
        f"relative={final_div_rel:.6e}"
    )
    print(
        "Final Schur saddle solve diagnostic: "
        f"iterations={static_model.last_pressure_iterations}, "
        f"abs_residual={static_model.last_pressure_residual:.6e}, "
        f"rel_residual={static_model.last_pressure_relative_residual:.6e}"
    )
    print(
        "Final wall normal momentum diagnostic: "
        f"max={final_wall_mom_max:.6e}, "
        f"rms={final_wall_mom_rms:.6e}"
    )
