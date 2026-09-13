"""Reusable 3D active-nematic channel model."""

from __future__ import annotations

import torch

from .control.active_force import active_force_divergence
from .models.active_nematics.fields import Q_COMPONENTS
from .solver import SpectralSolver


Q_BC = ("periodic", "neumann", "neumann")
U_BC = ("periodic", "dirichlet", "dirichlet")
PRESSURE_MODAL_BC = ("periodic", "neumann", "neumann")


class ActiveNematicNonlinearModel(torch.nn.Module):
    def __init__(self, b_q=-6.0, c_q=6.0, flow_alignment=1.0):
        super().__init__()
        self.b_q = float(b_q)
        self.c_q = float(c_q)
        self.flow_alignment = float(flow_alignment)

    def forward(self, fields, params):
        del params
        qxx = fields["Qxx"]
        qxy = fields["Qxy"]
        qxz = fields["Qxz"]
        qyy = fields["Qyy"]
        qyz = fields["Qyz"]
        q_squared = (
            qxx.square() + 2 * qxy.square() + 2 * qxz.square()
            + (qxx + qyy).square() + qyy.square() + 2 * qyz.square()
        )

        ux = fields["ux"]
        uy = fields["uy"]
        uz = fields["uz"]
        gxux = fields.gradient("ux", axis=0)
        gyux = fields.gradient("ux", axis=1)
        gzux = fields.gradient("ux", axis=2)
        gxuy = fields.gradient("uy", axis=0)
        gyuy = fields.gradient("uy", axis=1)
        gzuy = fields.gradient("uy", axis=2)
        gxuz = fields.gradient("uz", axis=0)
        gyuz = fields.gradient("uz", axis=1)
        gzuz = fields.gradient("uz", axis=2)

        wxy = -0.5 * (gxuy - gyux)
        wxz = -0.5 * (gxuz - gzux)
        wyz = -0.5 * (gyuz - gzuy)
        axx = gxux
        axy = 0.5 * (gxuy + gyux)
        axz = 0.5 * (gxuz + gzux)
        ayy = gyuy
        ayz = 0.5 * (gyuz + gzuy)
        tr_qa = (
            qxx * axx + 2 * qxy * axy + 2 * qxz * axz + qyy * ayy
            + 2 * qyz * ayz + (qxx + qyy) * (axx + ayy)
        )

        gradients = {
            name: [fields.gradient(name, axis=axis) for axis in range(3)]
            for name in Q_COMPONENTS
        }
        gx_qxx, gy_qxx, gz_qxx = gradients["Qxx"]
        gx_qxy, gy_qxy, gz_qxy = gradients["Qxy"]
        gx_qxz, gy_qxz, gz_qxz = gradients["Qxz"]
        gx_qyy, gy_qyy, gz_qyy = gradients["Qyy"]
        gx_qyz, gy_qyz, gz_qyz = gradients["Qyz"]
        lam = self.flow_alignment
        b_q = self.b_q
        c_q = self.c_q

        out0 = (
            b_q * (-(qxx.square() + qxy.square() + qxz.square()
                     - 2 * qxx * qyy - 2 * qyy.square() - 2 * qyz.square()) / 3)
            - c_q * q_squared * qxx
            - ux * gx_qxx - uy * gy_qxx - uz * gz_qxx
            + lam * (2 / 3 * axx + 2 * (qxx * axx + qxy * axy + qxz * axz)
                     - 2 / 3 * tr_qa)
            + 2 * (wxy * qxy + wxz * qxz)
        )
        out1 = (
            -b_q * (qxx * qxy + qxy * qyy + qxz * qyz)
            - c_q * q_squared * qxy
            - ux * gx_qxy - uy * gy_qxy - uz * gz_qxy
            + lam * (2 / 3 * axy + qxx * axy + qxy * ayy + qxz * ayz
                     + axx * qxy + axy * qyy + axz * qyz)
            + wxy * (qyy - qxx) + wxz * qyz + wyz * qxz
        )
        out2 = (
            -b_q * (qxy * qyz - qxz * qyy)
            - c_q * q_squared * qxz
            - ux * gx_qxz - uy * gy_qxz - uz * gz_qxz
            + lam * (2 / 3 * axz + qxy * ayz - qxz * ayy + axy * qyz - axz * qyy)
            - wxz * (qyy + 2 * qxx) + wxy * qyz - wyz * qxy
        )
        out3 = (
            b_q * (-(-2 * qxx.square() + qxy.square() - 2 * qxz.square()
                     - 2 * qxx * qyy + qyy.square() + qyz.square()) / 3)
            - c_q * q_squared * qyy
            - ux * gx_qyy - uy * gy_qyy - uz * gz_qyy
            + lam * (2 / 3 * ayy + 2 * (qxy * axy + qyy * ayy + qyz * ayz)
                     - 2 / 3 * tr_qa)
            - 2 * (wxy * qxy - wyz * qyz)
        )
        out4 = (
            -b_q * (qxy * qxz - qyz * qxx)
            - c_q * q_squared * qyz
            - ux * gx_qyz - uy * gy_qyz - uz * gz_qyz
            + lam * (2 / 3 * ayz + qxy * axz - qyz * axx + axy * qxz - ayz * qxx)
            - wyz * (qxx + 2 * qyy) - wxz * qxy - wxy * qxz
        )
        return fields.transform_tensor(torch.stack((out0, out1, out2, out3, out4)), Q_BC)


class ModalSaddleStokesCompute(torch.nn.Module):
    """Mixed FFT/DST/DCT matrix-free Stokes/Brinkman solve."""

    def __init__(
        self,
        solver,
        beta=-1.0,
        friction=0.0,
        viscosity=1.0,
        pressure_rel_tol=1e-6,
        pressure_max_iter=80,
        pressure_fixed_iterations=None,
    ):
        super().__init__()
        self.beta = float(beta)
        backend = solver.transform_backend
        spectral_dtype = backend.spectral_dtype
        real_dtype = backend.real_dtype
        device = solver.qx.device
        velocity_metadata = backend.get_metadata(U_BC)
        pressure_metadata = backend.get_metadata(PRESSURE_MODAL_BC)
        qx = velocity_metadata.axis_modes[0]
        ky_d = velocity_metadata.axis_modes[1]
        kz_d = velocity_metadata.axis_modes[2]
        ky_n = pressure_metadata.axis_modes[1]
        kz_n = pressure_metadata.axis_modes[2]
        nx, ny, nz = qx.numel(), ky_d.numel(), kz_d.numel()

        wall_ops = {}
        for axis, size, k_dirichlet in ((1, ny, ky_d), (2, nz, kz_d)):
            dst = backend._get_matrix("dst", size).to(device=device, dtype=real_dtype)
            dct = backend._get_matrix("dct", size).to(device=device, dtype=real_dtype)
            d_d_to_n = torch.zeros((size, size), device=device, dtype=real_dtype)
            if size > 1:
                index = torch.arange(size - 1, device=device)
                d_d_to_n[index, index + 1] = k_dirichlet[:-1]
            wall_ops[axis] = {
                "n_to_d": (dct @ dst.transpose(0, 1)).to(spectral_dtype),
                "d_to_n": (dst @ dct.transpose(0, 1)).to(spectral_dtype),
                "dn_to_dd": (-d_d_to_n.transpose(0, 1)).to(spectral_dtype),
                "dd_to_dn": d_d_to_n.to(spectral_dtype),
            }

        a_diag = friction + viscosity * (
            qx.square().view(nx, 1, 1)
            + ky_d.square().view(1, ny, 1)
            + kz_d.square().view(1, 1, nz)
        )
        pressure_q2 = (
            qx.square().view(nx, 1, 1)
            + ky_n.square().view(1, ny, 1)
            + kz_n.square().view(1, 1, nz)
        )
        schur_diag = pressure_q2 / (
            friction + viscosity * pressure_q2.clamp_min(torch.finfo(real_dtype).eps)
        )
        null_mask = torch.zeros((1, nx, ny, nz), device=device, dtype=torch.bool)
        null_mask[:, 0, 0, 0] = True
        schur_safe = schur_diag.unsqueeze(0).clone()
        schur_safe[null_mask] = 1.0

        self.register_buffer("ikx", (1j * qx).view(1, nx, 1, 1).to(spectral_dtype))
        self.register_buffer("a_inv", (1.0 / a_diag).unsqueeze(0))
        for axis_name, axis in (("y", 1), ("z", 2)):
            for operator_name, operator in wall_ops[axis].items():
                self.register_buffer(f"{operator_name}_{axis_name}", operator)
        self.register_buffer("schur_diag_safe", schur_safe)
        self.register_buffer("pressure_null_mask", null_mask)
        self.pressure_rel_tol = float(pressure_rel_tol)
        self.pressure_max_iter = int(pressure_max_iter)
        self.pressure_fixed_iterations = (
            None
            if pressure_fixed_iterations is None
            else int(pressure_fixed_iterations)
        )
        if (
            self.pressure_fixed_iterations is not None
            and self.pressure_fixed_iterations <= 0
        ):
            raise ValueError("pressure_fixed_iterations must be positive.")
        self.pressure_guess = None
        self.last_pressure_iterations = 0
        self.last_pressure_residual = 0.0
        self.last_pressure_relative_residual = 0.0

    def _apply_axis_matrix(self, tensor, matrix, axis):
        spectral_axis = tensor.ndim - 3 + axis
        moved = tensor.movedim(spectral_axis, -1)
        return torch.matmul(moved, matrix).movedim(-1, spectral_axis)

    def _project_pressure_gauge(self, pressure_hat):
        return pressure_hat.masked_fill(self.pressure_null_mask, 0)

    def _helmholtz_inverse(self, rhs_hat):
        return rhs_hat * self.a_inv

    def _pressure_to_velocity(self, pressure_hat):
        out = self._apply_axis_matrix(pressure_hat, self.n_to_d_y, axis=1)
        return self._apply_axis_matrix(out, self.n_to_d_z, axis=2)

    def _velocity_to_pressure(self, velocity_hat):
        out = self._apply_axis_matrix(velocity_hat, self.d_to_n_y, axis=1)
        return self._apply_axis_matrix(out, self.d_to_n_z, axis=2)

    def _pressure_gradient(self, pressure_hat, axis):
        if axis == 0:
            return self.ikx * self._pressure_to_velocity(pressure_hat)
        if axis == 1:
            out = self._apply_axis_matrix(pressure_hat, self.dn_to_dd_y, axis=1)
            return self._apply_axis_matrix(out, self.n_to_d_z, axis=2)
        if axis == 2:
            out = self._apply_axis_matrix(pressure_hat, self.n_to_d_y, axis=1)
            return self._apply_axis_matrix(out, self.dn_to_dd_z, axis=2)
        raise IndexError(f"Unsupported pressure-gradient axis {axis}.")

    def _velocity_divergence_component(self, velocity_hat, axis):
        if axis == 0:
            return self._velocity_to_pressure(self.ikx * velocity_hat)
        if axis == 1:
            out = self._apply_axis_matrix(velocity_hat, self.dd_to_dn_y, axis=1)
            return self._apply_axis_matrix(out, self.d_to_n_z, axis=2)
        if axis == 2:
            out = self._apply_axis_matrix(velocity_hat, self.d_to_n_y, axis=1)
            return self._apply_axis_matrix(out, self.dd_to_dn_z, axis=2)
        raise IndexError(f"Unsupported velocity-divergence axis {axis}.")

    def pressure_gradient_hats(self, pressure_hat):
        """Return pressure-gradient coefficients in the three velocity bases."""

        return tuple(
            self._pressure_gradient(pressure_hat, axis)
            for axis in range(3)
        )

    def divergence_hat(self, ux_hat, uy_hat, uz_hat):
        """Return velocity divergence in the native pressure basis."""

        return sum(
            self._velocity_divergence_component(velocity_hat, axis)
            for axis, velocity_hat in enumerate((ux_hat, uy_hat, uz_hat))
        )

    def _pressure_operator(self, pressure_hat):
        pressure_hat = self._project_pressure_gauge(pressure_hat)
        velocity_hats = [
            self._helmholtz_inverse(self._pressure_gradient(pressure_hat, axis))
            for axis in range(3)
        ]
        divergence_hat = self.divergence_hat(*velocity_hats)
        return self._project_pressure_gauge(-divergence_hat)

    def _solve_pressure(self, rhs_hat):
        rhs_hat = self._project_pressure_gauge(rhs_hat)
        rhs_norm = torch.linalg.vector_norm(rhs_hat.reshape(-1)).item()
        if rhs_norm == 0.0:
            self.last_pressure_iterations = 0
            self.last_pressure_residual = 0.0
            self.last_pressure_relative_residual = 0.0
            return torch.zeros_like(rhs_hat)
        if self.pressure_guess is None or self.pressure_guess.shape != rhs_hat.shape:
            pressure_hat = torch.zeros_like(rhs_hat)
        else:
            pressure_hat = self._project_pressure_gauge(
                self.pressure_guess.to(device=rhs_hat.device, dtype=rhs_hat.dtype)
            )
        residual = rhs_hat - self._pressure_operator(pressure_hat)
        preconditioned = residual / self.schur_diag_safe
        direction = preconditioned.clone()
        rz_old = torch.sum(torch.conj(residual) * preconditioned).real
        tolerance = self.pressure_rel_tol * rhs_norm
        residual_norm = torch.linalg.vector_norm(residual.reshape(-1)).item()
        iterations = 0
        iteration_limit = (
            self.pressure_fixed_iterations
            if self.pressure_fixed_iterations is not None
            else self.pressure_max_iter
        )
        while iterations < iteration_limit and (
            self.pressure_fixed_iterations is not None or residual_norm > tolerance
        ):
            operator_direction = self._pressure_operator(direction)
            denominator = torch.sum(torch.conj(direction) * operator_direction).real
            if denominator.abs().item() < 1e-30:
                break
            step = rz_old / denominator
            pressure_hat = self._project_pressure_gauge(pressure_hat + step * direction)
            residual = self._project_pressure_gauge(residual - step * operator_direction)
            residual_norm = torch.linalg.vector_norm(residual.reshape(-1)).item()
            iterations += 1
            if residual_norm <= tolerance:
                break
            preconditioned = residual / self.schur_diag_safe
            rz_new = torch.sum(torch.conj(residual) * preconditioned).real
            if rz_old.abs().item() < 1e-30:
                break
            direction = preconditioned + (rz_new / rz_old) * direction
            rz_old = rz_new
        self.pressure_guess = pressure_hat.detach()
        self.last_pressure_iterations = iterations
        self.last_pressure_residual = residual_norm
        self.last_pressure_relative_residual = residual_norm / rhs_norm
        return pressure_hat

    def solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        """Solve the channel saddle system for native force spectra."""

        force_hats = (fx_hat, fy_hat, fz_hat)
        free_velocity = [
            self._helmholtz_inverse(force_hat)
            for force_hat in force_hats
        ]
        provisional_divergence = self.divergence_hat(*free_velocity)
        pressure_hat = self._solve_pressure(
            self._project_pressure_gauge(-provisional_divergence)
        )
        velocity_hat = [
            free_velocity[axis]
            - self._helmholtz_inverse(
                self._pressure_gradient(pressure_hat, axis)
            )
            for axis in range(3)
        ]
        return (*velocity_hat, pressure_hat)

    def forward(self, fields, params):
        force = active_force_divergence(fields, params["alpha"], self.beta)
        force_hat = fields.transform_tensor(force, U_BC)
        return torch.stack(self.solve_force_hats(*force_hat))


def build_active_nematic_channel(
    shape,
    lengths,
    dt,
    initial_q,
    *,
    device="cuda",
    batchsize=1,
    rho=6.0,
    elastic_constant=1.0,
    beta=-1.0,
    friction=0.0,
    viscosity=1.0,
    pressure_rel_tol=1e-6,
    pressure_max_iter=80,
    pressure_fixed_iterations=None,
):
    """Build the channel solver without starting a simulation."""

    solver = SpectralSolver(
        shape=tuple(shape),
        L=tuple(lengths),
        dt=dt,
        device=device,
        batchsize=batchsize,
    )
    missing = set(Q_COMPONENTS) - set(initial_q)
    if missing:
        raise ValueError(f"initial_q is missing components {sorted(missing)}.")
    a_q = 1.0 - rho / 3.0
    b_q = -rho
    c_q = rho
    q2 = solver.get_q2(Q_BC)
    linear_operator = -(a_q + elastic_constant * q2)
    for name in Q_COMPONENTS:
        solver.model.add_dynamic_field(
            name,
            initial_q[name],
            L_hat=linear_operator,
            boundary_conditions=Q_BC,
        )
    solver.model.add_static_field("ux", boundary_conditions=U_BC)
    solver.model.add_static_field("uy", boundary_conditions=U_BC)
    solver.model.add_static_field("uz", boundary_conditions=U_BC)
    solver.model.add_static_field("p", boundary_conditions=PRESSURE_MODAL_BC)
    solver.model.set_nonlinear_model(ActiveNematicNonlinearModel(b_q=b_q, c_q=c_q))
    solver.model.set_static_compute_model(
        ModalSaddleStokesCompute(
            solver,
            beta=beta,
            friction=friction,
            viscosity=viscosity,
            pressure_rel_tol=pressure_rel_tol,
            pressure_max_iter=pressure_max_iter,
            pressure_fixed_iterations=pressure_fixed_iterations,
        )
    )
    alpha = torch.zeros((batchsize, *shape), device=device)
    solver.model.parameters.new_param("alpha", alpha)
    solver.build()
    return solver
