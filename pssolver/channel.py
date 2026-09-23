"""Reusable 3D active-nematic channel model."""

from __future__ import annotations

import torch

from .control.active_force import active_force_divergence
from .linear_solvers.stokes.channel_no_slip import (
    ChannelNoSlipModalStokesSolver,
)
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


class ModalSaddleStokesCompute(ChannelNoSlipModalStokesSolver):
    """Legacy active-force facade over the canonical Channel Stokes solver."""

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
        super().__init__(
            solver.transform_backend,
            friction=friction,
            viscosity=viscosity,
            pressure_relative_tolerance=pressure_rel_tol,
            pressure_max_iterations=pressure_max_iter,
            pressure_fixed_iterations=pressure_fixed_iterations,
        )
        self.beta = float(beta)

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
