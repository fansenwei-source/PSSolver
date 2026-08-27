"""Dealiased plane variant with free-slip velocity and free Q anchoring.

The tangential velocity and pressure multiplier use DCT modes in z; the normal
velocity uses DST modes. Tangential plug-flow modes can either be fixed by a
zero-mean pseudoinverse or retained using positive Brinkman friction.

Pointwise products are stabilized with a basis-aware tensor-product spectral
projection. The default two-thirds rule exactly protects compatible quadratic
convolutions, while cubic_half is the conservative no-padding choice for the
explicit cubic Q^2 Q term. See BasisAwareSpectralProjector below for the
FFT/DCT/DST cutoff convention and its mixed-basis limitations.
"""

import argparse
import time
import torch
from pssolver import (
    SpectralSolver,
    apply_snapshot_to_solver,
    load_snapshot,
    prepare_new_run_directory,
    representative_ordered_S,
    REPRESENTATIVE_ORDERED_S_DEFINITION,
    require_distinct_output_directory,
    write_run_metadata,
)
from pssolver.integrator import SemiImplicitEulerIntegrator
from tqdm import trange
from pssolver.models.active_nematics import Q_convention_metadata, create_initial_condition
import numpy as np
import os


Q_BC = ("periodic", "periodic", "neumann")
U_TANGENTIAL_BC = ("periodic", "periodic", "neumann")
U_NORMAL_BC = ("periodic", "periodic", "dirichlet")
# Pressure is used as a modal Lagrange multiplier for incompressibility, not as
# an independently prescribed wall boundary condition.  For free slip, div(u)
# lives in the DCT space: dx(ux) and dy(uy) retain the tangential DCT basis while
# dz(uz) maps the normal DST basis into DCT.  Pressure therefore uses the same
# DCT multiplier space.
PRESSURE_MODAL_BC = ("periodic", "periodic", "neumann")
DEFAULT_DEALIAS_RULE = "two_thirds"
DEALIAS_RULE_FRACTIONS = {
    "none": None,
    "two_thirds": 2.0 / 3.0,
    "cubic_half": 0.5,
}
ENABLE_DIAGNOSTICS = False
DIAGNOSTIC_INTERVAL = 10
SAVE_INTERVAL = 100
INITIALIZATION_MODE = "generated"  # "generated" or "snapshot"
SNAPSHOT_MODE = "resume"  # "resume" or "branch"
SNAPSHOT_DIRECTORY = "data02"
SNAPSHOT_STEP = 1000

# Tangential zero-mode handling:
# - "zero_mean": use fric=0 and the Stokes pseudoinverse, fixing
#   <ux>=<uy>=0.
# - "friction": retain the constant tangential modes and regularize them with
#   the strictly positive FRICTION_MODE_FRIC below. This Brinkman friction acts
#   on all three velocity components, not only ux and uy.
#
# Important physical caveat: free Q anchoring does not by itself guarantee
# zero volume-averaged active tangential force. For example,
#
#   <f_x> is proportional to
#       [<Q_xz>_{xy}]_{z=0}^{z=L_z}.
#
# Therefore, removing the tangential zero mode in "zero_mean" mode is an
# explicit modeling choice, not merely a pressure-gauge choice. To retain a
# possible mean flow, use positive friction, prescribe the mean flow rate, add
# an external mean pressure gradient, or impose mutually cancelling active
# tangential tractions at the two walls.
TANGENTIAL_ZERO_MODE_POLICY = "zero_mean"
FRICTION_MODE_FRIC = 0.1
if TANGENTIAL_ZERO_MODE_POLICY == "zero_mean":
    fric = 0.0
elif TANGENTIAL_ZERO_MODE_POLICY == "friction":
    if FRICTION_MODE_FRIC <= 0:
        raise ValueError("FRICTION_MODE_FRIC must be positive in friction mode.")
    fric = float(FRICTION_MODE_FRIC)
else:
    raise ValueError(
        "TANGENTIAL_ZERO_MODE_POLICY must be 'zero_mean' or 'friction'."
    )

# Nematic parameters
rho = 2.65
aQ = 1 - rho / 3
bQ = -rho
cQ = rho
KQ = 1.0
S_INITIAL = 2.0 / 3.0
S_BULK = None

# Fluid/active-stress parameters
beta = -1.0
eta = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the dealiased free-slip plane simulation."
    )
    parser.add_argument(
        "--zero-mode-policy",
        choices=("zero_mean", "friction"),
        default=TANGENTIAL_ZERO_MODE_POLICY,
    )
    parser.add_argument(
        "--friction-mode-fric",
        type=float,
        default=FRICTION_MODE_FRIC,
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--dealias-rule",
        choices=tuple(DEALIAS_RULE_FRACTIONS),
        default=DEFAULT_DEALIAS_RULE,
        help=(
            "Spectral truncation rule: two_thirds is the standard quadratic "
            "rule; cubic_half is conservative for Q^2 Q; none disables it."
        ),
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.dt <= 0:
        parser.error("--dt must be positive")
    if args.save_interval <= 0:
        parser.error("--save-interval must be positive")
    if args.zero_mode_policy == "friction" and args.friction_mode_fric <= 0:
        parser.error("--friction-mode-fric must be positive in friction mode")
    return args


class BasisAwareSpectralProjector:
    """Sharp tensor-product projector for the native FFT/DCT/DST bases.

    Each axis is compared with the grid Nyquist wavenumber pi*N/L and uses a
    strict cutoff. In integer mode numbers this is

      periodic FFT:  abs(k) < fraction*N/2,
      Neumann DCT:   m      < fraction*N,      m = 0, ..., N-1,
      Dirichlet DST: r      < fraction*N,      r = 1, ..., N.

    The strict inequality removes the cutoff mode itself: retaining it would
    allow a product of two cutoff modes to alias back onto that same retained
    mode. DCT and DST therefore share the same physical cutoff even though a
    DST array slot j represents physical mode r=j+1.

    This is an exact two-thirds dealiasing projector for compatible quadratic
    basis products. The model also contains the cubic term Q^2 Q and mixed
    DCT/DST projections, so two_thirds is best described as dealiasing
    stabilization, not a proof that every nonlinear coefficient is alias-free.
    cubic_half provides the conservative global truncation for the cubic term
    without the much more expensive 2x padded transforms.
    """

    def __init__(self, solver, rule=DEFAULT_DEALIAS_RULE):
        if rule not in DEALIAS_RULE_FRACTIONS:
            raise ValueError(
                f"Unknown dealias rule {rule!r}; expected one of "
                f"{tuple(DEALIAS_RULE_FRACTIONS)}."
            )
        self.rule = rule
        self.fraction = DEALIAS_RULE_FRACTIONS[rule]
        self.shape = tuple(solver.shape)
        self.device = solver.transform_backend.device
        self.real_dtype = solver.transform_backend.real_dtype
        self._mask_cache = {}
        self._axis_mask_cache = {}

    @property
    def enabled(self):
        return self.fraction is not None

    def _axis_mode_numbers(self, axis, boundary_condition):
        key = (axis, boundary_condition)
        if key in self._axis_mask_cache:
            return self._axis_mask_cache[key]

        n = self.shape[axis]
        if boundary_condition == "periodic":
            mode_numbers = torch.fft.fftfreq(n, d=1.0 / n).to(
                device=self.device,
                dtype=self.real_dtype,
            )
            nyquist_mode = n / 2.0
        elif boundary_condition == "neumann":
            mode_numbers = torch.arange(
                n,
                device=self.device,
                dtype=self.real_dtype,
            )
            nyquist_mode = float(n)
        elif boundary_condition == "dirichlet":
            mode_numbers = torch.arange(
                1,
                n + 1,
                device=self.device,
                dtype=self.real_dtype,
            )
            nyquist_mode = float(n)
        else:
            raise ValueError(
                f"Unsupported boundary condition {boundary_condition!r}."
            )

        keep = mode_numbers.abs() < self.fraction * nyquist_mode
        self._axis_mask_cache[key] = keep
        return keep

    def mask(self, boundary_conditions):
        if not self.enabled:
            raise RuntimeError("The spectral projector is disabled.")

        boundary_conditions = tuple(boundary_conditions)
        if len(boundary_conditions) != len(self.shape):
            raise ValueError(
                "Boundary-condition count must match the spectral dimension."
            )
        if boundary_conditions in self._mask_cache:
            return self._mask_cache[boundary_conditions]

        mask = torch.ones(self.shape, device=self.device, dtype=torch.bool)
        for axis, boundary_condition in enumerate(boundary_conditions):
            axis_keep = self._axis_mode_numbers(axis, boundary_condition)
            view_shape = [1] * len(self.shape)
            view_shape[axis] = self.shape[axis]
            mask &= axis_keep.reshape(view_shape)

        self._mask_cache[boundary_conditions] = mask
        return mask

    def project(self, spectral, boundary_conditions):
        """Project spectral data in its trailing spatial dimensions."""
        if not self.enabled:
            return spectral
        if tuple(spectral.shape[-len(self.shape):]) != self.shape:
            raise ValueError(
                f"Expected trailing spectral shape {self.shape}, got "
                f"{tuple(spectral.shape[-len(self.shape):])}."
            )
        mask = self.mask(boundary_conditions)
        return spectral * mask.to(dtype=spectral.dtype)

    def project_dynamic_fields(self, fields, *, sync_spatial):
        """Project dynamic transform groups and optionally sync real fields."""
        if not self.enabled:
            return
        groups = fields.group_indices_by_boundary_conditions(
            range(fields.dyn_count)
        )
        for group in groups:
            boundary_conditions = fields.get_boundary_conditions(group[0])
            fields.spectral[group] = self.project(
                fields.spectral[group],
                boundary_conditions,
            )
            if sync_spatial:
                fields.spatial[group] = fields.inverse_transform_group(group)

    def retained_axis_counts(self, boundary_conditions):
        if not self.enabled:
            return tuple(self.shape)
        return tuple(
            int(self._axis_mode_numbers(axis, bc).sum().item())
            for axis, bc in enumerate(boundary_conditions)
        )


class DealiasedSemiImplicitEulerIntegrator(SemiImplicitEulerIntegrator):
    """Semi-implicit Euler with projection before every inverse transform.

    This intentionally mirrors the core integrator so the nonlinear spectrum
    is truncated after the IMEX update and before Q returns to real space. The
    periodic real-to-spectral refresh is projected and synchronized as well.
    """

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        self.spectral_projector = model.spectral_projector

    def step(self, pre_update_callback=None):
        if self._static_fields_are_current:
            self._static_fields_are_current = False
        else:
            self.model.update_static_fields()

        if pre_update_callback is not None:
            pre_update_callback()

        nonlinear_hats = self.model.compute_nonlinear()
        dynamic_fields = self.model.fields.spectral[:self.dyn_count]
        dynamic_fields.add_(self.dt * nonlinear_hats)
        dynamic_fields.div_(self.denom)
        self.spectral_projector.project_dynamic_fields(
            self.model.fields,
            sync_spatial=False,
        )

        for group in self.dynamic_transform_groups:
            self.model.fields.spatial[group] = (
                self.model.fields.inverse_transform_group(group)
            )

        self.step_count += 1
        if self.step_count % self.spectral_refresh_interval == 0:
            for group in self.dynamic_transform_groups:
                self.model.fields.spectral[group] = (
                    self.model.fields.forward_transform_group(group)
                )
            self.spectral_projector.project_dynamic_fields(
                self.model.fields,
                sync_spatial=True,
            )
            self.step_count = 0


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


def wall_normal_momentum_stats(fields, params, spectral_projector):
    """Check normal momentum balance at the two z walls for the modal pressure."""
    alpha = params['alpha']
    force_prefactor = beta * alpha

    gxQxz = fields.gradient('Qxz', axis=0)
    gyQyz = fields.gradient('Qyz', axis=1)
    gzQxx = fields.gradient('Qxx', axis=2)
    gzQyy = fields.gradient('Qyy', axis=2)
    fz = force_prefactor * (gxQxz + gyQyz - gzQxx - gzQyy)
    if spectral_projector.enabled:
        fz_hat = fields.transform_tensor(fz, U_NORMAL_BC)
        fz = fields.inverse_transform_tensor(
            spectral_projector.project(fz_hat, U_NORMAL_BC),
            U_NORMAL_BC,
        )

    lap_uz = fields.laplacian('uz')
    dzp = fields.gradient('p', axis=2)
    residual = dzp - (fz + eta * lap_uz - fric * fields['uz'])

    wall_residual = torch.stack([residual[..., 0], residual[..., -1]], dim=-1)
    wall_abs = wall_residual.abs()
    return wall_abs.max().item(), torch.sqrt(torch.mean(wall_abs.square())).item()

class NonlinearModel(torch.nn.Module):
    def __init__(self, spectral_projector):
        super().__init__()
        self.spectral_projector = spectral_projector

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

        # Project the complete pseudo-spectral RHS; the state and Stokes
        # velocity inputs have already been restricted to the same mode box.
        nonlinear_hat = fields.transform_tensor(
            torch.stack([out0, out1, out2, out3, out4]),
            Q_BC,
        )
        return self.spectral_projector.project(nonlinear_hat, Q_BC)

class FreeSlipModalSaddleStokesCompute(torch.nn.Module):
    """
    Mixed DCT/DST Stokes-Brinkman solve for free-slip walls normal to z.

    The z bases are

        ux, uy : Neumann/DCT (cosine),
        uz     : Dirichlet/DST (sine),
        p      : Neumann/DCT (cosine multiplier).

    Since div(u) is DCT-valued, pressure uses the same modal space. In this
    pairing the pressure Schur complement is diagonal and is inverted directly.

    In "zero_mean" mode, zero friction makes spatially constant ux and uy
    rigid plug-flow null modes. The pseudoinverse fixes the reference frame by
    setting those modes to zero. In "friction" mode, positive friction retains
    and uniquely determines the constant tangential modes.
    """

    def __init__(
        self,
        solver,
        spectral_projector,
        beta_value=beta,
        friction=fric,
        viscosity=eta,
        zero_mode_policy=TANGENTIAL_ZERO_MODE_POLICY,
    ):
        super().__init__()
        if viscosity <= 0:
            raise ValueError("Free-slip Stokes solve requires viscosity > 0.")
        if friction < 0:
            raise ValueError("Free-slip Stokes solve requires friction >= 0.")
        if zero_mode_policy not in ("zero_mean", "friction"):
            raise ValueError(
                "zero_mode_policy must be 'zero_mean' or 'friction'."
            )
        if zero_mode_policy == "zero_mean" and friction != 0:
            raise ValueError("zero_mean mode requires friction == 0.")
        if zero_mode_policy == "friction" and friction <= 0:
            raise ValueError("friction mode requires friction > 0.")
        self.beta = float(beta_value)
        self.friction = float(friction)
        self.viscosity = float(viscosity)
        self.zero_mode_policy = zero_mode_policy
        self.spectral_projector = spectral_projector

        backend = solver.transform_backend
        spectral_dtype = backend.spectral_dtype
        real_dtype = backend.real_dtype
        device = solver.qx.device

        tangential_metadata = backend.get_metadata(U_TANGENTIAL_BC)
        normal_metadata = backend.get_metadata(U_NORMAL_BC)
        pressure_metadata = backend.get_metadata(PRESSURE_MODAL_BC)

        qx_xy, qy_xy = torch.meshgrid(
            pressure_metadata.axis_modes[0],
            pressure_metadata.axis_modes[1],
            indexing="ij",
        )
        kz_tangential = tangential_metadata.axis_modes[2]
        kz_normal = normal_metadata.axis_modes[2]
        kz_pressure = pressure_metadata.axis_modes[2]
        nz = kz_normal.numel()

        # Row-vector coefficient convention:
        #
        #   u_D @ dz_D_to_N = dz(u_D),
        #   p_N @ dz_N_to_D = dz(p_N).
        #
        # The last DST mode differentiates to the first unresolved DCT mode and
        # is therefore dropped, matching TensorProductTransformBackend.
        dz_dirichlet_to_neumann = torch.zeros(
            (nz, nz),
            device=device,
            dtype=real_dtype,
        )
        if nz > 1:
            index = torch.arange(nz - 1, device=device)
            dz_dirichlet_to_neumann[index, index + 1] = kz_normal[:-1]
        dz_neumann_to_dirichlet = -dz_dirichlet_to_neumann.transpose(0, 1)

        kxy2 = qx_xy.square() + qy_xy.square()
        a_tangential = self.friction + self.viscosity * (
            kxy2.unsqueeze(-1)
            + kz_tangential.square().view(1, 1, -1)
        )
        a_normal = self.friction + self.viscosity * (
            kxy2.unsqueeze(-1)
            + kz_normal.square().view(1, 1, -1)
        )

        tangential_null_mask = a_tangential == 0
        a_tangential_safe = a_tangential.masked_fill(
            tangential_null_mask,
            1.0,
        )
        a_tangential_inv = 1.0 / a_tangential_safe
        a_tangential_inv.masked_fill_(tangential_null_mask, 0.0)
        a_normal_inv = 1.0 / a_normal

        # S = -D A^{-1} G. The x/y terms remain in DCT. Pressure cosine
        # mode m>=1 maps to normal sine coefficient m-1 and back again.
        schur_diag = kxy2.unsqueeze(-1) * a_tangential_inv
        if nz > 1:
            schur_diag[..., 1:] += (
                kz_pressure[1:].square().view(1, 1, -1)
                * a_normal_inv[..., :-1]
            )

        pressure_null_mask = torch.zeros(
            (1, *qx_xy.shape, nz),
            device=device,
            dtype=torch.bool,
        )
        pressure_null_mask[:, 0, 0, 0] = True
        schur_diag_safe = schur_diag.unsqueeze(0).clone()
        schur_diag_safe.masked_fill_(pressure_null_mask, 1.0)
        unresolved = (schur_diag_safe == 0) & ~pressure_null_mask
        if unresolved.any():
            raise ValueError(
                "Free-slip pressure Schur complement contains a non-gauge "
                "null mode."
            )

        self.register_buffer(
            "ikx",
            (1j * qx_xy)
            .view(1, *qx_xy.shape, 1)
            .to(dtype=spectral_dtype),
        )
        self.register_buffer(
            "iky",
            (1j * qy_xy)
            .view(1, *qy_xy.shape, 1)
            .to(dtype=spectral_dtype),
        )
        self.register_buffer(
            "a_tangential_inv",
            a_tangential_inv.unsqueeze(0),
        )
        self.register_buffer("a_normal_inv", a_normal_inv.unsqueeze(0))
        self.register_buffer(
            "tangential_null_mask",
            tangential_null_mask.unsqueeze(0),
        )
        self.register_buffer(
            "dz_neumann_to_dirichlet",
            dz_neumann_to_dirichlet.to(dtype=spectral_dtype),
        )
        self.register_buffer(
            "dz_dirichlet_to_neumann",
            dz_dirichlet_to_neumann.to(dtype=spectral_dtype),
        )
        self.register_buffer("schur_diag_safe", schur_diag_safe)
        self.register_buffer("pressure_null_mask", pressure_null_mask)

        self.has_tangential_null_mode = bool(
            tangential_null_mask.any().item()
        )
        self.last_tangential_force_mean = None
        self.last_removed_tangential_force_mean = None
        self.last_pressure_hat = None
        self.last_pressure_iterations = 0
        self.last_pressure_residual = 0.0
        self.last_pressure_relative_residual = 0.0

    def _matmul_lastdim(self, tensor, matrix):
        return torch.matmul(tensor, matrix)

    def _project_pressure_gauge(self, pressure_hat):
        return pressure_hat.masked_fill(self.pressure_null_mask, 0)

    def _tangential_helmholtz_inverse(self, rhs_hat):
        return (rhs_hat * self.a_tangential_inv).masked_fill(
            self.tangential_null_mask,
            0,
        )

    def _normal_helmholtz_inverse(self, rhs_hat):
        return rhs_hat * self.a_normal_inv

    def _pressure_grad_z(self, pressure_hat):
        return self._matmul_lastdim(
            pressure_hat,
            self.dz_neumann_to_dirichlet,
        )

    def _velocity_div_z(self, velocity_hat):
        return self._matmul_lastdim(
            velocity_hat,
            self.dz_dirichlet_to_neumann,
        )

    def _pressure_operator(self, pressure_hat):
        pressure_hat = self._project_pressure_gauge(pressure_hat)
        ux_hat = self._tangential_helmholtz_inverse(
            self.ikx * pressure_hat
        )
        uy_hat = self._tangential_helmholtz_inverse(
            self.iky * pressure_hat
        )
        uz_hat = self._normal_helmholtz_inverse(
            self._pressure_grad_z(pressure_hat)
        )
        divergence_hat = (
            self.ikx * ux_hat
            + self.iky * uy_hat
            + self._velocity_div_z(uz_hat)
        )
        return self._project_pressure_gauge(-divergence_hat)

    def _solve_pressure(self, rhs_hat):
        rhs_hat = self._project_pressure_gauge(rhs_hat)
        rhs_norm = torch.linalg.vector_norm(rhs_hat.reshape(-1)).item()
        if rhs_norm == 0.0:
            self.last_pressure_iterations = 0
            self.last_pressure_residual = 0.0
            self.last_pressure_relative_residual = 0.0
            return torch.zeros_like(rhs_hat)

        pressure_hat = self._project_pressure_gauge(
            rhs_hat / self.schur_diag_safe
        )
        residual = rhs_hat - self._pressure_operator(pressure_hat)
        residual_norm = torch.linalg.vector_norm(residual.reshape(-1)).item()
        self.last_pressure_iterations = 1
        self.last_pressure_residual = residual_norm
        self.last_pressure_relative_residual = residual_norm / rhs_norm
        return pressure_hat

    def _solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        """Solve the mixed-basis saddle system for native force spectra."""
        ux_hat_free = self._tangential_helmholtz_inverse(fx_hat)
        uy_hat_free = self._tangential_helmholtz_inverse(fy_hat)
        uz_hat_free = self._normal_helmholtz_inverse(fz_hat)

        provisional_divergence = (
            self.ikx * ux_hat_free
            + self.iky * uy_hat_free
            + self._velocity_div_z(uz_hat_free)
        )
        pressure_rhs = self._project_pressure_gauge(
            -provisional_divergence
        )
        pressure_hat = self._solve_pressure(pressure_rhs)
        self.last_pressure_hat = pressure_hat.detach()

        ux_hat = ux_hat_free - self._tangential_helmholtz_inverse(
            self.ikx * pressure_hat
        )
        uy_hat = uy_hat_free - self._tangential_helmholtz_inverse(
            self.iky * pressure_hat
        )
        uz_hat = uz_hat_free - self._normal_helmholtz_inverse(
            self._pressure_grad_z(pressure_hat)
        )
        return ux_hat, uy_hat, uz_hat, pressure_hat

    def forward(self, fields, params):
        alpha = params["alpha"]
        force_prefactor = self.beta * alpha

        gxQxx = fields.gradient("Qxx", axis=0)
        gyQxy = fields.gradient("Qxy", axis=1)
        gzQxz = fields.gradient("Qxz", axis=2)

        gxQxy = fields.gradient("Qxy", axis=0)
        gyQyy = fields.gradient("Qyy", axis=1)
        gzQyz = fields.gradient("Qyz", axis=2)

        gxQxz = fields.gradient("Qxz", axis=0)
        gyQyz = fields.gradient("Qyz", axis=1)
        gzQxx = fields.gradient("Qxx", axis=2)
        gzQyy = fields.gradient("Qyy", axis=2)

        force = force_prefactor * torch.stack(
            (
                gxQxx + gyQxy + gzQxz,
                gxQxy + gyQyy + gzQyz,
                gxQxz + gyQyz - gzQxx - gzQyy,
            )
        )
        self.last_tangential_force_mean = (
            force[:2].mean(dim=(-3, -2, -1)).detach()
        )
        self.last_removed_tangential_force_mean = (
            self.last_tangential_force_mean
            if self.has_tangential_null_mode
            else torch.zeros_like(self.last_tangential_force_mean)
        )

        # Project force spectra before the coupled saddle solve. Post-filtering
        # u component by component could destroy the discrete div(u)=0 pairing.
        force_tangential_hat = self.spectral_projector.project(
            fields.transform_tensor(
                force[:2],
                U_TANGENTIAL_BC,
            ),
            U_TANGENTIAL_BC,
        )
        force_normal_hat = self.spectral_projector.project(
            fields.transform_tensor(
                force[2],
                U_NORMAL_BC,
            ),
            U_NORMAL_BC,
        )
        fx_hat, fy_hat = force_tangential_hat
        ux_hat, uy_hat, uz_hat, pressure_hat = self._solve_force_hats(
            fx_hat,
            fy_hat,
            force_normal_hat,
        )

        return torch.stack((ux_hat, uy_hat, uz_hat, pressure_hat))


def main():
    global fric
    args = parse_args()
    seed = 24
    dt = args.dt
    steps = args.steps
    save_interval = args.save_interval
    dealias_rule = args.dealias_rule
    zero_mode_policy = args.zero_mode_policy
    fric = (
        0.0
        if zero_mode_policy == "zero_mean"
        else float(args.friction_mode_fric)
    )
    if args.output_dir is None:
        base_output_dir = (
            "data_free_slip"
            if zero_mode_policy == "zero_mean"
            else "data_free_slip_friction"
        )
        generated_output_dir = f"{base_output_dir}_dealiased_{dealias_rule}"
        snapshot_output_dir = f"{generated_output_dir}_snapshot"
    else:
        generated_output_dir = args.output_dir
        snapshot_output_dir = args.output_dir
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batchsize = 1

    Nx, Ny, Nz = 256, 256, 60
    Lx, Ly, Lz = 64.0, 64.0, 15.0

    solver = SpectralSolver(
        shape=(Nx, Ny, Nz),
        L=(Lx, Ly, Lz),
        dt=dt,
        device=device,
        batchsize=batchsize,
    )
    spectral_projector = BasisAwareSpectralProjector(
        solver,
        rule=dealias_rule,
    )
    solver.model.spectral_projector = spectral_projector
    solver.integrator_cl = DealiasedSemiImplicitEulerIntegrator
    snapshot = None
    if INITIALIZATION_MODE == "generated":
        q_initial_condition = create_initial_condition(
            "aligned_x_smooth_noise",
            shape=(Nx, Ny, Nz),
            boundary_conditions=Q_BC,
            seed=seed,
            S_initial=S_INITIAL,
            noise_theta=0.01,
            noise_phi=0.01,
            sigma_x=1.0,
            sigma_y=1.0,
            sigma_z=1.0,
        )
    elif INITIALIZATION_MODE == "snapshot":
        if SNAPSHOT_MODE not in ("resume", "branch"):
            raise ValueError("SNAPSHOT_MODE must be 'resume' or 'branch'")
        snapshot_load_options = {
            "expected_shape": (Nx, Ny, Nz),
            "require_S_initial": True,
        }
        if SNAPSHOT_MODE == "resume":
            snapshot_load_options["expected_S_bulk"] = S_BULK
        snapshot = load_snapshot(
            SNAPSHOT_DIRECTORY,
            SNAPSHOT_STEP,
            **snapshot_load_options,
        )
        q_initial_condition = snapshot.q_fields
    else:
        raise ValueError("INITIALIZATION_MODE must be 'generated' or 'snapshot'")
    snapshot_ordered_S = (
        None
        if snapshot is None
        else representative_ordered_S(snapshot.q_fields)
    )
    Qxx_0 = q_initial_condition["Qxx"]
    Qxy_0 = q_initial_condition["Qxy"]
    Qxz_0 = q_initial_condition["Qxz"]
    Qyy_0 = q_initial_condition["Qyy"]
    Qyz_0 = q_initial_condition["Qyz"]

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
    # Free-slip mixed modal spaces: tangential velocity uses DCT in z, normal
    # velocity uses DST in z, and pressure is the DCT incompressibility multiplier.
    solver.model.add_static_field("ux", boundary_conditions=U_TANGENTIAL_BC)
    solver.model.add_static_field("uy", boundary_conditions=U_TANGENTIAL_BC)
    solver.model.add_static_field("uz", boundary_conditions=U_NORMAL_BC)
    solver.model.add_static_field("p", boundary_conditions=PRESSURE_MODAL_BC)



    solver.model.set_nonlinear_model(
        NonlinearModel(spectral_projector)
    )
    solver.model.set_static_compute_model(
        FreeSlipModalSaddleStokesCompute(
            solver,
            spectral_projector=spectral_projector,
            beta_value=beta,
            friction=fric,
            viscosity=eta,
            zero_mode_policy=zero_mode_policy,
        )
    )

    alpha = torch.tensor(5.0, device=device)
    solver.model.parameters.new_param('alpha', alpha)

    solver.build()

    start_step = 0
    if snapshot is None:
        output_dir = generated_output_dir
    else:
        output_dir = require_distinct_output_directory(
            snapshot,
            snapshot_output_dir,
        )

    diagnostic_history = []
    output_dir = prepare_new_run_directory(output_dir)
    dealias_fraction_text = (
        "disabled"
        if spectral_projector.fraction is None
        else f"{spectral_projector.fraction:.6g}"
    )
    q_mode_counts = spectral_projector.retained_axis_counts(Q_BC)
    normal_mode_counts = spectral_projector.retained_axis_counts(U_NORMAL_BC)
    initial_condition_metadata = (
        {
            "name": "aligned_x_smooth_noise",
            "seed": seed,
            "S_initial": S_INITIAL,
        }
        if snapshot is None
        else {
            "name": "snapshot",
            "source_directory": str(snapshot.directory),
            "source_step": snapshot.step,
            "mode": SNAPSHOT_MODE,
            "source_metadata": str(snapshot.source_metadata.path),
            "source_S_initial": snapshot.source_metadata.S_initial,
            "source_S_bulk": snapshot.source_metadata.S_bulk,
            "snapshot_ordered_S": snapshot_ordered_S,
            "snapshot_ordered_S_definition": (
                REPRESENTATIVE_ORDERED_S_DEFINITION
            ),
        }
    )
    run_metadata = {
        "schema_version": 1,
        "script": "Plane_free_slip_dealiased.py",
        "solver": {
            "shape": [Nx, Ny, Nz],
            "lengths": [Lx, Ly, Lz],
            "dt": dt,
            "steps": steps,
            "save_interval": save_interval,
        },
        "model": {
            "name": "active_nematics",
            "Q_convention": Q_convention_metadata(),
            "parameters": {
                "aQ": aQ,
                "bQ": bQ,
                "cQ": cQ,
                "KQ": KQ,
                "S_initial": (
                    S_INITIAL
                    if snapshot is None
                    else (
                        snapshot.source_metadata.S_initial
                        if SNAPSHOT_MODE == "resume"
                        else snapshot_ordered_S
                    )
                ),
                "S_bulk": S_BULK,
                "alpha": float(alpha.item()),
                "beta": beta,
                "fric": fric,
                "eta": eta,
            },
        },
        "boundary_conditions": {
            "Q": Q_BC,
            "velocity_tangential": U_TANGENTIAL_BC,
            "velocity_normal": U_NORMAL_BC,
            "pressure": PRESSURE_MODAL_BC,
        },
        "numerics": {
            "dealiasing": {
                "rule": dealias_rule,
                "fraction": spectral_projector.fraction,
                "Q_retained_modes": q_mode_counts,
                "normal_velocity_retained_modes": normal_mode_counts,
            },
            "velocity_zero_mode": zero_mode_policy,
            "pressure_solver": "free_slip_modal_schur_complement",
        },
        "initial_condition": initial_condition_metadata,
    }
    if snapshot is not None:
        start_step = apply_snapshot_to_solver(
            solver,
            snapshot,
            mode=SNAPSHOT_MODE,
            current_run_metadata=run_metadata,
        )
        print(
            f"Loaded {SNAPSHOT_MODE} snapshot at step {snapshot.step} "
            f"from {snapshot.directory}"
        )

    # Generated and branch Q are projected before their first nonlinear
    # product. A metadata-compatible resume skips this projection because its
    # saved Q is already part of the verified trajectory and its restored u/p
    # must remain current for the first continued step.
    if snapshot is None or SNAPSHOT_MODE == "branch":
        spectral_projector.project_dynamic_fields(
            solver.model.fields,
            sync_spatial=True,
        )
        if spectral_projector.enabled:
            solver.integrator._static_fields_are_current = False

    write_run_metadata(output_dir, run_metadata, status="running")
    print(
        f"Run configuration: zero_mode_policy={zero_mode_policy}, "
        f"fric={fric}, steps={steps}, dt={dt}, "
        f"dealias_rule={dealias_rule}, fraction={dealias_fraction_text}, "
        f"save_interval={save_interval}, output_dir={output_dir}"
    )
    print(
        "Retained modes per axis: "
        f"Q/ux/uy/p={q_mode_counts}, uz={normal_mode_counts}"
    )
    start = time.time()
    pbar = trange(steps)


    def save_step_state(i):
        state_names = ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz", "ux", "uy", "uz"]
        nonfinite = [
            name
            for name in state_names
            if not torch.isfinite(solver.model.fields[name]).all().item()
        ]
        if nonfinite:
            raise FloatingPointError(
                f"Non-finite state at step {i}: {', '.join(nonfinite)}"
            )

        q_snapshot = torch.stack([
            solver.model.fields[name].detach().cpu()
            for name in ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"]
        ])
        q_snapshot = q_snapshot.permute(1, 2, 3, 4, 0)
        np.save(f"{output_dir}/Q_{i}.npy", q_snapshot[0].numpy())

        u_snapshot = torch.stack([
            solver.model.fields[name].detach().cpu()
            for name in ["ux", "uy", "uz"]
        ])
        u_snapshot = u_snapshot.permute(1, 2, 3, 4, 0)
        np.save(f"{output_dir}/u_{i}.npy", u_snapshot[0].numpy())

        p_snapshot = solver.model.fields["p"].detach().cpu()
        np.save(f"{output_dir}/p_{i}.npy", p_snapshot[0].numpy())


    def record_step_state(i):
        if ENABLE_DIAGNOSTICS and i % DIAGNOSTIC_INTERVAL == 0:
            div_max, div_rms, div_rel = divergence_stats(solver.model.fields)
            wall_mom_max, wall_mom_rms = wall_normal_momentum_stats(
                solver.model.fields,
                solver.model.parameters,
                spectral_projector,
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
                div_max=f"{div_max:.2e}",
                div_rms=f"{div_rms:.2e}",
                div_rel=f"{div_rel:.2e}",
                schur_it=static_model.last_pressure_iterations,
                schur_rel=f"{static_model.last_pressure_relative_residual:.2e}",
                wall_n_rms=f"{wall_mom_rms:.2e}",
            )
        if i % save_interval == 0:
            save_step_state(i)


    for local_step in pbar:
        i = start_step + local_step
        solver.run(1, pre_update_callback=lambda _solver, _step, i=i: record_step_state(i))

    final_step = start_step + steps
    solver.refresh_static_fields()
    if final_step % save_interval == 0:
        save_step_state(final_step)

    end = time.time()
    print(f"Elapsed time: {end - start:.6f} seconds")
    if ENABLE_DIAGNOSTICS:
        solver.refresh_static_fields()
        final_div_max, final_div_rms, final_div_rel = divergence_stats(solver.model.fields)
        final_wall_mom_max, final_wall_mom_rms = wall_normal_momentum_stats(
            solver.model.fields,
            solver.model.parameters,
            spectral_projector,
        )
        static_model = solver.model.static_model
        diagnostic_history.append((
            start_step + steps,
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

    write_run_metadata(output_dir, run_metadata, status="complete")


if __name__ == "__main__":
    main()
