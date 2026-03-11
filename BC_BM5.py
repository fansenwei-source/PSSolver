import os
import math
import time
import numpy as np
import torch
from tqdm import trange
from scipy.ndimage import gaussian_filter1d

from pssolver import SpectralSolver
from pssolver.transforms import dct1, idct1, dst1, idst1, _k_axis_for_bc, _forward_1d_bc
from pssolver.boundary import BoundaryCondition
from pssolver.mixed_spectral import SpectralTransform3D

# =========================
# Boundary condition labels
# =========================
BC_PERIODIC  = BoundaryCondition.PERIODIC
BC_NEUMANN   = BoundaryCondition.NEUMANN
BC_DIRICHLET = BoundaryCondition.DIRICHLET


def _mixed_k2(shape, L, bcs, device, dtype):
    axes = []
    for N, Ld, axis in zip(shape, L, ("x", "y", "z")):
        bc = _as_bc(bcs.get(axis, BC_PERIODIC))
        k, _, _ = _k_axis_for_bc(bc, N, Ld, device, dtype)
        axes.append(k)
    grids = torch.meshgrid(*axes, indexing="ij")
    return sum(g ** 2 for g in grids)

# ============================================================
# BC-aware finite difference (diagnostic / real-space ops)
# ============================================================

def _as_bc(bc):
    if isinstance(bc, BoundaryCondition):
        return bc
    return BoundaryCondition(bc)


def _spectral_diff_1d(F, h, dim, bc):
    bc = _as_bc(bc)
    d = dim % F.ndim
    n = F.shape[d]
    if n <= 2:
        return torch.zeros_like(F)

    L = h * (n - 1)
    dev = F.device
    dtype = F.dtype

    if bc == BC_NEUMANN:
        # Cosine series -> derivative is sine series (Dirichlet at boundaries).
        A = dct1(F, dim=d)
        k, _, _ = _k_axis_for_bc(BC_NEUMANN, n, L, dev, dtype)
        shape = [1] * F.ndim
        shape[d] = n
        k = k.view(*shape)
        B_in = (-(k * A)).narrow(d, 1, n - 2)
        d_in = idst1(B_in, dim=d).real
        out = torch.zeros_like(F)
        idx = [slice(None)] * F.ndim
        idx[d] = slice(1, -1)
        out[tuple(idx)] = d_in
        return out

    if bc == BC_DIRICHLET:
        # Sine series -> derivative is cosine series (Neumann at boundaries).
        F_in = F.narrow(d, 1, n - 2)
        A = dst1(F_in, dim=d)
        k, _, _ = _k_axis_for_bc(BC_DIRICHLET, n, L, dev, dtype)
        shape = [1] * F.ndim
        shape[d] = n - 2
        k = k.view(*shape)
        C_in = k * A
        C_full = torch.zeros_like(F)
        idx = [slice(None)] * F.ndim
        idx[d] = slice(1, -1)
        C_full[tuple(idx)] = C_in
        return idct1(C_full, dim=d).real

    raise ValueError(f"Unknown BC: {bc}")


def diff_1d(F, h, dim, bc):
    """
    1st derivative along dim:
    - periodic : spectral derivative via FFT
    - neumann/dirichlet: spectral derivative via DCT-I/DST-I
    """
    bc = _as_bc(bc)
    if bc == BC_PERIODIC:
        d = dim % F.ndim
        n = F.shape[d]
        if n <= 1:
            return torch.zeros_like(F)
        L = h * n
        k, _, _ = _k_axis_for_bc(BC_PERIODIC, n, L, F.device, F.dtype)
        shape = [1] * F.ndim
        shape[d] = n
        k = k.view(*shape)
        Fh = torch.fft.fft(F, dim=d)
        return torch.fft.ifft(1j * k * Fh, dim=d).real
    return _spectral_diff_1d(F, h, dim, bc)


# Backwards-compatible wrappers (if your code still calls these)
# ============================================================
# Q initialization (BC-aware smoothing)
# ============================================================

def Q_init(shape, seed=42, noise=0.05*2*math.pi, sigma_xy=1.0, sigma_z=1.0, bcs_Q=None):
    """
    Q init:
    - Uniform in-plane alignment (no defect pair).
    - Add smooth 3D angle noise (theta, phi) with BC-aware gaussian smoothing.
    - Enforce periodic only on periodic axes; leave Neumann/Dirichlet to projection.
    """
    Nx, Ny, Nz = shape
    rng = np.random.default_rng(seed)

    if bcs_Q is None:
        bcs_Q = dict(x=BC_PERIODIC, y=BC_PERIODIC, z=BC_NEUMANN)

    def mode_from_bc(bc):
        # Q_init smoothing currently assumes periodic or Neumann; Dirichlet also maps to reflect.
        if bc == BC_PERIODIC:
            return "wrap"
        return "reflect"

    mode_x = mode_from_bc(bcs_Q.get("x", BC_PERIODIC))
    mode_y = mode_from_bc(bcs_Q.get("y", BC_PERIODIC))
    mode_z = mode_from_bc(bcs_Q.get("z", BC_NEUMANN))

    S0 = 1.0
    angle_phi_plane = np.zeros((Nx, Ny, Nz), dtype=np.float32)

    angle_theta_0 = math.pi / 2.0  # in-plane

    # smooth noise fields
    delta_theta = rng.uniform(-1.0, 1.0, size=(Nx, Ny, Nz)).astype(np.float32)
    delta_phi   = rng.uniform(-1.0, 1.0, size=(Nx, Ny, Nz)).astype(np.float32)

    for arr in (delta_theta, delta_phi):
        arr[:] = gaussian_filter1d(arr, sigma=sigma_xy, axis=0, mode=mode_x)
        arr[:] = gaussian_filter1d(arr, sigma=sigma_xy, axis=1, mode=mode_y)
        arr[:] = gaussian_filter1d(arr, sigma=sigma_z,  axis=2, mode=mode_z)

    def rescale_to_amp(arr, amp):
        std = arr.std()
        if std > 1e-12:
            return arr / std * amp
        return arr * 0.0

    delta_theta = rescale_to_amp(delta_theta, noise)
    delta_phi   = rescale_to_amp(delta_phi,   noise)

    angle_theta = angle_theta_0 + delta_theta
    angle_phi   = angle_phi_plane + delta_phi

    sin_theta = np.sin(angle_theta)
    cos_theta = np.cos(angle_theta)
    cos_phi   = np.cos(angle_phi)
    sin_phi   = np.sin(angle_phi)

    nx = sin_theta * cos_phi
    ny = sin_theta * sin_phi
    nz = cos_theta

    Qxx = S0 * (nx * nx - 1.0 / 3.0)
    Qyy = S0 * (ny * ny - 1.0 / 3.0)
    Qxy = S0 * (nx * ny)
    Qxz = S0 * (nx * nz)
    Qyz = S0 * (ny * nz)

    # Periodic continuity is already enforced by mode="wrap" smoothing.

    Qxx_t = torch.from_numpy(Qxx.astype(np.float32))
    Qxy_t = torch.from_numpy(Qxy.astype(np.float32))
    Qxz_t = torch.from_numpy(Qxz.astype(np.float32))
    Qyy_t = torch.from_numpy(Qyy.astype(np.float32))
    Qyz_t = torch.from_numpy(Qyz.astype(np.float32))

    return Qxx_t, Qxy_t, Qxz_t, Qyy_t, Qyz_t

# ============================================================
# Static compute model: BC-aware Q operators + Stokes (yz Dirichlet)
# ============================================================

class Static_compute_fn(torch.nn.Module):
    """
    Static compute:
      - Q Laplacians via mixed spectral transform
      - Q gradients: periodic x/y spectral derivative; z by real-space diff_1d w/ BC
      - Stress -> force -> Stokes solve with x periodic, y/z Dirichlet
      - Diagnostics: div(u), wall velocities (optional)
    """
    def __init__(
        self,
        solver,
        bcs_Q=None,
        bcs_u=None,
        beta=-1.0,
        fric=0.0,
        eta=1.0,
        enable_diag=False,
        diag_every=50,
        heavy_diag_every=200,
    ):
        super().__init__()
        self.Nx, self.Ny, self.Nz = solver.shape
        self.Lx, self.Ly, self.Lz = solver.L
        self.batchsize = solver.batchsize
        self.device = solver.q2.device
        self.dtype = solver.qx.dtype

        # store fluid params here (avoid globals)
        self.beta = float(beta)
        self.fric = float(fric)
        self.eta  = float(eta)
        # Convention: eta*Lap(u) - grad(p) - fric*u = f,   div(u)=0
        # f = div(sigma),  sigma = - beta * alpha * Q
        # (beta = -1 => sigma = +alpha Q => f = +alpha div(Q), same sign as Minu)

        # BC settings (prefer field-level bcs from solver if available)
        if bcs_Q is not None or bcs_u is not None:
            bcs_Q = bcs_Q or {}
            bcs_u = bcs_u or {}
            self.bcs_Q_tuple = (bcs_Q.get("x", BC_PERIODIC), bcs_Q.get("y", BC_PERIODIC), bcs_Q.get("z", BC_NEUMANN))
            self.bcs_u_tuple = (bcs_u.get("x", BC_PERIODIC), bcs_u.get("y", BC_DIRICHLET), bcs_u.get("z", BC_DIRICHLET))
        elif solver.fields.field_bcs:
            self.bcs_Q_tuple = solver.fields.field_bcs.get(
                "Qxx",
                (BC_PERIODIC, BC_PERIODIC, BC_NEUMANN),
            )
            self.bcs_u_tuple = solver.fields.field_bcs.get(
                "ux",
                (BC_PERIODIC, BC_DIRICHLET, BC_DIRICHLET),
            )
        else:
            self.bcs_Q_tuple = (BC_PERIODIC, BC_PERIODIC, BC_NEUMANN)
            self.bcs_u_tuple = (BC_PERIODIC, BC_DIRICHLET, BC_DIRICHLET)
        self.bcs_Q = {"x": self.bcs_Q_tuple[0], "y": self.bcs_Q_tuple[1], "z": self.bcs_Q_tuple[2]}
        self.bcs_u = {"x": self.bcs_u_tuple[0], "y": self.bcs_u_tuple[1], "z": self.bcs_u_tuple[2]}
        self.bcs_p_tuple = (BC_PERIODIC, BC_NEUMANN, BC_NEUMANN)
        self.bcs_p = {"x": self.bcs_p_tuple[0], "y": self.bcs_p_tuple[1], "z": self.bcs_p_tuple[2]}

        def _h_for_axis(L, N, bc):
            return L / (N if bc == BC_PERIODIC else (N - 1))

        self.dx_u = _h_for_axis(self.Lx, self.Nx, self.bcs_u_tuple[0])
        self.dy_u = _h_for_axis(self.Ly, self.Ny, self.bcs_u_tuple[1])
        self.dz_u = _h_for_axis(self.Lz, self.Nz, self.bcs_u_tuple[2])

        self.dx_Q = _h_for_axis(self.Lx, self.Nx, self.bcs_Q_tuple[0])
        self.dy_Q = _h_for_axis(self.Ly, self.Ny, self.bcs_Q_tuple[1])
        self.dz_Q = _h_for_axis(self.Lz, self.Nz, self.bcs_Q_tuple[2])

        self.dx_p = _h_for_axis(self.Lx, self.Nx, self.bcs_p_tuple[0])
        self.dy_p = _h_for_axis(self.Ly, self.Ny, self.bcs_p_tuple[1])
        self.dz_p = _h_for_axis(self.Lz, self.Nz, self.bcs_p_tuple[2])

        # Q transform (mixed BC)
        self.trans_Q = SpectralTransform3D(
            shape=solver.shape,
            L=solver.L,
            bcs=self.bcs_Q,
            device=self.device,
            dtype=self.dtype
        )
        self.trans_u = SpectralTransform3D(
            shape=solver.shape,
            L=solver.L,
            bcs=self.bcs_u,
            device=self.device,
            dtype=self.dtype
        )

        # diagnostics control
        self.enable_diag = bool(enable_diag)
        self.diag_every = int(diag_every)
        self.heavy_diag_every = int(heavy_diag_every)

        # diagnostics counter (FIXED)
        self._div_counter = 0
        self._pressure_iters = None
        self._p_full_buf = None
        self._tmp_full_buf = None
        self._ux_buf = None
        self._uy_buf = None
        self._uz_buf = None
        self._precond_diag = None
        self._precond_meta = None
        self._pressure_resid = None
        self._pressure_schur_resid = None
        self._pressure_tilde_resid = None
        self._pressure_bc_resid = None
        self._pressure_bc_final_resid = None
        self._u_wall_max = None
        self._divu_rms = None
        self._u_cache = None
        self._p_tilde_buf = None
        self.disable_lifting = False
        self.disable_p_warm = False
        self.enable_diag_detail = False

    # ---- Q laplacian via mixed spectral ----
    def _laplacian_Q(self, Q):
        Qh = self.trans_Q.forward_scalar(Q)
        k2 = self.trans_Q.k2()
        Qh2 = -k2 * Qh
        Q_new = self.trans_Q.inverse_scalar(Qh2)
        return Q_new.real

    # ---- Q gradient: x/y periodic spectral; z real-space diff ----
    def _grad_Q(self, Q):
        Qh = self.trans_Q.forward_scalar(Q)
        axes = [
            (self.trans_Q.ax_x, self.dx_Q, -3, self.bcs_Q["x"]),
            (self.trans_Q.ax_y, self.dy_Q, -2, self.bcs_Q["y"]),
            (self.trans_Q.ax_z, self.dz_Q, -1, self.bcs_Q["z"]),
        ]
        grads = []
        for ax, h, dim, bc in axes:
            k_factor = ax.deriv_factor()
            if k_factor is not None:
                grad = self.trans_Q.inverse_scalar(k_factor * Qh).real
            else:
                grad = diff_1d(Q, h, dim=dim, bc=bc)
            grads.append(grad)
        return grads[0], grads[1], grads[2]

    # ---- strict interior-only centered difference (for D/G consistency) ----
    def _cdiff_interior(self, F, h, dim):
        g = torch.zeros_like(F)
        d = dim % F.ndim
        n = F.shape[d]
        if n <= 2:
            return g

        s_mid = [slice(None)] * F.ndim
        s_p   = [slice(None)] * F.ndim
        s_m   = [slice(None)] * F.ndim
        s_mid[d] = slice(1, -1)
        s_p[d]   = slice(2, None)
        s_m[d]   = slice(None, -2)
        g[tuple(s_mid)] = (F[tuple(s_p)] - F[tuple(s_m)]) / (2 * h)
        return g

    def _yz_slices(self):
        sy = slice(1, -1) if self.bcs_u_tuple[1] == BC_DIRICHLET else slice(None)
        sz = slice(1, -1) if self.bcs_u_tuple[2] == BC_DIRICHLET else slice(None)
        return sy, sz

    def _restrict_interior_yz(self, F):
        sy, sz = self._yz_slices()
        return F[..., sy, sz]

    def _remove_mean(self, x):
        return x - x.mean(dim=(-3, -2, -1), keepdim=True)

    def _remove_mean_interior(self, x):
        sy, sz = self._yz_slices()
        m = x[..., sy, sz].mean(dim=(-3, -2, -1), keepdim=True)
        return x - m

    def _diff_axis(self, F, h, dim, bc):
        return diff_1d(F, h, dim=dim, bc=bc)

    def _replace_boundary_deriv(self, dF, F, h, dim, bc, mode="onesided"):
        # For Neumann fields, mode="zero" is often a cleaner wall treatment.
        if bc not in (BC_DIRICHLET, BC_NEUMANN):
            return dF
        if mode == "zero":
            out = dF.clone()
            d = dim % F.ndim
            idx0 = [slice(None)] * F.ndim
            idxm1 = [slice(None)] * F.ndim
            idx0[d] = 0
            idxm1[d] = -1
            out[tuple(idx0)] = 0.0
            out[tuple(idxm1)] = 0.0
            return out

        if mode != "onesided":
            raise ValueError(f"Unknown boundary-derivative mode: {mode}")

        d = dim % F.ndim
        n = F.shape[d]
        if n < 3:
            out = dF.clone()
            idx0 = [slice(None)] * F.ndim
            idx0[d] = 0
            idxm1 = [slice(None)] * F.ndim
            idxm1[d] = -1
            out[tuple(idx0)] = 0.0
            out[tuple(idxm1)] = 0.0
            return out

        out = dF.clone()
        idx0 = [slice(None)] * F.ndim
        idx1 = [slice(None)] * F.ndim
        idx2 = [slice(None)] * F.ndim
        idxm1 = [slice(None)] * F.ndim
        idxm2 = [slice(None)] * F.ndim
        idxm3 = [slice(None)] * F.ndim
        idx0[d] = 0
        idx1[d] = 1
        idx2[d] = 2
        idxm1[d] = -1
        idxm2[d] = -2
        idxm3[d] = -3

        out[tuple(idx0)] = (-3.0 * F[tuple(idx0)] + 4.0 * F[tuple(idx1)] - F[tuple(idx2)]) / (2.0 * h)
        out[tuple(idxm1)] = (3.0 * F[tuple(idxm1)] - 4.0 * F[tuple(idxm2)] + F[tuple(idxm3)]) / (2.0 * h)
        return out

    def _grad_p(self, p):
        px = diff_1d(p, self.dx_p, dim=-3, bc=self.bcs_p_tuple[0])
        py = diff_1d(p, self.dy_p, dim=-2, bc=self.bcs_p_tuple[1])
        pz = diff_1d(p, self.dz_p, dim=-1, bc=self.bcs_p_tuple[2])
        return px, py, pz

    def _div_u(self, ux, uy, uz):
        divx = diff_1d(ux, self.dx_u, dim=-3, bc=BC_PERIODIC)
        divy = self._diff_axis(uy, self.dy_u, dim=-2, bc=self.bcs_u_tuple[1])
        divz = self._diff_axis(uz, self.dz_u, dim=-1, bc=self.bcs_u_tuple[2])
        divy = self._replace_boundary_deriv(divy, uy, self.dy_u, dim=-2, bc=self.bcs_u_tuple[1])
        divz = self._replace_boundary_deriv(divz, uz, self.dz_u, dim=-1, bc=self.bcs_u_tuple[2])
        return divx + divy + divz

    def _laplacian_u(self, u):
        uh = self.trans_u.forward_scalar(u)
        uh2 = -self.trans_u.k2() * uh
        return self.trans_u.inverse_scalar(uh2).real

    def _apply_u_dirichlet(self, u):
        out = u
        if self.bcs_u_tuple[1] == BC_DIRICHLET:
            out = out.clone()
            out[..., 0, :] = 0.0
            out[..., -1, :] = 0.0
        if self.bcs_u_tuple[2] == BC_DIRICHLET:
            out = out.clone()
            out[..., :, 0] = 0.0
            out[..., :, -1] = 0.0
        return out

    def _pressure_lifting(self, fx, fy, fz, ufx, ufy, ufz, u_cache=None):
        # Non-homogeneous Neumann lifting from n·(eta∇^2 u - fric*u - f) on walls (no-slip).
        dev = fy.device
        dtype = fy.dtype
        B, Nx, Ny, Nz = fy.shape

        if u_cache is not None:
            ux_c, uy_c, uz_c = u_cache
            uy_use = self._apply_u_dirichlet(uy_c)
            uz_use = self._apply_u_dirichlet(uz_c)
        else:
            uy_use = self._apply_u_dirichlet(ufy)
            uz_use = self._apply_u_dirichlet(ufz)

        def _d2_onesided(F, h, dim):
            d = dim % F.ndim
            n = F.shape[d]
            out = torch.zeros_like(F)
            if n < 4:
                return out
            idx0 = [slice(None)] * F.ndim
            idx1 = [slice(None)] * F.ndim
            idx2 = [slice(None)] * F.ndim
            idx3 = [slice(None)] * F.ndim
            idxm1 = [slice(None)] * F.ndim
            idxm2 = [slice(None)] * F.ndim
            idxm3 = [slice(None)] * F.ndim
            idxm4 = [slice(None)] * F.ndim
            idx0[d] = 0
            idx1[d] = 1
            idx2[d] = 2
            idx3[d] = 3
            idxm1[d] = -1
            idxm2[d] = -2
            idxm3[d] = -3
            idxm4[d] = -4
            out[tuple(idx0)] = (2.0 * F[tuple(idx0)] - 5.0 * F[tuple(idx1)] + 4.0 * F[tuple(idx2)] - F[tuple(idx3)]) / (h * h)
            out[tuple(idxm1)] = (2.0 * F[tuple(idxm1)] - 5.0 * F[tuple(idxm2)] + 4.0 * F[tuple(idxm3)] - F[tuple(idxm4)]) / (h * h)
            return out

        def _d2_axis(F, h, dim, bc):
            return diff_1d(diff_1d(F, h, dim=dim, bc=bc), h, dim=dim, bc=bc)

        d2x_uy = _d2_axis(uy_use, self.dx_u, dim=-3, bc=BC_PERIODIC)
        d2z_uy = _d2_axis(uy_use, self.dz_u, dim=-1, bc=self.bcs_u_tuple[2])
        d2y_uy = _d2_onesided(uy_use, self.dy_u, dim=-2)

        d2x_uz = _d2_axis(uz_use, self.dx_u, dim=-3, bc=BC_PERIODIC)
        d2y_uz = _d2_axis(uz_use, self.dy_u, dim=-2, bc=self.bcs_u_tuple[1])
        d2z_uz = _d2_onesided(uz_use, self.dz_u, dim=-1)

        # outward normals: y=0 -> -y, y=Ly -> +y; z=0 -> -z, z=Lz -> +z
        lap_uy_y0 = d2x_uy[..., 0, :] + d2y_uy[..., 0, :] + d2z_uy[..., 0, :]
        lap_uy_y1 = d2x_uy[..., -1, :] + d2y_uy[..., -1, :] + d2z_uy[..., -1, :]
        lap_uz_z0 = d2x_uz[..., :, 0] + d2y_uz[..., :, 0] + d2z_uz[..., :, 0]
        lap_uz_z1 = d2x_uz[..., :, -1] + d2y_uz[..., :, -1] + d2z_uz[..., :, -1]

        g_y0 = -(self.eta * lap_uy_y0 - self.fric * uy_use[..., 0, :] - fy[..., 0, :])
        g_y1 = +(self.eta * lap_uy_y1 - self.fric * uy_use[..., -1, :] - fy[..., -1, :])
        g_z0 = -(self.eta * lap_uz_z0 - self.fric * uz_use[..., :, 0] - fz[..., :, 0])
        g_z1 = +(self.eta * lap_uz_z1 - self.fric * uz_use[..., :, -1] - fz[..., :, -1])
        # g = n · (eta * Lap(u) - fric*u - f)

        y = torch.linspace(0.0, self.Ly, Ny, device=dev, dtype=dtype).view(1, 1, Ny, 1)
        z = torch.linspace(0.0, self.Lz, Nz, device=dev, dtype=dtype).view(1, 1, 1, Nz)

        g_y0 = g_y0.view(B, Nx, 1, Nz)
        g_y1 = g_y1.view(B, Nx, 1, Nz)
        a_y = (g_y1 - g_y0) / (2.0 * self.Ly)
        p0_y = a_y * (y ** 2) + g_y0 * y

        g_z0 = g_z0.view(B, Nx, Ny, 1)
        g_z1 = g_z1.view(B, Nx, Ny, 1)
        a_z = (g_z1 - g_z0) / (2.0 * self.Lz)
        p0_z = a_z * (z ** 2) + g_z0 * z

        p0 = p0_y + p0_z
        return self._remove_mean(p0), (g_y0, g_y1, g_z0, g_z1)

    def _forward_axis(self, X, bc, dim):
        if bc == BC_PERIODIC:
            return torch.fft.fft(X, dim=dim)
        if bc == BC_NEUMANN:
            return dct1(X, dim=dim)
        if bc == BC_DIRICHLET:
            return dst1(X, dim=dim)
        raise ValueError(f"Unknown BC: {bc}")

    def _inverse_axis(self, X, bc, dim):
        if bc == BC_PERIODIC:
            return torch.fft.ifft(X, dim=dim).real
        if bc == BC_NEUMANN:
            return idct1(X, dim=dim).real
        if bc == BC_DIRICHLET:
            return idst1(X, dim=dim).real
        raise ValueError(f"Unknown BC: {bc}")

    # ---- A^{-1}: yz mixed-BC Helmholtz inverse in spectral space ----
    def _Ainv_mixed_yz(self, gx, gy, gz):
        """
        Solve (fric + eta k^2) u_hat = g_hat
        with x periodic; y/z periodic, Neumann, or Dirichlet.
        """
        B, Nx, Ny, Nz = gx.shape
        dev = gx.device
        dtype = gx.dtype

        bc_y = self.bcs_u_tuple[1]
        bc_z = self.bcs_u_tuple[2]

        sy, sz = self._yz_slices()
        Ny_eff = Ny - 2 if bc_y == BC_DIRICHLET else Ny
        Nz_eff = Nz - 2 if bc_z == BC_DIRICHLET else Nz
        assert Ny_eff > 0 and Nz_eff > 0

        # enforce compatibility: g=0 at walls
        gxs = gx.clone(); gys = gy.clone(); gzs = gz.clone()
        if bc_y == BC_DIRICHLET:
            idx_y = torch.tensor([0, Ny - 1], device=dev)
            gxs.index_fill_(-2, idx_y, 0.0)
            gys.index_fill_(-2, idx_y, 0.0)
            gzs.index_fill_(-2, idx_y, 0.0)
        if bc_z == BC_DIRICHLET:
            idx_z = torch.tensor([0, Nz - 1], device=dev)
            gxs.index_fill_(-1, idx_z, 0.0)
            gys.index_fill_(-1, idx_z, 0.0)
            gzs.index_fill_(-1, idx_z, 0.0)

        # interior
        gxs = gxs[..., sy, sz]
        gys = gys[..., sy, sz]
        gzs = gzs[..., sy, sz]

        # transform: x FFT, y/z mixed
        Gx = torch.fft.fft(gxs, dim=-3)
        Gy = torch.fft.fft(gys, dim=-3)
        Gz = torch.fft.fft(gzs, dim=-3)

        Gx = self._forward_axis(Gx, bc_y, dim=-2)
        Gx = self._forward_axis(Gx, bc_z, dim=-1)
        Gy = self._forward_axis(Gy, bc_y, dim=-2)
        Gy = self._forward_axis(Gy, bc_z, dim=-1)
        Gz = self._forward_axis(Gz, bc_y, dim=-2)
        Gz = self._forward_axis(Gz, bc_z, dim=-1)

        # k grids
        ky, Ny_k, _ = _k_axis_for_bc(bc_y, Ny, self.Ly, dev, dtype)
        kz, Nz_k, _ = _k_axis_for_bc(bc_z, Nz, self.Lz, dev, dtype)
        if Ny_k != Ny_eff or Nz_k != Nz_eff:
            raise ValueError(
                f"k-axis length mismatch: ky={Ny_k} vs Ny_eff={Ny_eff}, "
                f"kz={Nz_k} vs Nz_eff={Nz_eff}"
            )
        ky = ky.view(1, 1, Ny_k, 1)
        kz = kz.view(1, 1, 1, Nz_k)

        n = torch.arange(Nx, device=dev, dtype=self.dtype)
        n = torch.where(n <= Nx // 2, n, n - Nx)
        kx_1d = (2.0 * math.pi / self.Lx) * n
        kx = kx_1d.view(1, Nx, 1, 1).expand(B, Nx, Ny_eff, Nz_eff)

        k2 = kx * kx + ky * ky + kz * kz

        denom = self.fric + self.eta * k2
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        Ux = Gx / denom
        Uy = Gy / denom
        Uz = Gz / denom

        # inverse: z, y, then x
        Ux = self._inverse_axis(Ux, bc_z, dim=-1)
        Ux = self._inverse_axis(Ux, bc_y, dim=-2)
        Uy = self._inverse_axis(Uy, bc_z, dim=-1)
        Uy = self._inverse_axis(Uy, bc_y, dim=-2)
        Uz = self._inverse_axis(Uz, bc_z, dim=-1)
        Uz = self._inverse_axis(Uz, bc_y, dim=-2)

        ux_in = torch.fft.ifft(Ux, dim=-3).real
        uy_in = torch.fft.ifft(Uy, dim=-3).real
        uz_in = torch.fft.ifft(Uz, dim=-3).real

        # embed back to full grid
        if self._ux_buf is None or self._ux_buf.shape != gx.shape or self._ux_buf.dtype != ux_in.dtype:
            self._ux_buf = torch.zeros_like(gx, dtype=ux_in.dtype)
            self._uy_buf = torch.zeros_like(gx, dtype=uy_in.dtype)
            self._uz_buf = torch.zeros_like(gx, dtype=uz_in.dtype)
        ux = self._ux_buf.zero_()
        uy = self._uy_buf.zero_()
        uz = self._uz_buf.zero_()
        ux[..., sy, sz] = ux_in
        uy[..., sy, sz] = uy_in
        uz[..., sy, sz] = uz_in
        return ux, uy, uz

    # ---- Schur complement CG for pressure on interior ----
    def _cg_solve_pressure(self, rhs, max_iter=200, tol=1e-7, p0=None, Sp0=None, p_init=None):
        rhs_full = rhs.contiguous()
        rhs_full = self._remove_mean_interior(rhs_full)
        sy, sz = self._yz_slices()
        if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
            print(f"[cg entry] rhs finite={torch.isfinite(rhs_full).all().item()}")
            if p_init is not None:
                p_init_finite = torch.isfinite(p_init).all().item()
                p_init_max = p_init.abs().max().item()
                print(f"[cg entry] p_init finite={p_init_finite} p_init maxabs={p_init_max:.3e}")

        def apply_Minv_interior(r_in):
            sy, sz = self._yz_slices()
            r_int = r_in[..., sy, sz]
            B, Nx, Ny_in, Nz_in = r_int.shape
            meta = (Nx, Ny_in, Nz_in, r_in.device, r_in.dtype)
            bc_y = self.bcs_u_tuple[1]
            bc_z = self.bcs_u_tuple[2]
            if self._precond_meta != meta:
                kx, _, _ = _k_axis_for_bc(BC_PERIODIC, Nx, self.Lx, r_in.device, r_in.dtype)
                ky, _, _ = _k_axis_for_bc(bc_y, self.Ny, self.Ly, r_in.device, r_in.dtype)
                kz, _, _ = _k_axis_for_bc(bc_z, self.Nz, self.Lz, r_in.device, r_in.dtype)
                kx = kx.view(1, Nx, 1, 1)
                ky = ky.view(1, 1, Ny_in, 1)
                kz = kz.view(1, 1, 1, Nz_in)
                k2 = kx * kx + ky * ky + kz * kz
                k2_safe = torch.where(k2 == 0, torch.ones_like(k2), k2)
                inv_diag = (self.fric + self.eta * k2) / k2_safe
                inv_diag = torch.where(k2 == 0, torch.zeros_like(inv_diag), inv_diag)
                self._precond_diag = inv_diag
                self._precond_meta = meta

            Z = torch.fft.fft(r_int, dim=-3)
            Z = self._forward_axis(Z, bc_y, dim=-2)
            Z = self._forward_axis(Z, bc_z, dim=-1)
            Z = Z * self._precond_diag
            Z = self._inverse_axis(Z, bc_z, dim=-1)
            Z = self._inverse_axis(Z, bc_y, dim=-2)
            Z = torch.fft.ifft(Z, dim=-3).real

            if self._tmp_full_buf is None or self._tmp_full_buf.shape != r_in.shape or self._tmp_full_buf.dtype != Z.dtype:
                self._tmp_full_buf = torch.zeros_like(r_in, dtype=Z.dtype)
            Z_full = self._tmp_full_buf.zero_()
            Z_full[..., sy, sz] = Z
            return self._remove_mean_interior(Z_full)

        rhs_norm = rhs_full[..., sy, sz].square().mean().sqrt().item()
        rhs_norm = max(rhs_norm, 1e-30)
        if rhs_norm < tol:
            self._pressure_iters = 0
            self._pressure_resid = 0.0
            return torch.zeros_like(rhs_full)

        def _grad_p_cg(p_in_):
            px = diff_1d(p_in_, self.dx_p, dim=-3, bc=BC_PERIODIC)
            py = diff_1d(p_in_, self.dy_p, dim=-2, bc=self.bcs_p_tuple[1])
            pz = diff_1d(p_in_, self.dz_p, dim=-1, bc=self.bcs_p_tuple[2])
            py = self._replace_boundary_deriv(py, p_in_, self.dy_p, dim=-2, bc=self.bcs_p_tuple[1], mode="zero")
            pz = self._replace_boundary_deriv(pz, p_in_, self.dz_p, dim=-1, bc=self.bcs_p_tuple[2], mode="zero")
            return px, py, pz

        def _div_u_cg(ux, uy, uz):
            divx = diff_1d(ux, self.dx_u, dim=-3, bc=BC_PERIODIC)
            divy = self._diff_axis(uy, self.dy_u, dim=-2, bc=self.bcs_u_tuple[1])
            divz = self._diff_axis(uz, self.dz_u, dim=-1, bc=self.bcs_u_tuple[2])
            divy = self._replace_boundary_deriv(divy, uy, self.dy_u, dim=-2, bc=self.bcs_u_tuple[1])
            divz = self._replace_boundary_deriv(divz, uz, self.dz_u, dim=-1, bc=self.bcs_u_tuple[2])
            return divx + divy + divz

        def apply_S_interior(p_in_):
            px, py, pz = _grad_p_cg(p_in_)
            if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
                p_max = p_in_[..., sy, sz].abs().max().item()
                px_max = px[..., sy, sz].abs().max().item()
                py_max = py[..., sy, sz].abs().max().item()
                pz_max = pz[..., sy, sz].abs().max().item()
                denom = max(p_max, 1e-30)
                print(
                    f"[grad p diag] "
                    f"p={p_max:.3e} px={px_max:.3e} py={py_max:.3e} pz={pz_max:.3e} "
                    f"px/p={px_max/denom:.3e} py/p={py_max/denom:.3e} pz/p={pz_max/denom:.3e}"
                )
            vx, vy, vz = self._Ainv_mixed_yz(px, py, pz)
            if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
                vx_max = vx[..., sy, sz].abs().max().item()
                vy_max = vy[..., sy, sz].abs().max().item()
                vz_max = vz[..., sy, sz].abs().max().item()
                print(f"[Ainv diag] vx={vx_max:.3e} vy={vy_max:.3e} vz={vz_max:.3e}")
            out = -_div_u_cg(vx, vy, vz)
            if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
                out_max = out[..., sy, sz].abs().max().item()
                print(f"[S diag] maxabs={out_max:.3e}")
            return out

        if p0 is not None:
            if Sp0 is None:
                Sp0 = self._remove_mean_interior(apply_S_interior(p0))
            rhs_full = self._remove_mean_interior(rhs_full - Sp0)

        if p_init is None:
            p_in = torch.zeros_like(rhs_full)
        else:
            p_in = self._remove_mean_interior(p_init.contiguous())
        if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
            print(f"[cg before S] p_in finite={torch.isfinite(p_in).all().item()}")
        r = self._remove_mean_interior(rhs_full - apply_S_interior(p_in))
        if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
            print(f"[cg after S] Sp finite={torch.isfinite(r).all().item()}")
        if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
            rhs_norm = rhs_full[..., sy, sz].square().mean().sqrt().item()
            rnorm = r[..., sy, sz].square().mean().sqrt().item()
            print(f"[cg diag] rhs_norm={rhs_norm:.3e} r0_norm={rnorm:.3e}")
        z = apply_Minv_interior(r)
        if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
            rr = (r[..., sy, sz] * z[..., sy, sz]).sum()
            rr_plain = (r[..., sy, sz] * r[..., sy, sz]).sum()
            print(f"[cg rr check] rr={rr.item():.3e} rr_plain={rr_plain.item():.3e}")
        d = z.clone()
        rz_old = (r[..., sy, sz] * z[..., sy, sz]).sum()
        if self.enable_diag and self._div_counter <= 3:
            Sd = self._remove_mean_interior(apply_S_interior(d))
            q = (d[..., sy, sz] * Sd[..., sy, sz]).sum().item()
            print(f"[schur quad] dSd={q:.3e}")

        iters = 0
        rnorm = r[..., sy, sz].square().mean().sqrt().item()
        nan_reported = False
        if rnorm / rhs_norm >= tol:
            for _ in range(max_iter):
                if not torch.isfinite(p_in).all() or not torch.isfinite(r).all() or not torch.isfinite(d).all():
                    if not nan_reported:
                        def _maxabs(x):
                            return x.abs().max().item()
                        print(
                            "[cg nan] "
                            f"iter={iters} p={_maxabs(p_in):.3e} r={_maxabs(r):.3e} d={_maxabs(d):.3e}"
                        )
                        nan_reported = True
                    break
                Ad = self._remove_mean_interior(apply_S_interior(d))
                if not torch.isfinite(Ad).all():
                    if not nan_reported:
                        print(f"[cg nan] iter={iters} Ad=nan")
                        nan_reported = True
                    break
                denom = (d[..., sy, sz] * Ad[..., sy, sz]).sum()
                if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
                    denom_val = denom.item()
                    rr_val = rz_old.item()
                    print(f"[cg iter diag] denom={denom_val:.3e} rr={rr_val:.3e}")
                if denom.abs() < 1e-30:
                    break

                alpha = rz_old / denom
                if self.enable_diag and self._div_counter <= 3 and (iters % 40 == 0):
                    alpha_val = alpha.item()
                    print(f"[cg alpha] iter={iters} denom={denom.item():.3e} alpha={alpha_val:.3e}")
                if not torch.isfinite(alpha):
                    print(f"[cg break] iter={iters} alpha=nan denom={denom.item():.3e}")
                    break
                if denom.item() <= 0.0:
                    print(f"[cg break] iter={iters} denom<=0 denom={denom.item():.3e}")
                    break
                if alpha.abs().item() > 1e6:
                    print(f"[cg break] iter={iters} alpha>1e6 alpha={alpha.item():.3e}")
                    break
                p_in = self._remove_mean_interior(p_in + alpha * d)
                r = self._remove_mean_interior(r - alpha * Ad)

                rnorm = r[..., sy, sz].square().mean().sqrt().item()
                if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail and iters % 5 == 0:
                    print(f"[cg rnorm] iter={iters} rnorm={rnorm:.3e}")
                if rnorm / rhs_norm < tol:
                    break

                z = apply_Minv_interior(r)
                if self.enable_diag and self._div_counter <= 3 and self.enable_diag_detail:
                    rr = (r[..., sy, sz] * z[..., sy, sz]).sum()
                    rr_plain = (r[..., sy, sz] * r[..., sy, sz]).sum()
                    print(f"[cg rr check] rr={rr.item():.3e} rr_plain={rr_plain.item():.3e}")
                rz_new = (r[..., sy, sz] * z[..., sy, sz]).sum()
                beta = rz_new / (rz_old + 1e-30)
                d = z + beta * d
                rz_old = rz_new
                iters += 1

        self._pressure_iters = iters
        self._pressure_resid = rnorm / rhs_norm

        # embed to full p
        if p0 is None:
            return self._remove_mean_interior(p_in)
        return self._remove_mean_interior(p_in + p0)

    # ---- Stokes solve: yz mixed BC ----
    def _stokes_solve(self, fx, fy, fz, do_diag=False, do_heavy=False):
        assert self.bcs_u["x"] == BC_PERIODIC

        if do_heavy:
            def _nan_inf(x):
                return torch.isnan(x).any().item() or torch.isinf(x).any().item()

            def _minmax(x):
                return f"min={x.min().item():.3e} max={x.max().item():.3e}"

            def _need_print(x, thresh=1e6):
                return _nan_inf(x) or x.abs().max().item() > thresh

            if _need_print(fx) or _need_print(fy) or _need_print(fz):
                print(
                    "[stokes diag] "
                    f"fx={_minmax(fx)} fy={_minmax(fy)} fz={_minmax(fz)}"
                )

        ufx, ufy, ufz = self._Ainv_mixed_yz(fx, fy, fz)
        rhs = self._div_u(ufx, ufy, ufz)
        if self.enable_diag and self._div_counter <= 3:
            rhs_max = rhs.abs().max().item()
            rhs_nan = torch.isnan(rhs).any().item()
            rhs_inf = torch.isinf(rhs).any().item()
            print(f"[rhs diag] maxabs={rhs_max:.3e} nan={rhs_nan} inf={rhs_inf}")
        if do_heavy:
            if _need_print(ufx) or _need_print(ufy) or _need_print(ufz) or _need_print(rhs):
                print(
                    "[stokes diag] "
                    f"ufx={_minmax(ufx)} ufy={_minmax(ufy)} "
                    f"ufz={_minmax(ufz)} rhs={_minmax(rhs)}"
                )
        if self.disable_lifting:
            p_warm = None if self.disable_p_warm else self._p_full_buf
            if p_warm is not None and not torch.isfinite(p_warm).all():
                p_warm = None
            if p_warm is not None:
                p_warm = self._remove_mean_interior(p_warm)
            p0 = None
            p0_flux = None
            p = self._cg_solve_pressure(rhs, p0=None, Sp0=None, p_init=p_warm)
            self._p_tilde_buf = None
        else:
            p_warm = None if self.disable_p_warm else self._p_tilde_buf
            if p_warm is not None and not torch.isfinite(p_warm).all():
                p_warm = None
            if p_warm is not None:
                p_warm = self._remove_mean_interior(p_warm)
            # Picard-style two-pass lifting: use u* then corrected u.
            p0, p0_flux = self._pressure_lifting(fx, fy, fz, ufx, ufy, ufz, u_cache=None)
            p = self._cg_solve_pressure(rhs, p0=p0, Sp0=None, p_init=p_warm)
            p = self._remove_mean(p)

            px, py, pz = self._grad_p(p)
            gx = -(fx + px)
            gy = -(fy + py)
            gz = -(fz + pz)
            ux0, uy0, uz0 = self._Ainv_mixed_yz(gx, gy, gz)

            p0, p0_flux = self._pressure_lifting(fx, fy, fz, ufx, ufy, ufz, u_cache=(ux0, uy0, uz0))
            rhs = self._div_u(ux0, uy0, uz0)
            p_init = self._remove_mean(p - p0)
            p = self._cg_solve_pressure(rhs, p0=p0, Sp0=None, p_init=p_init)
        p = self._remove_mean(p)

        if do_heavy:
            # heavy diagnostics: Schur residuals + BC flux mismatch
            rhs_full = self._remove_mean(rhs.contiguous())
            sy, sz = self._yz_slices()

            def _grad_p_cg(p_in_):
                px = diff_1d(p_in_, self.dx_p, dim=-3, bc=BC_PERIODIC)
                py = self._cdiff_interior(p_in_, self.dy_p, dim=-2)
                pz = self._cdiff_interior(p_in_, self.dz_p, dim=-1)
                return px, py, pz

            def _div_u_cg(ux, uy, uz):
                divx = diff_1d(ux, self.dx_u, dim=-3, bc=BC_PERIODIC)
                divy = self._cdiff_interior(uy, self.dy_u, dim=-2)
                divz = self._cdiff_interior(uz, self.dz_u, dim=-1)
                return divx + divy + divz

            def _apply_S_interior_local(p_in):
                px, py, pz = _grad_p_cg(p_in)
                vx, vy, vz = self._Ainv_mixed_yz(px, py, pz)
                return _div_u_cg(vx, vy, vz)

            if p0 is None:
                self._pressure_tilde_resid = None
            else:
                Sp0 = self._remove_mean(_apply_S_interior_local(p0))
                pt = self._remove_mean(p - p0)
                St_full = self._remove_mean(_apply_S_interior_local(pt))
                rhs_t = self._remove_mean(rhs_full - Sp0)
                res_t = (St_full - rhs_t)[..., sy, sz].square().mean().sqrt().item()
                denom_t = rhs_t[..., sy, sz].square().mean().sqrt().item()
                self._pressure_tilde_resid = res_t / max(denom_t, 1e-30)

            Sp_full = self._remove_mean(_apply_S_interior_local(p))
            res = (Sp_full - rhs_full)[..., sy, sz].square().mean().sqrt().item()
            denom = rhs_full[..., sy, sz].square().mean().sqrt().item()
            self._pressure_schur_resid = res / max(denom, 1e-30)

            # boundary flux mismatch for lifting: check p0 only (one-sided FD)
            if p0_flux is None:
                self._pressure_bc_resid = None
                self._pressure_bc_final_resid = None
            else:
                g_y0, g_y1, g_z0, g_z1 = p0_flux
                dy = self.dy_p
                dz = self.dz_p
                p0y0 = (-3.0 * p0[..., 0, :] + 4.0 * p0[..., 1, :] - p0[..., 2, :]) / (2.0 * dy)
                p0y1 = (3.0 * p0[..., -1, :] - 4.0 * p0[..., -2, :] + p0[..., -3, :]) / (2.0 * dy)
                p0z0 = (-3.0 * p0[..., :, 0] + 4.0 * p0[..., :, 1] - p0[..., :, 2]) / (2.0 * dz)
                p0z1 = (3.0 * p0[..., :, -1] - 4.0 * p0[..., :, -2] + p0[..., :, -3]) / (2.0 * dz)
                e_y0 = ((-p0y0) - g_y0.squeeze(-2)).square().mean().sqrt().item()
                e_y1 = (p0y1 - g_y1.squeeze(-2)).square().mean().sqrt().item()
                e_z0 = ((-p0z0) - g_z0.squeeze(-1)).square().mean().sqrt().item()
                e_z1 = (p0z1 - g_z1.squeeze(-1)).square().mean().sqrt().item()
                self._pressure_bc_resid = max(e_y0, e_y1, e_z0, e_z1)

                # boundary flux check for final p (one-sided FD)
                p_y0 = (-3.0 * p[..., 0, :] + 4.0 * p[..., 1, :] - p[..., 2, :]) / (2.0 * dy)
                p_y1 = (3.0 * p[..., -1, :] - 4.0 * p[..., -2, :] + p[..., -3, :]) / (2.0 * dy)
                p_z0 = (-3.0 * p[..., :, 0] + 4.0 * p[..., :, 1] - p[..., :, 2]) / (2.0 * dz)
                p_z1 = (3.0 * p[..., :, -1] - 4.0 * p[..., :, -2] + p[..., :, -3]) / (2.0 * dz)
                e_y0_f = ((-p_y0) - g_y0.squeeze(-2)).square().mean().sqrt().item()
                e_y1_f = (p_y1 - g_y1.squeeze(-2)).square().mean().sqrt().item()
                e_z0_f = ((-p_z0) - g_z0.squeeze(-1)).square().mean().sqrt().item()
                e_z1_f = (p_z1 - g_z1.squeeze(-1)).square().mean().sqrt().item()
                self._pressure_bc_final_resid = max(e_y0_f, e_y1_f, e_z0_f, e_z1_f)

            def _fmt(val):
                return "None" if val is None else f"{val:.3e}"
            def _bad(val, thresh=1e6):
                if val is None:
                    return False
                return math.isnan(val) or abs(val) > thresh

            if (
                _bad(self._pressure_schur_resid)
                or _bad(self._pressure_tilde_resid)
                or _bad(self._pressure_bc_resid)
                or _bad(self._pressure_bc_final_resid)
            ):
                print(
                    "[pressure diag] "
                    f"schur={_fmt(self._pressure_schur_resid)} "
                    f"tilde={_fmt(self._pressure_tilde_resid)} "
                    f"bc={_fmt(self._pressure_bc_resid)} "
                    f"bc_final={_fmt(self._pressure_bc_final_resid)}"
                )

        px, py, pz = self._grad_p(p)
        gx = -(fx + px)
        gy = -(fy + py)
        gz = -(fz + pz)

        ux, uy, uz = self._Ainv_mixed_yz(gx, gy, gz)
        if do_diag:
            divu = self._div_u(ux, uy, uz)
            sy, sz = self._yz_slices()
            self._divu_rms = divu[..., sy, sz].square().mean().sqrt().item()
            uy_wall = max(uy[..., 0, :].abs().max().item(), uy[..., -1, :].abs().max().item())
            uz_wall = max(uz[..., :, 0].abs().max().item(), uz[..., :, -1].abs().max().item())
            self._u_wall_max = max(uy_wall, uz_wall)
        self._u_cache = (ux.detach(), uy.detach(), uz.detach())
        if p0 is None:
            self._p_tilde_buf = None
        else:
            self._p_tilde_buf = (p - p0).detach()
        self._p_full_buf = p.detach()
        return ux, uy, uz

    def forward(self, fields, params):
        # increment counter (FIXED)
        self._div_counter += 1
        step = self._div_counter
        do_diag = self.enable_diag and (self.diag_every > 0) and (step % self.diag_every == 0)
        do_heavy = self.enable_diag and (self.heavy_diag_every > 0) and (step % self.heavy_diag_every == 0)

        Qxx = fields['Qxx']; Qxy = fields['Qxy']; Qxz = fields['Qxz']; Qyy = fields['Qyy']; Qyz = fields['Qyz']
        alpha = params['alpha']

        # Laplacian
        lap_Qxx = self._laplacian_Q(Qxx)
        lap_Qxy = self._laplacian_Q(Qxy)
        lap_Qxz = self._laplacian_Q(Qxz)
        lap_Qyy = self._laplacian_Q(Qyy)
        lap_Qyz = self._laplacian_Q(Qyz)

        # grad Q
        gx_Qxx, gy_Qxx, gz_Qxx = self._grad_Q(Qxx)
        gx_Qxy, gy_Qxy, gz_Qxy = self._grad_Q(Qxy)
        gx_Qxz, gy_Qxz, gz_Qxz = self._grad_Q(Qxz)
        gx_Qyy, gy_Qyy, gz_Qyy = self._grad_Q(Qyy)
        gx_Qyz, gy_Qyz, gz_Qyz = self._grad_Q(Qyz)

        # active stress sigma = - beta * alpha * Q
        beta = self.beta
        sig_xx = - (beta * alpha * Qxx)
        sig_xy = - (beta * alpha * Qxy)
        sig_xz = - (beta * alpha * Qxz)
        sig_yy = - (beta * alpha * Qyy)
        sig_yz = - (beta * alpha * Qyz)

        if self.enable_diag and self.enable_diag_detail:
            def _nan_inf(x):
                return torch.isnan(x).any().item() or torch.isinf(x).any().item()

            def _maxabs(x):
                return x.abs().max().item()

            def _need_print(x, thresh=1e6):
                return _nan_inf(x) or _maxabs(x) > thresh

            force_print = self._div_counter <= 3
            if (
                force_print
                or _need_print(sig_xx) or _need_print(sig_xy) or _need_print(sig_xz)
                or _need_print(sig_yy) or _need_print(sig_yz)
            ):
                print(
                    "[sig diag] "
                    f"xx={_maxabs(sig_xx):.3e} xy={_maxabs(sig_xy):.3e} xz={_maxabs(sig_xz):.3e} "
                    f"yy={_maxabs(sig_yy):.3e} yz={_maxabs(sig_yz):.3e}"
                )

        # transform sigma for x/y spectral derivatives where available
        sig_xx_h = self.trans_Q.forward_scalar(sig_xx)
        sig_xy_h = self.trans_Q.forward_scalar(sig_xy)
        sig_xz_h = self.trans_Q.forward_scalar(sig_xz)
        sig_yy_h = self.trans_Q.forward_scalar(sig_yy)
        sig_yz_h = self.trans_Q.forward_scalar(sig_yz)

        kx_fac = self.trans_Q.ax_x.deriv_factor()
        ky_fac = self.trans_Q.ax_y.deriv_factor()

        if kx_fac is not None:
            dxx = self.trans_Q.inverse_scalar(kx_fac * sig_xx_h).real
            dyx = self.trans_Q.inverse_scalar(kx_fac * sig_xy_h).real
            dzx = self.trans_Q.inverse_scalar(kx_fac * sig_xz_h).real
        else:
            dxx = diff_1d(sig_xx, self.dx_Q, dim=-3, bc=self.bcs_Q["x"])
            dyx = diff_1d(sig_xy, self.dx_Q, dim=-3, bc=self.bcs_Q["x"])
            dzx = diff_1d(sig_xz, self.dx_Q, dim=-3, bc=self.bcs_Q["x"])

        if ky_fac is not None:
            dxy = self.trans_Q.inverse_scalar(ky_fac * sig_xy_h).real
            dyy = self.trans_Q.inverse_scalar(ky_fac * sig_yy_h).real
            dzy = self.trans_Q.inverse_scalar(ky_fac * sig_yz_h).real
        else:
            dxy = diff_1d(sig_xy, self.dy_Q, dim=-2, bc=self.bcs_Q["y"])
            dyy = diff_1d(sig_yy, self.dy_Q, dim=-2, bc=self.bcs_Q["y"])
            dzy = diff_1d(sig_yz, self.dy_Q, dim=-2, bc=self.bcs_Q["y"])

        dxz = diff_1d(sig_xz, self.dz_Q, dim=-1, bc=self.bcs_Q["z"])
        dyz = diff_1d(sig_yz, self.dz_Q, dim=-1, bc=self.bcs_Q["z"])
        dzz = diff_1d(-sig_xx-sig_yy, self.dz_Q, dim=-1, bc=self.bcs_Q["z"])

        if self.enable_diag and self.enable_diag_detail:
            force_print = self._div_counter <= 3
            if (
                force_print
                or _need_print(dxx) or _need_print(dxy) or _need_print(dxz)
                or _need_print(dyx) or _need_print(dyy) or _need_print(dyz)
                or _need_print(dzx) or _need_print(dzy) or _need_print(dzz)
            ):
                print(
                    "[grad sig diag] "
                    f"dxx={_maxabs(dxx):.3e} dxy={_maxabs(dxy):.3e} dxz={_maxabs(dxz):.3e} "
                    f"dyx={_maxabs(dyx):.3e} dyy={_maxabs(dyy):.3e} dyz={_maxabs(dyz):.3e} "
                    f"dzx={_maxabs(dzx):.3e} dzy={_maxabs(dzy):.3e} dzz={_maxabs(dzz):.3e}"
                )

        # force = div(sigma)
        fx = (dxx + dxy + dxz) 
        fy = (dyx + dyy + dyz) 
        fz = (dzx + dzy + dzz) 

        ux, uy, uz = self._stokes_solve(fx, fy, fz, do_diag=do_diag, do_heavy=do_heavy)

        # derived quantities (grad u)
        if do_heavy:
            def _nan_inf(x):
                return torch.isnan(x).any().item() or torch.isinf(x).any().item()

            def _minmax(x):
                return f"min={x.min().item():.3e} max={x.max().item():.3e}"

            def _need_print(x, thresh=1e6):
                return _nan_inf(x) or x.abs().max().item() > thresh

            if _need_print(ux) or _need_print(uy) or _need_print(uz):
                print(
                    "[u diag] "
                    f"ux={_minmax(ux)} uy={_minmax(uy)} uz={_minmax(uz)}"
                )
        ux_x = diff_1d(ux, self.dx_u, dim=-3, bc=self.bcs_u["x"])
        ux_y = diff_1d(ux, self.dy_u, dim=-2, bc=self.bcs_u["y"])
        ux_z = diff_1d(ux, self.dz_u, dim=-1, bc=self.bcs_u["z"])

        uy_x = diff_1d(uy, self.dx_u, dim=-3, bc=self.bcs_u["x"])
        uy_y = diff_1d(uy, self.dy_u, dim=-2, bc=self.bcs_u["y"])
        uy_z = diff_1d(uy, self.dz_u, dim=-1, bc=self.bcs_u["z"])

        uz_x = diff_1d(uz, self.dx_u, dim=-3, bc=self.bcs_u["x"])
        uz_y = diff_1d(uz, self.dy_u, dim=-2, bc=self.bcs_u["y"])
        uz_z = diff_1d(uz, self.dz_u, dim=-1, bc=self.bcs_u["z"])
        if do_heavy:
            if _need_print(ux_x) or _need_print(ux_y) or _need_print(ux_z):
                print(
                    "[u grad diag] "
                    f"ux_x={_minmax(ux_x)} ux_y={_minmax(ux_y)} ux_z={_minmax(ux_z)}"
                )
            if _need_print(uy_x) or _need_print(uy_y) or _need_print(uy_z):
                print(
                    "[u grad diag] "
                    f"uy_x={_minmax(uy_x)} uy_y={_minmax(uy_y)} uy_z={_minmax(uy_z)}"
                )
            if _need_print(uz_x) or _need_print(uz_y) or _need_print(uz_z):
                print(
                    "[u grad diag] "
                    f"uz_x={_minmax(uz_x)} uz_y={_minmax(uz_y)} uz_z={_minmax(uz_z)}"
                )

        ux_y = self._replace_boundary_deriv(ux_y, ux, self.dy_u, dim=-2, bc=self.bcs_u["y"])
        ux_z = self._replace_boundary_deriv(ux_z, ux, self.dz_u, dim=-1, bc=self.bcs_u["z"])
        uy_y = self._replace_boundary_deriv(uy_y, uy, self.dy_u, dim=-2, bc=self.bcs_u["y"])
        uy_z = self._replace_boundary_deriv(uy_z, uy, self.dz_u, dim=-1, bc=self.bcs_u["z"])
        uz_y = self._replace_boundary_deriv(uz_y, uz, self.dy_u, dim=-2, bc=self.bcs_u["y"])
        uz_z = self._replace_boundary_deriv(uz_z, uz, self.dz_u, dim=-1, bc=self.bcs_u["z"])

        wxy = 0.5 * (ux_y - uy_x)
        wxz = 0.5 * (ux_z - uz_x)
        wyz = 0.5 * (uy_z - uz_y)

        Axx = ux_x
        Ayy = uy_y
        Axy = 0.5 * (ux_y + uy_x)
        Axz = 0.5 * (ux_z + uz_x)
        Ayz = 0.5 * (uy_z + uz_y)

        def _fftn_mixed(x, bcs):
            out = x
            for axis, bc in enumerate(bcs):
                out = _forward_1d_bc(out, bc, dim=-(len(bcs) - axis))
            return out

        def _deriv_bc_tuple(bcs_tuple, axis):
            bcs = list(bcs_tuple)
            bc = _as_bc(bcs[axis])
            if bc == BC_NEUMANN:
                bcs[axis] = BC_DIRICHLET
            elif bc == BC_DIRICHLET:
                bcs[axis] = BC_NEUMANN
            return tuple(bcs)

        derived = {
            "ux": ux, "uy": uy, "uz": uz,
            "wxy": wxy, "wxz": wxz, "wyz": wyz,
            "Axx": Axx, "Axy": Axy, "Axz": Axz, "Ayy": Ayy, "Ayz": Ayz,
            "gradx_Qxx": gx_Qxx, "grady_Qxx": gy_Qxx, "gradz_Qxx": gz_Qxx,
            "gradx_Qxy": gx_Qxy, "grady_Qxy": gy_Qxy, "gradz_Qxy": gz_Qxy,
            "gradx_Qxz": gx_Qxz, "grady_Qxz": gy_Qxz, "gradz_Qxz": gz_Qxz,
            "gradx_Qyy": gx_Qyy, "grady_Qyy": gy_Qyy, "gradz_Qyy": gz_Qyy,
            "gradx_Qyz": gx_Qyz, "grady_Qyz": gy_Qyz, "gradz_Qyz": gz_Qyz,
            "lap_Qxx": lap_Qxx, "lap_Qxy": lap_Qxy, "lap_Qxz": lap_Qxz,
            "lap_Qyy": lap_Qyy, "lap_Qyz": lap_Qyz,
        }

        bcs_by_field = {}
        for name in ("ux", "uy", "uz", "wxy", "wxz", "wyz", "Axx", "Axy", "Axz", "Ayy", "Ayz"):
            bcs_by_field[name] = self.bcs_u_tuple
        gradx_bcs = _deriv_bc_tuple(self.bcs_Q_tuple, 0)
        grady_bcs = _deriv_bc_tuple(self.bcs_Q_tuple, 1)
        gradz_bcs = _deriv_bc_tuple(self.bcs_Q_tuple, 2)
        for name in ("gradx_Qxx", "gradx_Qxy", "gradx_Qxz", "gradx_Qyy", "gradx_Qyz"):
            bcs_by_field[name] = gradx_bcs
        for name in ("grady_Qxx", "grady_Qxy", "grady_Qxz", "grady_Qyy", "grady_Qyz"):
            bcs_by_field[name] = grady_bcs
        for name in ("gradz_Qxx", "gradz_Qxy", "gradz_Qxz", "gradz_Qyy", "gradz_Qyz"):
            bcs_by_field[name] = gradz_bcs
        for name in ("lap_Qxx", "lap_Qxy", "lap_Qxz", "lap_Qyy", "lap_Qyz"):
            bcs_by_field[name] = self.bcs_Q_tuple

        grouped = {}
        for name, bcs in bcs_by_field.items():
            grouped.setdefault(bcs, []).append(name)

        derived_hat = {}
        for bcs, names in grouped.items():
            stacked = torch.stack([derived[name] for name in names], dim=0)
            out = _fftn_mixed(stacked, bcs)
            for i, name in enumerate(names):
                derived_hat[name] = out[i]

        return derived_hat


# ============================================================
# Nonlinear model (kept close to your current form)
# NOTE: bQ, cQ are passed via closure in __main__ below.
# ============================================================

class NonlinearModel(torch.nn.Module):
    def __init__(self, solver, bQ, cQ, lam=1.0, bcs_Q=None):
        super().__init__()
        self.solver = solver
        self.bQ = float(bQ)
        self.cQ = float(cQ)
        self.lam = float(lam)
        self.freeze_q = False
        if bcs_Q is None:
            self.bcs_Q_tuple = (BC_PERIODIC, BC_PERIODIC, BC_NEUMANN)
        else:
            self.bcs_Q_tuple = (bcs_Q.get("x", BC_PERIODIC), bcs_Q.get("y", BC_PERIODIC), bcs_Q.get("z", BC_NEUMANN))

        Nx, Ny, Nz = solver.shape
        device = solver.q2.device

        def eff_n(N, bc):
            return N - 2 if bc == BC_DIRICHLET else N

        def make_idx(N, bc):
            if bc == BC_PERIODIC:
                m = torch.arange(N, device=device)
                return torch.where(m <= N // 2, m, m - N)
            return torch.arange(N, device=device)

        Nx_eff = eff_n(Nx, self.bcs_Q_tuple[0])
        Ny_eff = eff_n(Ny, self.bcs_Q_tuple[1])
        Nz_eff = eff_n(Nz, self.bcs_Q_tuple[2])

        mx = make_idx(Nx_eff, self.bcs_Q_tuple[0])
        my = make_idx(Ny_eff, self.bcs_Q_tuple[1])
        mz = make_idx(Nz_eff, self.bcs_Q_tuple[2])

        kx_ok = (mx.abs() <= Nx_eff // 3).view(1, 1, Nx_eff, 1, 1)
        ky_ok = (my.abs() <= Ny_eff // 3).view(1, 1, 1, Ny_eff, 1)
        kz_ok = (mz.abs() <= Nz_eff // 3).view(1, 1, 1, 1, Nz_eff)

        mask = (kx_ok & ky_ok & kz_ok).to(torch.float32)
        self.register_buffer("dealias_mask", mask)
        self.dealias_enabled = True

    def forward(self, fields, params):
        if self.freeze_q:
            zeros = {}
            for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz"):
                zeros[name] = self.solver.fields.spectral[name].new_zeros(
                    self.solver.fields.spectral[name].shape
                )
            return zeros
        bQ = self.bQ
        cQ = self.cQ
        lam = self.lam

        Qxx = fields['Qxx']; Qxy = fields['Qxy']; Qxz = fields['Qxz']; Qyy = fields['Qyy']; Qyz = fields['Qyz']
        Qsq = Qxx**2 + 2*Qxy**2 + 2*Qxz**2 + (Qxx+Qyy)**2 + Qyy**2 + 2*Qyz**2

        ux,uy,uz = fields['ux'], fields['uy'], fields['uz']
        wxy,wxz,wyz = fields['wxy'], fields['wxz'], fields['wyz']

        gxQxx,gyQxx,gzQxx = fields['gradx_Qxx'], fields['grady_Qxx'], fields['gradz_Qxx']
        gxQxy,gyQxy,gzQxy = fields['gradx_Qxy'], fields['grady_Qxy'], fields['gradz_Qxy']
        gxQxz,gyQxz,gzQxz = fields['gradx_Qxz'], fields['grady_Qxz'], fields['gradz_Qxz']
        gxQyy,gyQyy,gzQyy = fields['gradx_Qyy'], fields['grady_Qyy'], fields['gradz_Qyy']
        gxQyz,gyQyz,gzQyz = fields['gradx_Qyz'], fields['grady_Qyz'], fields['gradz_Qyz']

        Axx,Axy,Axz,Ayy,Ayz = fields['Axx'], fields['Axy'], fields['Axz'], fields['Ayy'], fields['Ayz']
        trQA = (
            Qxx*Axx + 2*Qxy*Axy + 2*Qxz*Axz +
            Qyy*Ayy + 2*Qyz*Ayz + (Qxx+Qyy)*(Axx+Ayy)
        )

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

        out_stack = torch.stack([out0, out1, out2, out3, out4], dim=0)
        F = out_stack
        for axis, bc in enumerate(self.bcs_Q_tuple):
            F = _forward_1d_bc(F, bc, dim=-(3 - axis))
        if self.dealias_enabled:
            F = F * self.dealias_mask
        return {
            "Qxx": F[0],
            "Qxy": F[1],
            "Qxz": F[2],
            "Qyy": F[3],
            "Qyz": F[4],
        }


# ============================================================
# MAIN: build solver & run simulation (safe-guarded)
# ============================================================

if __name__ == "__main__":
    # -------------------------
    # Run parameters
    # -------------------------
    # Run parameters
    # -------------------------
    seed = 24
    dt = 1e-2
    steps = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batchsize = 1

    Nx, Ny, Nz = 512, 40, 40
    Lx, Ly, Lz = 128.0, 10.0, 10.0

    # -------------------------
    # BC configuration
    # -------------------------
    bcs_Q = dict(x=BC_PERIODIC, y=BC_NEUMANN, z=BC_NEUMANN)    # Q: x periodic, y/z Neumann
    bcs_u = dict(x=BC_PERIODIC, y=BC_DIRICHLET, z=BC_DIRICHLET)  # u: x periodic, y/z no-slip

    # -------------------------
    # Initialize solver (field-level BCs)
    # -------------------------
    bcs_fields = {"default": bcs_Q}
    for name in ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"]:
        bcs_fields[name] = bcs_Q
    for name in ["ux", "uy", "uz", "wxy", "wxz", "wyz", "Axx", "Axy", "Axz", "Ayy", "Ayz"]:
        bcs_fields[name] = bcs_u
    bcs_Q_tuple = (bcs_Q["x"], bcs_Q["y"], bcs_Q["z"])

    def _flip_bc(bcs_tuple, axis):
        b = list(bcs_tuple)
        if b[axis] == BC_NEUMANN:
            b[axis] = BC_DIRICHLET
        elif b[axis] == BC_DIRICHLET:
            b[axis] = BC_NEUMANN
        return tuple(b)

    gradx_bcs = _flip_bc(bcs_Q_tuple, 0)
    grady_bcs = _flip_bc(bcs_Q_tuple, 1)
    gradz_bcs = _flip_bc(bcs_Q_tuple, 2)

    for name in ["gradx_Qxx", "gradx_Qxy", "gradx_Qxz", "gradx_Qyy", "gradx_Qyz"]:
        bcs_fields[name] = dict(x=gradx_bcs[0], y=gradx_bcs[1], z=gradx_bcs[2])
    for name in ["grady_Qxx", "grady_Qxy", "grady_Qxz", "grady_Qyy", "grady_Qyz"]:
        bcs_fields[name] = dict(x=grady_bcs[0], y=grady_bcs[1], z=grady_bcs[2])
    for name in ["gradz_Qxx", "gradz_Qxy", "gradz_Qxz", "gradz_Qyy", "gradz_Qyz"]:
        bcs_fields[name] = dict(x=gradz_bcs[0], y=gradz_bcs[1], z=gradz_bcs[2])
    for name in ["lap_Qxx", "lap_Qxy", "lap_Qxz", "lap_Qyy", "lap_Qyz"]:
        bcs_fields[name] = bcs_Q

    solver = SpectralSolver(
        shape=(Nx, Ny, Nz),
        L=(Lx, Ly, Lz),
        dt=dt,
        device=device,
        batchsize=batchsize,
        bcs=bcs_fields,
    )

    # Init Q
    Qxx_0, Qxy_0, Qxz_0, Qyy_0, Qyz_0 = Q_init((Nx, Ny, Nz), seed=seed, bcs_Q=bcs_Q)

    # -------------------------
    # Model parameters (example)
    # -------------------------
    aQ = -1.0
    bQ = -6.0
    cQ =  6.0
    KQ =  1.0

    beta = -1.0
    fric = 0.1
    eta  = 1.0

    # Active stress strength
    alpha = torch.tensor(5.0, device=device)
    solver.model.parameters.new_param('alpha', alpha)

    # -------------------------
    # Build mixed k^2 for linear part consistent with bcs_Q
    # -------------------------
    q2_mixed = _mixed_k2(
        shape=(Nx, Ny, Nz),
        L=(Lx, Ly, Lz),
        bcs=bcs_Q,
        device=solver.q2.device,
        dtype=solver.q2.dtype
    )

    # -------------------------
    # Register fields
    # -------------------------
    solver.model.add_dynamic_field("Qxx", init=Qxx_0, L_hat=-(aQ + KQ * q2_mixed))
    solver.model.add_dynamic_field("Qxy", init=Qxy_0, L_hat=-(aQ + KQ * q2_mixed))
    solver.model.add_dynamic_field("Qxz", init=Qxz_0, L_hat=-(aQ + KQ * q2_mixed))
    solver.model.add_dynamic_field("Qyy", init=Qyy_0, L_hat=-(aQ + KQ * q2_mixed))
    solver.model.add_dynamic_field("Qyz", init=Qyz_0, L_hat=-(aQ + KQ * q2_mixed))

    for name in [
        "ux","uy","uz","wxy","wxz","wyz","Axx","Axy","Axz","Ayy","Ayz",
        "gradx_Qxx","grady_Qxx","gradz_Qxx",
        "gradx_Qxy","grady_Qxy","gradz_Qxy",
        "gradx_Qxz","grady_Qxz","gradz_Qxz",
        "gradx_Qyy","grady_Qyy","gradz_Qyy",
        "gradx_Qyz","grady_Qyz","gradz_Qyz",
        "lap_Qxx","lap_Qxy","lap_Qxz","lap_Qyy","lap_Qyz"
    ]:
        solver.model.add_static_field(name)

    # -------------------------
    # Set models
    # -------------------------
    solver.model.set_nonlinear_model(NonlinearModel(solver, bQ=bQ, cQ=cQ, lam=0.0, bcs_Q=bcs_Q))
    solver.model.set_static_compute_model(
        Static_compute_fn(
            solver,
            bcs_Q=bcs_Q,
            bcs_u=bcs_u,
            beta=beta,
            fric=fric,
            eta=eta,
            enable_diag=False,  # build阶段建议 False；build后你可以手动改 True
        )
    )

    # -------------------------
    # Build
    # -------------------------
    solver.build()

    solver.model.static_model.disable_lifting = False
    solver.model.static_model.disable_p_warm = True
    freeze_q = False
    if freeze_q:
        solver.model.static_model._p_full_buf = None
        solver.model.static_model._u_cache = None

    # If you want diagnostics during run:
    solver.model.static_model.enable_diag = False
    solver.model.static_model.heavy_diag_every = 0
    solver.model.static_model.enable_diag_detail = False
    solver.model.nlmodel.enable_diag = False
    diag_steps = 3

    # -------------------------
    # Main loop
    # -------------------------
    os.makedirs("data_H=10", exist_ok=True)
    start = time.time()

    for i in trange(steps):
        if i == diag_steps:
            solver.model.static_model.enable_diag = False
        if i % 10 == 0:
            Q_stack = torch.stack(
                [
                    solver.model.fields["Qxx"],
                    solver.model.fields["Qxy"],
                    solver.model.fields["Qxz"],
                    solver.model.fields["Qyy"],
                    solver.model.fields["Qyz"],
                ],
                dim=-1,
            )
            Q_field = Q_stack.detach().cpu().numpy()
            np.save(f"data_H=10/Q_{i}.npy", Q_field)
        if i % 100 == 0 and solver.model.static_model.enable_diag:
            # print(
            #     solver.model.static_model._pressure_schur_resid,
            #     solver.model.static_model._pressure_tilde_resid,
            #     solver.model.static_model._pressure_bc_resid,
            #     solver.model.static_model._pressure_bc_final_resid,
            #     solver.model.static_model._u_wall_max,
            #     solver.model.static_model._divu_rms,
            # )

            print("iters", solver.model.static_model._pressure_iters, "pres", solver.model.static_model._pressure_resid, "divu", solver.model.static_model._divu_rms)

        if freeze_q:
            stat_out = solver.model.compute_static()
            if isinstance(solver.model.fields.spectral, dict):
                for name in solver.model.fields.stat_names:
                    solver.model.fields.spectral[name] = stat_out[name]
            else:
                solver.model.fields.spectral[solver.model.fields.dyn_count:] = stat_out
            solver.model.fields.spatial = solver.model.fields.ifftn()
        else:
            solver.run(1)
        if i % 100 == 0:
            u_cache = solver.model.static_model._u_cache
            if u_cache is None:
                max_u = float("nan")
            else:
                ux, uy, uz = u_cache
                max_u = max(ux.abs().max().item(), uy.abs().max().item(), uz.abs().max().item())

            p_full = solver.model.static_model._p_full_buf
            max_p = float("nan") if p_full is None else p_full.abs().max().item()

            max_q = max(
                solver.model.fields["Qxx"].abs().max().item(),
                solver.model.fields["Qxy"].abs().max().item(),
                solver.model.fields["Qxz"].abs().max().item(),
                solver.model.fields["Qyy"].abs().max().item(),
                solver.model.fields["Qyz"].abs().max().item(),
            )

            dx = solver.model.static_model.dx_u
            dy = solver.model.static_model.dy_u
            dz = solver.model.static_model.dz_u
            cfl = max_u * solver.dt / min(dx, dy, dz)
            print(f"[diag100] max|u|={max_u:.3e} max|p|={max_p:.3e} max|Q|={max_q:.3e} CFL={cfl:.3e}")

    end = time.time()
    print(f"Elapsed time: {end - start:.3f}s")
