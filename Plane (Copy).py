# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import torch
from pssolver import SpectralSolver
from tqdm import trange
import numpy as np
import math
import os
from scipy.ndimage import gaussian_filter1d

# def Q_init(shape, seed=42):
#     # For nematics aligned along x in 3D, director n = (1,0,0)
#     Nx, Ny, Nz = shape
#     generator = torch.Generator().manual_seed(seed)
#     # n = (1,0,0)
#     Qxx = 0.5 + 0.01 * (2 * torch.rand((Nx, Ny, Nz), generator=generator) - 1)
#     Qxy = 0.01 * (2 * torch.rand((Nx, Ny, Nz), generator=generator) - 1)
#     Qxz = 0.01 * (2 * torch.rand((Nx, Ny, Nz), generator=generator) - 1)
#     Qyy = -0.5 + 0.01 * (2 * torch.rand((Nx, Ny, Nz), generator=generator) - 1)
#     Qyz = 0.01 * (2 * torch.rand((Nx, Ny, Nz), generator=generator) - 1)
#     return Qxx, Qxy, Qxz, Qyy, Qyz

# def Q_init(shape, seed=42):
#     Nx, Ny, Nz = shape
#     rng = np.random.default_rng(seed)
#     noise_amplitude = 0.01
#     noise_precision = 1e-5

#     def filtered_noise(size):
#         vals = rng.uniform(-noise_amplitude, noise_amplitude, size=size)
#         mask = (np.abs(vals) < noise_precision)
#         while np.any(mask):
#             vals[mask] = rng.uniform(-noise_amplitude, noise_amplitude, size=np.sum(mask))
#             mask = (np.abs(vals) < noise_precision)
#         return vals

#     S = 1.0
#     angle_theta_0 = np.arccos(0.0)  # = π/2，in-plane director
#     angle_phi_0 = 0.0               # 这里只是基准，用不到也没关系

#     # 两个缺陷在平面中的位置（对所有 z 层都相同）
#     defect1_x = 5 * Nx // 8
#     defect1_y = Ny // 2
#     defect2_x = 3 * Nx // 8
#     defect2_y = Ny // 2

#     Qxx = np.zeros((Nx, Ny, Nz), dtype=np.float32)
#     Qxy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
#     Qxz = np.zeros((Nx, Ny, Nz), dtype=np.float32)
#     Qyy = np.zeros((Nx, Ny, Nz), dtype=np.float32)
#     Qyz = np.zeros((Nx, Ny, Nz), dtype=np.float32)

#     for k in range(Nz):
#         for j in range(Ny):
#             for i in range(Nx):
#                 # 每个点自己的小噪声
#                 angle_theta_noise = filtered_noise(1)[0]
#                 angle_phi_noise   = filtered_noise(1)[0]

#                 # 对应这两个缺陷的角度场（对所有 z 相同）
#                 theta1 = math.atan2(j - defect1_y, i - defect1_x)
#                 theta2 = math.atan2(j - defect2_y, i - defect2_x)
#                 angle_phi_plane = 0.5 * theta1 - 0.5 * (theta2 + math.pi)

#                 # 真正用于 director 的 polar angles
#                 angle_theta = angle_theta_0 + angle_theta_noise
#                 angle_phi   = angle_phi_plane + angle_phi_noise

#                 nx = math.sin(angle_theta) * math.cos(angle_phi)
#                 ny = math.sin(angle_theta) * math.sin(angle_phi)
#                 nz = math.cos(angle_theta)

#                 Qxx[i, j, k] = S * (nx * nx - 1.0 / 3.0)
#                 Qyy[i, j, k] = S * (ny * ny - 1.0 / 3.0)
#                 Qxy[i, j, k] = S * (nx * ny)
#                 Qxz[i, j, k] = S * (nx * nz)
#                 Qyz[i, j, k] = S * (ny * nz)

#     return (
#         torch.from_numpy(Qxx),
#         torch.from_numpy(Qxy),
#         torch.from_numpy(Qxz),
#         torch.from_numpy(Qyy),
#         torch.from_numpy(Qyz)
#     )

# def Q_init(shape, seed=42, noise=0.01, sigma_xy=1.0, sigma_z=1.0):
#     """
#     新版匹配的初始化（周期边界条件版本）：
#     - 每层 z 放一对 +1/2/-1/2 缺陷（平面角场 φ 相同并沿 z 复制）
#     - 在 θ, φ 上叠加 3D 平滑噪声（x/y 用 sigma_xy，z 用 sigma_z）
#     - 周期边界条件：用 mode='wrap' 做平滑，并显式 enforce 周期性
#     """
#     Nx, Ny, Nz = shape
#     rng = np.random.default_rng(seed)

#     # 两个平面缺陷位置（对所有 z 相同）
#     defect1_x = 5 * Nx // 8
#     defect1_y = Ny // 2
#     defect2_x = 3 * Nx // 8
#     defect2_y = Ny // 2

#     # 角场 φ_plane（二维→三维复制）
#     X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")
#     theta1 = np.arctan2(Y - defect1_y, X - defect1_x)
#     theta2 = np.arctan2(Y - defect2_y, X - defect2_x)
#     angle_phi_plane_xy = 0.5 * theta1 - 0.5 * (theta2 + math.pi)
#     angle_phi_plane = np.repeat(angle_phi_plane_xy[:, :, None], Nz, axis=2)  # (Nx,Ny,Nz)

#     # 极角 θ0 = π/2（平面取向）
#     angle_theta_0 = math.acos(0.0)  # = π/2

#     # 平滑噪声 δθ, δφ（周期包裹）
#     delta_theta = rng.uniform(-1.0, 1.0, size=(Nx, Ny, Nz)).astype(np.float32)
#     delta_phi   = rng.uniform(-1.0, 1.0, size=(Nx, Ny, Nz)).astype(np.float32)

#     for arr in (delta_theta, delta_phi):
#         arr[:] = gaussian_filter1d(arr, sigma=sigma_xy, axis=0, mode="wrap")
#         arr[:] = gaussian_filter1d(arr, sigma=sigma_xy, axis=1, mode="wrap")
#         arr[:] = gaussian_filter1d(arr, sigma=sigma_z,  axis=2, mode="wrap")

#     def rescale_to_amp(arr, amp):
#         std = arr.std()
#         return (arr / std * amp) if std > 1e-12 else (arr * 0.0)

#     delta_theta = rescale_to_amp(delta_theta, noise)
#     delta_phi   = rescale_to_amp(delta_phi,   noise)

#     # 最终角度场
#     angle_theta = angle_theta_0 + delta_theta
#     angle_phi   = angle_phi_plane + delta_phi

#     # 计算 director 与 Q（S0=1）
#     S0 = 1.0
#     sin_theta = np.sin(angle_theta); cos_theta = np.cos(angle_theta)
#     cos_phi   = np.cos(angle_phi);   sin_phi   = np.sin(angle_phi)

#     nx = sin_theta * cos_phi
#     ny = sin_theta * sin_phi
#     nz = cos_theta

#     Qxx = S0 * (nx * nx - 1.0 / 3.0)
#     Qyy = S0 * (ny * ny - 1.0 / 3.0)
#     Qxy = S0 * (nx * ny)
#     Qxz = S0 * (nx * nz)
#     Qyz = S0 * (ny * nz)

#     # 显式周期 enforce：首尾一致
#     def enforce_periodic_all(arr):
#         arr[0, :, :]  = arr[-1, :, :]
#         arr[:, 0, :]  = arr[:, -1, :]
#         arr[:, :, 0]  = arr[:, :, -1]
#         return arr

#     for A in (Qxx, Qxy, Qxz, Qyy, Qyz):
#         enforce_periodic_all(A)

#     # 转 torch
#     return (
#         torch.from_numpy(Qxx.astype(np.float32)),
#         torch.from_numpy(Qxy.astype(np.float32)),
#         torch.from_numpy(Qxz.astype(np.float32)),
#         torch.from_numpy(Qyy.astype(np.float32)),
#         torch.from_numpy(Qyz.astype(np.float32)),
#     )

def Q_init(shape, seed=42, noise=0.01, sigma_xy=1.0, sigma_z=1.0):
    """
    Mixed-BC 版本初始化：
    - 全场 director 初始沿 +x 方向
    - 在 theta, phi 上叠加 3D 平滑噪声（x/y 用 sigma_xy，z 用 sigma_z）
    - x/y 用 periodic 平滑，z 用 reflect 平滑，近似匹配 Q 的 Neumann 边界
    """
    Nx, Ny, Nz = shape
    rng = np.random.default_rng(seed)

    # 基准方向：nx=1, ny=nz=0 <=> theta=pi/2, phi=0
    angle_theta_0 = math.pi / 2.0
    angle_phi_0 = 0.0

    # 平滑噪声 dtheta, dphi（周期包裹）
    delta_theta = rng.uniform(-1.0, 1.0, size=(Nx, Ny, Nz)).astype(np.float32)
    delta_phi = rng.uniform(-1.0, 1.0, size=(Nx, Ny, Nz)).astype(np.float32)

    for arr in (delta_theta, delta_phi):
        arr[:] = gaussian_filter1d(arr, sigma=sigma_xy, axis=0, mode="wrap")
        arr[:] = gaussian_filter1d(arr, sigma=sigma_xy, axis=1, mode="wrap")
        arr[:] = gaussian_filter1d(arr, sigma=sigma_z, axis=2, mode="reflect")

    def rescale_to_amp(arr, amp):
        std = arr.std()
        return (arr / std * amp) if std > 1e-12 else (arr * 0.0)

    delta_theta = rescale_to_amp(delta_theta, noise)
    delta_phi = rescale_to_amp(delta_phi, noise)

    # 最终角度场
    angle_theta = angle_theta_0 + delta_theta
    angle_phi = angle_phi_0 + delta_phi

    # 计算 director 与 Q（S0=1）
    S0 = 1.0
    sin_theta = np.sin(angle_theta)
    cos_theta = np.cos(angle_theta)
    cos_phi = np.cos(angle_phi)
    sin_phi = np.sin(angle_phi)

    nx = sin_theta * cos_phi
    ny = sin_theta * sin_phi
    nz = cos_theta

    Qxx = S0 * (nx * nx - 1.0 / 3.0)
    Qyy = S0 * (ny * ny - 1.0 / 3.0)
    Qxy = S0 * (nx * ny)
    Qxz = S0 * (nx * nz)
    Qyz = S0 * (ny * nz)

    return (
        torch.from_numpy(Qxx.astype(np.float32)),
        torch.from_numpy(Qxy.astype(np.float32)),
        torch.from_numpy(Qxz.astype(np.float32)),
        torch.from_numpy(Qyy.astype(np.float32)),
        torch.from_numpy(Qyz.astype(np.float32)),
    )


Q_BC = ("periodic", "periodic", "neumann")
U_BC = ("periodic", "periodic", "dirichlet")
P_BC = ("periodic", "periodic", "neumann")
ENABLE_DIAGNOSTICS = True
DIAGNOSTIC_INTERVAL = 10
SAVE_INTERVAL = 10


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
        # gzuz = fields.gradient('uz', axis=2)

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

class Static_compute_fn(torch.nn.Module):
    def __init__(self, solver, pressure_rel_tol=1e-6, pressure_max_iter=80):
        super().__init__()
        backend = solver.transform_backend
        spectral_dtype = backend.spectral_dtype
        real_dtype = backend.real_dtype
        device = solver.qx.device

        velocity_metadata = backend.get_metadata(U_BC)
        pressure_metadata = backend.get_metadata(P_BC)

        qx_xy, qy_xy = torch.meshgrid(
            velocity_metadata.axis_modes[0],
            velocity_metadata.axis_modes[1],
            indexing="ij",
        )
        kz_dirichlet = velocity_metadata.axis_modes[2]
        kz_neumann = pressure_metadata.axis_modes[2]
        nz = kz_dirichlet.numel()

        dirichlet_matrix = backend._get_matrix("dst", nz).to(device=device, dtype=real_dtype)
        neumann_matrix = backend._get_matrix("dct", nz).to(device=device, dtype=real_dtype)

        basis_neumann_to_dirichlet = neumann_matrix @ dirichlet_matrix.transpose(0, 1)
        basis_dirichlet_to_neumann = dirichlet_matrix @ neumann_matrix.transpose(0, 1)

        dz_dirichlet_to_neumann = torch.zeros((nz, nz), device=device, dtype=real_dtype)
        if nz > 1:
            idx = torch.arange(nz - 1, device=device)
            dz_dirichlet_to_neumann[idx, idx + 1] = kz_dirichlet[:-1]
        dz_neumann_to_dirichlet = -dz_dirichlet_to_neumann.transpose(0, 1)

        a_diag = fric + eta * (
            (qx_xy.square() + qy_xy.square()).unsqueeze(-1)
            + kz_dirichlet.square().view(1, 1, -1)
        )
        a_inv = 1.0 / a_diag

        xy_diag_base = torch.einsum(
            'ij,...j->...i',
            basis_neumann_to_dirichlet.square(),
            a_inv,
        )
        z_diag_base = torch.einsum(
            'ij,...j->...i',
            dz_neumann_to_dirichlet.square(),
            a_inv,
        )
        schur_diag = (qx_xy.square() + qy_xy.square()).unsqueeze(-1) * xy_diag_base + z_diag_base

        pressure_null_mask = torch.zeros((1, *qx_xy.shape, nz), device=device, dtype=torch.bool)
        pressure_null_mask[:, 0, 0, 0] = True
        schur_diag_safe = schur_diag.unsqueeze(0).clone()
        schur_diag_safe[pressure_null_mask] = 1.0

        self.register_buffer("ikx", (1j * qx_xy).view(1, *qx_xy.shape, 1).to(dtype=spectral_dtype))
        self.register_buffer("iky", (1j * qy_xy).view(1, *qy_xy.shape, 1).to(dtype=spectral_dtype))
        self.register_buffer("a_inv", a_inv.unsqueeze(0).to(dtype=real_dtype))
        self.register_buffer(
            "basis_neumann_to_dirichlet",
            basis_neumann_to_dirichlet.to(dtype=spectral_dtype),
        )
        self.register_buffer(
            "basis_dirichlet_to_neumann",
            basis_dirichlet_to_neumann.to(dtype=spectral_dtype),
        )
        self.register_buffer(
            "dz_neumann_to_dirichlet",
            dz_neumann_to_dirichlet.to(dtype=spectral_dtype),
        )
        self.register_buffer(
            "dz_dirichlet_to_neumann",
            dz_dirichlet_to_neumann.to(dtype=spectral_dtype),
        )
        self.register_buffer("schur_diag_safe", schur_diag_safe.to(dtype=real_dtype))
        self.register_buffer("pressure_null_mask", pressure_null_mask)

        self.pressure_rel_tol = pressure_rel_tol
        self.pressure_max_iter = pressure_max_iter
        self.pressure_guess = None
        self.last_pressure_iterations = 0
        self.last_pressure_residual = 0.0
        self.last_pressure_relative_residual = 0.0

    def _matmul_lastdim(self, tensor, matrix):
        return torch.matmul(tensor, matrix)

    def _project_pressure_gauge(self, p_hat):
        return p_hat.masked_fill(self.pressure_null_mask, 0)

    def _dirichlet_helmholtz_inverse(self, rhs_hat):
        return rhs_hat * self.a_inv

    def _pressure_to_dirichlet(self, p_hat):
        return self._matmul_lastdim(p_hat, self.basis_neumann_to_dirichlet)

    def _dirichlet_to_pressure(self, spectral):
        return self._matmul_lastdim(spectral, self.basis_dirichlet_to_neumann)

    def _pressure_grad_z(self, p_hat):
        return self._matmul_lastdim(p_hat, self.dz_neumann_to_dirichlet)

    def _velocity_div_z(self, velocity_hat):
        return self._matmul_lastdim(velocity_hat, self.dz_dirichlet_to_neumann)

    def _pressure_operator(self, p_hat):
        p_hat = self._project_pressure_gauge(p_hat)
        p_hat_dirichlet = self._pressure_to_dirichlet(p_hat)

        ux_hat = self._dirichlet_helmholtz_inverse(self.ikx * p_hat_dirichlet)
        uy_hat = self._dirichlet_helmholtz_inverse(self.iky * p_hat_dirichlet)
        uz_hat = self._dirichlet_helmholtz_inverse(self._pressure_grad_z(p_hat))

        divergence_hat = (
            self._dirichlet_to_pressure(self.ikx * ux_hat)
            + self._dirichlet_to_pressure(self.iky * uy_hat)
            + self._velocity_div_z(uz_hat)
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
        alpha = params['alpha']
        force_prefactor = beta * alpha

        gxQxx = fields.gradient('Qxx', axis=0)
        gyQxy = fields.gradient('Qxy', axis=1)
        gzQxz = fields.gradient('Qxz', axis=2)

        gxQxy = fields.gradient('Qxy', axis=0)
        gyQyy = fields.gradient('Qyy', axis=1)
        gzQyz = fields.gradient('Qyz', axis=2)

        gxQxz = fields.gradient('Qxz', axis=0)
        gyQyz = fields.gradient('Qyz', axis=1)
        gzQxx = fields.gradient('Qxx', axis=2)
        gzQyy = fields.gradient('Qyy', axis=2)

        force = force_prefactor * torch.stack([
            gxQxx + gyQxy + gzQxz,
            gxQxy + gyQyy + gzQyz,
            gxQxz + gyQyz - gzQxx - gzQyy,
        ])
        force_hat = fields.transform_tensor(force, U_BC)
        fx_hat, fy_hat, fz_hat = force_hat[0], force_hat[1], force_hat[2]

        ux_hat_free = self._dirichlet_helmholtz_inverse(fx_hat)
        uy_hat_free = self._dirichlet_helmholtz_inverse(fy_hat)
        uz_hat_free = self._dirichlet_helmholtz_inverse(fz_hat)

        provisional_divergence = (
            self._dirichlet_to_pressure(self.ikx * ux_hat_free)
            + self._dirichlet_to_pressure(self.iky * uy_hat_free)
            + self._velocity_div_z(uz_hat_free)
        )
        pressure_rhs = self._project_pressure_gauge(-provisional_divergence)
        pressure_hat = self._solve_pressure(pressure_rhs)
        pressure_hat_dirichlet = self._pressure_to_dirichlet(pressure_hat)

        ux_hat = ux_hat_free - self._dirichlet_helmholtz_inverse(self.ikx * pressure_hat_dirichlet)
        uy_hat = uy_hat_free - self._dirichlet_helmholtz_inverse(self.iky * pressure_hat_dirichlet)
        uz_hat = uz_hat_free - self._dirichlet_helmholtz_inverse(self._pressure_grad_z(pressure_hat))

        return torch.stack([ux_hat, uy_hat, uz_hat])

# seed = 24
# N = 64
# L = 64
# dt = 0.001
# steps = 20000
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# batch = 1

# solver = SpectralSolver(shape = (N,N,N), L=L, dt=dt, device=device, batch_size = batch)

# Qxx_0, Qxy_0, Qxz_0, Qyy_0, Qyz_0 = Q_init(shape = (N,N,N), seed = seed)

# # # --- Parameters ---
# aQ = -5
# bQ = 6
# cQ = 6
# KQ = 6

# beta = -1

seed = 24
dt = 1e-3
steps = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batchsize = 1

Nx, Ny, Nz = 256, 256, 120
Lx, Ly, Lz = 64.0, 64.0, 30.0

solver = SpectralSolver(shape=(Nx,Ny,Nz), L=(Lx,Ly,Lz), dt=dt, device=device, batchsize=batchsize)
Qxx_0, Qxy_0, Qxz_0, Qyy_0, Qyz_0 = Q_init((Nx,Ny,Nz), seed)

# Nematic 参数
aQ = -1.0
bQ = -6.0
cQ = 6.0
KQ = 1.0

# 流体/应力参数
beta = -1.0
fric = 0.1
eta  = 1.0

q2_Q = solver.get_q2(Q_BC)

# print("Max value of -(aQ + q2_Q * KQ):", torch.max(-(aQ + q2_Q * KQ)).item())

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
# Static_compute_fn now solves a mixed spectral Stokes projection using
# periodic FFTs in x/y and a z-collocation Schur solve consistent with U_BC.
solver.model.add_static_field("ux", boundary_conditions=U_BC)
solver.model.add_static_field("uy", boundary_conditions=U_BC)
solver.model.add_static_field("uz", boundary_conditions=U_BC)



# compiled_nl_model = torch.compile(NonlinearModel(solver),  mode="max-autotune")
# compiled_static_model = torch.compile(Static_compute_fn(solver), mode="max-autotune")
# solver.model.set_nonlinear_model(compiled_nl_model)
# solver.model.set_static_compute_model(compiled_static_model)
solver.model.set_nonlinear_model(NonlinearModel(solver))
solver.model.set_static_compute_model(Static_compute_fn(solver))

alpha = torch.tensor(5.0, device=device)
solver.model.parameters.new_param('alpha', alpha)

solver.build()
# print(solver.model.fields.dyn_count)
# print(solver.model.fields.name_to_idx)

div_history = []
output_dir = "test"
os.makedirs(output_dir, exist_ok=True)
start = time.time()
pbar = trange(steps)
for i in pbar:
    if i % 1 == 0:
        solver.refresh_static_fields()
        if ENABLE_DIAGNOSTICS and i % DIAGNOSTIC_INTERVAL == 0:
            div_max, div_rms, div_rel = divergence_stats(solver.model.fields)
            div_history.append((i, div_max, div_rms, div_rel))
            static_model = solver.model.static_model
            pbar.set_postfix(
                div_max=f"{div_max:.2e}",
                div_rms=f"{div_rms:.2e}",
                div_rel=f"{div_rel:.2e}",
                p_it=static_model.last_pressure_iterations,
                p_res_rel=f"{static_model.last_pressure_relative_residual:.2e}",
            )
        if i % SAVE_INTERVAL == 0:
            snapshot = torch.stack([
                solver.model.fields[name].detach().cpu()
                for name in ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"]
            ])  # shape -> (5, batch, N, N, N)
            snapshot = snapshot.permute(1, 2, 3, 4, 0)  # shape -> (batch, N, N, N, 5)
            np.save(f"{output_dir}/Q_{i}.npy", snapshot[0].numpy())

            u_snapshot = torch.stack([
                solver.model.fields[name].detach().cpu()
                for name in ["ux", "uy", "uz"]
            ])  # shape -> (3, batch, N, N, N)
            u_snapshot = u_snapshot.permute(1, 2, 3, 4, 0)  # shape -> (batch, N, N, N, 3)
            np.save(f"{output_dir}/u_{i}.npy", u_snapshot[0].numpy())
    # if i==steps//2:
    #     alpha.fill_(0)
    solver.run(1)

	    
end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
if ENABLE_DIAGNOSTICS:
    solver.refresh_static_fields()
    final_div_max, final_div_rms, final_div_rel = divergence_stats(solver.model.fields)
    static_model = solver.model.static_model
    print(
        "Final div(u) diagnostic: "
        f"max={final_div_max:.6e}, "
        f"rms={final_div_rms:.6e}, "
        f"relative={final_div_rel:.6e}"
    )
    print(
        "Final pressure solve diagnostic: "
        f"iterations={static_model.last_pressure_iterations}, "
        f"abs_residual={static_model.last_pressure_residual:.6e}, "
        f"rel_residual={static_model.last_pressure_relative_residual:.6e}"
    )
