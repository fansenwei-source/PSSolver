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
    周期边界条件版本初始化：
    - 全场 director 初始沿 +x 方向
    - 在 theta, phi 上叠加 3D 平滑噪声（x/y 用 sigma_xy，z 用 sigma_z）
    - 用 mode='wrap' 平滑，并显式 enforce 周期性
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
        arr[:] = gaussian_filter1d(arr, sigma=sigma_z, axis=2, mode="wrap")

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

class NonlinearModel(torch.nn.Module):
    def __init__(self, solver):
        super().__init__()
        self.qx = solver.qx
        self.qy = solver.qy
        self.q2 = solver.q2
        
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
        wxy  = fields['wxy']
        wxz  = fields['wxz']
        wyz  = fields['wyz']
        gxQxx = fields['gradx_Qxx']
        gyQxx = fields['grady_Qxx']
        gzQxx = fields['gradz_Qxx']
        gxQxy = fields['gradx_Qxy']
        gyQxy = fields['grady_Qxy']
        gzQxy = fields['gradz_Qxy']
        gxQxz = fields['gradx_Qxz']
        gyQxz = fields['grady_Qxz']
        gzQxz = fields['gradz_Qxz']
        gxQyy = fields['gradx_Qyy']
        gyQyy = fields['grady_Qyy']
        gzQyy = fields['gradz_Qyy']
        gxQyz = fields['gradx_Qyz']
        gyQyz = fields['grady_Qyz']
        gzQyz = fields['gradz_Qyz']
        
        Axx = fields['Axx']
        Axy = fields['Axy']
        Axz = fields['Axz']
        Ayy = fields['Ayy']
        Ayz = fields['Ayz']

        trQA = Qxx*Axx + 2*Qxy*Axy + 2*Qxz*Axz + Qyy*Ayy + 2*Qyz*Ayz + (Qxx + Qyy)*(Axx + Ayy)

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

        return torch.fft.fftn(torch.stack([out0, out1, out2, out3, out4]), dim=(-3, -2, -1))  

class Static_compute_fn(torch.nn.Module):
    def __init__(self, solver):
        super().__init__()
        qx = solver.qx
        qy = solver.qy
        qz = solver.qz
        q2 = solver.q2
        batch_size = solver.batchsize

        iqx = 1j * qx
        iqy = 1j * qy
        iqz = 1j * qz
        self.iqx = iqx
        self.iqy = iqy
        self.iqz = iqz

        # 3D projection operator P_ij = δ_ij - q_i q_j / q^2
        P = torch.zeros((3, 3, batch_size, *q2.shape), dtype=torch.cfloat, device=q2.device)
        P[0, 0] = 1 - (qx * qx) / q2
        P[0, 1] = - (qx * qy) / q2
        P[0, 2] = - (qx * qz) / q2
        P[1, 0] = - (qy * qx) / q2
        P[1, 1] = 1 - (qy * qy) / q2
        P[1, 2] = - (qy * qz) / q2
        P[2, 0] = - (qz * qx) / q2
        P[2, 1] = - (qz * qy) / q2
        P[2, 2] = 1 - (qz * qz) / q2
        self.P = P * 1/(fric + eta * q2)

        # For stress-to-force conversion in 3D
        sig_to_f = torch.zeros((3, 5, batch_size, *q2.shape), dtype=torch.cfloat, device=q2.device)
        sig_to_f[0, 0] = iqx
        sig_to_f[0, 1] = iqy
        sig_to_f[0, 2] = iqz
        sig_to_f[1, 1] = iqx
        sig_to_f[1, 3] = iqy
        sig_to_f[1, 4] = iqz
        sig_to_f[2, 0] = -iqz
        sig_to_f[2, 2] = iqx
        sig_to_f[2, 3] = -iqz
        sig_to_f[2, 4] = iqy
        
        self.sig_to_f = sig_to_f
        # self.P = self.P.unsqueeze(0).expand(batch_size, -1, -1, -1, -1).contiguous()

    ### avoid repeating same calculations
    def forward(self, fields, params): 
        Qxx = fields['Qxx']  
        Qxy = fields['Qxy']
        Qxz = fields['Qxz']
        Qyy = fields['Qyy']
        Qyz = fields['Qyz']
        alpha = params['alpha']
        Q = torch.stack([Qxx, Qxy, Qxz, Qyy, Qyz], dim=0) # shape -> (5,B, *shape)
        sig =  beta * alpha * Q  
        sig_hat = torch.fft.fftn(sig, dim=(-3, -2, -1))

        f_hat = torch.einsum('ijBxyz,jBxyz->iBxyz', self.sig_to_f, sig_hat)  
        u_hat = torch.einsum('ijBxyz,jBxyz->iBxyz', self.P, f_hat)

        # u_hat = 0*u_hat
        ux_hat = u_hat[0]
        uy_hat = u_hat[1]
        uz_hat = u_hat[2]
        # Compute vorticity components in Fourier space
        wxy_hat = -0.5 * (self.iqx * uy_hat - self.iqy * ux_hat)
        wxz_hat = -0.5 * (self.iqx * uz_hat - self.iqz * ux_hat)
        wyz_hat = -0.5 * (self.iqy * uz_hat - self.iqz * uy_hat)

        # Compute symmetric velocity gradient tensor components (A = sym(grad u))
        Axx_hat = self.iqx * ux_hat
        Axy_hat = 0.5 * (self.iqx * uy_hat + self.iqy * ux_hat)
        Axz_hat = 0.5 * (self.iqx * uz_hat + self.iqz * ux_hat)
        Ayy_hat = self.iqy * uy_hat
        Ayz_hat = 0.5 * (self.iqy * uz_hat + self.iqz * uy_hat)
        
        Qxx_hat = fields['Qxx.hat']
        Qxy_hat = fields['Qxy.hat']
        Qxz_hat = fields['Qxz.hat']
        Qyy_hat = fields['Qyy.hat']
        Qyz_hat = fields['Qyz.hat']

        gradx_Qxx_hat = self.iqx * Qxx_hat
        grady_Qxx_hat = self.iqy * Qxx_hat
        gradz_Qxx_hat = self.iqz * Qxx_hat

        gradx_Qxy_hat = self.iqx * Qxy_hat
        grady_Qxy_hat = self.iqy * Qxy_hat
        gradz_Qxy_hat = self.iqz * Qxy_hat

        gradx_Qxz_hat = self.iqx * Qxz_hat
        grady_Qxz_hat = self.iqy * Qxz_hat
        gradz_Qxz_hat = self.iqz * Qxz_hat

        gradx_Qyy_hat = self.iqx * Qyy_hat
        grady_Qyy_hat = self.iqy * Qyy_hat
        gradz_Qyy_hat = self.iqz * Qyy_hat

        gradx_Qyz_hat = self.iqx * Qyz_hat
        grady_Qyz_hat = self.iqy * Qyz_hat
        gradz_Qyz_hat = self.iqz * Qyz_hat
    
        return torch.stack([
            ux_hat,         # 0
            uy_hat,         # 1
            uz_hat,         # 2
            wxy_hat,        # 3
            wxz_hat,        # 4
            wyz_hat,        # 5
            Axx_hat,        # 6
            Axy_hat,        # 7
            Axz_hat,        # 8
            Ayy_hat,        # 9
            Ayz_hat,        # 10
            gradx_Qxx_hat,  # 11
            grady_Qxx_hat,  # 12
            gradz_Qxx_hat,  # 13
            gradx_Qxy_hat,  # 14
            grady_Qxy_hat,  # 15
            gradz_Qxy_hat,  # 16
            gradx_Qxz_hat,  # 17
            grady_Qxz_hat,  # 18
            gradz_Qxz_hat,  # 19
            gradx_Qyy_hat,  # 20
            grady_Qyy_hat,  # 21
            gradz_Qyy_hat,  # 22
            gradx_Qyz_hat,  # 23
            grady_Qyz_hat,  # 24
            gradz_Qyz_hat   # 25
        ])

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
dt = 1e-2
steps = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batchsize = 1

Nx, Ny, Nz = 256, 40, 40
Lx, Ly, Lz = 64.0, 10.0, 10.0

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

# print("Max value of -(aQ + solver.q2 * KQ):", torch.max(-(aQ + solver.q2 * KQ)).item())

# --- Add active fields ---
solver.model.add_dynamic_field(
    "Qxx",
    init = Qxx_0,
    L_hat = -( aQ + solver.q2 * KQ)
)
solver.model.add_dynamic_field(
    "Qxy",
    init =  Qxy_0,
    L_hat = -( aQ + solver.q2 * KQ)
)
solver.model.add_dynamic_field(
    "Qxz",
    init = Qxz_0,
    L_hat = -( aQ + solver.q2 * KQ)
)
solver.model.add_dynamic_field(
    "Qyy",
    init =  Qyy_0,
    L_hat = -( aQ + solver.q2 * KQ)
)
solver.model.add_dynamic_field(
    "Qyz",
    init =  Qyz_0,
    L_hat = -( aQ + solver.q2 * KQ)
)

# --- Add static fields ---
solver.model.add_static_field("ux")
solver.model.add_static_field("uy")
solver.model.add_static_field("uz")
solver.model.add_static_field("wxy")
solver.model.add_static_field("wxz")
solver.model.add_static_field("wyz")
solver.model.add_static_field("Axx")
solver.model.add_static_field("Axy")
solver.model.add_static_field("Axz")
solver.model.add_static_field("Ayy")
solver.model.add_static_field("Ayz")
solver.model.add_static_field("gradx_Qxx")
solver.model.add_static_field("grady_Qxx")
solver.model.add_static_field("gradz_Qxx")
solver.model.add_static_field("gradx_Qxy")
solver.model.add_static_field("grady_Qxy")
solver.model.add_static_field("gradz_Qxy")
solver.model.add_static_field("gradx_Qxz")
solver.model.add_static_field("grady_Qxz")
solver.model.add_static_field("gradz_Qxz")
solver.model.add_static_field("gradx_Qyy")
solver.model.add_static_field("grady_Qyy")
solver.model.add_static_field("gradz_Qyy")
solver.model.add_static_field("gradx_Qyz")
solver.model.add_static_field("grady_Qyz")
solver.model.add_static_field("gradz_Qyz")



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

traj = []
u_traj = []
start = time.time()
for i in trange(steps):
    if i % 1 == 0:
        solver.refresh_static_fields()
        snapshot = torch.stack([solver.model.fields[name].clone().detach().cpu() for name in ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"]]) # shape -> (5, batch, N, N, N)
        traj.append(snapshot)
        u_snapshot = torch.stack([solver.model.fields[name].clone().detach().cpu() for name in ["ux", "uy", "uz"]])  # shape -> (3, batch, N, N, N)
        u_traj.append(u_snapshot)
    # if i==steps//2:
    #     alpha.fill_(0)
    solver.run(1)

    
end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
traj = torch.stack(traj) # shape -> (time, 5, batch, N,N,N)
traj = traj.permute(2,0,3,4,5,1) # shape -> (batch, time, N,N,N, 5)
u_traj = torch.stack(u_traj) # shape -> (time, 3, batch, N,N,N)
u_traj = u_traj.permute(2,0,3,4,5,1) # shape -> (batch, time, N,N,N, 3)

# print(traj.shape)

os.makedirs("data01", exist_ok=True)
batch_traj = traj[0]  # shape -> (time, N,N,N, 5)
for t in range(batch_traj.shape[0]):
    np.save(f"data01/Q_{t}.npy", batch_traj[t].cpu().numpy())
batch_u_traj = u_traj[0]  # shape -> (time, N,N,N, 3)
for t in range(batch_u_traj.shape[0]):
    np.save(f"data01/u_{t}.npy", batch_u_traj[t].cpu().numpy())
# qxx = batch_traj[0]  # shape: (time, nx, ny)
# qxy = batch_traj[1]  # shape: (time, nx, ny)

# # Calculate scalar order parameter s
# s = torch.sqrt(qxx**2 + qxy**2)  # shape: (time, nx, ny)

# solver.visualize(data = s)
