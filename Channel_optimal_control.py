# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sys
import time
import torch
from pssolver import SpectralSolver, prepare_new_run_directory, write_run_metadata
from pssolver.models.active_nematics import (
    Q_convention_metadata,
    create_initial_condition,
    positive_equilibrium_S,
)
from pssolver.models.active_nematics.nematics3d_adapter import director_from_Q
from tqdm import trange
import numpy as np
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NEMATICS_SRC_CANDIDATES = (
    Path(__file__).resolve().parent / "Nematics3D" / "src",
    REPO_ROOT / "Nematics3D" / "src",
)
for local_nematics_src in LOCAL_NEMATICS_SRC_CANDIDATES:
    if local_nematics_src.exists():
        sys.path.insert(0, str(local_nematics_src))
        break

try:
    import nematics3d as n3d
except ImportError as exc:
    n3d = None
    NEMATICS3D_IMPORT_ERROR = exc
else:
    NEMATICS3D_IMPORT_ERROR = None


def fields_to_q_numpy(fields):
    q = torch.stack([
        fields[name].detach().cpu()
        for name in ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"]
    ])
    q = q.permute(1, 2, 3, 4, 0)
    return np.asarray(q[0].numpy(), dtype=np.float64)


def detect_single_loop_metrics(q5):
    if n3d is None:
        raise RuntimeError(
            "Nematics3D is required for loop control but could not be imported. "
            f"Original import error: {NEMATICS3D_IMPORT_ERROR}"
        )

    director = director_from_Q(q5)
    defects = n3d.defect_detect(
        director,
        threshold=CONTROL_DEFECT_THRESHOLD,
        is_boundary_periodic=CONTROL_PERIODIC_BOUNDARY,
        planes=CONTROL_DETECTION_PLANES,
    )
    box_size_periodic = [
        director.shape[axis] if CONTROL_PERIODIC_BOUNDARY[axis] else np.inf
        for axis in range(3)
    ]
    lines = n3d.defect_classify_into_lines(
        defects,
        box_size_periodic=box_size_periodic,
        grid_offset=np.zeros(3),
        grid_transform=np.eye(3),
    )
    loop_lines = [line for line in lines if getattr(line, "kind", None) == "loop"]
    if len(loop_lines) != 1 or len(lines) != 1:
        return {
            "is_single_loop": False,
            "defect_points": len(defects),
            "line_count": len(lines),
            "loop_count": len(loop_lines),
        }

    line = loop_lines[0]
    try:
        normal = np.asarray(line.act_calc_norm(), dtype=float)
    except Exception:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm < 1e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / normal_norm

    if line.calc_defect_num >= CONTROL_MIN_LINE_LENGTH:
        try:
            smooth = line.act_smooth(
                window_length=CONTROL_SMOOTH_WINDOW,
                min_line_length=CONTROL_MIN_LINE_LENGTH,
                is_window_warning=False,
            )
            coords = np.asarray(smooth.result, dtype=float)
        except Exception:
            coords = np.asarray(line.calc_defect_coords, dtype=float)
    else:
        coords = np.asarray(line.calc_defect_coords, dtype=float)

    center = coords.mean(axis=0)
    delta = coords - center
    normal_distance = delta @ normal
    radial_vectors = delta - normal_distance[:, None] * normal
    radius = float(np.sqrt(np.mean(np.sum(radial_vectors * radial_vectors, axis=1))))
    if not np.isfinite(radius) or radius <= 0.0:
        radius = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))

    center_wrapped = center.copy()
    for axis, periodic in enumerate(CONTROL_PERIODIC_BOUNDARY):
        if periodic:
            center_wrapped[axis] = center_wrapped[axis] % q5.shape[axis]

    return {
        "is_single_loop": True,
        "defect_points": len(defects),
        "line_count": len(lines),
        "loop_count": len(loop_lines),
        "line_points": line.calc_defect_num,
        "center": center_wrapped,
        "radius": radius,
        "normal": normal,
    }


class RadiusHoldController:
    def __init__(self, alpha_field, shape, device):
        self.alpha_field = alpha_field
        self.base_alpha_field = alpha_field.detach().clone()
        self.shape = shape
        self.device = device
        self.phase = "grow"
        self.confirm_count = 0
        self.target_center = None
        self.target_radius = None
        self.target_normal = None
        self.last_metrics = None

        x = torch.arange(shape[0], device=device, dtype=alpha_field.dtype).view(1, shape[0], 1, 1)
        y = torch.arange(shape[1], device=device, dtype=alpha_field.dtype).view(1, 1, shape[1], 1)
        z = torch.arange(shape[2], device=device, dtype=alpha_field.dtype).view(1, 1, 1, shape[2])
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z

    def observe_and_update(self, step, fields):
        q5 = fields_to_q_numpy(fields)
        metrics = detect_single_loop_metrics(q5)
        self.last_metrics = metrics

        if not metrics["is_single_loop"]:
            self.confirm_count = 0
            print(
                f"[control] step={step} phase={self.phase} "
                f"single_loop=0 lines={metrics['line_count']} loops={metrics['loop_count']}",
                flush=True,
            )
            return

        radius = metrics["radius"]
        center = metrics["center"]
        print(
            f"[control] step={step} phase={self.phase} "
            f"radius={radius:.3f} center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})",
            flush=True,
        )

        if self.phase == "grow":
            if radius >= CONTROL_START_RADIUS:
                self.confirm_count += 1
            else:
                self.confirm_count = 0

            if self.confirm_count >= CONTROL_CONFIRM_COUNT:
                self.phase = "hold"
                self.target_center = center.copy()
                self.target_radius = float(radius)
                self.target_normal = metrics["normal"].copy()
                self.apply_radius_hold(radius)
                print(
                    f"[control] switched_to_hold step={step} "
                    f"target_radius={self.target_radius:.3f} "
                    f"target_center=({self.target_center[0]:.2f},{self.target_center[1]:.2f},{self.target_center[2]:.2f})",
                    flush=True,
                )
            return

        self.apply_radius_hold(radius)

    def apply_radius_hold(self, current_radius):
        center = torch.as_tensor(self.target_center, device=self.device, dtype=self.alpha_field.dtype)
        normal = torch.as_tensor(self.target_normal, device=self.device, dtype=self.alpha_field.dtype)
        normal = normal / torch.linalg.vector_norm(normal).clamp_min(1e-12)

        dx = self.grid_x - center[0]
        if CONTROL_PERIODIC_BOUNDARY[0]:
            box_x = torch.as_tensor(float(self.shape[0]), device=self.device, dtype=self.alpha_field.dtype)
            dx = torch.remainder(dx + 0.5 * box_x, box_x) - 0.5 * box_x
        dy = self.grid_y - center[1]
        dz = self.grid_z - center[2]

        normal_distance = dx * normal[0] + dy * normal[1] + dz * normal[2]
        distance_sq = dx.square() + dy.square() + dz.square()
        radial_sq = (distance_sq - normal_distance.square()).clamp_min(0.0)
        radial_distance = torch.sqrt(radial_sq)

        target_radius = torch.as_tensor(self.target_radius, device=self.device, dtype=self.alpha_field.dtype)
        shell = torch.exp(
            -0.5 * ((radial_distance - target_radius) / CONTROL_RADIAL_WIDTH).square()
            -0.5 * (normal_distance / CONTROL_NORMAL_WIDTH).square()
        )

        radius_error = float(current_radius - self.target_radius) / max(self.target_radius, 1e-12)
        error_gain = CONTROL_QUENCH_GAIN if radius_error >= 0.0 else 0.5 * CONTROL_QUENCH_GAIN
        strength = CONTROL_QUENCH_BASE + error_gain * radius_error
        strength = min(CONTROL_QUENCH_MAX, max(0.0, strength))
        alpha_next = self.base_alpha_field[0] * (1.0 - strength * shell)
        alpha_next = alpha_next.clamp(min=0.0, max=CONTROL_ALPHA_MAX)
        self.alpha_field.copy_(alpha_next.expand_as(self.alpha_field))


Q_BC = ("periodic", "neumann", "neumann")
U_BC = ("periodic", "dirichlet", "dirichlet")
# Pressure is used as a modal Lagrange multiplier for incompressibility, not as
# an independently prescribed wall boundary condition. The DCT space supplies the
# pressure gauge/null mode and pairs with div(u) for the Schur complement.
PRESSURE_MODAL_BC = ("periodic", "neumann", "neumann")
SAVE_INTERVAL = 10

ENABLE_RADIUS_CONTROL = True
CONTROL_INTERVAL = 1
CONTROL_START_RADIUS = 8.0
CONTROL_CONFIRM_COUNT = 2
CONTROL_ALPHA0 = 5.0
CONTROL_ALPHA_MAX = CONTROL_ALPHA0
CONTROL_ALPHA_BOX_X = (246, 266)
CONTROL_ALPHA_BOX_Y = (10, 30)
CONTROL_ALPHA_BOX_Z = (10, 30)
CONTROL_QUENCH_BASE = 0.55
CONTROL_QUENCH_GAIN = 1.5
CONTROL_QUENCH_MAX = 0.95
CONTROL_RADIAL_WIDTH = 2.0
CONTROL_NORMAL_WIDTH = 3.0
CONTROL_SMOOTH_WINDOW = 9
CONTROL_MIN_LINE_LENGTH = 6
CONTROL_DEFECT_THRESHOLD = 0.0
CONTROL_PERIODIC_BOUNDARY = (True, False, False)
CONTROL_DETECTION_PLANES = (True, True, True)

if ENABLE_RADIUS_CONTROL and n3d is None:
    raise SystemExit(
        "ENABLE_RADIUS_CONTROL=True requires Nematics3D. Run this script with the "
        "Nematics3D environment or install its visualization dependencies. "
        f"Original import error: {NEMATICS3D_IMPORT_ERROR}"
    )


def make_local_alpha_box(batchsize, shape, device):
    Nx, Ny, Nz = shape
    alpha_field = torch.zeros((batchsize, Nx, Ny, Nz), device=device)
    x0, x1 = CONTROL_ALPHA_BOX_X
    y0, y1 = CONTROL_ALPHA_BOX_Y
    z0, z1 = CONTROL_ALPHA_BOX_Z
    alpha_field[:, x0:x1, y0:y1, z0:z1] = CONTROL_ALPHA0
    return alpha_field


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
            self._velocity_divergence_component(ux_hat_free, axis=0)
            + self._velocity_divergence_component(uy_hat_free, axis=1)
            + self._velocity_divergence_component(uz_hat_free, axis=2)
        )
        pressure_rhs = self._project_pressure_gauge(-provisional_divergence)
        pressure_hat = self._solve_pressure(pressure_rhs)

        ux_hat = ux_hat_free - self._dirichlet_helmholtz_inverse(self._pressure_gradient(pressure_hat, axis=0))
        uy_hat = uy_hat_free - self._dirichlet_helmholtz_inverse(self._pressure_gradient(pressure_hat, axis=1))
        uz_hat = uz_hat_free - self._dirichlet_helmholtz_inverse(self._pressure_gradient(pressure_hat, axis=2))

        return torch.stack([ux_hat, uy_hat, uz_hat, pressure_hat])

# seed = 24
# N = 64
# L = 64
# dt = 0.001
# steps = 20000
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# batch = 1

# solver = SpectralSolver(shape = (N,N,N), L=L, dt=dt, device=device, batch_size = batch)


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

Nx, Ny, Nz = 512, 40, 40
Lx, Ly, Lz = 128.0, 10.0, 10.0

solver = SpectralSolver(shape=(Nx,Ny,Nz), L=(Lx,Ly,Lz), dt=dt, device=device, batchsize=batchsize)
q_initial_condition = create_initial_condition(
    "aligned_x_smooth_noise",
    shape=(Nx, Ny, Nz),
    boundary_conditions=Q_BC,
    seed=seed,
    S_initial=2.0 / 3.0,
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
S_bulk = positive_equilibrium_S(aQ, bQ, cQ)

# 流体/应力参数
beta = -1.0
fric = 0.0
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
# ModalSaddleStokesCompute solves the pure FFT/DST/DCT modal saddle-point
# discretization through its pressure Schur complement.
solver.model.add_static_field("ux", boundary_conditions=U_BC)
solver.model.add_static_field("uy", boundary_conditions=U_BC)
solver.model.add_static_field("uz", boundary_conditions=U_BC)
solver.model.add_static_field("p", boundary_conditions=PRESSURE_MODAL_BC)



# compiled_nl_model = torch.compile(NonlinearModel(solver),  mode="max-autotune")
# compiled_static_model = torch.compile(ModalSaddleStokesCompute(solver), mode="max-autotune")
# solver.model.set_nonlinear_model(compiled_nl_model)
# solver.model.set_static_compute_model(compiled_static_model)
solver.model.set_nonlinear_model(NonlinearModel(solver))
solver.model.set_static_compute_model(ModalSaddleStokesCompute(solver))

alpha = make_local_alpha_box(batchsize, (Nx, Ny, Nz), device)
solver.model.parameters.new_param('alpha', alpha)
active_alpha_voxels = int((alpha[0] > 0).sum().item())
print(
    "[control] local_alpha_box "
    f"x={CONTROL_ALPHA_BOX_X} y={CONTROL_ALPHA_BOX_Y} z={CONTROL_ALPHA_BOX_Z} "
    f"alpha0={CONTROL_ALPHA0} active_voxels={active_alpha_voxels}"
)

solver.build()
# print(solver.model.fields.dyn_count)
# print(solver.model.fields.name_to_idx)

output_dir = prepare_new_run_directory("data_channel_optimal_control")
run_metadata = {
    "schema_version": 1,
    "script": "Channel_optimal_control.py",
    "solver": {
        "shape": [Nx, Ny, Nz],
        "lengths": [Lx, Ly, Lz],
        "dt": dt,
        "steps": steps,
        "save_interval": SAVE_INTERVAL,
    },
    "model": {
        "name": "active_nematics",
        "Q_convention": Q_convention_metadata(),
        "parameters": {
            "aQ": aQ,
            "bQ": bQ,
            "cQ": cQ,
            "KQ": KQ,
            "S_initial": 2.0 / 3.0,
            "S_bulk": S_bulk,
            "alpha": {
                "type": "box_with_radius_feedback",
                "initial": CONTROL_ALPHA0,
                "maximum": CONTROL_ALPHA_MAX,
                "box_x": CONTROL_ALPHA_BOX_X,
                "box_y": CONTROL_ALPHA_BOX_Y,
                "box_z": CONTROL_ALPHA_BOX_Z,
                "active_voxels": active_alpha_voxels,
                "control_enabled": ENABLE_RADIUS_CONTROL,
                "control_interval": CONTROL_INTERVAL,
            },
            "beta": beta,
            "fric": fric,
            "eta": eta,
        },
    },
    "boundary_conditions": {
        "Q": Q_BC,
        "velocity": U_BC,
        "pressure": PRESSURE_MODAL_BC,
    },
    "numerics": {
        "dealiasing": "none",
        "velocity_zero_mode": "wall_constrained",
        "pressure_solver": "modal_schur_complement",
    },
    "initial_condition": {
        "name": "aligned_x_smooth_noise",
        "seed": seed,
        "S_initial": 2.0 / 3.0,
    },
}
write_run_metadata(output_dir, run_metadata, status="running")
start = time.time()
pbar = trange(steps)
controller = RadiusHoldController(alpha, (Nx, Ny, Nz), device) if ENABLE_RADIUS_CONTROL else None


def save_step_state(i):
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

        p_snapshot = solver.model.fields["p"].detach().cpu()
        np.save(f"{output_dir}/p_{i}.npy", p_snapshot[0].numpy())


for i in pbar:
    # if i==steps//2:
    #     alpha.fill_(0)
    if controller is not None and i % CONTROL_INTERVAL == 0:
        controller.observe_and_update(i, solver.model.fields)
    solver.run(1, pre_update_callback=lambda _solver, _step, i=i: save_step_state(i))


end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")

write_run_metadata(output_dir, run_metadata, status="complete")
