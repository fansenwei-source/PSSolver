"""H=15 evolution probe for an extruded 2D defect field with weak twist.

This is an independent descendant of Plane.py's current Fig. 4 solver. It
runs one deliberately well-resolved diagnostic case rather than a parameter
scan. A saved 2D Q field containing neutral +/-1/2 defects is extruded into
vertical defect lines, then perturbed by a weak Neumann-compatible twist.

    A = H * sqrt(zeta / K),

The default A=25 lies above the reported crossover near A=18 for H=15.
"""

import argparse
import json
import math
from pathlib import Path
import time
import torch
from pssolver import SpectralSolver, write_run_metadata
from pssolver.models.active_nematics import (
    Q_convention_metadata,
    Q_magnitude,
    S_from_Q,
    create_initial_condition,
    positive_equilibrium_S,
    sample_periodic_neutral_defects_2d,
)
from tqdm import trange
import numpy as np


# Deliberately retain the free/Neumann Q wall condition from Plane.py for this
# first benchmark pass, as requested.  Fig. 4's no-slip data used strong planar
# anchoring, so this remaining model mismatch is recorded in metadata below.
Q_BC = ("periodic", "periodic", "neumann")
U_BC = ("periodic", "periodic", "dirichlet")
# Pressure is used as a modal Lagrange multiplier for incompressibility, not as
# an independently prescribed wall boundary condition. The DCT space supplies the
# pressure gauge/null mode and pairs with div(u) for the Schur complement.
PRESSURE_MODAL_BC = ("periodic", "periodic", "neumann")
ENABLE_DIAGNOSTICS = False
DIAGNOSTIC_INTERVAL = 100
SAVE_INTERVAL = 500
SAVE_HYDRODYNAMICS = False
ALIGNMENT_PARAMETER = 0.3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evolve an H=15 extruded 2D defect field with a weak twist seed."
        )
    )
    parser.add_argument(
        "--q2d-path",
        type=Path,
        default=None,
        help=(
            "Optional precomputed Q2D .npy file. If omitted, construct a "
            "neutral periodic analytic defect gas in memory."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_H15_twist_A25"))
    parser.add_argument("--activity-number", type=float, default=25.0)
    parser.add_argument("--height", type=float, default=15.0)
    parser.add_argument(
        "--parameterization",
        choices=("paper-window", "fixed-k"),
        default="paper-window",
        help=(
            "paper-window keeps K and zeta inside the paper's coefficient "
            "window; fixed-k uses --frank-k for every A."
        ),
    )
    parser.add_argument("--frank-k", type=float, default=0.01)
    parser.add_argument("--coefficient-min", type=float, default=0.01)
    parser.add_argument("--coefficient-max", type=float, default=0.05)
    parser.add_argument("--lx", type=float, default=100.0)
    parser.add_argument("--ly", type=float, default=100.0)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save-start-step", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--diagnostic-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--num-defect-pairs", type=int, default=6)
    parser.add_argument("--defect-min-separation", type=float, default=10.0)
    parser.add_argument("--defect-core-radius", type=float, default=1.5)
    parser.add_argument(
        "--initial-s",
        dest="S_initial",
        type=float,
        default=1.0 / 3.0,
        help="Initial S in Q=(3S/2)(nn-I/3); default 1/3.",
    )
    parser.add_argument("--background-angle", type=float, default=0.0)
    parser.add_argument("--ldg-a", type=float, default=0.0)
    parser.add_argument("--ldg-b", type=float, default=-0.3)
    parser.add_argument("--ldg-c", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=2.94)
    parser.add_argument("--flow-alignment", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=2.0 / 3.0)
    parser.add_argument("--friction", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument(
        "--twist-amplitude",
        type=float,
        default=0.01,
        help="RMS layer-rotation angle in radians.",
    )
    parser.add_argument(
        "--twist-modes",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="Positive Neumann/DCT mode indices used for the z-dependent twist.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--save-hydrodynamics", action="store_true")
    parser.add_argument(
        "--save-pressure",
        action="store_true",
        help="Save pressure snapshots in addition to Q and optional velocity.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved parameters without allocating the solver.",
    )
    args = parser.parse_args()
    positive_values = {
        "activity_number": args.activity_number,
        "height": args.height,
        "frank_k": args.frank_k,
        "coefficient_min": args.coefficient_min,
        "coefficient_max": args.coefficient_max,
        "lx": args.lx,
        "ly": args.ly,
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "dt": args.dt,
        "steps": args.steps,
        "save_interval": args.save_interval,
        "diagnostic_interval": args.diagnostic_interval,
        "gamma": args.gamma,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        parser.error(f"these values must be positive: {invalid}")
    if args.twist_amplitude < 0:
        parser.error("--twist-amplitude must be non-negative")
    if args.num_defect_pairs <= 0:
        parser.error("--num-defect-pairs must be positive")
    if args.defect_min_separation <= 0:
        parser.error("--defect-min-separation must be positive")
    if args.defect_core_radius <= 0:
        parser.error("--defect-core-radius must be positive")
    if args.S_initial <= 0:
        parser.error("--initial-s must be positive")
    if args.coefficient_min >= args.coefficient_max:
        parser.error("--coefficient-min must be smaller than --coefficient-max")
    if args.save_start_step < 0 or args.save_start_step > args.steps:
        parser.error("--save-start-step must lie between 0 and --steps")
    if not math.isclose(args.height, 15.0, rel_tol=0.0, abs_tol=1e-12):
        parser.error("this diagnostic script fixes H=Lz=15")
    if args.q2d_path is not None and not args.q2d_path.is_file():
        parser.error(f"--q2d-path does not exist or is not a file: {args.q2d_path}")
    return args


def load_canonical_q2d_input(
    q_path,
    *,
    expected_shape,
    expected_S_bulk,
    expected_S_initial,
):
    """Load a Q2D field only when its active-nematic metadata is canonical."""
    q_path = Path(q_path)
    q_values = np.load(q_path, mmap_mode="r")
    accepted_shapes = {
        (*expected_shape, 5),
        (*expected_shape, 1, 5),
    }
    if q_values.shape not in accepted_shapes:
        raise ValueError(
            f"Expected Q2D shape {(*expected_shape, 5)} or "
            f"{(*expected_shape, 1, 5)}, got {q_values.shape} from {q_path}."
        )
    if not np.isfinite(q_values).all():
        raise ValueError(f"Q2D tensor contains non-finite values: {q_path}")

    metadata_path = q_path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Canonical Q2D input metadata is required at "
            f"{metadata_path}. Regenerate this field with the active-nematics "
            "model instead of relying on a legacy convention fallback."
        )
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(source_metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object.")
    if source_metadata.get("schema_version") != 1:
        raise ValueError(f"{metadata_path} must declare schema_version=1.")
    model_metadata = source_metadata.get("model")
    if not isinstance(model_metadata, dict):
        raise ValueError(f"{metadata_path} is missing model metadata.")
    if model_metadata.get("name") != "active_nematics":
        raise ValueError(
            f"{metadata_path} must declare model.name='active_nematics'."
        )

    convention = model_metadata.get("Q_convention")
    expected_convention = Q_convention_metadata()
    if convention != expected_convention:
        raise ValueError(
            f"{metadata_path} does not declare the complete canonical Q "
            f"convention {expected_convention!r}."
        )

    parameters = model_metadata.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{metadata_path} is missing model.parameters.")
    missing_order_parameters = {
        name for name in ("S_initial", "S_bulk") if name not in parameters
    }
    if missing_order_parameters:
        raise ValueError(
            f"{metadata_path} is missing model.parameters fields "
            f"{sorted(missing_order_parameters)}."
        )

    declared_order = {}
    for name in ("S_initial", "S_bulk"):
        raw_value = parameters[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(
                f"{metadata_path} has non-numeric {name}={raw_value!r}."
            )
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{metadata_path} has invalid {name}={value!r}.")
        declared_order[name] = value
    declared_S_initial = declared_order["S_initial"]
    declared_S_bulk = declared_order["S_bulk"]

    diagnostic_values = q_values[..., 0, :] if q_values.ndim == 4 else q_values
    S_values = np.asarray(S_from_Q(diagnostic_values))
    magnitudes = np.asarray(Q_magnitude(diagnostic_values))
    ordered_cutoff = float(np.quantile(magnitudes, 0.75))
    observed_S_initial = float(np.median(S_values[magnitudes >= ordered_cutoff]))
    declared_initial_tolerance = max(5.0e-3, 0.05 * declared_S_initial)
    expected_initial_tolerance = max(5.0e-3, 0.05 * expected_S_initial)
    if abs(observed_S_initial - declared_S_initial) > declared_initial_tolerance:
        raise ValueError(
            f"{q_path} has ordered-region median S={observed_S_initial:.8g}, "
            f"inconsistent with metadata S_initial={declared_S_initial:.8g}."
        )
    if not np.isclose(
        declared_S_bulk,
        expected_S_bulk,
        rtol=0.05,
        atol=5.0e-3,
    ):
        raise ValueError(
            f"{q_path} declares S_bulk={declared_S_bulk:.8g}, but this H15 "
            f"run uses S_bulk={expected_S_bulk:.8g}."
        )
    if not np.isclose(
        declared_S_initial,
        expected_S_initial,
        rtol=0.05,
        atol=5.0e-3,
    ):
        raise ValueError(
            f"{q_path} declares S_initial={declared_S_initial:.8g}, but this "
            f"H15 run uses S_initial={expected_S_initial:.8g}."
        )
    if (
        abs(observed_S_initial - expected_S_initial)
        > expected_initial_tolerance
    ):
        raise ValueError(
            f"{q_path} has ordered-region median S={observed_S_initial:.8g}, "
            f"but this H15 run uses S_initial={expected_S_initial:.8g}."
        )

    return q_values, {
        "path": str(q_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "shape": list(q_values.shape),
        "Q_convention": convention,
        "S_initial": declared_S_initial,
        "S_bulk": declared_S_bulk,
        "observed_ordered_S": observed_S_initial,
    }


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


def wall_normal_momentum_stats(fields, params):
    """Check normal momentum balance at the two z walls for the modal pressure."""
    alpha = params['alpha']
    force_prefactor = beta * alpha

    gxQxz = fields.gradient('Qxz', axis=0)
    gyQyz = fields.gradient('Qyz', axis=1)
    gzQxx = fields.gradient('Qxx', axis=2)
    gzQyy = fields.gradient('Qyy', axis=2)
    fz = force_prefactor * (gxQxz + gyQyz - gzQxx - gzQyy)

    lap_uz = fields.laplacian('uz')
    dzp = fields.gradient('p', axis=2)
    residual = dzp - (fz + eta * lap_uz - fric * fields['uz'])

    wall_residual = torch.stack([residual[..., 0], residual[..., -1]], dim=-1)
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

        lam = ALIGNMENT_PARAMETER
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

    u is represented in the Dirichlet/DST velocity space in z. p is represented
    in a DCT multiplier space so that D u and pressure test functions live in the
    same modal space. This should be interpreted as the pressure space of the
    saddle-point discretization, not as a physical homogeneous Neumann pressure
    wall condition.
    """

    def __init__(self, solver, pressure_rel_tol=1e-6, pressure_max_iter=80):
        super().__init__()
        backend = solver.transform_backend
        spectral_dtype = backend.spectral_dtype
        real_dtype = backend.real_dtype
        device = solver.qx.device

        velocity_metadata = backend.get_metadata(U_BC)
        pressure_metadata = backend.get_metadata(PRESSURE_MODAL_BC)

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
        self.last_pressure_hat = None
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
        self.last_pressure_hat = pressure_hat.detach()
        pressure_hat_dirichlet = self._pressure_to_dirichlet(pressure_hat)

        ux_hat = ux_hat_free - self._dirichlet_helmholtz_inverse(self.ikx * pressure_hat_dirichlet)
        uy_hat = uy_hat_free - self._dirichlet_helmholtz_inverse(self.iky * pressure_hat_dirichlet)
        uz_hat = uz_hat_free - self._dirichlet_helmholtz_inverse(self._pressure_grad_z(pressure_hat))

        return torch.stack([ux_hat, uy_hat, uz_hat, pressure_hat])

args = parse_args()

seed = args.seed
dt = args.dt
steps = args.steps
device = (
    "cuda" if args.device == "auto" and torch.cuda.is_available()
    else "cpu" if args.device == "auto"
    else args.device
)
batchsize = 1

Nx, Ny, Nz = args.nx, args.ny, args.nz
Lx, Ly, Lz = args.lx, args.ly, args.height
SAVE_INTERVAL = args.save_interval
SAVE_START_STEP = args.save_start_step
DIAGNOSTIC_INTERVAL = args.diagnostic_interval
ENABLE_DIAGNOSTICS = args.diagnostics
SAVE_HYDRODYNAMICS = args.save_hydrodynamics
SAVE_PRESSURE = args.save_pressure
ALIGNMENT_PARAMETER = args.flow_alignment

# Resolve the one-variable A scan into physical K and zeta values.  In
# paper-window mode one coefficient is held at the lower edge of the paper's
# [0.01, 0.05] interval while the other supplies zeta/K=(A/H)^2.
activity_ratio = (args.activity_number / args.height) ** 2
if args.parameterization == "paper-window":
    ratio_min = args.coefficient_min / args.coefficient_max
    ratio_max = args.coefficient_max / args.coefficient_min
    if not ratio_min <= activity_ratio <= ratio_max:
        activity_min = args.height * ratio_min**0.5
        activity_max = args.height * ratio_max**0.5
        raise ValueError(
            f"A={args.activity_number} is outside the paper-window interval "
            f"[{activity_min:.6g}, {activity_max:.6g}] for H={args.height}"
        )
    if activity_ratio <= 1.0:
        zeta = args.coefficient_min
        frank_k = zeta / activity_ratio
    else:
        frank_k = args.coefficient_min
        zeta = frank_k * activity_ratio
else:
    frank_k = args.frank_k
    zeta = frank_k * activity_ratio

# For Q = 3 S (nn-I/3)/2 and S_eq=1/3, the paper's one-constant
# relation is L1 = 2 K.  Only molecular-field relaxation is divided by
# gamma; advection and flow alignment in NonlinearModel are not.
landau_l1 = 2.0 * frank_k
aQ = args.ldg_a / args.gamma
bQ = args.ldg_b / args.gamma
cQ = args.ldg_c / args.gamma
KQ = landau_l1 / args.gamma
S_bulk = positive_equilibrium_S(args.ldg_a, args.ldg_b, args.ldg_c)

# The active stress in this code is beta * alpha * Q.  beta=-1 therefore
# matches the paper's -zeta Q convention when alpha=zeta.
alpha_value = zeta
beta = args.beta
fric = args.friction
eta = args.eta
q2d_source = "file" if args.q2d_path is not None else "analytic_periodic_defect_gas_2d"
q_2d = None
q2d_input_summary = None
if args.q2d_path is not None:
    q_2d, q2d_input_summary = load_canonical_q2d_input(
        args.q2d_path,
        expected_shape=(Nx, Ny),
        expected_S_bulk=S_bulk,
        expected_S_initial=args.S_initial,
    )

metadata = {
    "schema_version": 1,
    "script": "Plane_H15_twist_evolution.py",
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
            "alpha": alpha_value,
            "beta": beta,
            "S_initial": args.S_initial,
            "S_bulk": S_bulk,
            "fric": fric,
            "eta": eta,
            "flow_alignment": ALIGNMENT_PARAMETER,
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
    "experiment": "H=15 vertical defect lines with weak z-dependent twist",
    "source_solver": "Plane.py / Plane_fig4_benchmark.py",
    "parameterization": args.parameterization,
    "activity_number": args.activity_number,
    "activity_number_definition": "H*sqrt(zeta/K)",
    "zeta": zeta,
    "frank_k": frank_k,
    "landau_l1": landau_l1,
    "height": Lz,
    "domain": [Lx, Ly, Lz],
    "shape": [Nx, Ny, Nz],
    "dt": dt,
    "steps": steps,
    "save_start_step": SAVE_START_STEP,
    "save_interval": SAVE_INTERVAL,
    "diagnostic_interval": DIAGNOSTIC_INTERVAL,
    "seed": seed,
    "device": device,
    "q_boundary_conditions": Q_BC,
    "S_initial": args.S_initial,
    "S_bulk": S_bulk,
    "velocity_boundary_conditions": U_BC,
    "q_wall_model_note": "free/Neumann; differs from Fig. 4 strong planar anchoring",
    "ldg_coefficients": {"A": args.ldg_a, "B": args.ldg_b, "C": args.ldg_c},
    "gamma": args.gamma,
    "flow_alignment": ALIGNMENT_PARAMETER,
    "eta": eta,
    "friction": fric,
    "active_stress_beta": beta,
    "initial_condition": {
        "name": "extruded_2d_twist",
        "seed": seed,
        "S_initial": args.S_initial,
        "Q_2d_source": q2d_source,
        "source": None if args.q2d_path is None else str(args.q2d_path.resolve()),
        "Q_2d_input": q2d_input_summary,
        "twist_amplitude": args.twist_amplitude,
        "twist_modes": list(args.twist_modes),
    },
    "q2d_source": q2d_source,
    "q2d_path": None if args.q2d_path is None else str(args.q2d_path.resolve()),
    "analytic_q2d_parameters": None if args.q2d_path is not None else {
        "num_defect_pairs": args.num_defect_pairs,
        "min_separation": args.defect_min_separation,
        "core_radius": args.defect_core_radius,
        "S_initial": args.S_initial,
        "S_bulk": S_bulk,
        "background_angle": args.background_angle,
        "seed": seed,
    },
    "twist_amplitude_rms_radians": args.twist_amplitude,
    "twist_modes": list(args.twist_modes),
    "save_hydrodynamics": SAVE_HYDRODYNAMICS,
    "save_pressure": SAVE_PRESSURE,
    "model_limitations": [
        "quasistatic Stokes rather than the paper's full momentum equation",
        "passive elastic/reactive nematic stresses are not included in the Stokes solve",
        "Q wall anchoring intentionally not matched in this first pass",
    ],
}
print(json.dumps(metadata, indent=2))
if args.dry_run:
    raise SystemExit(0)

output_dir = args.output_dir.resolve()
if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError(f"Refusing to mix benchmark outputs in nonempty {output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)
write_run_metadata(output_dir, metadata, status="running")

if args.q2d_path is None:
    q_2d = create_initial_condition(
        "analytic_periodic_defect_gas_2d",
        shape=(Nx, Ny),
        lengths=(Lx, Ly),
        num_defect_pairs=args.num_defect_pairs,
        min_separation=args.defect_min_separation,
        core_radius=args.defect_core_radius,
        seed=seed,
        S_initial=args.S_initial,
        background_angle=args.background_angle,
    )
    defect_positions, defect_charges = sample_periodic_neutral_defects_2d(
        lengths=(Lx, Ly),
        num_defect_pairs=args.num_defect_pairs,
        min_separation=args.defect_min_separation,
        seed=seed,
    )
    q2d_stacked = np.stack(
        [q_2d[name].numpy() for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")],
        axis=-1,
    )
    np.save(output_dir / "Q2D_initial.npy", q2d_stacked)
    np.savetxt(
        output_dir / "Q2D_defects.csv",
        np.column_stack((defect_positions, defect_charges)),
        delimiter=",",
        header="x,y,charge",
        comments="",
    )
else:
    assert q_2d is not None

solver = SpectralSolver(
    shape=(Nx, Ny, Nz),
    L=(Lx, Ly, Lz),
    dt=dt,
    device=device,
    batchsize=batchsize,
)
q_initial_condition = create_initial_condition(
    "extruded_2d_twist",
    shape=(Nx, Ny, Nz),
    Q_2d=q_2d,
    boundary_conditions=Q_BC,
    seed=seed,
    twist_amplitude=args.twist_amplitude,
    twist_modes=tuple(args.twist_modes),
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
# ModalSaddleStokesCompute solves the pure FFT/DST/DCT modal saddle-point
# discretization through its pressure Schur complement.
solver.model.add_static_field("ux", boundary_conditions=U_BC)
solver.model.add_static_field("uy", boundary_conditions=U_BC)
solver.model.add_static_field("uz", boundary_conditions=U_BC)
solver.model.add_static_field("p", boundary_conditions=PRESSURE_MODAL_BC)



solver.model.set_nonlinear_model(NonlinearModel(solver))
solver.model.set_static_compute_model(ModalSaddleStokesCompute(solver))

alpha = torch.tensor(alpha_value, device=device)
solver.model.parameters.new_param('alpha', alpha)

solver.build()

start_step = 0

diagnostic_history = []
start = time.time()
pbar = trange(steps)


def save_snapshot(i):
    q_snapshot = torch.stack([
        solver.model.fields[name].detach().cpu()
        for name in ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"]
    ])  # shape -> (5, batch, Nx, Ny, Nz)
    q_snapshot = q_snapshot.permute(1, 2, 3, 4, 0)
    np.save(output_dir / f"Q_{i}.npy", q_snapshot[0].numpy())

    if SAVE_HYDRODYNAMICS:
        u_snapshot = torch.stack([
            solver.model.fields[name].detach().cpu()
            for name in ["ux", "uy", "uz"]
        ])  # shape -> (3, batch, Nx, Ny, Nz)
        u_snapshot = u_snapshot.permute(1, 2, 3, 4, 0)
        np.save(output_dir / f"u_{i}.npy", u_snapshot[0].numpy())

    if SAVE_PRESSURE:
        p_snapshot = solver.model.fields["p"].detach().cpu()
        np.save(output_dir / f"p_{i}.npy", p_snapshot[0].numpy())


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
            div_max=f"{div_max:.2e}",
            div_rms=f"{div_rms:.2e}",
            div_rel=f"{div_rel:.2e}",
            schur_it=static_model.last_pressure_iterations,
            schur_rel=f"{static_model.last_pressure_relative_residual:.2e}",
            wall_n_rms=f"{wall_mom_rms:.2e}",
        )
    if i >= SAVE_START_STEP and i % SAVE_INTERVAL == 0:
        save_snapshot(i)


for local_step in pbar:
    i = start_step + local_step
    solver.run(1, pre_update_callback=lambda _solver, _step, i=i: record_step_state(i))

final_step = start_step + steps
solver.refresh_static_fields()
save_snapshot(final_step)

end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
if ENABLE_DIAGNOSTICS:
    final_div_max, final_div_rms, final_div_rel = divergence_stats(solver.model.fields)
    final_wall_mom_max, final_wall_mom_rms = wall_normal_momentum_stats(
        solver.model.fields,
        solver.model.parameters,
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
    np.save(output_dir / "diagnostics.npy", diagnostic_array)
    np.savetxt(
        output_dir / "diagnostics.csv",
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

metadata["completed_steps"] = final_step
metadata["elapsed_seconds"] = end - start
(output_dir / "COMPLETE").write_text("complete\n")
write_run_metadata(output_dir, metadata, status="complete")
