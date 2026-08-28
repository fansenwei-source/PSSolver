"""Single-point free-slip/free-Q Shendruk et al. Fig. 4 benchmark.

The independent scan variable is the paper's dimensionless activity number

    A = H * sqrt(zeta / K),

where ``H`` is the channel height and ``K`` is the one-constant Frank elastic
constant. This file runs exactly one value of A using the mixed FFT/DCT/DST
free-slip architecture from ``Plane_free_slip_dealiased.py``:

* Q and tangential velocity use Neumann/DCT modes at the z walls;
* normal velocity uses Dirichlet/DST modes at the z walls;
* pressure is a Neumann/DCT modal incompressibility multiplier;
* the default plug-flow convention is the zero-mean pseudoinverse;
* the default no-padding dealiasing rule is ``cubic_half``.

Use ``scripts_plane/run_fig4_scan.sh`` to launch a sweep.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from pssolver import SpectralSolver, write_run_metadata
from pssolver.models.active_nematics import (
    Q_convention_metadata,
    create_initial_condition,
    positive_equilibrium_S,
    sample_periodic_neutral_defects_2d,
)
from pssolver.integrator import SemiImplicitEulerIntegrator
from tqdm import trange
import numpy as np


# Free Q anchoring and free-slip velocity at walls normal to z.
Q_BC = ("periodic", "periodic", "neumann")
U_TANGENTIAL_BC = ("periodic", "periodic", "neumann")
U_NORMAL_BC = ("periodic", "periodic", "dirichlet")
# div(u) lives in the DCT space, so pressure uses the same DCT modal multiplier
# space rather than an independently prescribed physical wall value.
PRESSURE_MODAL_BC = ("periodic", "periodic", "neumann")
# Tangential plug-flow handling. The recommended benchmark default uses the
# fric=0 Stokes pseudoinverse and explicitly fixes <ux>=<uy>=0. This is a
# reference-frame/modeling choice, not a pressure gauge: free Q anchoring does
# not guarantee zero volume-averaged active tangential force, e.g.
#
#   <f_x> is proportional to [<Q_xz>_{xy}]_{z=0}^{z=L_z}.
#
# A strict reproduction of the paper's inertial mean-momentum dynamics would
# require evolving the two tangential mean modes separately. Positive friction
# is retained as an optional sensitivity model but is not the benchmark default.
DEFAULT_ZERO_MODE_POLICY = "zero_mean"
DEFAULT_FRICTION_MODE_FRIC = 0.1
DEFAULT_DEALIAS_RULE = "cubic_half"
DEFAULT_SPECTRAL_REFRESH_TIME = 0.2
DEALIAS_RULE_FRACTIONS = {
    "none": None,
    "two_thirds": 2.0 / 3.0,
    "cubic_half": 0.5,
}
ENABLE_DIAGNOSTICS = False
DIAGNOSTIC_INTERVAL = 100
SAVE_INTERVAL = 500
SAVE_HYDRODYNAMICS = False
ALIGNMENT_PARAMETER = 0.3


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run one activity-number value for the Shendruk Fig. 4 benchmark. "
            "Use scripts_plane/run_fig4_scan.sh for a sweep."
        )
    )
    parser.add_argument("--activity-number", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=float, default=20.0)
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
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--save-start-step", type=int, default=5_000)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--diagnostic-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--num-defect-pairs", type=int, default=6)
    parser.add_argument("--defect-min-separation", type=float, default=10.0)
    parser.add_argument("--defect-core-radius", type=float, default=1.5)
    parser.add_argument("--background-angle", type=float, default=0.0)
    parser.add_argument(
        "--twist-amplitude",
        type=float,
        default=0.01,
        help="RMS Neumann-compatible layer-rotation angle in radians.",
    )
    parser.add_argument(
        "--twist-modes",
        type=int,
        nargs="+",
        default=(1, 2, 3),
        help="Positive DCT mode indices used for the wall-normal twist.",
    )
    parser.add_argument("--ldg-a", type=float, default=0.0)
    parser.add_argument("--ldg-b", type=float, default=-0.3)
    parser.add_argument("--ldg-c", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=2.94)
    parser.add_argument("--flow-alignment", type=float, default=0.3)
    parser.add_argument("--eta", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--zero-mode-policy",
        choices=("zero_mean", "friction"),
        default=DEFAULT_ZERO_MODE_POLICY,
        help=(
            "Tangential plug-flow convention. zero_mean uses fric=0 and fixes "
            "<ux>=<uy>=0; friction retains plug modes with positive drag."
        ),
    )
    parser.add_argument(
        "--friction-mode-fric",
        type=float,
        default=DEFAULT_FRICTION_MODE_FRIC,
    )
    parser.add_argument(
        "--dealias-rule",
        choices=tuple(DEALIAS_RULE_FRACTIONS),
        default=DEFAULT_DEALIAS_RULE,
        help=(
            "cubic_half is conservative for Q^2 Q; two_thirds protects "
            "compatible quadratic products; none disables projection."
        ),
    )
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument(
        "--initial-s",
        dest="S_initial",
        type=float,
        default=1.0 / 3.0,
        help=(
            "Initial S in Q=(3S/2)(nn-I/3); default 1/3, equal to the "
            "benchmark bulk equilibrium."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Real arithmetic precision used by fields and transforms.",
    )
    parser.add_argument(
        "--tf32",
        choices=("off", "on"),
        default="off",
        help=(
            "Explicit CUDA TF32 policy. It is effective only for float32 "
            "CUDA runs and is recorded in the run metadata."
        ),
    )
    refresh_group = parser.add_mutually_exclusive_group()
    refresh_group.add_argument(
        "--spectral-refresh-time",
        type=float,
        default=None,
        help=(
            "Physical-time interval between dynamic real-to-spectral rebuilds. "
            f"The default is {DEFAULT_SPECTRAL_REFRESH_TIME:g}. The interval "
            "must be an integer multiple of dt."
        ),
    )
    refresh_group.add_argument(
        "--spectral-refresh-steps",
        type=int,
        default=None,
        help="Legacy step-count interval between dynamic spectral rebuilds.",
    )
    refresh_group.add_argument(
        "--disable-spectral-refresh",
        action="store_true",
        help="Disable only the periodic dynamic real-to-spectral rebuild.",
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--save-hydrodynamics", action="store_true")
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
    if args.S_initial <= 0:
        parser.error("--initial-s must be positive")
    if args.num_defect_pairs <= 0:
        parser.error("--num-defect-pairs must be positive")
    if args.defect_min_separation <= 0:
        parser.error("--defect-min-separation must be positive")
    if args.defect_core_radius <= 0:
        parser.error("--defect-core-radius must be positive")
    if args.twist_amplitude < 0:
        parser.error("--twist-amplitude must be non-negative")
    if len(set(args.twist_modes)) != len(args.twist_modes):
        parser.error("--twist-modes must be unique")
    if any(mode <= 0 or mode >= args.nz for mode in args.twist_modes):
        parser.error("--twist-modes must satisfy 1 <= mode < nz")
    if args.zero_mode_policy == "friction" and args.friction_mode_fric <= 0:
        parser.error("--friction-mode-fric must be positive in friction mode")
    if args.coefficient_min >= args.coefficient_max:
        parser.error("--coefficient-min must be smaller than --coefficient-max")
    if args.save_start_step < 0 or args.save_start_step > args.steps:
        parser.error("--save-start-step must lie between 0 and --steps")

    if args.spectral_refresh_steps is not None:
        if args.spectral_refresh_steps <= 0:
            parser.error("--spectral-refresh-steps must be positive")
        args.spectral_refresh_mode = "steps"
        args.spectral_refresh_interval_steps = args.spectral_refresh_steps
        args.spectral_refresh_requested_time = None
        args.spectral_refresh_requested_steps = args.spectral_refresh_steps
    elif args.disable_spectral_refresh:
        args.spectral_refresh_mode = "disabled"
        args.spectral_refresh_interval_steps = None
        args.spectral_refresh_requested_time = None
        args.spectral_refresh_requested_steps = None
    else:
        requested_time = (
            DEFAULT_SPECTRAL_REFRESH_TIME
            if args.spectral_refresh_time is None
            else args.spectral_refresh_time
        )
        if not math.isfinite(requested_time) or requested_time <= 0:
            parser.error("--spectral-refresh-time must be positive and finite")
        interval_ratio = requested_time / args.dt
        interval_steps = int(round(interval_ratio))
        if interval_steps <= 0 or not math.isclose(
            interval_ratio,
            interval_steps,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            parser.error(
                "--spectral-refresh-time must be an integer multiple of --dt; "
                f"got time/dt={interval_ratio:.17g}"
            )
        args.spectral_refresh_mode = "physical_time"
        args.spectral_refresh_interval_steps = interval_steps
        args.spectral_refresh_requested_time = requested_time
        args.spectral_refresh_requested_steps = None

    args.spectral_refresh_effective_time = (
        None
        if args.spectral_refresh_interval_steps is None
        else args.spectral_refresh_interval_steps * args.dt
    )
    return args


def tensor_sha256(tensors):
    """Hash an ordered collection of contiguous CPU tensor byte strings."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


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

    def _refresh_dynamic_spectra(self):
        super()._refresh_dynamic_spectra()
        self.spectral_projector.project_dynamic_fields(
            self.model.fields,
            sync_spatial=True,
        )

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

        self._advance_spectral_refresh_clock()


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
        beta_value=-1.0,
        friction=0.0,
        viscosity=2.0 / 3.0,
        zero_mode_policy=DEFAULT_ZERO_MODE_POLICY,
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


args = parse_args()

seed = args.seed
dt = args.dt
steps = args.steps
device = (
    "cuda" if args.device == "auto" and torch.cuda.is_available()
    else "cpu" if args.device == "auto"
    else args.device
)
real_dtype = {
    "float32": torch.float32,
    "float64": torch.float64,
}[args.dtype]
spectral_dtype_name = "complex64" if real_dtype == torch.float32 else "complex128"
tf32_requested = args.tf32 == "on"
tf32_effective = (
    tf32_requested
    and real_dtype == torch.float32
    and torch.device(device).type == "cuda"
)
torch.set_float32_matmul_precision("high" if tf32_effective else "highest")
torch.backends.cuda.matmul.allow_tf32 = tf32_effective
torch.backends.cudnn.allow_tf32 = tf32_effective
batchsize = 1

Nx, Ny, Nz = args.nx, args.ny, args.nz
Lx, Ly, Lz = args.lx, args.ly, args.height
SAVE_INTERVAL = args.save_interval
SAVE_START_STEP = args.save_start_step
DIAGNOSTIC_INTERVAL = args.diagnostic_interval
ENABLE_DIAGNOSTICS = args.diagnostics
SAVE_HYDRODYNAMICS = args.save_hydrodynamics
ALIGNMENT_PARAMETER = args.flow_alignment
zero_mode_policy = args.zero_mode_policy
dealias_rule = args.dealias_rule

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
fric = (
    0.0
    if zero_mode_policy == "zero_mean"
    else float(args.friction_mode_fric)
)
eta = args.eta

metadata = {
    "schema_version": 1,
    "script": "Plane_fig4_benchmark.py",
    "solver": {
        "shape": [Nx, Ny, Nz],
        "lengths": [Lx, Ly, Lz],
        "dt": dt,
        "steps": steps,
        "save_interval": SAVE_INTERVAL,
        "real_dtype": args.dtype,
        "spectral_dtype": spectral_dtype_name,
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
        "velocity_tangential": U_TANGENTIAL_BC,
        "velocity_normal": U_NORMAL_BC,
        "pressure": PRESSURE_MODAL_BC,
    },
    "numerics": {
        "dealiasing": {
            "rule": dealias_rule,
            "fraction": DEALIAS_RULE_FRACTIONS[dealias_rule],
        },
        "velocity_zero_mode": zero_mode_policy,
        "pressure_solver": "free_slip_modal_schur_complement",
        "precision": {
            "real_dtype": args.dtype,
            "spectral_dtype": spectral_dtype_name,
            "tf32_requested": args.tf32,
            "tf32_effective": tf32_effective,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "spectral_refresh": {
            "mode": args.spectral_refresh_mode,
            "requested_interval_time": args.spectral_refresh_requested_time,
            "requested_interval_steps": args.spectral_refresh_requested_steps,
            "effective_interval_steps": args.spectral_refresh_interval_steps,
            "effective_interval_time": args.spectral_refresh_effective_time,
            "phase_origin_step": 0,
        },
    },
    "benchmark": "Shendruk et al. PRE 98, 010601(R) (2018), Fig. 4",
    "scan_variable": "activity_number",
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
    "dtype": args.dtype,
    "tf32": args.tf32,
    "q_boundary_conditions": Q_BC,
    "tangential_velocity_boundary_conditions": U_TANGENTIAL_BC,
    "normal_velocity_boundary_conditions": U_NORMAL_BC,
    "velocity_wall_model": "free-slip",
    "q_wall_model_note": "free/Neumann; matches the Fig. 4 free-anchoring branch",
    "zero_mode_policy": zero_mode_policy,
    "dealias_rule": dealias_rule,
    "dealias_fraction": DEALIAS_RULE_FRACTIONS[dealias_rule],
    "ldg_coefficients": {"A": args.ldg_a, "B": args.ldg_b, "C": args.ldg_c},
    "gamma": args.gamma,
    "flow_alignment": ALIGNMENT_PARAMETER,
    "eta": eta,
    "friction": fric,
    "active_stress_beta": beta,
    "S_initial": args.S_initial,
    "S_bulk": S_bulk,
    "initial_condition": {
        "name": "extruded_analytic_periodic_defect_gas_2d",
        "seed": seed,
        "S_initial": args.S_initial,
        "twist_amplitude": args.twist_amplitude,
        "twist_modes": args.twist_modes,
    },
    "initial_defect_gas": {
        "num_pairs": args.num_defect_pairs,
        "minimum_separation": args.defect_min_separation,
        "core_radius": args.defect_core_radius,
        "S_initial": args.S_initial,
        "background_angle": args.background_angle,
    },
    "initial_neumann_twist": {
        "rms_amplitude_radians": args.twist_amplitude,
        "dct_modes": args.twist_modes,
    },
    "save_hydrodynamics": SAVE_HYDRODYNAMICS,
    "model_limitations": [
        "quasistatic Stokes rather than the paper's full momentum equation",
        "passive elastic/reactive nematic stresses are not included in the Stokes solve",
        "zero_mean fixes the free-slip tangential plug mode in a chosen reference frame",
        "spectral dealiasing differs from the paper's finite-difference/LB discretization",
    ],
}
print(json.dumps(metadata, indent=2))
if args.dry_run:
    raise SystemExit(0)

output_dir = args.output_dir.resolve()
if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError(f"Refusing to mix benchmark outputs in nonempty {output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

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
    dtype=real_dtype,
)
defect_positions, defect_charges = sample_periodic_neutral_defects_2d(
    lengths=(Lx, Ly),
    num_defect_pairs=args.num_defect_pairs,
    min_separation=args.defect_min_separation,
    seed=seed,
)
np.save(
    output_dir / "Q2D_initial.npy",
    np.stack(
        [q_2d[name].numpy() for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")],
        axis=-1,
    ),
)
np.savetxt(
    output_dir / "Q2D_defects.csv",
    np.column_stack((defect_positions, defect_charges)),
    delimiter=",",
    header="x,y,charge",
    comments="",
)

solver = SpectralSolver(
    shape=(Nx, Ny, Nz),
    L=(Lx, Ly, Lz),
    dt=dt,
    device=device,
    batchsize=batchsize,
    dtype=real_dtype,
)
spectral_projector = BasisAwareSpectralProjector(
    solver,
    rule=dealias_rule,
)
solver.model.spectral_projector = spectral_projector
solver.integrator_cl = DealiasedSemiImplicitEulerIntegrator
q_initial_condition = create_initial_condition(
    "extruded_2d_twist",
    shape=(Nx, Ny, Nz),
    Q_2d=q_2d,
    boundary_conditions=Q_BC,
    seed=seed,
    twist_amplitude=args.twist_amplitude,
    twist_modes=tuple(args.twist_modes),
    dtype=real_dtype,
)
Qxx_0 = q_initial_condition["Qxx"]
Qxy_0 = q_initial_condition["Qxy"]
Qxz_0 = q_initial_condition["Qxz"]
Qyy_0 = q_initial_condition["Qyy"]
Qyz_0 = q_initial_condition["Qyz"]
metadata["initial_condition"]["raw_q_sha256"] = tensor_sha256(
    (Qxx_0, Qxy_0, Qxz_0, Qyy_0, Qyz_0)
)

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

solver.model.set_nonlinear_model(NonlinearModel(spectral_projector))
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

alpha = torch.tensor(alpha_value, device=device, dtype=real_dtype)
solver.model.parameters.new_param('alpha', alpha)

solver.build()
solver.integrator.set_spectral_refresh_interval(
    args.spectral_refresh_interval_steps
)

# Band-limit the generated Q field before it participates in pointwise products.
spectral_projector.project_dynamic_fields(
    solver.model.fields,
    sync_spatial=True,
)
if spectral_projector.enabled:
    solver.integrator._static_fields_are_current = False
metadata["initial_condition"]["projected_q_sha256"] = tensor_sha256(
    tuple(
        solver.model.fields[name]
        for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
    )
)
metadata["retained_q_modes"] = spectral_projector.retained_axis_counts(Q_BC)
metadata["retained_normal_velocity_modes"] = spectral_projector.retained_axis_counts(
    U_NORMAL_BC
)
write_run_metadata(output_dir, metadata, status="running")

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

        p_snapshot = solver.model.fields["p"].detach().cpu()
        np.save(output_dir / f"p_{i}.npy", p_snapshot[0].numpy())


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
metadata["numerics"]["spectral_refresh"]["actual_count"] = (
    solver.integrator.refresh_count
)
(output_dir / "COMPLETE").write_text("complete\n")
write_run_metadata(output_dir, metadata, status="complete")
