"""Reusable Beris--Edwards model with complete-stress Stokes flow.

This independent script is copied from Plane_shendruk_stokes.py. It retains
the Shendruk benchmark parameterization and mixed-basis geometry while using
the reusable PSSolver Beris--Edwards Q model for the equation

    (partial_t + u dot grad) Q - S(E, Omega, Q) = H / gamma

with the full co-rotational flow-alignment term and the one-constant
Landau--de Gennes molecular field. The quasistatic momentum equation is

    0 = -grad(p) + eta lap(u) - fric u + div(Pi_nem),

where, up to an isotropic term absorbed into pressure,

    Pi_nem = 2 lambda M (Q:H) - lambda (H M + M H)
             + Q H - H Q - L1 partial_i Q:partial_j Q
             + beta alpha Q,                 M = Q + I/3.

Thus beta=-1 and alpha=zeta give the paper's active stress -zeta Q while the
reactive and one-constant distortion stresses are retained as well.

The independent scan variable is the paper's dimensionless activity number

    A = H * sqrt(zeta / K),

where H is the channel height and K is the one-constant Frank elastic
constant. This file runs exactly one value of A using the mixed FFT/DCT/DST
free-slip architecture from Plane_free_slip_dealiased.py:

* Q and tangential velocity use Neumann/DCT modes at the z walls;
* normal velocity uses Dirichlet/DST modes at the z walls;
* pressure is a Neumann/DCT modal incompressibility multiplier;
* the default plug-flow convention is the zero-mean pseudoinverse;
* the default no-padding dealiasing rule is cubic_half.

This is not yet a strict Fig. 4 reproduction: zero-Reynolds-number Stokes
replaces the paper's inertial momentum dynamics, and the initialization and
spectral discretization are not paper-identical.
"""

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import time
import torch
from pssolver import SpectralSolver, write_run_metadata
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsQNonlinearModel,
    Q_COMPONENTS,
    Q_convention_metadata,
    S_from_Q,
    aligned_x_band_limited_noise_2d,
    beris_edwards_free_energy_density,
    beris_edwards_linear_operator,
    create_initial_condition,
    extruded_2d_unbiased_rotation,
    positive_equilibrium_S,
    sample_periodic_neutral_defects_2d,
)
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.transforms import (
    BasisAwareSpectralProjector,
    DEALIAS_RULE_FRACTIONS,
    DEFAULT_DEALIAS_RULE,
)
from tqdm import trange
import numpy as np


# Free Q anchoring and free-slip velocity at walls normal to z.
Q_BC = ("periodic", "periodic", "neumann")
U_TANGENTIAL_BC = ("periodic", "periodic", "neumann")
U_NORMAL_BC = ("periodic", "periodic", "dirichlet")
DISTORTION_ODD_Z_BC = ("periodic", "periodic", "dirichlet")
# div(u) lives in the DCT space, so pressure uses the same DCT modal multiplier
# space rather than an independently prescribed physical wall value.
PRESSURE_MODAL_BC = ("periodic", "periodic", "neumann")
# Tangential plug-flow handling. The recommended pilot default uses the
# fric=0 Stokes pseudoinverse and explicitly fixes <ux>=<uy>=0. This is a
# reference-frame/modeling choice, not a pressure gauge: free Q anchoring does
# not guarantee zero volume-averaged total nematic tangential force. Its active
# part alone can already be nonzero, e.g.
#
#   <f_x> is proportional to [<Q_xz>_{xy}]_{z=0}^{z=L_z}.
#
# A strict reproduction of the paper's inertial mean-momentum dynamics would
# require evolving the two tangential mean modes separately. Positive friction
# is retained as an optional sensitivity model but is not the default.
DEFAULT_ZERO_MODE_POLICY = "zero_mean"
DEFAULT_FRICTION_MODE_FRIC = 0.1
DEFAULT_SPECTRAL_REFRESH_TIME = 0.2
ENABLE_DIAGNOSTICS = False
DIAGNOSTIC_INTERVAL = 100
SAVE_INTERVAL = 500
SAVE_HYDRODYNAMICS = False
ALIGNMENT_PARAMETER = 0.3

# Local source files whose contents determine the discrete Q/Stokes dynamics.
# Keep repository-relative names in metadata so runs made in different checkout
# locations remain directly comparable.
IMPLEMENTATION_SOURCE_FILES = (
    "Plane_beris_edwards_stokes.py",
    "pssolver/solver.py",
    "pssolver/Field.py",
    "pssolver/PDEmodel.py",
    "pssolver/integrator.py",
    "pssolver/transforms.py",
    "pssolver/__init__.py",
    "pssolver/models/active_nematics/__init__.py",
    "pssolver/models/active_nematics/fields.py",
    "pssolver/models/active_nematics/q_tensor.py",
    "pssolver/models/active_nematics/beris_edwards.py",
    "pssolver/models/active_nematics/stokes.py",
    "pssolver/models/active_nematics/initial_conditions.py",
)


def file_sha256(path):
    """Hash one implementation file for run-to-run provenance checks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the reusable Beris--Edwards Q model with PSSolver's "
            "complete one-constant nematic-stress quasistatic free-slip "
            "Stokes solver using the Shendruk benchmark parameterization."
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
    parser.add_argument(
        "--initialization-protocol",
        choices=("v1", "v3-mother", "v3-extruded"),
        default="v1",
        help=(
            "v1 uses the analytic defect gas and coherent twist; v3-mother "
            "evolves a target-parameter, strictly z-independent 2D seed; "
            "v3-extruded loads a qualified target-parameter Q2D checkpoint "
            "and adds an unbiased local 3D rotation."
        ),
    )
    parser.add_argument(
        "--initial-q2d",
        type=Path,
        default=None,
        help="Qualified Q2D checkpoint required by v3-extruded.",
    )
    parser.add_argument(
        "--v3-mother-manifest",
        type=Path,
        default=None,
        help=(
            "V3 qualification manifest binding the Q2D checkpoint and its "
            "target K,zeta pair; required by an executed v3-extruded run."
        ),
    )
    parser.add_argument(
        "--v3-mother-restart-q2d",
        type=Path,
        default=None,
        help="Optional Q2D checkpoint for a target-parameter mother continuation.",
    )
    parser.add_argument(
        "--v3-mother-start-step",
        type=int,
        default=0,
        help="Absolute step represented by --v3-mother-restart-q2d.",
    )
    parser.add_argument("--v3-mother-noise-rms", type=float, default=0.01)
    parser.add_argument("--v3-mother-max-mode-x", type=int, default=4)
    parser.add_argument("--v3-mother-max-mode-y", type=int, default=4)
    parser.add_argument("--v3-rotation-rms", type=float, default=1.0e-3)
    parser.add_argument("--v3-rotation-max-mode-x", type=int, default=3)
    parser.add_argument("--v3-rotation-max-mode-y", type=int, default=3)
    parser.add_argument(
        "--v3-rotation-z-modes",
        type=int,
        nargs="+",
        default=(1, 2, 3),
    )
    parser.add_argument(
        "--v3-perturbation-seed",
        type=int,
        default=None,
        help="Independent 3D perturbation seed; defaults to --seed.",
    )
    parser.add_argument(
        "--save-layout",
        choices=("auto", "q3d", "q2d"),
        default="auto",
        help=(
            "auto selects q2d for v3-mother and q3d otherwise. q2d saves the "
            "z-mean only after a strict z-invariance check."
        ),
    )
    parser.add_argument(
        "--q2d-z-invariance-rtol",
        type=float,
        default=1.0e-10,
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
            "cubic_half is used with projected H and complete stress/force "
            "stages; two_thirds protects compatible quadratic products; "
            "none disables projection."
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
            "Shendruk bulk equilibrium."
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
        "--validation-config-sha256",
        default=None,
        help=(
            "Optional canonical SHA-256 of the complete validation-run "
            "configuration. It is recorded verbatim in metadata so a "
            "validation runner can reject accidental result reuse."
        ),
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
    if not math.isfinite(args.v3_mother_noise_rms) or args.v3_mother_noise_rms < 0:
        parser.error("--v3-mother-noise-rms must be non-negative and finite")
    if not math.isfinite(args.v3_rotation_rms) or args.v3_rotation_rms < 0:
        parser.error("--v3-rotation-rms must be non-negative and finite")
    if any(
        mode < 0
        for mode in (
            args.v3_mother_max_mode_x,
            args.v3_mother_max_mode_y,
            args.v3_rotation_max_mode_x,
            args.v3_rotation_max_mode_y,
        )
    ):
        parser.error("V3 maximum mode indices must be non-negative")
    if args.initialization_protocol == "v3-extruded":
        if args.initial_q2d is None:
            parser.error("v3-extruded requires --initial-q2d")
        if args.v3_mother_manifest is None:
            parser.error("v3-extruded requires --v3-mother-manifest")
        if len(set(args.v3_rotation_z_modes)) != len(args.v3_rotation_z_modes):
            parser.error("--v3-rotation-z-modes must be unique")
        if any(mode <= 0 or mode >= args.nz for mode in args.v3_rotation_z_modes):
            parser.error("--v3-rotation-z-modes must satisfy 1 <= mode < nz")
    elif args.initial_q2d is not None or args.v3_mother_manifest is not None:
        parser.error(
            "--initial-q2d and --v3-mother-manifest are valid only with "
            "v3-extruded"
        )
    if args.initialization_protocol != "v3-mother" and (
        args.v3_mother_restart_q2d is not None
        or args.v3_mother_start_step != 0
    ):
        parser.error("V3 mother restart options require v3-mother")
    if args.initialization_protocol == "v3-mother":
        if args.v3_mother_restart_q2d is None and args.v3_mother_start_step != 0:
            parser.error("a positive V3 mother start step requires a restart Q2D")
        if args.v3_mother_restart_q2d is not None and args.v3_mother_start_step <= 0:
            parser.error("a V3 mother restart requires a positive absolute start step")
    args.save_layout = (
        "q2d"
        if args.save_layout == "auto" and args.initialization_protocol == "v3-mother"
        else "q3d"
        if args.save_layout == "auto"
        else args.save_layout
    )
    if args.save_layout == "q2d" and args.initialization_protocol != "v3-mother":
        parser.error("q2d save layout is restricted to v3-mother runs")
    if args.save_layout == "q2d" and args.save_hydrodynamics:
        parser.error("--save-hydrodynamics is incompatible with q2d save layout")
    if (
        not math.isfinite(args.q2d_z_invariance_rtol)
        or args.q2d_z_invariance_rtol <= 0
    ):
        parser.error("--q2d-z-invariance-rtol must be positive and finite")
    if args.v3_perturbation_seed is None:
        args.v3_perturbation_seed = args.seed
    if args.zero_mode_policy == "friction" and args.friction_mode_fric <= 0:
        parser.error("--friction-mode-fric must be positive in friction mode")
    if args.coefficient_min >= args.coefficient_max:
        parser.error("--coefficient-min must be smaller than --coefficient-max")
    absolute_start_step = (
        args.v3_mother_start_step
        if args.initialization_protocol == "v3-mother"
        else 0
    )
    if (
        args.save_start_step < absolute_start_step
        or args.save_start_step > absolute_start_step + args.steps
    ):
        parser.error(
            "--save-start-step must lie between the absolute start and final steps"
        )
    if args.validation_config_sha256 is not None and (
        len(args.validation_config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.validation_config_sha256
        )
    ):
        parser.error(
            "--validation-config-sha256 must be exactly 64 lowercase "
            "hexadecimal characters"
        )

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
    if (
        args.v3_mother_restart_q2d is not None
        and args.spectral_refresh_interval_steps is not None
    ):
        parser.error(
            "V3 mother continuation currently requires "
            "--disable-spectral-refresh so refresh phase cannot reset"
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


def load_v3_mother_manifest(path, q2d_path, *, frank_k, zeta):
    """Validate the small V3 contract that binds a qualified Q2D mother."""
    manifest_path = Path(path).expanduser().resolve()
    checkpoint_path = Path(q2d_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"V3 mother manifest does not exist: {manifest_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V3 Q2D checkpoint does not exist: {checkpoint_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid V3 mother manifest JSON: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("V3 mother manifest root must be a JSON object")
    required = {
        "schema_version",
        "protocol",
        "qualified",
        "checkpoint",
        "parameters",
        "qualification_report",
    }
    missing = required - manifest.keys()
    if missing:
        raise ValueError(f"V3 mother manifest is missing {sorted(missing)}")
    if manifest["schema_version"] != 1 or manifest["protocol"] != "V3":
        raise ValueError("V3 mother manifest must declare schema_version=1 and protocol=V3")
    if manifest["qualified"] is not True:
        raise ValueError("V3 mother manifest does not qualify the 2D statistical state")
    checkpoint = manifest["checkpoint"]
    parameters = manifest["parameters"]
    qualification_report = manifest["qualification_report"]
    if not all(
        isinstance(value, dict)
        for value in (checkpoint, parameters, qualification_report)
    ):
        raise ValueError(
            "V3 checkpoint, parameters, and qualification_report entries "
            "must be JSON objects"
        )
    declared_path = Path(checkpoint.get("path", "")).expanduser().resolve()
    if declared_path != checkpoint_path:
        raise ValueError(
            f"V3 checkpoint path mismatch: manifest={declared_path}, "
            f"requested={checkpoint_path}"
        )
    actual_checkpoint_sha = file_sha256(checkpoint_path)
    if checkpoint.get("sha256") != actual_checkpoint_sha:
        raise ValueError("V3 checkpoint SHA-256 does not match its manifest")
    qualification_path = Path(
        qualification_report.get("path", "")
    ).expanduser().resolve()
    if not qualification_path.is_file():
        raise FileNotFoundError(
            f"V3 qualification report does not exist: {qualification_path}"
        )
    actual_qualification_sha = file_sha256(qualification_path)
    if qualification_report.get("sha256") != actual_qualification_sha:
        raise ValueError("V3 qualification report SHA-256 does not match its manifest")
    try:
        qualification = json.loads(qualification_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid V3 qualification report JSON: {qualification_path}"
        ) from error
    if not isinstance(qualification, dict) or qualification.get(
        "candidate_2d_statistical_steady_state"
    ) is not True:
        raise ValueError("V3 qualification report does not approve the 2D mother")
    recommended = qualification.get("recommended_checkpoint")
    if not isinstance(recommended, dict):
        raise ValueError("V3 qualification report has no recommended checkpoint")
    if Path(recommended.get("path", "")).expanduser().resolve() != checkpoint_path:
        raise ValueError("V3 qualification report recommends a different checkpoint")
    if recommended.get("sha256") != actual_checkpoint_sha:
        raise ValueError("V3 qualification report checkpoint SHA-256 is incorrect")
    for name, expected in (("frank_k", frank_k), ("zeta", zeta)):
        try:
            declared = float(parameters[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"V3 manifest parameter {name!r} is missing or invalid") from error
        if not math.isclose(declared, expected, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise ValueError(
                f"V3 mother {name}={declared:.17g} does not match target "
                f"{expected:.17g}"
            )
        qualification_parameters = qualification.get("parameters")
        if not isinstance(qualification_parameters, dict):
            raise ValueError("V3 qualification report has no parameters object")
        try:
            qualified_value = float(qualification_parameters[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"V3 qualification parameter {name!r} is missing or invalid"
            ) from error
        if not math.isclose(
            qualified_value,
            expected,
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        ):
            raise ValueError(
                f"V3 qualification {name}={qualified_value:.17g} does not "
                f"match target {expected:.17g}"
            )
    return manifest, {
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": actual_checkpoint_sha,
        "qualification_report_path": str(qualification_path),
        "qualification_report_sha256": actual_qualification_sha,
    }


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


def wall_normal_momentum_stats(fields, static_model):
    """Check normal momentum on the first and last cell-center planes."""
    fz = static_model.last_projected_normal_force
    if fz is None:
        raise RuntimeError(
            "Normal-force diagnostics were not enabled in the static model."
        )

    lap_uz = fields.laplacian('uz')
    dzp = fields.gradient('p', axis=2)
    residual = dzp - (
        fz
        + static_model.viscosity * lap_uz
        - static_model.friction * fields['uz']
    )

    wall_residual = torch.stack([residual[..., 0], residual[..., -1]], dim=-1)
    wall_abs = wall_residual.abs()
    return wall_abs.max().item(), torch.sqrt(torch.mean(wall_abs.square())).item()

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

resolved_torch_device = torch.device(device)
cuda_device_name = None
cuda_total_memory_bytes = None
if resolved_torch_device.type == "cuda":
    cuda_properties = torch.cuda.get_device_properties(resolved_torch_device)
    cuda_device_name = cuda_properties.name
    cuda_total_memory_bytes = int(cuda_properties.total_memory)
runtime_environment = {
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "numpy_version": np.__version__,
    "torch_version": str(torch.__version__),
    "torch_cuda_runtime": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "resolved_device": str(resolved_torch_device),
    "device_type": resolved_torch_device.type,
    "cuda_device_name": cuda_device_name,
    "cuda_total_memory_bytes": cuda_total_memory_bytes,
}
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

# Write Q = q (nn-I/3), where q = 3 S/2 in the declared convention.
# Equating (L1/2) |grad Q|^2 with (K/2) |grad n|^2 gives
# K = 2 L1 q_eq^2 and therefore L1 = K/(2 q_eq^2).  At the Shendruk
# bulk equilibrium S_eq=1/3, q_eq=1/2 and L1=2K.  Using q_eq explicitly
# avoids the scalar-amplitude ambiguity in the paper's printed mapping.
S_bulk = positive_equilibrium_S(args.ldg_a, args.ldg_b, args.ldg_c)
q_equilibrium_amplitude = 1.5 * S_bulk
ldg_l1 = frank_k / (2.0 * q_equilibrium_amplitude**2)
rotational_viscosity = args.gamma
ldg_a_over_gamma = args.ldg_a / rotational_viscosity
ldg_b_over_gamma = args.ldg_b / rotational_viscosity
ldg_c_over_gamma = args.ldg_c / rotational_viscosity
ldg_l1_over_gamma = ldg_l1 / rotational_viscosity

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

implementation_provenance = {
    "schema_version": 1,
    "files": {
        relative_path: file_sha256(
            Path(__file__).resolve().parent / relative_path
        )
        for relative_path in IMPLEMENTATION_SOURCE_FILES
    },
}

metadata = {
    "schema_version": 1,
    "script": "Plane_beris_edwards_stokes.py",
    "validation_config_sha256": args.validation_config_sha256,
    "implementation_provenance": implementation_provenance,
    "runtime_environment": runtime_environment,
    "solver": {
        "shape": [Nx, Ny, Nz],
        "lengths": [Lx, Ly, Lz],
        "dt": dt,
        "steps": steps,
        "start_step": (
            args.v3_mother_start_step
            if args.initialization_protocol == "v3-mother"
            else 0
        ),
        "save_interval": SAVE_INTERVAL,
        "save_layout": args.save_layout,
        "real_dtype": args.dtype,
        "spectral_dtype": spectral_dtype_name,
    },
    "model": {
        "name": "active_nematics",
        "variant": "beris_edwards_complete_nematic_stress_stokes",
        "stage": "reusable_beris_edwards_complete_stress_stokes",
        "Q_convention": Q_convention_metadata(),
        "q_dynamics": {
            "name": "Beris-Edwards (Shendruk parameterization)",
            "implementation": (
                "pssolver.models.active_nematics.BerisEdwardsQNonlinearModel"
            ),
            "equation": "(partial_t+u.grad)Q-S(E,Omega,Q)=H/gamma",
            "flow_alignment_form": "full_beris_edwards",
            "molecular_field": "one_constant_landau_de_gennes",
            "raw_coefficients": {
                "A": args.ldg_a,
                "B": args.ldg_b,
                "C": args.ldg_c,
                "L1": ldg_l1,
                "rotational_viscosity_gamma": rotational_viscosity,
                "flow_alignment_lambda": ALIGNMENT_PARAMETER,
            },
            "evolution_coefficients": {
                "A_over_gamma": ldg_a_over_gamma,
                "B_over_gamma": ldg_b_over_gamma,
                "C_over_gamma": ldg_c_over_gamma,
                "L1_over_gamma": ldg_l1_over_gamma,
            },
            "frank_to_ldg_mapping": {
                "frank_K": frank_k,
                "tensor_amplitude_q_eq": q_equilibrium_amplitude,
                "formula": "L1=K/(2*q_eq^2), q_eq=3*S_eq/2",
                "paper_notation_note": (
                    "The printed mapping uses ambiguous scalar-amplitude "
                    "notation; this script uses the declared Q convention."
                ),
            },
        },
        "flow_dynamics": {
            "momentum_equation": (
                "0=-grad(p)+eta*lap(u)-fric*u+div(Pi_nematic)"
            ),
            "regime": "quasistatic_incompressible_stokes_brinkman",
            "nematic_stress": (
                "complete_one_constant_beris_edwards_up_to_isotropic_pressure"
            ),
            "stress_components": [
                "reactive",
                "one_constant_distortion",
                "active",
            ],
            "reactive_stress": (
                "2*lambda*M*(Q:H)-lambda*(H*M+M*H)+Q*H-H*Q; "
                "M=Q+I/3"
            ),
            "distortion_stress": (
                "-L1*(partial_i Q_kl)*(partial_j Q_kl)"
            ),
            "active_stress": "beta*alpha*Q; beta=-1 gives -zeta*Q",
            "molecular_field_in_stress": "raw_H_not_H_over_gamma",
            "isotropic_stress": "absorbed_into_incompressible_pressure",
            "viscous_stress": "handled_by_eta_laplacian_in_stokes_operator",
        },
        "parameters": {
            "activity_number": args.activity_number,
            "zeta": zeta,
            "frank_K": frank_k,
            "ldg_A": args.ldg_a,
            "ldg_B": args.ldg_b,
            "ldg_C": args.ldg_c,
            "ldg_L1": ldg_l1,
            "rotational_viscosity_gamma": rotational_viscosity,
            "flow_alignment_lambda": ALIGNMENT_PARAMETER,
            "alpha": alpha_value,
            "beta": beta,
            "S_initial": args.S_initial,
            "S_bulk": S_bulk,
            "fric": fric,
            "eta": eta,
        },
    },
    "boundary_conditions": {
        "Q": Q_BC,
        "velocity_tangential": U_TANGENTIAL_BC,
        "velocity_normal": U_NORMAL_BC,
        "pressure": PRESSURE_MODAL_BC,
        "velocity_wall_model_note": (
            "homogeneous kinematic free-slip; not zero total nematic traction"
        ),
    },
    "numerics": {
        "dealiasing": {
            "rule": dealias_rule,
            "fraction": DEALIAS_RULE_FRACTIONS[dealias_rule],
            "nematic_force_evaluation": (
                "project_H_then_algebraic_and_distortion_stresses_then_force"
            ),
            "monolithic_quintic_galerkin": False,
            "distortion_stress_z_parity": {
                "even_components": ["xx", "xy", "yx", "yy", "zz"],
                "odd_components": ["xz", "yz", "zx", "zy"],
                "even_basis": Q_BC,
                "odd_basis": DISTORTION_ODD_Z_BC,
            },
        },
        "velocity_zero_mode": zero_mode_policy,
        "zero_mode_force": "total_nematic_tangential_force",
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
    "benchmark_target": "Shendruk et al. PRE 98, 010601(R) (2018), Fig. 4",
    "reproduction_status": "development_reusable_beris_edwards_stokes_stage",
    "scan_variable": "activity_number",
    "parameterization": args.parameterization,
    "activity_number": args.activity_number,
    "activity_number_definition": "H*sqrt(zeta/K)",
    "zeta": zeta,
    "frank_k": frank_k,
    "ldg_l1": ldg_l1,
    "q_equilibrium_amplitude": q_equilibrium_amplitude,
    "q_elastic_relaxation": ldg_l1_over_gamma,
    "height": Lz,
    "domain": [Lx, Ly, Lz],
    "shape": [Nx, Ny, Nz],
    "dt": dt,
    "steps": steps,
    "save_start_step": SAVE_START_STEP,
    "save_interval": SAVE_INTERVAL,
    "diagnostic_interval": DIAGNOSTIC_INTERVAL,
    "seed": seed,
    "initialization_protocol": args.initialization_protocol,
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
        "protocol": args.initialization_protocol,
        "source": "PSSolver constructed",
        "paper_identical": False,
        "seed": seed,
        "S_initial": args.S_initial,
    },
    "save_hydrodynamics": SAVE_HYDRODYNAMICS,
    "model_limitations": [
        (
            "quasistatic zero-Reynolds-number Stokes rather than the paper's "
            "inertial momentum equation"
        ),
        (
            "homogeneous velocity free-slip is kinematic and does not impose "
            "zero total nematic traction"
        ),
        (
            "tangential plug-mode treatment is a modeling choice: zero_mean "
            "removes total forcing, while friction adds drag"
        ),
        "initial condition is PSSolver-constructed and not verified paper-identical",
        (
            "staged spectral filtering differs from the paper's "
            "finite-difference/LB discretization"
        ),
    ],
}
if args.initialization_protocol == "v1":
    metadata["initial_condition"].update({
        "name": "extruded_analytic_periodic_defect_gas_2d",
        "twist_amplitude": args.twist_amplitude,
        "twist_modes": args.twist_modes,
    })
    metadata["initial_defect_gas"] = {
        "num_pairs": args.num_defect_pairs,
        "minimum_separation": args.defect_min_separation,
        "core_radius": args.defect_core_radius,
        "S_initial": args.S_initial,
        "background_angle": args.background_angle,
    }
    metadata["initial_neumann_twist"] = {
        "rms_amplitude_radians": args.twist_amplitude,
        "dct_modes": args.twist_modes,
    }
elif args.initialization_protocol == "v3-mother":
    metadata["initial_condition"].update({
        "name": "V3_target_parameter_2d_mother",
        "z_independent": True,
        "angle_noise_rms_radians": args.v3_mother_noise_rms,
        "max_periodic_modes": [
            args.v3_mother_max_mode_x,
            args.v3_mother_max_mode_y,
        ],
        "target_parameter_matched": True,
        "continuation": args.v3_mother_restart_q2d is not None,
        "absolute_start_step": args.v3_mother_start_step,
        "restart_q2d_requested_path": (
            None
            if args.v3_mother_restart_q2d is None
            else str(args.v3_mother_restart_q2d)
        ),
    })
elif args.initialization_protocol == "v3-extruded":
    metadata["initial_condition"].update({
        "name": "V3_qualified_2d_mother_unbiased_3d_rotation",
        "source": "qualified external Q2D checkpoint",
        "initial_q2d_requested_path": str(args.initial_q2d),
        "mother_manifest_requested_path": str(args.v3_mother_manifest),
        "target_parameter_matched": True,
        "rotation_rms_radians": args.v3_rotation_rms,
        "rotation_max_periodic_modes": [
            args.v3_rotation_max_mode_x,
            args.v3_rotation_max_mode_y,
        ],
        "rotation_positive_dct_modes": args.v3_rotation_z_modes,
        "perturbation_seed": args.v3_perturbation_seed,
        "contains_kz_zero": False,
        "preserves_pointwise_Q_eigenvalues_before_spectral_projection": True,
    })
print(json.dumps(metadata, indent=2))
if args.dry_run:
    raise SystemExit(0)

output_dir = args.output_dir.resolve()
if output_dir.exists() and any(output_dir.iterdir()):
    raise FileExistsError(f"Refusing to mix pilot outputs in nonempty {output_dir}")
output_dir.mkdir(parents=True, exist_ok=True)

v3_manifest = None
if args.initialization_protocol == "v1":
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
elif args.initialization_protocol == "v3-mother":
    if args.v3_mother_restart_q2d is None:
        q_2d = aligned_x_band_limited_noise_2d(
            (Nx, Ny),
            S_initial=args.S_initial,
            seed=seed,
            angle_rms=args.v3_mother_noise_rms,
            max_mode_x=args.v3_mother_max_mode_x,
            max_mode_y=args.v3_mother_max_mode_y,
            dtype=real_dtype,
        )
    else:
        restart_path = args.v3_mother_restart_q2d.expanduser().resolve()
        if not restart_path.is_file():
            raise FileNotFoundError(f"V3 mother restart does not exist: {restart_path}")
        restart_sha256 = file_sha256(restart_path)
        q_2d = np.load(restart_path, allow_pickle=False)
        if not np.issubdtype(q_2d.dtype, np.floating):
            raise ValueError(f"V3 mother restart must be floating point, got {q_2d.dtype}")
        if not np.all(np.isfinite(q_2d)):
            raise ValueError("V3 mother restart contains NaN or Inf")
        metadata["initial_condition"]["restart_q2d"] = {
            "path": str(restart_path),
            "sha256": restart_sha256,
        }
else:
    v3_manifest, v3_source_provenance = load_v3_mother_manifest(
        args.v3_mother_manifest,
        args.initial_q2d,
        frank_k=frank_k,
        zeta=zeta,
    )
    q_2d_array = np.load(Path(args.initial_q2d).expanduser().resolve(), allow_pickle=False)
    if not np.issubdtype(q_2d_array.dtype, np.floating):
        raise ValueError(f"V3 Q2D checkpoint must be real floating point, got {q_2d_array.dtype}")
    if not np.all(np.isfinite(q_2d_array)):
        raise ValueError("V3 Q2D checkpoint contains NaN or Inf")
    q_2d = q_2d_array
    metadata["initial_condition"]["qualified_mother"] = v3_source_provenance
    metadata["initial_condition"]["qualification_manifest"] = v3_manifest

np.save(
    output_dir / "Q2D_initial.npy",
    np.asarray(q_2d)
    if not isinstance(q_2d, dict)
    else np.stack(
        [q_2d[name].numpy() for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")],
        axis=-1,
    ),
)
if args.initialization_protocol == "v1":
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
if args.initialization_protocol == "v1":
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
elif args.initialization_protocol == "v3-mother":
    q_initial_condition = extruded_2d_unbiased_rotation(
        (Nx, Ny, Nz),
        Q_2d=q_2d,
        boundary_conditions=Q_BC,
        rotation_rms=0.0,
        z_modes=(1,),
        seed=seed,
        dtype=real_dtype,
    )
else:
    q_initial_condition = extruded_2d_unbiased_rotation(
        (Nx, Ny, Nz),
        Q_2d=q_2d,
        boundary_conditions=Q_BC,
        rotation_rms=args.v3_rotation_rms,
        max_mode_x=args.v3_rotation_max_mode_x,
        max_mode_y=args.v3_rotation_max_mode_y,
        z_modes=tuple(args.v3_rotation_z_modes),
        seed=args.v3_perturbation_seed,
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
q_linear_operator = beris_edwards_linear_operator(
    q2_Q,
    ldg_a=args.ldg_a,
    ldg_l1=ldg_l1,
    rotational_viscosity=rotational_viscosity,
)

# --- Add active fields ---
solver.model.add_dynamic_field(
    "Qxx",
    init = Qxx_0,
    L_hat = q_linear_operator,
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qxy",
    init =  Qxy_0,
    L_hat = q_linear_operator,
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qxz",
    init = Qxz_0,
    L_hat = q_linear_operator,
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qyy",
    init =  Qyy_0,
    L_hat = q_linear_operator,
    boundary_conditions = Q_BC,
)
solver.model.add_dynamic_field(
    "Qyz",
    init =  Qyz_0,
    L_hat = q_linear_operator,
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
    BerisEdwardsQNonlinearModel(
        spectral_projector,
        Q_BC,
        ldg_b=args.ldg_b,
        ldg_c=args.ldg_c,
        rotational_viscosity=rotational_viscosity,
        flow_alignment=ALIGNMENT_PARAMETER,
    )
)
solver.model.set_static_compute_model(
    BerisEdwardsFreeSlipStokes(
        solver,
        spectral_projector=spectral_projector,
        beta_value=beta,
        friction=fric,
        viscosity=eta,
        ldg_a=args.ldg_a,
        ldg_b=args.ldg_b,
        ldg_c=args.ldg_c,
        ldg_l1=ldg_l1,
        flow_alignment=ALIGNMENT_PARAMETER,
        cache_force_diagnostics=ENABLE_DIAGNOSTICS,
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

start_step = (
    args.v3_mother_start_step
    if args.initialization_protocol == "v3-mother"
    else 0
)

diagnostic_history = []
v3_mother_diagnostic_history = []
q2d_z_invariance_history = []
start = time.time()
pbar = trange(steps)


def save_snapshot(i):
    q_snapshot = torch.stack([
        solver.model.fields[name].detach().cpu()
        for name in ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"]
    ])  # shape -> (5, batch, Nx, Ny, Nz)
    q_snapshot = q_snapshot.permute(1, 2, 3, 4, 0)
    q_array = q_snapshot[0].numpy()
    if args.save_layout == "q2d":
        q_mean = np.mean(q_array, axis=2)
        deviation = q_array - q_mean[:, :, None, :]
        deviation_rms = float(np.sqrt(np.mean(deviation * deviation)))
        field_rms = float(np.sqrt(np.mean(q_mean * q_mean)))
        relative_rms = deviation_rms / max(field_rms, np.finfo(q_array.dtype).tiny)
        maximum_absolute = float(np.max(np.abs(deviation)))
        q2d_z_invariance_history.append({
            "step": int(i),
            "relative_rms": relative_rms,
            "maximum_absolute": maximum_absolute,
        })
        if relative_rms > args.q2d_z_invariance_rtol:
            raise RuntimeError(
                f"V3 mother lost z invariance at step {i}: relative RMS "
                f"{relative_rms:.6e} exceeds {args.q2d_z_invariance_rtol:.6e}"
            )
        np.save(output_dir / f"Q2D_{i}.npy", q_mean)
    else:
        np.save(output_dir / f"Q_{i}.npy", q_array)

    if SAVE_HYDRODYNAMICS:
        u_snapshot = torch.stack([
            solver.model.fields[name].detach().cpu()
            for name in ["ux", "uy", "uz"]
        ])  # shape -> (3, batch, Nx, Ny, Nz)
        u_snapshot = u_snapshot.permute(1, 2, 3, 4, 0)
        np.save(output_dir / f"u_{i}.npy", u_snapshot[0].numpy())

        p_snapshot = solver.model.fields["p"].detach().cpu()
        np.save(output_dir / f"p_{i}.npy", p_snapshot[0].numpy())


def record_v3_mother_observables(i):
    fields = solver.model.fields
    q_components = tuple(fields[name] for name in Q_COMPONENTS)
    q_gradients = tuple(
        tuple(fields.gradient(name, axis=axis) for name in Q_COMPONENTS)
        for axis in range(3)
    )
    free_energy = beris_edwards_free_energy_density(
        q_components,
        q_gradients,
        ldg_a=args.ldg_a,
        ldg_b=args.ldg_b,
        ldg_c=args.ldg_c,
        ldg_l1=ldg_l1,
    )
    ux, uy, uz = (fields[name] for name in ("ux", "uy", "uz"))
    vorticity_x = fields.gradient("uz", axis=1) - fields.gradient("uy", axis=2)
    vorticity_y = fields.gradient("ux", axis=2) - fields.gradient("uz", axis=0)
    vorticity_z = fields.gradient("uy", axis=0) - fields.gradient("ux", axis=1)
    speed_squared = ux.square() + uy.square() + uz.square()
    vorticity_squared = (
        vorticity_x.square() + vorticity_y.square() + vorticity_z.square()
    )
    v3_mother_diagnostic_history.append((
        int(i),
        float(i * dt),
        torch.sqrt(torch.mean(speed_squared)).item(),
        (0.5 * torch.mean(speed_squared)).item(),
        torch.sqrt(torch.mean(vorticity_squared)).item(),
        torch.mean(S_from_Q(dict(zip(Q_COMPONENTS, q_components)))).item(),
        torch.mean(free_energy).item(),
    ))


def record_step_state(i):
    if ENABLE_DIAGNOSTICS and i % DIAGNOSTIC_INTERVAL == 0:
        div_max, div_rms, div_rel = divergence_stats(solver.model.fields)
        static_model = solver.model.static_model
        wall_mom_max, wall_mom_rms = wall_normal_momentum_stats(
            solver.model.fields,
            static_model,
        )
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
    if (
        args.initialization_protocol == "v3-mother"
        and i % DIAGNOSTIC_INTERVAL == 0
    ):
        record_v3_mother_observables(i)
    if i >= SAVE_START_STEP and i % SAVE_INTERVAL == 0:
        save_snapshot(i)


for local_step in pbar:
    i = start_step + local_step
    solver.run(1, pre_update_callback=lambda _solver, _step, i=i: record_step_state(i))

final_step = start_step + steps
solver.refresh_static_fields()
if (
    args.initialization_protocol == "v3-mother"
    and final_step % DIAGNOSTIC_INTERVAL == 0
):
    record_v3_mother_observables(final_step)
save_snapshot(final_step)

end = time.time()
print(f"Elapsed time: {end - start:.6f} seconds")
if ENABLE_DIAGNOSTICS:
    final_div_max, final_div_rms, final_div_rel = divergence_stats(solver.model.fields)
    static_model = solver.model.static_model
    final_wall_mom_max, final_wall_mom_rms = wall_normal_momentum_stats(
        solver.model.fields,
        static_model,
    )
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

if args.initialization_protocol == "v3-mother":
    mother_diagnostics = np.array(
        v3_mother_diagnostic_history,
        dtype=[
            ("step", np.int64),
            ("time", np.float64),
            ("u_rms", np.float64),
            ("kinetic_energy_density", np.float64),
            ("vorticity_rms", np.float64),
            ("mean_S", np.float64),
            ("ldg_free_energy_density", np.float64),
        ],
    )
    np.save(output_dir / "mother_online_diagnostics.npy", mother_diagnostics)
    np.savetxt(
        output_dir / "mother_online_diagnostics.csv",
        mother_diagnostics,
        delimiter=",",
        header=(
            "step,time,u_rms,kinetic_energy_density,vorticity_rms,mean_S,"
            "ldg_free_energy_density"
        ),
        comments="",
    )

metadata["completed_steps"] = final_step
metadata["elapsed_seconds"] = end - start
if args.save_layout == "q2d":
    metadata["q2d_z_invariance"] = {
        "definition": "RMS_z(Q-mean_z(Q))/RMS(mean_z(Q))",
        "relative_tolerance": args.q2d_z_invariance_rtol,
        "all_saved_frames_passed": True,
        "maximum_relative_rms": max(
            entry["relative_rms"] for entry in q2d_z_invariance_history
        ),
        "history": q2d_z_invariance_history,
    }
    (output_dir / "z_invariance.json").write_text(
        json.dumps(metadata["q2d_z_invariance"], indent=2) + "\n"
    )
metadata["numerics"]["spectral_refresh"]["actual_count"] = (
    solver.integrator.refresh_count
)
if args.initialization_protocol == "v3-extruded":
    final_v3_identities = {
        "manifest_sha256": file_sha256(
            Path(v3_source_provenance["manifest_path"])
        ),
        "checkpoint_sha256": file_sha256(
            Path(v3_source_provenance["checkpoint_path"])
        ),
        "qualification_report_sha256": file_sha256(
            Path(v3_source_provenance["qualification_report_path"])
        ),
    }
    expected_v3_identities = {
        name: v3_source_provenance[name]
        for name in final_v3_identities
    }
    if final_v3_identities != expected_v3_identities:
        raise RuntimeError("a bound V3 mother input changed during the 3D run")
    metadata["initial_condition"]["bound_inputs_unchanged_at_completion"] = True
if args.v3_mother_restart_q2d is not None:
    final_restart_sha256 = file_sha256(restart_path)
    if final_restart_sha256 != restart_sha256:
        raise RuntimeError("V3 mother restart Q2D changed during continuation")
    metadata["initial_condition"]["restart_input_unchanged_at_completion"] = True
(output_dir / "COMPLETE").write_text("complete\n")
write_run_metadata(output_dir, metadata, status="complete")
