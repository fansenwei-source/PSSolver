"""Shendruk benchmark driver with inertial Beris--Edwards hydrodynamics.

This independent script is derived from Plane_beris_edwards_stokes.py.  It
retains the V1 initial condition, Shendruk parameterization, mixed-basis
free-slip geometry, dealiasing, and output layout while replacing the
quasistatic Stokes solve by the incompressible momentum equation

    rho (partial_t u + u dot grad u)
        = -grad(p) + eta lap(u) - fric u + div(Pi_nem),
    div(u) = 0.

Once inertia is present this is a Navier--Stokes--Beris--Edwards model, not a
Stokes model.  The Q equation remains

    (partial_t + u dot grad) Q - S(E, Omega, Q) = H / gamma

with the full co-rotational flow-alignment term and the one-constant
Landau--de Gennes molecular field.  Up to an isotropic term absorbed into
pressure, the complete nematic stress is

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
* uniform tangential velocity modes evolve dynamically by default;
* the default no-padding dealiasing rule is cubic_half.

The temporal discretization is first-order IMEX Euler.  Nematic and convective
forces are evaluated at time n, while viscosity, optional substrate friction,
pressure, and incompressibility are solved together at time n+1 using the
existing mixed DCT/DST Schur solver with rho/dt added to its Helmholtz
coefficient.  The convective term is explicitly dealiased.  This removes the
largest equation-level difference from the paper, but the V1 initialization
and spectral discretization are still not paper-identical.
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
# Tangential plug-flow handling.  In contrast to the quasistatic fric=0
# problem, rho/dt makes the uniform tangential modes nonsingular.  The default
# therefore evolves their physical mean momentum.  An optional zero_mean mode
# remains an explicit modeling sensitivity, not a pressure gauge: free Q
# anchoring does not guarantee zero volume-averaged total nematic tangential
# force. Its active part alone can already be nonzero, e.g.
#
#   <f_x> is proportional to [<Q_xz>_{xy}]_{z=0}^{z=L_z}.
#
# The paper-compatible choice in this script is mean_flow_policy=evolve and
# fric=0.  Positive friction and zero_mean are retained only as sensitivities.
DEFAULT_MEAN_FLOW_POLICY = "evolve"
DEFAULT_FRIC = 0.0
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
    "Plane_beris_edwards_navier_stokes.py",
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
            "Run the reusable Beris--Edwards Q model with inertial, "
            "incompressible free-slip Navier--Stokes dynamics and the "
            "Shendruk benchmark parameterization."
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
        "--density",
        type=float,
        default=1.0,
        help="Constant mass density rho; the Shendruk value is 1.",
    )
    parser.add_argument(
        "--fric",
        type=float,
        default=DEFAULT_FRIC,
        help=(
            "Optional linear substrate drag in the momentum equation. "
            "The Shendruk benchmark value is 0."
        ),
    )
    parser.add_argument(
        "--mean-flow-policy",
        choices=("evolve", "zero_mean"),
        default=DEFAULT_MEAN_FLOW_POLICY,
        help=(
            "evolve advances the two uniform tangential momentum modes; "
            "zero_mean explicitly removes them after every momentum solve."
        ),
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
        "density": args.density,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        parser.error(f"these values must be positive: {invalid}")
    if args.S_initial <= 0:
        parser.error("--initial-s must be positive")
    if not math.isfinite(args.eta) or args.eta <= 0:
        parser.error("--eta must be finite and positive")
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
    if not math.isfinite(args.fric) or args.fric < 0:
        parser.error("--fric must be finite and non-negative")
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


class StoredPressureModel(torch.nn.Module):
    """Keep the pressure written by the coupled momentum integrator current."""

    def forward(self, fields, params):
        del params
        return fields["p.hat"].unsqueeze(0)


class BerisEdwardsNavierStokesRHS(torch.nn.Module):
    """Explicit Q, nematic-force, and advective parts of the coupled PDE.

    The returned velocity entries have acceleration units,

        N_u = div(Pi_nem)/rho - (u dot grad)u.

    Viscosity, friction, pressure, and the time derivative are deliberately
    absent here because the integrator treats them in one implicit saddle
    solve.
    """

    def __init__(
        self,
        solver,
        spectral_projector,
        *,
        density,
        dt,
        beta_value,
        physical_friction,
        viscosity,
        ldg_a,
        ldg_b,
        ldg_c,
        ldg_l1,
        rotational_viscosity,
        flow_alignment,
        mean_flow_policy,
        cache_diagnostics=False,
    ):
        super().__init__()
        if not math.isfinite(float(density)) or density <= 0:
            raise ValueError("density must be finite and positive")
        if not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(float(physical_friction)) or physical_friction < 0:
            raise ValueError("physical friction must be finite and non-negative")
        if mean_flow_policy not in ("evolve", "zero_mean"):
            raise ValueError("mean_flow_policy must be 'evolve' or 'zero_mean'")

        self.density = float(density)
        self.dt = float(dt)
        self.physical_friction = float(physical_friction)
        self.viscosity = float(viscosity)
        self.mean_flow_policy = mean_flow_policy
        self.cache_diagnostics = bool(cache_diagnostics)
        self.spectral_projector = spectral_projector

        self.q_model = BerisEdwardsQNonlinearModel(
            spectral_projector,
            Q_BC,
            ldg_b=ldg_b,
            ldg_c=ldg_c,
            rotational_viscosity=rotational_viscosity,
            flow_alignment=flow_alignment,
        )

        # Reuse one adapter both to evaluate the complete nematic force and to
        # hold the backward-Euler saddle operator.  Its Helmholtz ``friction``
        # is the effective rho/dt + physical_friction coefficient, not an
        # extra physical drag.  This avoids allocating a duplicate 3D Schur
        # solver on the GPU.
        self.force_evaluator = BerisEdwardsFreeSlipStokes(
            solver,
            spectral_projector=spectral_projector,
            beta_value=beta_value,
            friction=self.density / self.dt + self.physical_friction,
            viscosity=viscosity,
            ldg_a=ldg_a,
            ldg_b=ldg_b,
            ldg_c=ldg_c,
            ldg_l1=ldg_l1,
            flow_alignment=flow_alignment,
            cache_force_diagnostics=False,
            zero_mode_policy="friction",
        )

        # Backward Euler gives the momentum Helmholtz coefficient
        # rho/dt + fric + eta*k^2.  It is strictly positive even at k=0, so
        # the old quasistatic plug-flow null space is absent.
        self.momentum_solver = self.force_evaluator

        self.last_total_tangential_force_mean = None
        self.last_active_tangential_force_mean = None
        self.last_passive_tangential_force_mean = None
        self.last_advective_tangential_mean = None
        self.last_projected_normal_force = None

    def _project_hat(self, fields, tensor, boundary_conditions):
        return self.spectral_projector.project(
            fields.transform_tensor(tensor, boundary_conditions),
            boundary_conditions,
        )

    @staticmethod
    def _advective_acceleration(fields):
        ux, uy, uz = (fields[name] for name in ("ux", "uy", "uz"))
        velocities = (ux, uy, uz)
        acceleration = []
        for name in ("ux", "uy", "uz"):
            acceleration.append(
                sum(
                    velocity * fields.gradient(name, axis=axis)
                    for axis, velocity in enumerate(velocities)
                )
            )
        return torch.stack(acceleration)

    def forward(self, fields, params):
        q_rhs_hats = self.q_model(fields, params)
        force, active_tangential_force = self.force_evaluator.compute_nematic_force(
            fields,
            params["alpha"],
        )
        advective = self._advective_acceleration(fields)

        total_mean = force[:2].mean(dim=(-3, -2, -1)).detach()
        active_mean = active_tangential_force.mean(
            dim=(-3, -2, -1)
        ).detach()
        self.last_total_tangential_force_mean = total_mean
        self.last_active_tangential_force_mean = active_mean
        self.last_passive_tangential_force_mean = total_mean - active_mean
        self.last_advective_tangential_mean = advective[:2].mean(
            dim=(-3, -2, -1)
        ).detach()

        force_t_hat = self._project_hat(fields, force[:2], U_TANGENTIAL_BC)
        force_n_hat = self._project_hat(fields, force[2], U_NORMAL_BC)
        advective_t_hat = self._project_hat(
            fields,
            advective[:2],
            U_TANGENTIAL_BC,
        )
        advective_n_hat = self._project_hat(
            fields,
            advective[2],
            U_NORMAL_BC,
        )

        if self.cache_diagnostics:
            self.last_projected_normal_force = fields.inverse_transform_tensor(
                force_n_hat,
                U_NORMAL_BC,
            ).detach()
        else:
            self.last_projected_normal_force = None

        velocity_rhs_hats = torch.cat(
            (
                force_t_hat / self.density - advective_t_hat,
                (force_n_hat / self.density - advective_n_hat).unsqueeze(0),
            ),
            dim=0,
        )
        return torch.cat((q_rhs_hats, velocity_rhs_hats), dim=0)


class IncompressibleNavierStokesEulerIntegrator(SemiImplicitEulerIntegrator):
    """First-order IMEX step with an exact mixed-basis momentum saddle solve."""

    Q_NAMES = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
    VELOCITY_NAMES = ("ux", "uy", "uz")

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        expected = (*self.Q_NAMES, *self.VELOCITY_NAMES)
        actual = tuple(model.get_field_order())
        if actual != expected:
            raise ValueError(
                "inertial integrator requires dynamic field order "
                f"{expected}, got {actual}"
            )
        if tuple(model.stat_fields[0][:1]) != ("p",):
            raise ValueError("inertial integrator requires one static pressure field")

        self.rhs_model = model.nlmodel
        if not isinstance(self.rhs_model, BerisEdwardsNavierStokesRHS):
            raise TypeError("unexpected nonlinear model for inertial integrator")
        self.spectral_projector = model.spectral_projector
        self.momentum_solver = self.rhs_model.momentum_solver
        self.density = self.rhs_model.density
        self.mass_coefficient = self.density / self.dt
        self.mean_flow_policy = self.rhs_model.mean_flow_policy
        self.q_indices = tuple(range(5))
        self.velocity_indices = tuple(range(5, 8))
        self.pressure_index = model.fields.name_to_idx["p"]
        self.last_momentum_residual_max = math.nan
        self.last_momentum_residual_rms = math.nan

    def _apply_mean_flow_policy(self, ux_hat, uy_hat):
        if self.mean_flow_policy == "zero_mean":
            ux_hat[..., 0, 0, 0] = 0
            uy_hat[..., 0, 0, 0] = 0

    def _helmholtz_project_velocity(self, velocity_hats):
        """Restore div-free velocity after an optional spectral refresh."""
        ux_hat, uy_hat, uz_hat = velocity_hats
        # A^{-1} is strictly positive because rho/dt > 0.
        fx_hat = ux_hat / self.momentum_solver.a_tangential_inv
        fy_hat = uy_hat / self.momentum_solver.a_tangential_inv
        fz_hat = uz_hat / self.momentum_solver.a_normal_inv

        saved_diagnostics = (
            self.momentum_solver.last_pressure_hat,
            self.momentum_solver.last_pressure_iterations,
            self.momentum_solver.last_pressure_residual,
            self.momentum_solver.last_pressure_relative_residual,
        )
        projected = self.momentum_solver.solve_force_hats(
            fx_hat,
            fy_hat,
            fz_hat,
        )[:3]
        (
            self.momentum_solver.last_pressure_hat,
            self.momentum_solver.last_pressure_iterations,
            self.momentum_solver.last_pressure_residual,
            self.momentum_solver.last_pressure_relative_residual,
        ) = saved_diagnostics
        self._apply_mean_flow_policy(projected[0], projected[1])
        return projected

    def _refresh_dynamic_spectra(self):
        super()._refresh_dynamic_spectra()
        fields = self.model.fields
        fields.spectral[:5] = self.spectral_projector.project(
            fields.spectral[:5],
            Q_BC,
        )
        projected_velocity = self._helmholtz_project_velocity(
            fields.spectral[5:8]
        )
        fields.spectral[5:8] = torch.stack(projected_velocity)
        for group in self.dynamic_transform_groups:
            fields.spatial[group] = fields.inverse_transform_group(group)

    def _cache_momentum_residual(
        self,
        rhs_hats,
        velocity_hats,
        pressure_hat,
    ):
        if not self.rhs_model.cache_diagnostics:
            return
        ux_hat, uy_hat, uz_hat = velocity_hats
        grad_px, grad_py, grad_pz = self.momentum_solver.pressure_gradient_hats(
            pressure_hat
        )
        residual_hats = (
            ux_hat / self.momentum_solver.a_tangential_inv + grad_px - rhs_hats[0],
            uy_hat / self.momentum_solver.a_tangential_inv + grad_py - rhs_hats[1],
            uz_hat / self.momentum_solver.a_normal_inv + grad_pz - rhs_hats[2],
        )
        fields = self.model.fields
        residual = torch.stack(
            (
                fields.inverse_transform_tensor(residual_hats[0], U_TANGENTIAL_BC),
                fields.inverse_transform_tensor(residual_hats[1], U_TANGENTIAL_BC),
                fields.inverse_transform_tensor(residual_hats[2], U_NORMAL_BC),
            )
        )
        residual_abs = residual.abs()
        self.last_momentum_residual_max = residual_abs.max().item()
        self.last_momentum_residual_rms = torch.sqrt(
            torch.mean(residual_abs.square())
        ).item()

    def step(self, pre_update_callback=None):
        if pre_update_callback is not None:
            pre_update_callback()

        fields = self.model.fields
        nonlinear_hats = self.model.compute_nonlinear()

        # Q: explicit nonlinear terms, implicit linear molecular relaxation.
        fields.spectral[:5].add_(self.dt * nonlinear_hats[:5])
        fields.spectral[:5].div_(self.denom[:5])
        fields.spectral[:5] = self.spectral_projector.project(
            fields.spectral[:5],
            Q_BC,
        )

        # Momentum: [rho/dt + fric - eta*lap]u^{n+1}+grad(p^{n+1})
        #           = rho/dt*u^n + rho*N_u(Q^n,u^n).
        old_velocity_hats = fields.spectral[5:8]
        momentum_rhs_hats = (
            self.mass_coefficient * old_velocity_hats
            + self.density * nonlinear_hats[5:8]
        )
        if self.mean_flow_policy == "zero_mean":
            # Solve the explicitly constrained problem on the zero-mean
            # subspace, rather than solving first and silently overwriting a
            # nonzero physical residual afterwards.
            momentum_rhs_hats[0, ..., 0, 0, 0] = 0
            momentum_rhs_hats[1, ..., 0, 0, 0] = 0
        ux_hat, uy_hat, uz_hat, pressure_hat = (
            self.momentum_solver.solve_force_hats(
                momentum_rhs_hats[0],
                momentum_rhs_hats[1],
                momentum_rhs_hats[2],
            )
        )
        self._apply_mean_flow_policy(ux_hat, uy_hat)
        self._cache_momentum_residual(
            momentum_rhs_hats,
            (ux_hat, uy_hat, uz_hat),
            pressure_hat,
        )
        fields.spectral[5:8] = torch.stack((ux_hat, uy_hat, uz_hat))
        fields.spectral[self.pressure_index] = pressure_hat

        for group in self.dynamic_transform_groups:
            fields.spatial[group] = fields.inverse_transform_group(group)
        fields.spatial[self.pressure_index] = fields.inverse_transform(
            self.pressure_index
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
mean_flow_policy = args.mean_flow_policy
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
rho = float(args.density)
fric = float(args.fric)
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
    "script": "Plane_beris_edwards_navier_stokes.py",
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
        "variant": "beris_edwards_complete_nematic_stress_navier_stokes",
        "stage": "inertial_beris_edwards_navier_stokes_v1_pilot",
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
                "rho*(partial_t*u+u.grad(u))="
                "-grad(p)+eta*lap(u)-fric*u+div(Pi_nematic)"
            ),
            "regime": "incompressible_navier_stokes_beris_edwards",
            "constant_density": rho,
            "initial_velocity": "u=0",
            "time_discretization": (
                "first_order_IMEX_Euler; explicit_nematic_and_advection; "
                "implicit_viscosity_friction_pressure"
            ),
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
            "rho": rho,
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
        "mean_flow_policy": mean_flow_policy,
        "uniform_tangential_mode": (
            "advanced_by_mean_momentum_equation"
            if mean_flow_policy == "evolve"
            else "explicitly_projected_to_zero"
        ),
        "pressure_solver": "backward_euler_free_slip_modal_schur_complement",
        "time_integrator": "first_order_coupled_IMEX_Euler",
        "implicit_momentum_helmholtz": "rho/dt+fric+eta*k^2",
        "explicit_terms": ["nematic_force", "u_dot_grad_u"],
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
    "reproduction_status": "development_inertial_navier_stokes_v1_stage",
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
    "mean_flow_policy": mean_flow_policy,
    "dealias_rule": dealias_rule,
    "dealias_fraction": DEALIAS_RULE_FRACTIONS[dealias_rule],
    "ldg_coefficients": {"A": args.ldg_a, "B": args.ldg_b, "C": args.ldg_c},
    "gamma": args.gamma,
    "flow_alignment": ALIGNMENT_PARAMETER,
    "density": rho,
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
            "homogeneous velocity free-slip is kinematic and does not impose "
            "zero total nematic traction"
        ),
        (
            "the first-order IMEX spectral time discretization differs from "
            "the paper's hybrid finite-difference/lattice-Boltzmann method"
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
solver.integrator_cl = IncompressibleNavierStokesEulerIntegrator
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

# --- Add inertial velocity fields and the pressure multiplier ---
# The paper does not prescribe a nonzero initial flow, so V1 starts from rest.
# Velocity has no diagonal PDEModel linear operator here: viscosity and drag
# are treated together with pressure by the coupled momentum saddle solve.
zero_velocity = torch.zeros(
    (Nx, Ny, Nz),
    device=device,
    dtype=real_dtype,
)
zero_tangential_operator = torch.zeros_like(solver.get_q2(U_TANGENTIAL_BC))
zero_normal_operator = torch.zeros_like(solver.get_q2(U_NORMAL_BC))
solver.model.add_dynamic_field(
    "ux",
    init=zero_velocity,
    L_hat=zero_tangential_operator,
    boundary_conditions=U_TANGENTIAL_BC,
)
solver.model.add_dynamic_field(
    "uy",
    init=zero_velocity.clone(),
    L_hat=zero_tangential_operator,
    boundary_conditions=U_TANGENTIAL_BC,
)
solver.model.add_dynamic_field(
    "uz",
    init=zero_velocity.clone(),
    L_hat=zero_normal_operator,
    boundary_conditions=U_NORMAL_BC,
)
solver.model.add_static_field("p", boundary_conditions=PRESSURE_MODAL_BC)

solver.model.set_nonlinear_model(
    BerisEdwardsNavierStokesRHS(
        solver,
        spectral_projector,
        density=rho,
        dt=dt,
        beta_value=beta,
        physical_friction=fric,
        viscosity=eta,
        ldg_a=args.ldg_a,
        ldg_b=args.ldg_b,
        ldg_c=args.ldg_c,
        ldg_l1=ldg_l1,
        rotational_viscosity=rotational_viscosity,
        flow_alignment=ALIGNMENT_PARAMETER,
        mean_flow_policy=mean_flow_policy,
        cache_diagnostics=ENABLE_DIAGNOSTICS,
    )
)
solver.model.set_static_compute_model(StoredPressureModel())

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
metadata["initial_condition"]["projected_q_sha256"] = tensor_sha256(
    tuple(
        solver.model.fields[name]
        for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
    )
)
metadata["initial_condition"]["velocity"] = {
    "name": "quiescent",
    "components": [0.0, 0.0, 0.0],
}
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
        (0.5 * rho * torch.mean(speed_squared)).item(),
        torch.sqrt(torch.mean(vorticity_squared)).item(),
        torch.mean(S_from_Q(dict(zip(Q_COMPONENTS, q_components)))).item(),
        torch.mean(free_energy).item(),
    ))


def record_step_state(i):
    if ENABLE_DIAGNOSTICS and i % DIAGNOSTIC_INTERVAL == 0:
        div_max, div_rms, div_rel = divergence_stats(solver.model.fields)
        integrator = solver.integrator
        momentum_solver = integrator.momentum_solver
        diagnostic_history.append((
            i,
            div_max,
            div_rms,
            div_rel,
            momentum_solver.last_pressure_iterations,
            momentum_solver.last_pressure_residual,
            momentum_solver.last_pressure_relative_residual,
            integrator.last_momentum_residual_max,
            integrator.last_momentum_residual_rms,
        ))
        pbar.set_postfix(
            div_max=f"{div_max:.2e}",
            div_rms=f"{div_rms:.2e}",
            div_rel=f"{div_rel:.2e}",
            schur_it=momentum_solver.last_pressure_iterations,
            schur_rel=f"{momentum_solver.last_pressure_relative_residual:.2e}",
            mom_rms=f"{integrator.last_momentum_residual_rms:.2e}",
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
    integrator = solver.integrator
    momentum_solver = integrator.momentum_solver
    diagnostic_history.append((
        start_step + steps,
        final_div_max,
        final_div_rms,
        final_div_rel,
        momentum_solver.last_pressure_iterations,
        momentum_solver.last_pressure_residual,
        momentum_solver.last_pressure_relative_residual,
        integrator.last_momentum_residual_max,
        integrator.last_momentum_residual_rms,
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
            ("time_discrete_momentum_residual_max", np.float64),
            ("time_discrete_momentum_residual_rms", np.float64),
        ],
    )
    np.save(output_dir / "diagnostics.npy", diagnostic_array)
    np.savetxt(
        output_dir / "diagnostics.csv",
        diagnostic_array,
        delimiter=",",
        header="step,div_max,div_rms,div_rel,schur_iterations,schur_abs_residual,schur_rel_residual,time_discrete_momentum_residual_max,time_discrete_momentum_residual_rms",
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
        f"iterations={momentum_solver.last_pressure_iterations}, "
        f"abs_residual={momentum_solver.last_pressure_residual:.6e}, "
        f"rel_residual={momentum_solver.last_pressure_relative_residual:.6e}"
    )
    print(
        "Final time-discrete momentum residual: "
        f"max={integrator.last_momentum_residual_max:.6e}, "
        f"rms={integrator.last_momentum_residual_rms:.6e}"
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
