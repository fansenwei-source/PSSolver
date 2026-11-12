"""Run the qualified Plane model through the public simulation API."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from tqdm import trange

from pssolver.api.declarations import (
    GeneratedInitialCondition,
    Output,
    SemiImplicitEuler,
    SpectralNumerics,
    TorchSpectralExecution,
)
from pssolver.api.models import ShendrukPlaneBerisEdwards
from pssolver.api.runner import run_simulation
from pssolver.api.simulation import Simulation, _free_slip_plane_declaration
from pssolver.core.numerics import (
    DEALIAS_RULE_FRACTIONS,
    DEFAULT_DEALIAS_RULE,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    PROJECTED_TRANSFORM_EXECUTION_MODES,
    SPECTRAL_STORAGE_MODES,
    SpectralStorage,
)


DEFAULT_ZERO_MODE_POLICY = "zero_mean"
DEFAULT_FRICTION_MODE_FRIC = 0.1
DEFAULT_SPECTRAL_REFRESH_TIME = 0.2
DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE = "spectral"
DEFAULT_STRESS_DIVERGENCE_SUM_SPACE = "spectral"
DEFAULT_PLANE_SPECTRAL_STORAGE = SpectralStorage.HERMITIAN_HALF.value
DEFAULT_POINTWISE_EXECUTION = "compile"
POINTWISE_EXECUTION_MODES = ("eager", "compile")


def _parser() -> argparse.ArgumentParser:
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
            "Tangential plug-flow convention. zero_mean uses fric=0 and "
            "fixes <ux>=<uy>=0; friction retains plug modes with positive "
            "drag."
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
    parser.add_argument(
        "--projected-transform-execution",
        choices=PROJECTED_TRANSFORM_EXECUTION_MODES,
        default=DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
        help=(
            "A/B control for transforms directly coupled to spectral "
            "projection. truncated is the H100-qualified production default "
            "and skips discarded DCT/DST modes while preserving the selected "
            "backend's native storage shape; full is the validated rollback "
            "path."
        ),
    )
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument(
        "--initial-s",
        dest="initial_s",
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
        "--molecular-field-linear-space",
        choices=("physical", "spectral"),
        default=DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE,
        help=(
            "Evaluation space for the linear L1 laplacian in the raw "
            "molecular field. spectral is the H100-qualified production "
            "default; physical retains the validated compatibility path."
        ),
    )
    parser.add_argument(
        "--stress-divergence-sum-space",
        choices=("physical", "spectral"),
        default=DEFAULT_STRESS_DIVERGENCE_SUM_SPACE,
        help=(
            "Assembly space for compatible stress-divergence derivatives. "
            "spectral is the H100-qualified production default and reduces "
            "inverse transforms; physical retains the validated compatibility "
            "path."
        ),
    )
    parser.add_argument(
        "--pointwise-execution",
        choices=POINTWISE_EXECUTION_MODES,
        default=DEFAULT_POINTWISE_EXECUTION,
        help=(
            "Execution policy for the four pure Beris--Edwards pointwise "
            "kernels. compile is the H100-qualified production default and "
            "uses fixed-shape, full-graph TorchInductor with no silent "
            "fallback; eager retains the validated compatibility path."
        ),
    )
    parser.add_argument(
        "--transform-execution-order",
        choices=("legacy", "real_first"),
        default=DEFAULT_TRANSFORM_EXECUTION_ORDER,
        help=(
            "Tensor-product transform execution plan (default: real_first). "
            "real_first applies DCT/DST axes before periodic FFTs so their "
            "matrix products use real arithmetic; legacy retains the "
            "historical axis order."
        ),
    )
    parser.add_argument(
        "--spectral-storage",
        choices=SPECTRAL_STORAGE_MODES,
        default=DEFAULT_PLANE_SPECTRAL_STORAGE,
        help=(
            "Native modal storage. hermitian_half is the H100-qualified "
            "Plane default and packs the positive-y spectrum; full_complex "
            "is the validated rollback."
        ),
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
            "Physical-time interval between dynamic real-to-spectral "
            f"rebuilds. The default is {DEFAULT_SPECTRAL_REFRESH_TIME:g}. "
            "The interval must be an integer multiple of dt."
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
    parser.add_argument(
        "--disable-q-gradient-reuse",
        action="store_true",
        help=(
            "Recompute Q gradients independently in the static and nonlinear "
            "models instead of using the guarded single-step cache."
        ),
    )
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
        "--checkpoint-interval",
        type=int,
        default=None,
        help=(
            "Write an exact same-backend workflow checkpoint after each "
            "positive multiple of this many completed steps."
        ),
    )
    parser.add_argument(
        "--restart-from",
        type=Path,
        default=None,
        help=(
            "Start a new output directory from a complete Stage O.3 "
            "same-backend checkpoint; --steps is the number of additional "
            "steps."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved parameters without allocating the solver.",
    )
    return parser


def _simulation(args: argparse.Namespace) -> Simulation:
    refresh = None
    if not args.disable_spectral_refresh:
        if args.spectral_refresh_steps is not None:
            refresh = {"steps": args.spectral_refresh_steps}
        else:
            interval = args.spectral_refresh_time
            refresh = {"physical_time": DEFAULT_SPECTRAL_REFRESH_TIME if interval is None else interval}
    geometry, boundaries = _free_slip_plane_declaration(
        (args.nx, args.ny, args.nz), (args.lx, args.ly, args.height),
    )
    values = vars(args)
    return Simulation(
        model=ShendrukPlaneBerisEdwards(**{
            key: values[key] for key in (
                "activity_number", "parameterization", "frank_k", "coefficient_min",
                "coefficient_max", "ldg_a", "ldg_b", "ldg_c", "gamma", "flow_alignment",
                "eta", "beta", "zero_mode_policy", "friction_mode_fric",
            )
        }),
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype=args.dtype, dealias_rule=args.dealias_rule,
            projected_transform_execution=args.projected_transform_execution,
            spectral_storage=args.spectral_storage,
            hermitian_axis=1 if args.spectral_storage == "hermitian_half" else None,
            transform_execution_order=args.transform_execution_order,
        ),
        time=SemiImplicitEuler(args.dt, refresh=refresh),
        initial_condition=GeneratedInitialCondition("extruded_defect_gas", {
            key: values[key] for key in (
                "seed", "num_defect_pairs", "defect_min_separation", "defect_core_radius",
                "background_angle", "twist_amplitude", "twist_modes", "initial_s",
            )
        }),
        execution=TorchSpectralExecution(device=args.device, options={
            key: values[key] for key in (
                "tf32", "molecular_field_linear_space", "stress_divergence_sum_space",
                "pointwise_execution", "disable_q_gradient_reuse",
            )
        }),
        output=Output(
            directory=args.output_dir, steps=args.steps, save_interval=args.save_interval,
            diagnostic_interval=args.diagnostic_interval, save_start_step=args.save_start_step,
            diagnostics=args.diagnostics, save_hydrodynamics=args.save_hydrodynamics,
            checkpoint_interval=args.checkpoint_interval, restart_from=args.restart_from,
            dry_run=args.dry_run, validation_config_sha256=args.validation_config_sha256,
        ),
    )


def _print_workflow_summary(workflow_result) -> None:
    print(f"Elapsed time: {workflow_result.elapsed_seconds:.6f} seconds")
    if workflow_result.diagnostics:
        final_diagnostic = workflow_result.diagnostics[-1]
        print(
            "Final div(u) diagnostic: "
            f"max={final_diagnostic.div_max:.6e}, "
            f"rms={final_diagnostic.div_rms:.6e}, "
            f"relative={final_diagnostic.div_rel:.6e}"
        )
        print(
            "Final Schur saddle solve diagnostic: "
            f"iterations={int(final_diagnostic.schur_iterations)}, "
            f"abs_residual={final_diagnostic.schur_abs_residual:.6e}, "
            f"rel_residual={final_diagnostic.schur_rel_residual:.6e}"
        )
        print(
            "Final wall normal momentum diagnostic: "
            f"max={final_diagnostic.wall_normal_momentum_max:.6e}, "
            f"rms={final_diagnostic.wall_normal_momentum_rms:.6e}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Translate the command options into a public declaration and run it."""
    parser = _parser()
    argv = sys.argv[1:] if argv is None else argv
    if any(arg.split("=", 1)[0] == "--runtime-path" for arg in argv):
        parser.error("--runtime-path was removed; see docs/migration_0_4_to_0_5.md")
    args = parser.parse_args(argv)
    digest = args.validation_config_sha256
    if digest is not None and (
        len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
    ):
        parser.error("--validation-config-sha256 must be exactly 64 lowercase hexadecimal characters")
    try:
        simulation = _simulation(args)
        result = run_simulation(
            simulation, progress=None if args.dry_run else trange(args.steps), emit_metadata=True,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if result.elapsed_seconds is not None:
        _print_workflow_summary(result)
    return 0
