"""DAL control between saved pure-splay-loop Q tensors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from pssolver import prepare_new_run_directory, write_run_metadata
from pssolver.models.active_nematics import (
    Q_convention_metadata,
    Q_magnitude,
    S_from_Q,
    positive_equilibrium_S,
)
from pssolver.channel import (
    PRESSURE_MODAL_BC,
    Q_BC,
    Q_COMPONENTS,
    U_BC,
    build_active_nematic_channel,
)
from pssolver.control import (
    DiscreteAdjointLoop,
    FunctionalSemiImplicitStep,
    TemporalMaskControl,
    available_loss_functions,
    build_loss_function,
    conservative_block_average,
    core_centerline_geometry,
    loop_core_metrics,
    periodic_profile_shift,
    periodic_translate_axis,
    partition_mask_along_axis,
    partition_mask_rbf,
    smooth_box_mask,
    x_disturbance_profile,
)


DEFAULT_INITIAL = Path("data_optimal_control_loop_translation/Q_initial_pure_loop.npy")
DEFAULT_TARGET = Path("data_pure_splay_loop_x62/Q_pure_splay_loop.npy")
SOURCE_LENGTHS = (128.0, 10.0, 10.0)
DEFAULT_S_BULK = positive_equilibrium_S(-1.0, -6.0, 6.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize a bounded local activity protocol that translates a saved "
            "pure-splay loop from its initial Q tensor to its target Q tensor."
        )
    )
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Optional saved spatial mask. By default a path-covering mask is generated.",
    )
    parser.add_argument(
        "--auto-mask-margin",
        type=int,
        nargs=3,
        default=(16, 4, 4),
        metavar=("MX", "MY", "MZ"),
        help=(
            "Source-grid margin around the initial/target difference bounding box. "
            "With loop metadata, MX sets the x path padding while y-z use the localized envelope."
        ),
    )
    parser.add_argument(
        "--auto-mask-relative-threshold",
        type=float,
        default=0.2,
        help="Fraction of the maximum initial/target Q difference used to localize the mask.",
    )
    parser.add_argument("--mask-transition-width", type=float, default=1.5)
    parser.add_argument(
        "--transverse-mask-margin",
        type=float,
        default=1.5,
        help="Physical padding outside the loop radius in the localized y-z envelope.",
    )
    parser.add_argument(
        "--transverse-mask-transition-width",
        type=float,
        default=0.5,
        help="Physical smoothing width of the localized y-z envelope.",
    )
    parser.add_argument(
        "--control-wall-buffer",
        type=float,
        default=0.5,
        help="Physical y-z wall buffer over which the automatic control mask decays.",
    )
    parser.add_argument(
        "--mask-type",
        choices=("path_gaussian", "rbf_3d"),
        default="path_gaussian",
        help="Spatial control basis: localized x-path windows or a full 3D RBF grid.",
    )
    parser.add_argument(
        "--num-masks",
        type=int,
        default=8,
        help="Number of x windows used by --mask-type path_gaussian.",
    )
    parser.add_argument(
        "--rbf-counts",
        type=int,
        nargs=3,
        default=(8, 3, 3),
        metavar=("NX", "NY", "NZ"),
        help="Numbers of Gaussian centers used by --mask-type rbf_3d.",
    )
    parser.add_argument(
        "--mask-overlap",
        type=float,
        default=0.75,
        help="Gaussian mask width divided by the spacing between mask centers.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=3,
        default=(8, 2, 2),
        metavar=("SX", "SY", "SZ"),
        help="Downsampling stride. Use 1 1 1 for the full 512x40x40 target.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("translated_initial", "file"),
        default="translated_initial",
        help=(
            "Construct the coarse target by translating the restricted initial field, "
            "or independently restrict the target file."
        ),
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--checkpoint-stride", type=int, default=5)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--pressure-rel-tol", type=float, default=1e-7)
    parser.add_argument("--pressure-max-iter", type=int, default=100)
    parser.add_argument(
        "--pressure-fixed-iterations",
        type=int,
        default=30,
        help="Fixed pressure-CG iterations for a smooth optimization forward map.",
    )
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=10.0)
    parser.add_argument("--initial-alpha", type=float, default=4.5)
    parser.add_argument(
        "--initial-control-amplitudes",
        type=Path,
        default=None,
        help="Optional (num_time_blocks, num_masks) .npy warm start.",
    )
    parser.add_argument(
        "--activity-mode",
        choices=("absolute", "perturbation"),
        default="absolute",
        help=(
            "In absolute mode the uncontrolled region and effort reference are zero. "
            "In perturbation mode they equal --activity-baseline."
        ),
    )
    parser.add_argument("--activity-baseline", type=float, default=4.5)
    parser.add_argument(
        "--loss-function",
        choices=available_loss_functions(),
        default="core_aware_q_tracking",
        help="Registered objective used by the direct-adjoint loop.",
    )
    parser.add_argument("--running-weight", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--control-weight", type=float, default=1e-5)
    parser.add_argument(
        "--bulk-s",
        dest="S_bulk",
        type=float,
        default=DEFAULT_S_BULK,
        help="Bulk S in Q=(3S/2)(nn-I/3).",
    )
    parser.add_argument("--core-threshold-fraction", type=float, default=0.65)
    parser.add_argument("--core-transition-fraction", type=float, default=0.08)
    parser.add_argument("--target-core-q-weight", type=float, default=4.0)
    parser.add_argument("--running-core-weight", type=float, default=0.05)
    parser.add_argument("--terminal-core-weight", type=float, default=0.5)
    parser.add_argument("--running-core-mass-weight", type=float, default=1.0)
    parser.add_argument("--terminal-core-mass-weight", type=float, default=10.0)
    parser.add_argument("--running-moment-weight", type=float, default=0.05)
    parser.add_argument("--terminal-moment-weight", type=float, default=1.0)
    parser.add_argument("--running-aligned-shape-weight", type=float, default=0.1)
    parser.add_argument("--terminal-aligned-shape-weight", type=float, default=2.0)
    parser.add_argument("--running-comoving-q-weight", type=float, default=0.1)
    parser.add_argument("--terminal-comoving-q-weight", type=float, default=2.0)
    parser.add_argument("--free-position-tolerance", type=float, default=0.25)
    parser.add_argument("--free-mass-tolerance", type=float, default=0.05)
    parser.add_argument("--free-moment-tolerance", type=float, default=0.05)
    parser.add_argument("--free-q-shape-tolerance", type=float, default=0.05)
    parser.add_argument("--free-running-mass-weight", type=float, default=0.02)
    parser.add_argument("--free-terminal-mass-weight", type=float, default=0.2)
    parser.add_argument("--free-running-moment-weight", type=float, default=0.02)
    parser.add_argument("--free-terminal-moment-weight", type=float, default=0.2)
    parser.add_argument("--free-running-q-weight", type=float, default=0.05)
    parser.add_argument("--free-terminal-q-weight", type=float, default=0.5)
    parser.add_argument("--temporal-weight", type=float, default=1e-3)
    parser.add_argument("--spatial-weight", type=float, default=1e-3)
    parser.add_argument("--armijo-step", type=float, default=1.0)
    parser.add_argument(
        "--optimizer",
        choices=("steepest_descent", "lbfgs"),
        default="lbfgs",
    )
    parser.add_argument("--lbfgs-history-size", type=int, default=10)
    parser.add_argument("--gradient-epsilon", type=float, default=1e-1)
    parser.add_argument(
        "--gradient-epsilons",
        type=float,
        nargs="+",
        default=(1e-1, 5e-2, 2.5e-2),
    )
    parser.add_argument("--gradient-directions", type=int, default=5)
    parser.add_argument("--gradient-check-seed", type=int, default=137)
    parser.add_argument("--min-iterations", type=int, default=5)
    parser.add_argument("--cost-relative-tol", type=float, default=1e-4)
    parser.add_argument("--gradient-tol", type=float, default=1e-6)
    parser.add_argument("--convergence-patience", type=int, default=5)
    parser.add_argument("--skip-gradient-check", action="store_true")
    parser.add_argument(
        "--skip-trajectory-diagnostics",
        action="store_true",
        help=(
            "Skip checkpoint-by-checkpoint geometry diagnostics during each DAL "
            "iteration. Terminal diagnostics and saved forward states are unaffected."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_dal_pure_splay_x64_to_x62"),
    )
    args = parser.parse_args()
    return args


def validate_args(args):
    if any(stride <= 0 for stride in args.stride):
        raise ValueError("All downsampling strides must be positive.")
    if any(margin < 0 for margin in args.auto_mask_margin):
        raise ValueError("All automatic-mask margins must be non-negative.")
    if not 0.0 < args.auto_mask_relative_threshold <= 1.0:
        raise ValueError("--auto-mask-relative-threshold must lie in (0, 1].")
    if args.mask_transition_width <= 0:
        raise ValueError("--mask-transition-width must be positive.")
    if args.transverse_mask_margin < 0:
        raise ValueError("--transverse-mask-margin must be non-negative.")
    if args.transverse_mask_transition_width <= 0:
        raise ValueError("--transverse-mask-transition-width must be positive.")
    if args.control_wall_buffer < 0:
        raise ValueError("--control-wall-buffer must be non-negative.")
    if args.num_masks <= 0:
        raise ValueError("--num-masks must be positive.")
    if any(count <= 0 for count in args.rbf_counts):
        raise ValueError("All --rbf-counts values must be positive.")
    if args.mask_overlap <= 0:
        raise ValueError("--mask-overlap must be positive.")
    if args.S_bulk <= 0:
        raise ValueError("--bulk-s must be positive.")
    if not 0 < args.core_threshold_fraction < 1:
        raise ValueError("--core-threshold-fraction must lie in (0, 1).")
    if args.core_transition_fraction <= 0:
        raise ValueError("--core-transition-fraction must be positive.")
    for name in (
        "running_comoving_q_weight",
        "terminal_comoving_q_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative.")
    for name in (
        "steps",
        "block_size",
        "checkpoint_stride",
        "pressure_max_iter",
        "pressure_fixed_iterations",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.iterations < 0:
        raise ValueError("--iterations must be non-negative.")
    if args.gradient_directions < 0:
        raise ValueError("--gradient-directions must be non-negative.")
    if any(value <= 0 for value in args.gradient_epsilons):
        raise ValueError("--gradient-epsilons must be positive.")
    if args.min_iterations < 0 or args.convergence_patience <= 0:
        raise ValueError("Invalid convergence iteration settings.")
    if args.cost_relative_tol < 0 or args.gradient_tol < 0:
        raise ValueError("Convergence tolerances must be non-negative.")
    if args.lbfgs_history_size <= 0:
        raise ValueError("--lbfgs-history-size must be positive.")
    if not args.initial.is_file():
        raise FileNotFoundError(f"Initial Q snapshot does not exist: {args.initial}")
    if not args.target.is_file():
        raise FileNotFoundError(f"Target Q snapshot does not exist: {args.target}")
    if args.mask is not None and not args.mask.is_file():
        raise FileNotFoundError(f"Activity mask does not exist: {args.mask}")
    if (
        args.initial_control_amplitudes is not None
        and not args.initial_control_amplitudes.is_file()
    ):
        raise FileNotFoundError(
            f"Control warm start does not exist: {args.initial_control_amplitudes}"
        )


def load_q(path):
    values = np.load(path)
    if values.ndim != 4 or values.shape[-1] != 5:
        raise ValueError(f"Expected Q shape (Nx, Ny, Nz, 5), got {values.shape} from {path}.")
    if not np.isfinite(values).all():
        raise ValueError(f"Q tensor contains non-finite values: {path}")
    return values


def automatic_path_mask(
    initial,
    target,
    stride,
    margin,
    relative_threshold,
    transition_width,
    device,
    initial_loop=None,
    target_loop=None,
    transverse_margin=1.5,
    transverse_transition_width=0.5,
    wall_buffer=0.5,
):
    difference = np.sqrt(np.sum((initial - target) ** 2, axis=-1))
    threshold = max(1e-5, relative_threshold * float(difference.max()))
    changed = np.argwhere(difference > threshold)
    if changed.size == 0:
        raise ValueError("Initial and target Q tensors are indistinguishable.")

    source_shape = np.asarray(initial.shape[:3], dtype=int)
    lower = np.maximum(changed.min(axis=0) - np.asarray(margin), 0)
    upper = np.minimum(changed.max(axis=0) + 1 + np.asarray(margin), source_shape)
    downsampled_shape = tuple(
        (size + axis_stride - 1) // axis_stride
        for size, axis_stride in zip(source_shape, stride)
    )
    bounds = tuple(
        (float(lo) / axis_stride, float(hi) / axis_stride)
        for lo, hi, axis_stride in zip(lower, upper, stride)
    )
    has_loop_geometry = all(
        loop is not None and loop.get("center") is not None and loop.get("radius") is not None
        for loop in (initial_loop, target_loop)
    )
    if has_loop_geometry:
        x = torch.arange(downsampled_shape[0], device=device, dtype=torch.float32) + 0.5
        x_window = 0.5 * (
            torch.tanh((x - bounds[0][0]) / transition_width)
            - torch.tanh((x - bounds[0][1]) / transition_width)
        )
        transverse = localized_transverse_path_envelope(
            downsampled_shape,
            SOURCE_LENGTHS,
            initial_loop["center"],
            target_loop["center"],
            max(float(initial_loop["radius"]), float(target_loop["radius"])),
            margin=transverse_margin,
            transition_width=transverse_transition_width,
            wall_buffer=wall_buffer,
            device=device,
        )
        mask = x_window[:, None, None] * transverse[None, :, :]
        mode = "localized_transverse_path"
    else:
        mask = smooth_box_mask(
            downsampled_shape,
            bounds,
            transition_width=transition_width,
            device=device,
        )
        mode = "automatic_difference_path"
    metadata = {
        "mode": mode,
        "difference_threshold": threshold,
        "relative_threshold": relative_threshold,
        "source_bounds": [[int(lo), int(hi)] for lo, hi in zip(lower, upper)],
        "downsampled_bounds": [list(bound) for bound in bounds],
        "margin": list(margin),
    }
    if has_loop_geometry:
        metadata["transverse_path"] = {
            "initial_center": list(initial_loop["center"]),
            "target_center": list(target_loop["center"]),
            "loop_radius": max(float(initial_loop["radius"]), float(target_loop["radius"])),
            "margin": float(transverse_margin),
            "envelope_radius": max(float(initial_loop["radius"]), float(target_loop["radius"]))
            + float(transverse_margin),
            "transition_width": float(transverse_transition_width),
            "wall_buffer": float(wall_buffer),
        }
    return mask, metadata


def localized_transverse_path_envelope(
    shape,
    lengths,
    initial_center,
    target_center,
    loop_radius,
    *,
    margin,
    transition_width,
    wall_buffer,
    device,
):
    """Smooth y-z envelope around the line joining two loop centers."""

    if loop_radius <= 0 or margin < 0 or transition_width <= 0 or wall_buffer < 0:
        raise ValueError("Invalid localized transverse-mask geometry.")
    dtype = torch.float32
    y = (torch.arange(shape[1], device=device, dtype=dtype) + 0.5) * lengths[1] / shape[1]
    z = (torch.arange(shape[2], device=device, dtype=dtype) + 0.5) * lengths[2] / shape[2]
    yy, zz = torch.meshgrid(y, z, indexing="ij")
    points = torch.stack((yy, zz), dim=-1)
    start = torch.as_tensor(initial_center[1:3], device=device, dtype=dtype)
    stop = torch.as_tensor(target_center[1:3], device=device, dtype=dtype)
    segment = stop - start
    length_squared = torch.dot(segment, segment)
    if length_squared.item() > torch.finfo(dtype).eps:
        fraction = ((points - start) * segment).sum(dim=-1) / length_squared
        closest = start + fraction.clamp(0.0, 1.0).unsqueeze(-1) * segment
    else:
        closest = start
    distance = torch.linalg.vector_norm(points - closest, dim=-1)
    envelope_radius = float(loop_radius) + float(margin)
    path_window = 0.5 * (1.0 - torch.tanh((distance - envelope_radius) / transition_width))

    wall_distance = torch.minimum(
        torch.minimum(yy, lengths[1] - yy),
        torch.minimum(zz, lengths[2] - zz),
    )
    wall_window = 0.5 * (
        1.0 + torch.tanh((wall_distance - wall_buffer) / transition_width)
    )
    return path_window * wall_window


def load_initial_target_and_mask(args, device, initial_loop=None, target_loop=None):
    initial = load_q(args.initial)
    target = load_q(args.target)
    if initial.shape != target.shape:
        raise ValueError(
            f"Initial shape {initial.shape} does not match target shape {target.shape}."
        )

    stride = tuple(args.stride)
    if args.mask is None:
        mask_tensor, mask_metadata = automatic_path_mask(
            initial,
            target,
            stride,
            tuple(args.auto_mask_margin),
            args.auto_mask_relative_threshold,
            args.mask_transition_width,
            device,
            initial_loop=initial_loop,
            target_loop=target_loop,
            transverse_margin=args.transverse_mask_margin,
            transverse_transition_width=args.transverse_mask_transition_width,
            wall_buffer=args.control_wall_buffer,
        )
    else:
        mask = np.load(args.mask)
        if mask.shape != initial.shape[:3]:
            raise ValueError(f"Mask shape {mask.shape} does not match Q shape {initial.shape[:3]}.")
        mask_tensor = torch.from_numpy(
            conservative_block_average(mask, stride).copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )
        mask_metadata = {"mode": "file", "path": str(args.mask)}

    initial = conservative_block_average(initial, stride).copy()
    restricted_file_target = conservative_block_average(target, stride).copy()
    grid_preparation = {
        "restriction": "conservative_block_average",
        "source_shape": list(target.shape[:3]),
        "factors": list(stride),
    }
    if args.target_mode == "translated_initial":
        if initial_loop is None or target_loop is None:
            raise ValueError("Loop metadata is required for translated_initial target mode.")
        initial_center = initial_loop.get("center")
        target_center = target_loop.get("center")
        if initial_center is None or target_center is None:
            raise ValueError(
                "Initial and target loop centers are required for translated_initial mode."
            )
        displacement = np.asarray(target_center, dtype=float) - np.asarray(
            initial_center, dtype=float
        )
        if np.linalg.norm(displacement[1:]) > 1e-10:
            raise ValueError(
                "translated_initial currently supports translation along periodic x only; "
                f"got displacement={displacement.tolist()}."
            )
        target = periodic_translate_axis(
            initial,
            float(displacement[0]),
            SOURCE_LENGTHS[0],
            axis=0,
        )
        comparison_scale = max(float(np.mean(restricted_file_target**2)), 1e-30)
        grid_preparation.update(
            {
                "target_mode": "translated_initial",
                "target_displacement": displacement.tolist(),
                "restricted_file_target_relative_mse": float(
                    np.mean((target - restricted_file_target) ** 2) / comparison_scale
                ),
            }
        )
    else:
        target = restricted_file_target
        grid_preparation["target_mode"] = "file"
    mask_metadata["grid_preparation"] = grid_preparation
    initial_q = torch.from_numpy(initial).movedim(-1, 0).unsqueeze(1).to(device)
    target_q = torch.from_numpy(target).movedim(-1, 0).unsqueeze(1).to(device)
    return initial_q, target_q, mask_tensor, tuple(target.shape[:3]), mask_metadata


def active_mask_bounds(mask, threshold_fraction=0.05):
    """Return grid-index bounds containing the active part of a spatial mask."""

    bounds = []
    for axis in range(mask.ndim):
        reduction_axes = tuple(index for index in range(mask.ndim) if index != axis)
        profile = mask.amax(dim=reduction_axes) if reduction_axes else mask
        active = torch.nonzero(
            profile > threshold_fraction * profile.max(), as_tuple=False
        ).flatten()
        if active.numel() == 0:
            raise ValueError("Cannot construct a control basis from an empty mask.")
        bounds.append((float(active[0]), float(active[-1] + 1)))
    return tuple(bounds)


def control_basis(args, control_mask, mask_metadata):
    """Build the selected normalized spatial control basis."""

    inferred_bounds = active_mask_bounds(control_mask)
    x_bounds = inferred_bounds[0]
    if "downsampled_bounds" in mask_metadata:
        x_bounds = tuple(mask_metadata["downsampled_bounds"][0])

    if args.mask_type == "path_gaussian":
        masks = partition_mask_along_axis(
            control_mask,
            args.num_masks,
            axis=0,
            bounds=x_bounds,
            overlap=args.mask_overlap,
            periodic=True,
        )
        return masks, (args.num_masks,), (x_bounds,), (True,)

    bounds = list(inferred_bounds)
    bounds[0] = x_bounds
    transverse = mask_metadata.get("transverse_path")
    if transverse is not None:
        envelope_radius = float(transverse["envelope_radius"])
        centers = np.asarray(
            [transverse["initial_center"], transverse["target_center"]],
            dtype=float,
        )
        for axis in (1, 2):
            spacing = SOURCE_LENGTHS[axis] / control_mask.shape[axis]
            lower = max(0.0, float(centers[:, axis].min()) - envelope_radius)
            upper = min(
                SOURCE_LENGTHS[axis],
                float(centers[:, axis].max()) + envelope_radius,
            )
            bounds[axis] = (lower / spacing, upper / spacing)
    bounds = tuple(bounds)
    grid_shape = tuple(args.rbf_counts)
    masks = partition_mask_rbf(
        control_mask,
        grid_shape,
        bounds=bounds,
        overlap=args.mask_overlap,
        periodic=(True, False, False),
    )
    return masks, grid_shape, bounds, (True, False, False)


def save_q(path, q):
    values = q.detach().cpu().squeeze(1).movedim(0, -1).numpy()
    np.save(path, values)


def q_input_summary(q_path, *, expected_S_bulk):
    q_path = Path(q_path)
    q_values = load_q(q_path)
    metadata_path = q_path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Canonical Q input metadata is required at "
            f"{metadata_path}. Regenerate this Q field with the active-nematics "
            "model instead of relying on a legacy convention fallback."
        )

    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object.")
    if metadata.get("schema_version") != 1:
        raise ValueError(f"{metadata_path} must declare schema_version=1.")
    model_metadata = metadata.get("model")
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

    S_values = np.asarray(S_from_Q(q_values))
    magnitudes = np.asarray(Q_magnitude(q_values))
    ordered_cutoff = float(np.quantile(magnitudes, 0.75))
    ordered_S = S_values[magnitudes >= ordered_cutoff]
    observed_S_initial = float(np.median(ordered_S))
    tolerance = max(5.0e-3, 0.05 * declared_S_initial)
    if abs(observed_S_initial - declared_S_initial) > tolerance:
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
            f"{q_path} declares S_bulk={declared_S_bulk:.8g}, but this DAL run "
            f"uses S_bulk={expected_S_bulk:.8g}."
        )

    classification = metadata.get("classification", {})
    loop = metadata.get("loop", {})
    validation = metadata.get("validation", {})
    return {
        "shape": list(q_values.shape),
        "finite": True,
        "metadata": str(metadata_path),
        "Q_convention": convention,
        "S_initial": declared_S_initial,
        "S_bulk": declared_S_bulk,
        "observed_ordered_S": observed_S_initial,
        "classification": classification.get("type"),
        "classification_regime": classification.get("classification_regime"),
        "center": loop.get("center"),
        "radius": loop.get("radius"),
        "defect_plaquette_counts": validation.get("defect_plaquette_counts"),
        "yz_plane_topology": validation.get("yz_plane_topology"),
    }


def uniform_x_Q_like(q, S):
    uniform = torch.zeros_like(q)
    uniform[0] = S
    uniform[3] = -S / 2.0
    return uniform


def objective_component_values(objective, q):
    values = {
        name: float(value.detach().item())
        for name, value in objective.components(q).items()
    }
    values["terminal_cost"] = float(objective.terminal_cost(q).detach().item())
    return values


def write_history_csv(path, history):
    fieldnames = (
        list(dict.fromkeys(name for row in history for name in row))
        if history
        else ["iteration"]
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_centerline_geometry(path, q, *, S_bulk):
    q_numpy = q.detach().cpu().squeeze(1).movedim(0, -1).numpy()
    try:
        geometry = core_centerline_geometry(
            q_numpy,
            SOURCE_LENGTHS,
            S_bulk=S_bulk,
        )
    except ValueError as error:
        serializable = {"detected": False, "error": str(error)}
        path.with_suffix(".json").write_text(json.dumps(serializable, indent=2))
        return serializable
    centerline_path = path.with_suffix(".npy")
    np.save(centerline_path, geometry.pop("centerline"))
    serializable = {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in geometry.items()
    }
    serializable["centerline_path"] = str(centerline_path)
    path.with_suffix(".json").write_text(json.dumps(serializable, indent=2))
    return serializable


def trajectory_diagnostics(
    objective,
    record,
    *,
    initial_geometry,
    S_bulk,
):
    """Summarize shape and topology over every stored forward checkpoint."""

    samples = []
    reference_radius = max(float(initial_geometry["radius_rms"]), 1e-12)
    initial_thickness = float(initial_geometry["x_thickness_rms"])
    thickness_scale = SOURCE_LENGTHS[0] / record.checkpoints[0].shape[2]
    initial_topology = bool(initial_geometry["yz_plane_topology"])
    for step in record.boundaries:
        q = record.checkpoints[step]
        q_device = q.to(objective.target_q.device)
        components = objective.components(q_device, step=step)
        q_numpy = q.squeeze(1).movedim(0, -1).numpy()
        geometry = loop_core_metrics(
            q_numpy,
            SOURCE_LENGTHS,
            S_bulk=S_bulk,
        )
        radius_y = float(geometry["radius_y_rms"])
        radius_z = float(geometry["radius_z_rms"])
        plaquettes = geometry["defect_plaquettes"]
        samples.append(
            {
                "step": int(step),
                "trajectory_position_penalty": float(
                    components["trajectory_position_penalty"].item()
                ),
                "comoving_q_shape_penalty": float(
                    components["comoving_q_shape_penalty"].item()
                ),
                "aligned_core_shape_penalty": float(
                    components["aligned_core_shape_penalty"].item()
                ),
                "core_moment_penalty": float(
                    components["core_moment_penalty"].item()
                ),
                "radius_relative_error": abs(
                    float(geometry["radius_rms"]) - reference_radius
                )
                / reference_radius,
                "transverse_anisotropy": abs(radius_y - radius_z)
                / max(radius_y + radius_z, 1e-12),
                "thickness_relative_error": abs(
                    float(geometry["x_thickness_rms"]) - initial_thickness
                )
                / thickness_scale,
                "plaquettes_xy": int(plaquettes["xy"]),
                "plaquettes_xz": int(plaquettes["xz"]),
                "plaquettes_yz": int(plaquettes["yz"]),
                "topology_matches_initial": (
                    bool(geometry["yz_plane_topology"]) == initial_topology
                ),
            }
        )
    return {
        "samples": samples,
        "max_position_penalty": max(
            sample["trajectory_position_penalty"] for sample in samples
        ),
        "max_comoving_q_shape_penalty": max(
            sample["comoving_q_shape_penalty"] for sample in samples
        ),
        "max_aligned_core_shape_penalty": max(
            sample["aligned_core_shape_penalty"] for sample in samples
        ),
        "max_radius_relative_error": max(
            sample["radius_relative_error"] for sample in samples
        ),
        "max_transverse_anisotropy": max(
            sample["transverse_anisotropy"] for sample in samples
        ),
        "max_thickness_relative_error": max(
            sample["thickness_relative_error"] for sample in samples
        ),
        "topology_violation_count": sum(
            not sample["topology_matches_initial"] for sample in samples
        ),
    }


def main():
    args = parse_args()
    validate_args(args)
    args.output = prepare_new_run_directory(args.output)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    initial_loop_summary = q_input_summary(args.initial, expected_S_bulk=args.S_bulk)
    target_loop_summary = q_input_summary(args.target, expected_S_bulk=args.S_bulk)
    print(f"initial_loop={json.dumps(initial_loop_summary)}", flush=True)
    print(f"target_loop={json.dumps(target_loop_summary)}", flush=True)

    initial_q, target_q, control_mask, shape, mask_metadata = (
        load_initial_target_and_mask(
            args,
            device,
            initial_loop=initial_loop_summary,
            target_loop=target_loop_summary,
        )
    )
    print(
        f"initial={args.initial} target={args.target} shape={shape} "
        f"stride={tuple(args.stride)} device={device}",
        flush=True,
    )
    print(f"control_mask={json.dumps(mask_metadata)}", flush=True)
    print(f"loss_function={args.loss_function}", flush=True)
    control_masks, mask_grid_shape, basis_bounds, basis_periodic = control_basis(
        args,
        control_mask,
        mask_metadata,
    )
    num_control_masks = control_masks.shape[0]
    print(
        f"control_basis=type:{args.mask_type} grid:{mask_grid_shape} "
        f"num_masks:{num_control_masks} num_blocks:"
        f"{int(np.ceil(args.steps / args.block_size))} "
        f"parameters:{num_control_masks * int(np.ceil(args.steps / args.block_size))}",
        flush=True,
    )
    initial_q_fields = {
        name: initial_q[index, 0]
        for index, name in enumerate(Q_COMPONENTS)
    }
    solver = build_active_nematic_channel(
        shape,
        SOURCE_LENGTHS,
        args.dt,
        initial_q_fields,
        device=device,
        batchsize=1,
        pressure_rel_tol=args.pressure_rel_tol,
        pressure_max_iter=args.pressure_max_iter,
        pressure_fixed_iterations=args.pressure_fixed_iterations,
    )
    initial_q = torch.stack([solver.fields[name] for name in Q_COMPONENTS]).detach()

    control = TemporalMaskControl(
        control_masks,
        num_steps=args.steps,
        block_size=args.block_size,
        batchsize=1,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        initial_alpha=args.initial_alpha,
        baseline=(args.activity_baseline if args.activity_mode == "perturbation" else 0.0),
        mask_grid_shape=mask_grid_shape,
    )
    if args.initial_control_amplitudes is not None:
        control.set_amplitudes(np.load(args.initial_control_amplitudes))
        print(
            f"control_warm_start={args.initial_control_amplitudes}", flush=True
        )
    objective_kwargs = dict(
        target_q=target_q,
        dt=args.dt,
        domain_lengths=SOURCE_LENGTHS,
        spatial_mask=control_mask,
        running_weight=args.running_weight,
        terminal_weight=args.terminal_weight,
        control_weight=args.control_weight,
        reference_alpha=(
            args.activity_baseline if args.activity_mode == "perturbation" else 0.0
        ),
        S_bulk=args.S_bulk,
        core_threshold_fraction=args.core_threshold_fraction,
        core_transition_fraction=args.core_transition_fraction,
        target_core_q_weight=args.target_core_q_weight,
        running_core_weight=args.running_core_weight,
        terminal_core_weight=args.terminal_core_weight,
        running_core_mass_weight=args.running_core_mass_weight,
        terminal_core_mass_weight=args.terminal_core_mass_weight,
        running_moment_weight=args.running_moment_weight,
        terminal_moment_weight=args.terminal_moment_weight,
        running_aligned_shape_weight=args.running_aligned_shape_weight,
        terminal_aligned_shape_weight=args.terminal_aligned_shape_weight,
    )
    if args.loss_function == "core_aware_comoving_q_trajectory":
        objective_kwargs.update(
            initial_q=initial_q,
            num_steps=args.steps,
            running_comoving_q_weight=args.running_comoving_q_weight,
            terminal_comoving_q_weight=args.terminal_comoving_q_weight,
        )
    elif args.loss_function == "free_path_shape_preserving":
        objective_kwargs.update(
            initial_q=initial_q,
            position_tolerance=args.free_position_tolerance,
            mass_tolerance=args.free_mass_tolerance,
            moment_tolerance=args.free_moment_tolerance,
            q_shape_tolerance=args.free_q_shape_tolerance,
            free_running_mass_weight=args.free_running_mass_weight,
            free_terminal_mass_weight=args.free_terminal_mass_weight,
            free_running_moment_weight=args.free_running_moment_weight,
            free_terminal_moment_weight=args.free_terminal_moment_weight,
            free_running_q_weight=args.free_running_q_weight,
            free_terminal_q_weight=args.free_terminal_q_weight,
        )
    objective = build_loss_function(args.loss_function, **objective_kwargs)
    dal = DiscreteAdjointLoop(
        FunctionalSemiImplicitStep(solver),
        objective,
        control,
        num_steps=args.steps,
        checkpoint_stride=args.checkpoint_stride,
        temporal_control_weight=args.temporal_weight,
        spatial_control_weight=args.spatial_weight,
    )

    save_q(args.output / "Q_initial.npy", initial_q)
    save_q(args.output / "Q_target_downsampled.npy", target_q)
    initial_line_geometry = save_centerline_geometry(
        args.output / "centerline_initial",
        initial_q,
        S_bulk=args.S_bulk,
    )
    target_line_geometry = save_centerline_geometry(
        args.output / "centerline_target",
        target_q,
        S_bulk=args.S_bulk,
    )
    np.save(args.output / "control_mask.npy", control_mask.detach().cpu().numpy())
    np.save(args.output / "control_masks.npy", control_masks.detach().cpu().numpy())

    arguments = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    run_metadata = {
        "schema_version": 1,
        "script": Path(__file__).name,
        "solver": {
            "shape": list(shape),
            "lengths": list(SOURCE_LENGTHS),
            "dt": args.dt,
            "steps": args.steps,
            "save_interval": None,
            "checkpoint_stride": args.checkpoint_stride,
        },
        "model": {
            "name": "active_nematics",
            "Q_convention": Q_convention_metadata(),
            "parameters": {
                "A": -1.0,
                "B": -6.0,
                "C": 6.0,
                "L1": 1.0,
                "S_initial": initial_loop_summary["S_initial"],
                "S_bulk": args.S_bulk,
                "flow_alignment": 1.0,
                "active_stress_sign": -1.0,
                "fric": 0.0,
                "eta": 1.0,
            },
        },
        "boundary_conditions": {
            "Q": list(Q_BC),
            "velocity": list(U_BC),
            "pressure_modal": list(PRESSURE_MODAL_BC),
        },
        "numerics": {
            "integrator": "semi_implicit_euler",
            "dealiasing": "none",
            "velocity_zero_mode": "not_applicable_with_dirichlet_walls",
            "pressure_solver": {
                "relative_tolerance": args.pressure_rel_tol,
                "maximum_iterations": args.pressure_max_iter,
                "fixed_iterations": args.pressure_fixed_iterations,
            },
        },
        "initial_condition": {
            "name": "external_Q",
            "path": str(args.initial),
            "input": initial_loop_summary,
        },
        "target": {
            "path": str(args.target),
            "mode": args.target_mode,
            "input": target_loop_summary,
        },
        "optimization": {
            "arguments": arguments,
            "mask": str(args.mask) if args.mask is not None else None,
            "mask_metadata": mask_metadata,
            "control_basis": {
                "type": args.mask_type,
                "num_masks": num_control_masks,
                "grid_shape": list(mask_grid_shape),
                "num_time_blocks": control.num_blocks,
                "num_parameters": control.logits.numel(),
                "bounds": basis_bounds,
                "periodic": basis_periodic,
                "overlap": args.mask_overlap,
            },
        },
        "diagnostics": {
            "centerline_geometry": {
                "initial": initial_line_geometry,
                "target": target_line_geometry,
            },
        },
    }
    write_run_metadata(args.output, run_metadata, status="running")

    if not args.skip_gradient_check:
        check = dal.gradient_check_suite(
            initial_q,
            epsilons=args.gradient_epsilons,
            num_random_directions=args.gradient_directions,
            seed=args.gradient_check_seed,
        )
        (args.output / "gradient_check.json").write_text(json.dumps(check, indent=2))
        print(f"gradient_check={json.dumps(check)}", flush=True)

    iteration_dir = args.output / "iterations"
    iteration_dir.mkdir(parents=True, exist_ok=True)
    initial_q_numpy = initial_q.detach().cpu().squeeze(1).movedim(0, -1).numpy()
    initial_profile = x_disturbance_profile(
        initial_q_numpy,
        S_bulk=args.S_bulk,
    )
    initial_geometry = loop_core_metrics(
        initial_q_numpy,
        SOURCE_LENGTHS,
        S_bulk=args.S_bulk,
    )
    history = []
    convergence_streak = 0
    stop_reason = "maximum_iterations"
    for iteration in range(args.iterations):
        result = dal.armijo_step(
            initial_q,
            initial_step=args.armijo_step,
            direction_method=args.optimizer,
            lbfgs_history_size=args.lbfgs_history_size,
        )
        row = {"iteration": iteration, **result.__dict__}
        record = dal.last_accepted_record
        if record is not None:
            iteration_q = record.checkpoints[args.steps]
            save_q(iteration_dir / f"Q_iteration_{iteration:03d}.npy", iteration_q)
            np.save(
                iteration_dir / f"alpha_iteration_{iteration:03d}.npy",
                control.amplitudes().detach().cpu().numpy(),
            )
            iteration_q_device = iteration_q.to(device)
            row.update({
                f"terminal_{name}": float(value.detach().item())
                for name, value in objective.components(iteration_q_device).items()
            })
            iteration_q_numpy = (
                iteration_q.detach().cpu().squeeze(1).movedim(0, -1).numpy()
            )
            geometry = loop_core_metrics(
                iteration_q_numpy,
                SOURCE_LENGTHS,
                S_bulk=args.S_bulk,
            )
            for name in (
                "core_mass",
                "radius_rms",
                "radius_y_rms",
                "radius_z_rms",
                "x_thickness_rms",
                "center_x",
                "center_y",
                "center_z",
            ):
                row[name] = geometry[name]
            plaquettes = geometry["defect_plaquettes"]
            row.update({
                "plaquettes_xy": plaquettes["xy"],
                "plaquettes_xz": plaquettes["xz"],
                "plaquettes_yz": plaquettes["yz"],
                "yz_plane_topology": geometry["yz_plane_topology"],
            })
            line_geometry = save_centerline_geometry(
                iteration_dir / f"centerline_iteration_{iteration:03d}",
                iteration_q,
                S_bulk=args.S_bulk,
            )
            row["line_detected"] = line_geometry["detected"]
            if line_geometry["detected"]:
                line_center = line_geometry["center"]
                row.update(
                    {
                        "line_center_x": line_center[0],
                        "line_center_y": line_center[1],
                        "line_center_z": line_center[2],
                        "line_length": line_geometry["line_length"],
                        "line_radius_mean": line_geometry["radius_mean"],
                        "line_ellipticity": line_geometry["ellipticity"],
                        "line_planarity_rms": line_geometry["planarity_rms"],
                        "line_curvature_max": line_geometry["curvature_max"],
                        "line_core_component_count": line_geometry["core_component_count"],
                    }
                )
            shift = periodic_profile_shift(
                initial_profile,
                x_disturbance_profile(
                    iteration_q_numpy,
                    S_bulk=args.S_bulk,
                ),
                SOURCE_LENGTHS[0],
            )
            row.update({
                "profile_shift": shift["shift_physical"],
                "profile_correlation": shift["correlation"],
            })
            if hasattr(objective, "num_steps") and not args.skip_trajectory_diagnostics:
                trajectory = trajectory_diagnostics(
                    objective,
                    record,
                    initial_geometry=initial_geometry,
                    S_bulk=args.S_bulk,
                )
                trajectory_path = (
                    iteration_dir / f"trajectory_iteration_{iteration:03d}.json"
                )
                trajectory_path.write_text(json.dumps(trajectory, indent=2))
                for name in (
                    "max_position_penalty",
                    "max_comoving_q_shape_penalty",
                    "max_aligned_core_shape_penalty",
                    "max_radius_relative_error",
                    "max_transverse_anisotropy",
                    "max_thickness_relative_error",
                    "topology_violation_count",
                ):
                    row[f"trajectory_{name}"] = trajectory[name]
        relative_change = abs(result.cost_before - result.cost_after) / max(
            abs(result.cost_before), 1e-12
        )
        row["relative_cost_change"] = relative_change
        if result.accepted and relative_change <= args.cost_relative_tol:
            convergence_streak += 1
        else:
            convergence_streak = 0
        row["convergence_streak"] = convergence_streak
        history.append(row)
        write_history_csv(args.output / "history.csv", history)
        print(f"dal_iteration={json.dumps(row)}", flush=True)
        if not result.accepted:
            stop_reason = "armijo_rejected"
            break
        if iteration + 1 >= args.min_iterations:
            if result.gradient_norm <= args.gradient_tol:
                stop_reason = "gradient_tolerance"
                break
            if convergence_streak >= args.convergence_patience:
                stop_reason = "cost_tolerance"
                break

    final_record = dal.forward(initial_q)
    final_q = final_record.checkpoints[args.steps]
    save_q(args.output / "Q_optimized_final.npy", final_q)
    final_line_geometry = save_centerline_geometry(
        args.output / "centerline_optimized_final",
        final_q,
        S_bulk=args.S_bulk,
    )
    loss_components = {
        "target": objective_component_values(objective, target_q),
        "initial": objective_component_values(objective, initial_q),
        "uniform_no_loop": objective_component_values(
            objective,
            uniform_x_Q_like(initial_q, args.S_bulk),
        ),
        "optimized_final": objective_component_values(
            objective,
            final_q.to(device),
        ),
    }
    with (args.output / "loss_components.json").open("w") as stream:
        json.dump(loss_components, stream, indent=2)
    amplitudes = control.amplitudes().detach().cpu().numpy()
    np.save(args.output / "alpha_amplitudes.npy", amplitudes)
    with (args.output / "alpha_amplitudes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["time_block", *[f"mask_{index}" for index in range(num_control_masks)]]
        )
        for block, values in enumerate(amplitudes):
            writer.writerow([block, *values.tolist()])
    torch.save(control.state_dict(), args.output / "control_state.pt")
    history_path = args.output / "history.csv"
    existing_history_count = 0
    if history:
        write_history_csv(history_path, history)
    elif history_path.is_file():
        with history_path.open(newline="") as stream:
            existing_history_count = sum(1 for _ in csv.DictReader(stream))
    else:
        write_history_csv(history_path, history)
    run_metadata["results"] = {
        "output": str(args.output),
        "final_cost": final_record.cost,
        "stop_reason": (
            stop_reason
            if history or not existing_history_count
            else "finalized_existing_control"
        ),
        "completed_iterations": len(history) or existing_history_count,
        "loss_components": loss_components,
    }
    run_metadata["diagnostics"]["centerline_geometry"][
        "optimized_final"
    ] = final_line_geometry
    write_run_metadata(args.output, run_metadata, status="complete")
    print(f"saved={args.output} final_cost={final_record.cost:.8e}", flush=True)


if __name__ == "__main__":
    main()
