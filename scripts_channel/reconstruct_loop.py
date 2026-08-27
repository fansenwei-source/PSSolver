#!/usr/bin/env python3
"""Reconstruct an ideal-loop director field from one saved Q-tensor frame.

The script implements the ideal initial disclination-loop construction from
``3D Dry Uniaxial Active Nematics in Bulk``.  A detected loop supplies its
best-fit common-plane normal ``nu`` and planar geometry.  The surrounding Q
texture supplies the local principal frame ``{N, M, L}``, the dominant Fourier
direction ``k``, the disturbance half-width, and the outer fluctuation
amplitude.  The reconstructed angle field is

    theta(r) = f(l) = pi*l/(2*h) + pi/2              (projection inside loop)
    theta(r) = g(l) = -epsilon(p)*sin(pi*l/h)        (projection outside loop)
    theta(r) = 0                                     (|l| > h)

with

    n_ideal(r) = cos(theta)*N + sin(theta)*M.

Here ``r = p + l*k`` and ``p`` lies in the fitted loop plane.  This use of k
also invokes the paper's ideal-loop Assumption 14, ``k parallel nu``; it is not
a general identity for arbitrary loops.  The loop is represented by a planar
ellipse, matching the paper's ideal-loop assumption.
The sign of M is selected by minimizing the nematic-angle mismatch with the
input Q field in the reconstructed neighborhood.

The director is physically undefined at a disclination core.  Therefore the
solver-ready result is ``Q_ideal``, built with the input S field and
an optional analytic suppression at the fitted model loop.  ``n_ideal`` is
also saved for inspection, together with a mask identifying points where the
output S is large enough for the director to be meaningful.

After reconstruction, the script independently redetects disclination lines,
refits the output loop normal, recomputes a Fourier k direction, and reports
topology, k-nu agreement, nematic-angle errors, and wall changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_loop as loop_analysis

PERIODIC_BOUNDARY = loop_analysis.PERIODIC_BOUNDARY

from pssolver.models.active_nematics.nematics3d_adapter import (
    S_and_director_from_Q,
    director_from_Q,
)
from pssolver.snapshots import validate_active_nematic_q_source
from pssolver.models.active_nematics import (
    Q_components,
    Q_convention_metadata,
    Q_magnitude,
    uniaxial_Q,
)


@dataclass(frozen=True)
class EllipseFit:
    center: np.ndarray
    axis1: np.ndarray
    axis2: np.ndarray
    normal: np.ndarray
    semiaxis1: float
    semiaxis2: float
    rms_equation_residual: float
    rms_plane_residual: float


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return loop_analysis.unsigned_angle_deg(a, b)


def fit_planar_ellipse(coords: np.ndarray, normal: np.ndarray) -> EllipseFit:
    """Fit a centered, principal-axis ellipse in the paper's common loop plane."""
    coords = np.asarray(coords, dtype=float)
    center = np.mean(coords, axis=0)
    normal = loop_analysis.unit(normal, name="ellipse normal")
    centered = coords - center
    projected = centered - (centered @ normal)[:, None] * normal
    covariance = projected.T @ projected / len(projected)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    axis1 = loop_analysis.unit(vectors[:, order[0]], name="ellipse axis1")
    axis1 -= np.dot(axis1, normal) * normal
    axis1 = loop_analysis.unit(axis1, name="projected ellipse axis1")
    axis2 = loop_analysis.unit(np.cross(normal, axis1), name="ellipse axis2")

    u = centered @ axis1
    v = centered @ axis2
    design = np.column_stack((u * u, v * v))
    coefficients, *_ = np.linalg.lstsq(design, np.ones(len(coords)), rcond=None)
    if np.any(coefficients <= 0.0) or not np.all(np.isfinite(coefficients)):
        semiaxis1 = float(np.sqrt(2.0 * np.mean(u * u)))
        semiaxis2 = float(np.sqrt(2.0 * np.mean(v * v)))
    else:
        semiaxis1 = float(1.0 / np.sqrt(coefficients[0]))
        semiaxis2 = float(1.0 / np.sqrt(coefficients[1]))
    if semiaxis1 < semiaxis2:
        semiaxis1, semiaxis2 = semiaxis2, semiaxis1
        axis1, axis2 = axis2, -axis1

    equation = (u / semiaxis1) ** 2 + (v / semiaxis2) ** 2
    plane_residual = centered @ normal
    return EllipseFit(
        center=center,
        axis1=axis1,
        axis2=axis2,
        normal=normal,
        semiaxis1=semiaxis1,
        semiaxis2=semiaxis2,
        rms_equation_residual=float(np.sqrt(np.mean((equation - 1.0) ** 2))),
        rms_plane_residual=float(np.sqrt(np.mean(plane_residual**2))),
    )


def sample_director_lines(
    q_obj,
    plane_points: np.ndarray,
    k_hat: np.ndarray,
    axis_n: np.ndarray,
    *,
    max_distance: float,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    l_values = np.linspace(-max_distance, max_distance, samples)
    all_angles = []
    for point in plane_points:
        sample_points = point + l_values[:, None] * k_hat
        q_values = np.asarray(q_obj.act_interpolate(sample_points, is_index=True))
        directors = director_from_Q(q_values)
        cosine = np.clip(np.abs(directors @ axis_n), 0.0, 1.0)
        all_angles.append(np.degrees(np.arccos(cosine)))
    return l_values, np.asarray(all_angles)


def estimate_half_thickness(
    q_obj,
    ellipse: EllipseFit,
    k_hat: np.ndarray,
    axis_n: np.ndarray,
    *,
    max_distance: float,
    samples: int,
    threshold_deg: float,
) -> tuple[float, dict[str, object]]:
    offsets = np.asarray(
        [
            np.zeros(3),
            0.25 * ellipse.semiaxis1 * ellipse.axis1,
            -0.25 * ellipse.semiaxis1 * ellipse.axis1,
            0.25 * ellipse.semiaxis2 * ellipse.axis2,
            -0.25 * ellipse.semiaxis2 * ellipse.axis2,
        ]
    )
    plane_points = ellipse.center + offsets
    l_values, angle_lines = sample_director_lines(
        q_obj,
        plane_points,
        k_hat,
        axis_n,
        max_distance=max_distance,
        samples=samples,
    )
    estimates = []
    details = []
    center_window = np.abs(l_values) <= 0.25 * max_distance
    for line_index, angles in enumerate(angle_lines):
        candidates = np.where(center_window)[0]
        seed = int(candidates[np.argmax(angles[candidates])])
        active = angles >= threshold_deg
        if not active[seed]:
            continue
        left = seed
        right = seed
        while left > 0 and active[left - 1]:
            left -= 1
        while right < len(active) - 1 and active[right + 1]:
            right += 1
        if left == 0 or right == len(active) - 1:
            continue
        h_left = abs(float(l_values[left]))
        h_right = abs(float(l_values[right]))
        estimate = 0.5 * (h_left + h_right)
        if estimate <= 0.0:
            continue
        estimates.append(estimate)
        details.append(
            {
                "line_index": line_index,
                "h_left": h_left,
                "h_right": h_right,
                "estimate": estimate,
                "max_angle_deg": float(np.max(angles)),
            }
        )
    if not estimates:
        raise RuntimeError(
            "Automatic half-thickness estimation failed. Pass a positive "
            "--half-thickness or increase --half-thickness-search-distance."
        )
    return float(np.median(estimates)), {
        "method": "median threshold crossing on five interior k-lines",
        "threshold_deg": threshold_deg,
        "search_distance": max_distance,
        "samples": samples,
        "line_estimates": details,
    }


def estimate_epsilon0(
    q_obj,
    ellipse: EllipseFit,
    k_hat: np.ndarray,
    axis_n: np.ndarray,
    *,
    half_thickness: float,
    radial_factor: float,
    azimuth_samples: int,
    line_samples: int,
    max_epsilon_deg: float,
) -> tuple[float, dict[str, object]]:
    azimuth = np.linspace(0.0, 2.0 * np.pi, azimuth_samples, endpoint=False)
    plane_points = ellipse.center + radial_factor * (
        np.cos(azimuth)[:, None] * ellipse.semiaxis1 * ellipse.axis1
        + np.sin(azimuth)[:, None] * ellipse.semiaxis2 * ellipse.axis2
    )
    _, angle_lines = sample_director_lines(
        q_obj,
        plane_points,
        k_hat,
        axis_n,
        max_distance=half_thickness,
        samples=line_samples,
    )
    amplitudes_deg = np.max(angle_lines, axis=1)
    raw_median_deg = float(np.median(amplitudes_deg))
    epsilon_deg = float(np.clip(raw_median_deg, 1.0, max_epsilon_deg))
    return float(np.radians(epsilon_deg)), {
        "method": "median maximum background angle on external k-lines",
        "radial_factor": radial_factor,
        "amplitudes_deg": amplitudes_deg,
        "raw_median_deg": raw_median_deg,
        "max_epsilon_deg": max_epsilon_deg,
        "epsilon_deg_clipped": epsilon_deg,
    }


def minimum_image_axis(values: np.ndarray, size: int, is_periodic: bool) -> np.ndarray:
    if not is_periodic:
        return values
    return (values + 0.5 * size) % size - 0.5 * size


def build_geometry_fields(
    shape: tuple[int, int, int],
    ellipse: EllipseFit,
    k_hat: np.ndarray,
) -> dict[str, np.ndarray]:
    coordinates = [
        np.arange(shape[axis], dtype=float) - ellipse.center[axis]
        for axis in range(3)
    ]
    coordinates = [
        minimum_image_axis(values, shape[axis], PERIODIC_BOUNDARY[axis])
        for axis, values in enumerate(coordinates)
    ]
    dx = coordinates[0][:, None, None]
    dy = coordinates[1][None, :, None]
    dz = coordinates[2][None, None, :]

    k_dot_nu = float(np.dot(k_hat, ellipse.normal))
    if abs(k_dot_nu) < 0.25:
        raise RuntimeError(
            f"k and nu are too far from parallel for the ideal-loop projection: dot={k_dot_nu}"
        )
    ell = (
        dx * ellipse.normal[0]
        + dy * ellipse.normal[1]
        + dz * ellipse.normal[2]
    ) / k_dot_nu
    px = dx - ell * k_hat[0]
    py = dy - ell * k_hat[1]
    pz = dz - ell * k_hat[2]
    u = px * ellipse.axis1[0] + py * ellipse.axis1[1] + pz * ellipse.axis1[2]
    v = px * ellipse.axis2[0] + py * ellipse.axis2[1] + pz * ellipse.axis2[2]
    rho = np.sqrt((u / ellipse.semiaxis1) ** 2 + (v / ellipse.semiaxis2) ** 2)

    phi = np.arctan2(v, u)
    radial_distance = np.sqrt(u * u + v * v)
    boundary_radius = 1.0 / np.sqrt(
        (np.cos(phi) / ellipse.semiaxis1) ** 2
        + (np.sin(phi) / ellipse.semiaxis2) ** 2
    )
    distance_to_ellipse = np.abs(radial_distance - boundary_radius)
    return {
        "ell": ell,
        "u": u,
        "v": v,
        "rho": rho,
        "distance_to_ellipse": distance_to_ellipse,
    }


def build_angle_field(
    geometry: dict[str, np.ndarray],
    *,
    half_thickness: float,
    epsilon0: float,
    epsilon_decay_length: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ell = geometry["ell"]
    inside = geometry["rho"] < 1.0
    disturbed = np.abs(ell) <= half_thickness
    theta = np.zeros_like(ell, dtype=float)
    inside_disturbed = inside & disturbed
    outside_disturbed = (~inside) & disturbed
    theta[inside_disturbed] = 0.5 * np.pi * (
        ell[inside_disturbed] / half_thickness + 1.0
    )
    epsilon = epsilon0 * np.exp(
        -0.5
        * (geometry["distance_to_ellipse"] / epsilon_decay_length) ** 2
    )
    theta[outside_disturbed] = -epsilon[outside_disturbed] * np.sin(
        np.pi * ell[outside_disturbed] / half_thickness
    )
    return theta, {
        "inside": inside,
        "disturbed": disturbed,
        "inside_disturbed": inside_disturbed,
        "outside_disturbed": outside_disturbed,
    }


def director_from_angle(
    theta: np.ndarray,
    axis_n: np.ndarray,
    axis_m: np.ndarray,
) -> np.ndarray:
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return cosine[..., None] * axis_n + sine[..., None] * axis_m


def nematic_angle_field_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cosine = np.sum(a * b, axis=-1)
    return np.degrees(np.arccos(np.clip(np.abs(cosine), 0.0, 1.0)))


def choose_m_sign(
    theta: np.ndarray,
    axis_n: np.ndarray,
    axis_m: np.ndarray,
    observed_director: np.ndarray,
    S: np.ndarray,
    local_mask: np.ndarray,
    *,
    S_threshold: float,
) -> tuple[np.ndarray, dict[str, float]]:
    compare_mask = local_mask & (S >= S_threshold)
    if not np.any(compare_mask):
        raise RuntimeError("No ordered local points are available to select the M sign.")
    director_plus = director_from_angle(theta, axis_n, axis_m)
    director_minus = director_from_angle(theta, axis_n, -axis_m)
    error_plus = nematic_angle_field_deg(director_plus, observed_director)
    error_minus = nematic_angle_field_deg(director_minus, observed_director)
    mean_plus = float(np.mean(error_plus[compare_mask]))
    mean_minus = float(np.mean(error_minus[compare_mask]))
    selected = axis_m if mean_plus <= mean_minus else -axis_m
    return selected, {
        "mean_error_M_plus_deg": mean_plus,
        "mean_error_M_minus_deg": mean_minus,
        "selected_sign": 1 if mean_plus <= mean_minus else -1,
        "comparison_points": int(np.count_nonzero(compare_mask)),
    }


def q5_from_director(director: np.ndarray, S: np.ndarray) -> np.ndarray:
    components = Q_components(uniaxial_Q(director, S))
    return np.stack(
        tuple(components[name] for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")),
        axis=-1,
    ).astype(np.float32)


def apply_model_core(
    S: np.ndarray,
    geometry: dict[str, np.ndarray],
    *,
    core_size: float,
) -> np.ndarray:
    if core_size <= 0.0:
        return np.asarray(S, dtype=float)
    distance_3d = np.sqrt(
        geometry["distance_to_ellipse"] ** 2 + geometry["ell"] ** 2
    )
    suppression = np.tanh(distance_3d / core_size)
    return np.asarray(S, dtype=float) * suppression


def blend_input_walls(
    q_model: np.ndarray,
    q_input: np.ndarray,
    *,
    wall_buffer: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = q_model.shape[:3]
    distances = []
    for axis, periodic in enumerate(PERIODIC_BOUNDARY):
        if periodic:
            continue
        coordinate = np.arange(shape[axis], dtype=float)
        distance = np.minimum(coordinate, shape[axis] - 1.0 - coordinate)
        reshape = [1, 1, 1]
        reshape[axis] = shape[axis]
        distances.append(np.broadcast_to(distance.reshape(reshape), shape))
    if not distances or wall_buffer <= 0.0:
        weight = np.ones(shape, dtype=float)
        return q_model, weight
    wall_distance = np.minimum.reduce(distances)
    t = np.clip(wall_distance / wall_buffer, 0.0, 1.0)
    weight = t * t * (3.0 - 2.0 * t)
    blended = weight[..., None] * q_model + (1.0 - weight[..., None]) * q_input
    return np.asarray(blended, dtype=np.float32), weight


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def validate_reconstruction(
    q_output: np.ndarray,
    observed_director: np.ndarray,
    target_ellipse: EllipseFit,
    axes_used: np.ndarray,
    target_k: np.ndarray,
    geometry_masks: dict[str, np.ndarray],
    *,
    step: int,
    S_threshold: float,
    fourier_crop_margin: int,
    grid_spacing: tuple[float, float, float],
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    S_output, director_output = S_and_director_from_Q(q_output)
    defined = S_output >= S_threshold
    angle_error = nematic_angle_field_deg(director_output, observed_director)
    error_regions = {
        "whole_ordered_channel_deg": distribution_summary(angle_error[defined]),
        "disturbed_ordered_region_deg": distribution_summary(
            angle_error[defined & geometry_masks["disturbed"]]
        ),
        "inside_ordered_region_deg": distribution_summary(
            angle_error[defined & geometry_masks["inside_disturbed"]]
        ),
        "outside_ordered_region_deg": distribution_summary(
            angle_error[defined & geometry_masks["outside_disturbed"]]
        ),
    }

    q_obj = loop_analysis.build_q_object(q_output, step)
    loops = [line for line in q_obj.lines if line.kind == "loop"]
    validation: dict[str, object] = {
        "finite_Q": bool(np.all(np.isfinite(q_output))),
        "detected_loop_count": len(loops),
        "director_defined_fraction": float(np.mean(defined)),
        "nematic_angle_error": error_regions,
    }
    if loops:
        target_center = target_ellipse.center
        selected = min(
            loops,
            key=lambda line: float(
                np.linalg.norm(np.mean(line.calc_defect_coords, axis=0) - target_center)
            ),
        )
        output_coords = np.asarray(selected.calc_defect_coords, dtype=float)
        output_nu, output_nu_metrics = loop_analysis.measure_nu(selected)
        output_k = loop_analysis.measure_dominant_k(
            q_output,
            output_coords,
            axes_used,
            output_nu,
            crop_margin=fourier_crop_margin,
            grid_spacing=grid_spacing,
        )
        output_center = np.mean(output_coords, axis=0)
        validation.update(
            {
                "selected_output_loop_points": int(selected.calc_defect_num),
                "selected_output_loop_center": output_center,
                "center_shift": float(np.linalg.norm(output_center - target_center)),
                "output_nu": output_nu,
                "output_nu_metrics": output_nu_metrics,
                "angle_output_nu_target_nu_deg": vector_angle_deg(
                    output_nu, target_ellipse.normal
                ),
                "output_k": output_k,
                "angle_output_k_target_k_deg": vector_angle_deg(
                    output_k["k_hat_lab"], target_k
                ),
            }
        )
    return validation, S_output, director_output


def write_error_csv(path: Path, validation: dict[str, object]) -> None:
    rows = validation["nematic_angle_error"]
    fields = ["region", "count", "mean", "std", "min", "q50", "q90", "q95", "q99", "max"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for region, summary in rows.items():
            writer.writerow({"region": region, **summary})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data_control_box" / "Q_6200.npy",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--loop-index", type=int, default=None)
    parser.add_argument("--grid-spacing", type=float, nargs=3, default=(0.25, 0.25, 0.25))
    parser.add_argument("--nml-sampling-spacing", type=float, default=1.5)
    parser.add_argument("--fourier-crop-margin", type=int, default=6)
    parser.add_argument("--k-nu-tolerance-deg", type=float, default=15.0)
    parser.add_argument(
        "--half-thickness",
        type=float,
        default=0.0,
        help="Positive fixed value in grid units; 0 estimates it from input Q.",
    )
    parser.add_argument("--half-thickness-search-distance", type=float, default=12.0)
    parser.add_argument("--half-thickness-samples", type=int, default=241)
    parser.add_argument("--half-thickness-angle-threshold-deg", type=float, default=10.0)
    parser.add_argument(
        "--epsilon0",
        type=float,
        default=-1.0,
        help="Nonnegative fixed outer amplitude in radians; negative estimates it.",
    )
    parser.add_argument("--epsilon-radial-factor", type=float, default=1.3)
    parser.add_argument(
        "--epsilon-max-deg",
        type=float,
        default=20.0,
        help="Upper bound for auto-estimated outer fluctuations; the paper assumes epsilon is small.",
    )
    parser.add_argument("--epsilon-decay-length", type=float, default=0.0)
    parser.add_argument("--core-size", type=float, default=0.75)
    parser.add_argument(
        "--s-threshold",
        dest="S_threshold",
        type=float,
        default=2.0 / 30.0,
    )
    parser.add_argument(
        "--wall-mode",
        choices=("background", "preserve-input"),
        default="preserve-input",
    )
    parser.add_argument("--wall-buffer", type=float, default=4.0)
    parser.add_argument("--skip-output-validation", action="store_true")
    return parser.parse_args()


def infer_step(path: Path) -> int:
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def load_input_model_metadata(Q_path: Path) -> tuple[Path, float, float]:
    source = validate_active_nematic_q_source(
        Q_path.parent,
        require_S_initial=True,
    )
    if source.S_bulk is None:
        raise ValueError(
            f"{source.path} must declare a positive S_bulk for reconstruction"
        )
    return source.path, source.S_initial, source.S_bulk


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    input_metadata_path, source_S_initial, S_bulk = load_input_model_metadata(args.input)
    step = infer_step(args.input)
    output_dir = args.output_dir or args.input.parent / "loop_reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)

    q_input = loop_analysis.load_q(args.input)
    S_input, director_input = S_and_director_from_Q(q_input)
    magnitudes = np.asarray(Q_magnitude(q_input))
    ordered_cutoff = float(np.quantile(magnitudes, 0.75))
    observed_input_ordered_S = float(
        np.median(np.asarray(S_input)[magnitudes >= ordered_cutoff])
    )
    q_obj = loop_analysis.build_q_object(q_input, step)
    line, loop_index, loop_count = loop_analysis.choose_single_loop(q_obj, args.loop_index)
    _, smooth_coords = loop_analysis.smooth_loop_coords(line)
    raw_coords = np.asarray(line.calc_defect_coords, dtype=float)

    nu, nu_metrics = loop_analysis.measure_nu(line)
    axes_nml, nml_diagnostics = loop_analysis.measure_nml_frame(
        q_obj,
        raw_coords,
        spacing=float(args.nml_sampling_spacing),
    )
    k_result = loop_analysis.measure_dominant_k(
        q_input,
        raw_coords,
        axes_nml,
        nu,
        crop_margin=int(args.fourier_crop_margin),
        grid_spacing=tuple(float(value) for value in args.grid_spacing),
    )
    k_hat = np.asarray(k_result["k_hat_lab"], dtype=float)
    if np.dot(k_hat, nu) < 0.0:
        k_hat = -k_hat
    k_nu_angle = vector_angle_deg(k_hat, nu)
    if k_nu_angle > float(args.k_nu_tolerance_deg):
        raise RuntimeError(
            f"Ideal-loop assumption failed: angle(k, nu)={k_nu_angle:.3f} deg "
            f"> tolerance {args.k_nu_tolerance_deg:.3f} deg"
        )

    ellipse = fit_planar_ellipse(smooth_coords, nu)
    axis_n = np.asarray(axes_nml[:, 0], dtype=float)
    axis_m_initial = np.asarray(axes_nml[:, 1], dtype=float)

    if args.half_thickness > 0.0:
        half_thickness = float(args.half_thickness)
        h_diagnostics = {"method": "user supplied"}
    else:
        half_thickness, h_diagnostics = estimate_half_thickness(
            q_obj,
            ellipse,
            k_hat,
            axis_n,
            max_distance=float(args.half_thickness_search_distance),
            samples=int(args.half_thickness_samples),
            threshold_deg=float(args.half_thickness_angle_threshold_deg),
        )

    if args.epsilon0 >= 0.0:
        epsilon0 = float(args.epsilon0)
        epsilon_diagnostics = {"method": "user supplied"}
    else:
        epsilon0, epsilon_diagnostics = estimate_epsilon0(
            q_obj,
            ellipse,
            k_hat,
            axis_n,
            half_thickness=half_thickness,
            radial_factor=float(args.epsilon_radial_factor),
            azimuth_samples=12,
            line_samples=101,
            max_epsilon_deg=float(args.epsilon_max_deg),
        )
    epsilon_decay_length = (
        float(args.epsilon_decay_length)
        if args.epsilon_decay_length > 0.0
        else 0.5 * (ellipse.semiaxis1 + ellipse.semiaxis2)
    )

    geometry = build_geometry_fields(q_input.shape[:3], ellipse, k_hat)
    theta, geometry_masks = build_angle_field(
        geometry,
        half_thickness=half_thickness,
        epsilon0=epsilon0,
        epsilon_decay_length=epsilon_decay_length,
    )
    axis_m, m_sign_diagnostics = choose_m_sign(
        theta,
        axis_n,
        axis_m_initial,
        director_input,
        S_input,
        geometry_masks["disturbed"],
        S_threshold=float(args.S_threshold),
    )
    axis_l = loop_analysis.unit(np.cross(axis_n, axis_m), name="selected L")
    axes_used = np.column_stack((axis_n, axis_m, axis_l))
    director_model = director_from_angle(theta, axis_n, axis_m)
    S_model = apply_model_core(
        S_input,
        geometry,
        core_size=float(args.core_size),
    )
    q_model = q5_from_director(director_model, S_model)

    if args.wall_mode == "preserve-input":
        q_output, wall_weight = blend_input_walls(
            q_model,
            q_input,
            wall_buffer=float(args.wall_buffer),
        )
    else:
        q_output = q_model
        wall_weight = np.ones(q_input.shape[:3], dtype=float)

    wall_mask = wall_weight < 1.0
    wall_delta = np.abs(np.asarray(q_output, dtype=float) - np.asarray(q_input, dtype=float))
    wall_metrics = {
        "mode": args.wall_mode,
        "buffer": float(args.wall_buffer),
        "affected_fraction": float(np.mean(wall_mask)),
        "max_abs_Q_change_in_blend_region": (
            float(np.max(wall_delta[wall_mask])) if np.any(wall_mask) else 0.0
        ),
        "rms_Q_change_in_blend_region": (
            float(np.sqrt(np.mean(wall_delta[wall_mask] ** 2)))
            if np.any(wall_mask)
            else 0.0
        ),
    }

    if args.skip_output_validation:
        S_output, director_output = S_and_director_from_Q(q_output)
        validation = {
            "skipped": True,
            "finite_Q": bool(np.all(np.isfinite(q_output))),
        }
    else:
        validation, S_output, director_output = validate_reconstruction(
            q_output,
            director_input,
            ellipse,
            axes_used,
            k_hat,
            geometry_masks,
            step=step,
            S_threshold=float(args.S_threshold),
            fourier_crop_margin=int(args.fourier_crop_margin),
            grid_spacing=tuple(float(value) for value in args.grid_spacing),
        )

    output_magnitudes = np.asarray(Q_magnitude(q_output))
    output_ordered_cutoff = float(np.quantile(output_magnitudes, 0.75))
    output_S_initial = float(
        np.median(np.asarray(S_output)[output_magnitudes >= output_ordered_cutoff])
    )
    if not np.isfinite(output_S_initial) or output_S_initial <= 0.0:
        raise ValueError(
            "Reconstructed Q does not have a positive finite ordered-region S."
        )

    defined_mask = S_output >= float(args.S_threshold)
    prefix = f"step_{step:05d}"
    q_path = output_dir / f"Q_ideal_{prefix}.npy"
    n_path = output_dir / f"n_ideal_{prefix}.npy"
    mask_path = output_dir / f"director_defined_{prefix}.npy"
    json_path = output_dir / "metadata.json"
    csv_path = output_dir / f"loop_reconstruction_error_{prefix}.csv"
    np.save(q_path, np.asarray(q_output, dtype=np.float32))
    np.save(n_path, np.asarray(director_output, dtype=np.float32))
    np.save(mask_path, defined_mask)

    summary = {
        "schema_version": 1,
        "model": {
            "name": "active_nematics",
            "Q_convention": Q_convention_metadata(),
            "parameters": {
                "S_initial": output_S_initial,
                "S_bulk": S_bulk,
                "S_threshold": float(args.S_threshold),
            },
        },
        "input": {
            "path": str(args.input.resolve()),
            "metadata": str(input_metadata_path.resolve()),
            "shape": q_input.shape,
            "source_run_S_initial": source_S_initial,
            "observed_ordered_S": observed_input_ordered_S,
            "step": step,
            "loop_count": loop_count,
            "selected_loop_index": loop_index,
            "grid_spacing": args.grid_spacing,
            "periodic_boundary": PERIODIC_BOUNDARY,
        },
        "paper_model": {
            "convention": loop_analysis.CONVENTION_NAME,
            "source": loop_analysis.CONVENTION_SOURCE,
            "classification_regime": "ideal-loop one-dimensional skyrmion",
            "assumptions_used": [
                "N, M, L diagonalize q_box = <n n>_box",
                "k parallel nu (Assumption 14)",
                "Omega parallel L",
            ],
            "inside_profile": "f(l)=pi*l/(2*h)+pi/2",
            "outside_profile": "g(l)=-epsilon(p)*sin(pi*l/h)",
            "half_thickness": half_thickness,
            "half_thickness_diagnostics": h_diagnostics,
            "epsilon0_rad": epsilon0,
            "epsilon0_deg": float(np.degrees(epsilon0)),
            "epsilon_diagnostics": epsilon_diagnostics,
            "epsilon_decay_length": epsilon_decay_length,
            "core_size": float(args.core_size),
        },
        "loop_geometry": {
            "center": ellipse.center,
            "nu": ellipse.normal,
            "nu_metrics": nu_metrics,
            "axis1": ellipse.axis1,
            "axis2": ellipse.axis2,
            "semiaxis1": ellipse.semiaxis1,
            "semiaxis2": ellipse.semiaxis2,
            "ellipse_rms_equation_residual": ellipse.rms_equation_residual,
            "plane_rms_residual": ellipse.rms_plane_residual,
        },
        "frame": {
            "N": axis_n,
            "M_initial": axis_m_initial,
            "M_selected": axis_m,
            "L_selected": axis_l,
            "definition": "decreasing-eigenvalue eigenvectors of q_box = <n n>_box",
            "principal_plane": "N-M",
            "principal_plane_normal": "L",
            "NML_diagnostics": nml_diagnostics,
            "M_sign_selection": m_sign_diagnostics,
        },
        "input_k": {
            **k_result,
            "k_hat_oriented_toward_nu": k_hat,
            "angle_k_nu_deg": k_nu_angle,
        },
        "wall": wall_metrics,
        "validation": validation,
        "outputs": {
            "Q_ideal": str(q_path.resolve()),
            "n_ideal": str(n_path.resolve()),
            "director_defined_mask": str(mask_path.resolve()),
            "error_csv": str(csv_path.resolve()) if not args.skip_output_validation else None,
        },
    }
    with json_path.open("w") as handle:
        json.dump(loop_analysis.json_value(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    if not args.skip_output_validation:
        write_error_csv(csv_path, validation)

    print(f"input={args.input}")
    print(
        f"loop_index={loop_index} points={line.calc_defect_num} "
        f"planarity={nu_metrics['planarity_score']:.6f}"
    )
    print(
        f"ellipse_semiaxes=({ellipse.semiaxis1:.6f}, {ellipse.semiaxis2:.6f}) "
        f"half_thickness={half_thickness:.6f} epsilon0_deg={np.degrees(epsilon0):.6f}"
    )
    print(f"input_k_hat={np.round(k_hat, 8).tolist()}")
    print(f"input_nu={np.round(nu, 8).tolist()} angle_k_nu_deg={k_nu_angle:.6f}")
    print(
        f"M_sign={m_sign_diagnostics['selected_sign']} "
        f"local_error_plus_deg={m_sign_diagnostics['mean_error_M_plus_deg']:.6f} "
        f"local_error_minus_deg={m_sign_diagnostics['mean_error_M_minus_deg']:.6f}"
    )
    if not args.skip_output_validation:
        print(
            f"output_loop_count={validation['detected_loop_count']} "
            f"output_k_nu_angle_deg="
            f"{validation.get('output_k', {}).get('angle_k_nu_deg', float('nan')):.6f}"
        )
    print(f"Q_ideal={q_path}")
    print(f"n_ideal={n_path}")
    print(f"defined_mask={mask_path}")
    print(f"json={json_path}")
    if not args.skip_output_validation:
        print(f"error_csv={csv_path}")


if __name__ == "__main__":
    main()
