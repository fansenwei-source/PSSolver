#!/usr/bin/env python3
"""Measure gamma, sigma, dominant k, and the k-nu agreement for one loop.

Calculation method
------------------
This script analyzes one saved Q-tensor frame containing one closed
disclination loop.  The implementation follows the definitions in
``3D Dry Uniaxial Active Nematics in Bulk`` and keeps the Fourier estimate of
the dominant wave vector independent of the geometric loop normal.

1. Detect and select the loop
   Diagonalize the component-last Q field to obtain the director field, detect
   disclination points with Nematics3D, and classify the connected defect line.
   By default, the frame must contain exactly one closed loop.  If several
   loops are present, ``--loop-index`` must select the target explicitly.

2. Compute the loop normal ``nu``
   The paper approximates an initial loop as a planar ellipse and defines
   ``nu`` as the normal of its common plane.  Numerically, the script performs
   a least-squares plane fit to the detected loop points.  Equivalently, ``nu``
   is the eigenvector associated with the smallest eigenvalue of the centered
   point-cloud covariance matrix.  The planarity score, RMS thickness, and
   eigenvalues are saved as quality diagnostics.  The sign of ``nu`` is
   immaterial in all unsigned comparisons.

3. Construct the local ``{N, M, L}`` frame
   A self-consistent principal-plane calculation samples the local director
   field and diagonalizes ``q_box = <n n>_box``.  In decreasing eigenvalue
   order, the axes are ``N`` (background director), ``M`` (principal
   perturbation), and ``L`` (normal of the N-M principal plane).  These are
   box-level texture axes, not the eigenvectors of pointwise Q and not the
   geometric principal axes of the loop.

4. Compute ``gamma``
   Divide the smoothed loop into 24 equally spaced sections, as in the paper.
   On a small polar ring normal to the local loop tangent, align neighboring
   directors and obtain the local rotation axis ``Omega`` from the least-
   variance eigenvector of their orientation matrix.  For every valid section,

       gamma_j = arccos(abs(nu dot Omega_j)).

   Rings outside the domain, rings containing another defect, or rings whose
   center is not detected as a defect are retained in the output but excluded
   from the mean and standard deviation.

5. Compute ``sigma``
   Slice the loop with several planes parallel to the N-M principal plane.
   Each accepted plane must intersect the loop twice.  Around each intersection
   sample a director ring and fit the two nematic winding modes

       theta(phi) = +(1/2) phi + theta0,
       theta(phi) = -(1/2) phi + theta0.

   This identifies the +1/2 and -1/2 defects.  For the +1/2 defect, the fitted
   phase gives its symmetry axis ``p``.  Define ``chi`` from the +1/2 defect to
   the -1/2 defect and calculate the principal angle

       sigma_j = arccos(chi dot p),    0 <= sigma_j <= pi.

   Only pairs passing the winding-coherence and coherence-gap thresholds enter
   the loop average.  The signed angle and all fit-quality values are also
   saved for diagnosis.

6. Compute the dominant wave vector ``k`` independently
   Crop a local box around the loop, express Q in the local N-M-L frame, and
   retain the reorientation components ``Q_NM`` and ``Q_NL`` used in the
   paper's instability analysis.  Subtract their spatial means, apply a 3D
   Hann window, perform a 3D FFT, and select the largest nonzero combined-power
   mode.  Its normalized direction is ``k_hat``.  The reported wavelength is
   window dependent; loop classification uses only ``k_hat``.

7. Compare ``k`` with ``nu`` and classify the loop
   Because ``k`` and ``nu`` are computed independently, the ideal-loop
   assumption ``k parallel nu`` is tested with

       angle_k_nu = arccos(abs(k_hat dot nu)).

   The default consistency tolerance is 15 degrees.  In the local frame,
   proximity of ``k_hat`` to ``N``, ``M``, or ``L`` indicates a pure-splay,
   pure-bend, or pure-twist loop, respectively.  The ideal-model mapping is

       gamma_k = arccos(abs(k_L)),
       sigma_k = 2 atan2(abs(k_M), abs(k_N)).

Outputs
-------
The summary JSON contains the global results, vectors, quality metrics, and all
local measurements.  The companion CSV contains one row per gamma or sigma
section, including validity and fit-quality columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LOCAL_NEMATICS_SRC_CANDIDATES = [
    REPO_ROOT.parent / "Nematics3D" / "src",
    REPO_ROOT / "Nematics3D" / "src",
]
for candidate in LOCAL_NEMATICS_SRC_CANDIDATES:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import nematics3d as n3d
from pssolver.snapshots import load_q_snapshot
from pssolver.models.active_nematics.nematics3d_adapter import (
    director_from_Q,
    q_field_object_from_Q,
)
from pssolver.loop_conventions import (
    CONVENTION_NAME,
    CONVENTION_SOURCE,
    classify_ideal_loop,
)


PERIODIC_BOUNDARY = (True, False, False)
DEFAULT_STEP = 6200
DEFAULT_GRID_SPACING = (0.25, 0.25, 0.25)


def unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Cannot normalize {name}: norm={norm!r}")
    return vector / norm


def unsigned_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(unit(a, name="a"), unit(b, name="b"))), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def load_q(path: Path) -> np.ndarray:
    return load_q_snapshot(path, require_S_initial=True).values


def build_q_object(q5: np.ndarray, step: int) -> n3d.QFieldObject:
    return q_field_object_from_Q(
        q5,
        box_periodic_flag=PERIODIC_BOUNDARY,
        name=f"loop parameter analysis step {step}",
    )


def choose_single_loop(q_obj: n3d.QFieldObject, loop_index: int | None):
    loops = [line for line in q_obj.lines if line.kind == "loop"]
    if not loops:
        raise RuntimeError("No closed disclination loop was detected in this frame.")
    if loop_index is None:
        if len(loops) != 1:
            raise RuntimeError(
                f"Detected {len(loops)} loops. Pass --loop-index to select one explicitly."
            )
        return loops[0], 0, len(loops)
    if loop_index < 0 or loop_index >= len(loops):
        raise IndexError(f"--loop-index={loop_index} but only {len(loops)} loops were detected")
    return loops[loop_index], loop_index, len(loops)


def smooth_loop_coords(line) -> tuple[object, np.ndarray]:
    if line.calc_defect_num < 5:
        raise RuntimeError("At least five defect points are required for loop analysis.")
    smooth = line.act_smooth(
        window_length=5,
        min_line_length=5,
        is_window_warning=False,
    )
    coords = np.asarray(smooth.result, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise RuntimeError(f"Unexpected smoothed loop shape {coords.shape}")
    return smooth, coords


def measure_nu(line) -> tuple[np.ndarray, dict[str, object]]:
    """Paper definition: normal of the common plane of the near-elliptical loop."""
    nu = unit(np.asarray(line.act_calc_norm(), dtype=float), name="nu")
    return nu, dict(line.calc_norm_metric)


def measure_nml_frame(
    q_obj: n3d.QFieldObject,
    raw_coords: np.ndarray,
    *,
    spacing: float,
) -> tuple[np.ndarray, dict[str, object]]:
    result = n3d.nml_principal_plane_analysis(
        q_obj,
        required_points=raw_coords,
        expand_factors=(2.0, 1.5, 1.5),
        min_lengths=(12.0, 14.0, 14.0),
        spacing=spacing,
        angle_tol_deg=0.25,
        max_iterations=20,
    )
    axes = np.asarray(result.axes, dtype=float)
    final = result.iterations[-1]
    diagnostics = {
        "converged": bool(result.converged),
        "iterations": len(result.iterations),
        "sample_count_final": int(final.sample_count),
        "eigenvalues_final": final.eigenvalues,
        "axis_change_deg_final": final.angle_changes_deg,
    }
    return axes, diagnostics


def measure_gamma(
    smooth,
    nu: np.ndarray,
    *,
    sections: int,
    ring_layers: int,
    ring_dr: float,
    ring_arc_dist: float,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    for section_index, u_percent in enumerate(
        np.arange(sections, dtype=float) * 100.0 / sections
    ):
        result = smooth.act_calc_omega(
            float(u_percent),
            layers=ring_layers,
            dr=ring_dr,
            arc_dist=ring_arc_dist,
        )
        omega = unit(np.asarray(result["omega"], dtype=float), name="omega")
        gamma_deg = unsigned_angle_deg(nu, omega)
        metric = result["metric"]
        is_valid = bool(
            np.isfinite(gamma_deg)
            and not metric["is_out_of_domain"]
            and not metric["is_defect_inside_R"]
            and metric["is_defect_at_center"]
        )
        rows.append(
            {
                "section_index": section_index,
                "u_percent": float(u_percent),
                "position": np.asarray(result["position"], dtype=float),
                "omega": omega,
                "gamma_deg": gamma_deg,
                "omega_orthogonality_score": float(metric["orthogonality_score"]),
                "omega_rotation_consistency": float(metric["rotation_consistency"]),
                "omega_tilt_deg": float(metric["tilt_angle_degrees"]),
                "omega_ring_radius": float(result["R"]),
                "omega_out_of_domain": bool(metric["is_out_of_domain"]),
                "omega_extra_defect_inside_ring": bool(metric["is_defect_inside_R"]),
                "omega_defect_at_center": bool(metric["is_defect_at_center"]),
                "is_valid": is_valid,
            }
        )

    values = np.asarray(
        [row["gamma_deg"] for row in rows if row["is_valid"]], dtype=float
    )
    if values.size == 0:
        raise RuntimeError("No valid local omega section remained for gamma analysis.")
    summary = {
        "total_sections": len(rows),
        "accepted_sections": int(values.size),
        "rejected_sections": int(len(rows) - values.size),
        "mean_deg": float(np.mean(values)),
        "std_deg": float(np.std(values)),
        "min_deg": float(np.min(values)),
        "median_deg": float(np.median(values)),
        "max_deg": float(np.max(values)),
    }
    return rows, summary


def closed_polyline_plane_intersections(
    coords: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    level: float,
    *,
    dedupe_distance: float = 0.25,
) -> list[np.ndarray]:
    points = np.vstack((coords, coords[0]))
    signed = (points - center) @ normal - level
    intersections: list[np.ndarray] = []
    for index in range(len(coords)):
        value_a = float(signed[index])
        value_b = float(signed[index + 1])
        if value_a == 0.0:
            fraction = 0.0
        elif value_b == 0.0 or value_a * value_b < 0.0:
            fraction = value_a / (value_a - value_b)
        else:
            continue
        point = points[index] + fraction * (points[index + 1] - points[index])
        if not any(np.linalg.norm(point - old) < dedupe_distance for old in intersections):
            intersections.append(point)
    return intersections


def fit_wedge_profile(
    q_obj: n3d.QFieldObject,
    origin: np.ndarray,
    axis_n: np.ndarray,
    axis_m: np.ndarray,
    *,
    radius: float,
    samples: int,
) -> dict[str, object]:
    phi = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    sample_points = origin + radius * (
        np.cos(phi)[:, None] * axis_n + np.sin(phi)[:, None] * axis_m
    )
    q_values = np.asarray(q_obj.act_interpolate(sample_points, is_index=True), dtype=float)
    directors = director_from_Q(q_values)
    projected = directors @ axis_n + 1j * (directors @ axis_m)
    projected_norm = np.abs(projected)
    projected /= np.maximum(projected_norm, 1e-12)
    nematic_phase = projected * projected

    coefficient_plus = np.mean(nematic_phase * np.exp(-1j * phi))
    coefficient_minus = np.mean(nematic_phase * np.exp(+1j * phi))
    coherence_plus = float(abs(coefficient_plus))
    coherence_minus = float(abs(coefficient_minus))
    charge_sign = 1 if coherence_plus >= coherence_minus else -1

    return {
        "charge_sign": charge_sign,
        "coherence_plus": coherence_plus,
        "coherence_minus": coherence_minus,
        "selected_coherence": max(coherence_plus, coherence_minus),
        "coherence_gap": abs(coherence_plus - coherence_minus),
        # For theta(phi)=phi/2+theta0, the +1/2 symmetry-axis angle is 2 theta0.
        "p_angle_rad": float(np.angle(coefficient_plus)),
        "mean_nm_projection": float(np.mean(projected_norm)),
    }


def measure_sigma(
    q_obj: n3d.QFieldObject,
    smooth_coords: np.ndarray,
    axes_nml: np.ndarray,
    *,
    slices: int,
    slice_fraction: float,
    ring_radius: float,
    ring_samples: int,
    min_coherence: float,
    min_coherence_gap: float,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    axis_n, axis_m, axis_l = axes_nml.T
    center = np.mean(smooth_coords, axis=0)
    l_coords = (smooth_coords - center) @ axis_l
    levels = np.linspace(
        float(np.min(l_coords)) * slice_fraction,
        float(np.max(l_coords)) * slice_fraction,
        slices,
    )

    rows: list[dict[str, object]] = []
    for slice_index, level in enumerate(levels):
        intersections = closed_polyline_plane_intersections(
            smooth_coords, center, axis_l, float(level)
        )
        if len(intersections) != 2:
            continue
        profiles = [
            fit_wedge_profile(
                q_obj,
                point,
                axis_n,
                axis_m,
                radius=ring_radius,
                samples=ring_samples,
            )
            for point in intersections
        ]
        signs = [int(profile["charge_sign"]) for profile in profiles]
        if sorted(signs) != [-1, 1]:
            continue
        if any(float(profile["selected_coherence"]) < min_coherence for profile in profiles):
            continue
        if any(float(profile["coherence_gap"]) < min_coherence_gap for profile in profiles):
            continue

        plus_index = signs.index(1)
        minus_index = signs.index(-1)
        chi = intersections[minus_index] - intersections[plus_index]
        chi -= np.dot(chi, axis_l) * axis_l
        chi = unit(chi, name="chi")

        p_angle = float(profiles[plus_index]["p_angle_rad"])
        p_axis = unit(
            np.cos(p_angle) * axis_n + np.sin(p_angle) * axis_m,
            name="p",
        )
        dot_value = float(np.clip(np.dot(chi, p_axis), -1.0, 1.0))
        sigma_deg = float(np.degrees(np.arccos(dot_value)))
        sigma_signed_deg = float(
            np.degrees(
                np.arctan2(
                    np.dot(axis_l, np.cross(chi, p_axis)),
                    dot_value,
                )
            )
            % 360.0
        )
        rows.append(
            {
                "slice_index": slice_index,
                "l_level": float(level),
                "plus_position": intersections[plus_index],
                "minus_position": intersections[minus_index],
                "chi": chi,
                "p": p_axis,
                "sigma_deg": sigma_deg,
                "sigma_signed_deg": sigma_signed_deg,
                "plus_coherence": float(profiles[plus_index]["selected_coherence"]),
                "minus_coherence": float(profiles[minus_index]["selected_coherence"]),
                "plus_coherence_gap": float(profiles[plus_index]["coherence_gap"]),
                "minus_coherence_gap": float(profiles[minus_index]["coherence_gap"]),
                "plus_mean_nm_projection": float(profiles[plus_index]["mean_nm_projection"]),
                "minus_mean_nm_projection": float(profiles[minus_index]["mean_nm_projection"]),
            }
        )

    if not rows:
        raise RuntimeError(
            "No valid +/-1/2 pair was obtained for sigma. Try changing "
            "--sigma-ring-radius, --sigma-slice-fraction, or coherence thresholds."
        )

    values = np.asarray([row["sigma_deg"] for row in rows], dtype=float)
    signed_rad = np.radians([row["sigma_signed_deg"] for row in rows])
    signed_mean = float(
        np.degrees(np.arctan2(np.mean(np.sin(signed_rad)), np.mean(np.cos(signed_rad))))
        % 360.0
    )
    summary = {
        "valid_slices": len(rows),
        "mean_deg": float(np.mean(values)),
        "std_deg": float(np.std(values)),
        "min_deg": float(np.min(values)),
        "median_deg": float(np.median(values)),
        "max_deg": float(np.max(values)),
        "signed_circular_mean_deg": signed_mean,
    }
    return rows, summary


def q5_component(q5: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    qxx, qxy, qxz, qyy, qyz = np.moveaxis(q5, -1, 0)
    qzz = -qxx - qyy
    return (
        left[0] * right[0] * qxx
        + (left[0] * right[1] + left[1] * right[0]) * qxy
        + (left[0] * right[2] + left[2] * right[0]) * qxz
        + left[1] * right[1] * qyy
        + (left[1] * right[2] + left[2] * right[1]) * qyz
        + left[2] * right[2] * qzz
    )


def canonical_vector_sign(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    pivot = int(np.argmax(np.abs(vector)))
    return -vector if vector[pivot] < 0.0 else vector


def measure_dominant_k(
    q5: np.ndarray,
    loop_coords: np.ndarray,
    axes_nml: np.ndarray,
    nu: np.ndarray,
    *,
    crop_margin: int,
    grid_spacing: tuple[float, float, float],
) -> dict[str, object]:
    lower = np.floor(np.min(loop_coords, axis=0) - crop_margin).astype(int)
    upper = np.ceil(np.max(loop_coords, axis=0) + crop_margin + 1).astype(int)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, np.asarray(q5.shape[:3], dtype=int))
    if np.any(upper - lower < 5):
        raise RuntimeError(f"Fourier crop is too small: lower={lower}, upper={upper}")
    crop_slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
    q_crop = q5[crop_slices]

    axis_n, axis_m, axis_l = axes_nml.T
    # The paper identifies director reorientation through the two components
    # coupling the background direction N to the transverse directions M and L.
    fluctuations = np.stack(
        (
            q5_component(q_crop, axis_n, axis_m),
            q5_component(q_crop, axis_n, axis_l),
        ),
        axis=-1,
    ).astype(float)
    fluctuations -= np.mean(fluctuations, axis=(0, 1, 2), keepdims=True)

    window = np.ones(fluctuations.shape[:3], dtype=float)
    for axis, size in enumerate(fluctuations.shape[:3]):
        reshape = [1, 1, 1]
        reshape[axis] = size
        window *= np.hanning(size).reshape(reshape)
    fft_values = np.fft.fftn(fluctuations * window[..., None], axes=(0, 1, 2))
    power = np.sum(np.abs(fft_values) ** 2, axis=-1)
    power[(0, 0, 0)] = 0.0
    peak_index = np.unravel_index(int(np.argmax(power)), power.shape)

    k_axes = [
        2.0 * np.pi * np.fft.fftfreq(size, d=grid_spacing[axis])
        for axis, size in enumerate(power.shape)
    ]
    k_vector = np.asarray([k_axes[axis][peak_index[axis]] for axis in range(3)])
    k_hat = canonical_vector_sign(unit(k_vector, name="dominant k"))
    k_nml = axes_nml.T @ k_hat
    k_nml = k_nml / np.linalg.norm(k_nml)

    gamma_from_k = unsigned_angle_deg(k_hat, axis_l)
    sigma_from_k = float(
        2.0 * np.degrees(np.arctan2(abs(k_nml[1]), abs(k_nml[0])))
    )
    k_nu_angle = unsigned_angle_deg(k_hat, nu)
    nearest_type, ideal_axis_angles = classify_ideal_loop(k_hat, axes_nml)
    component_angles = {
        "pure_splay_axis_N_deg": ideal_axis_angles["pure-splay"],
        "pure_bend_axis_M_deg": ideal_axis_angles["pure-bend"],
        "pure_twist_axis_L_deg": ideal_axis_angles["pure-twist"],
    }

    positive_power = power[power > 0.0]
    return {
        "crop_lower_index": lower,
        "crop_upper_index_exclusive": upper,
        "crop_shape": np.asarray(power.shape, dtype=int),
        "peak_fft_index": np.asarray(peak_index, dtype=int),
        "k_vector_lab": k_vector,
        "k_magnitude": float(np.linalg.norm(k_vector)),
        "wavelength": float(2.0 * np.pi / np.linalg.norm(k_vector)),
        "k_hat_lab": k_hat,
        "k_hat_NML": k_nml,
        "peak_power_fraction": float(power[peak_index] / np.sum(positive_power)),
        "gamma_from_k_deg": gamma_from_k,
        "sigma_from_k_deg": sigma_from_k,
        "angle_k_nu_deg": k_nu_angle,
        "abs_dot_k_nu": float(abs(np.dot(k_hat, nu))),
        "axis_angles_deg": component_angles,
        "nearest_pure_type": nearest_type,
        "classification_regime": "ideal-loop one-dimensional skyrmion",
    }


def flatten_vector(prefix: str, vector: np.ndarray) -> dict[str, float]:
    vector = np.asarray(vector, dtype=float)
    return {f"{prefix}_{axis}": float(vector[index]) for index, axis in enumerate("xyz")}


def write_local_csv(
    path: Path,
    gamma_rows: list[dict[str, object]],
    sigma_rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = [
            "measurement",
            "index",
            "coordinate",
            "value_deg",
            "signed_value_deg",
            "quality_1",
            "quality_2",
            "is_valid",
            "x",
            "y",
            "z",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in gamma_rows:
            position = np.asarray(row["position"], dtype=float)
            writer.writerow(
                {
                    "measurement": "gamma",
                    "index": row["section_index"],
                    "coordinate": row["u_percent"],
                    "value_deg": row["gamma_deg"],
                    "signed_value_deg": "",
                    "quality_1": row["omega_orthogonality_score"],
                    "quality_2": row["omega_rotation_consistency"],
                    "is_valid": row["is_valid"],
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                }
            )
        for row in sigma_rows:
            position = np.asarray(row["plus_position"], dtype=float)
            writer.writerow(
                {
                    "measurement": "sigma",
                    "index": row["slice_index"],
                    "coordinate": row["l_level"],
                    "value_deg": row["sigma_deg"],
                    "signed_value_deg": row["sigma_signed_deg"],
                    "quality_1": row["plus_coherence"],
                    "quality_2": row["minus_coherence"],
                    "is_valid": True,
                    "x": position[0],
                    "y": position[1],
                    "z": position[2],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data_control_box")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--loop-index", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--grid-spacing", type=float, nargs=3, default=DEFAULT_GRID_SPACING)
    parser.add_argument("--nml-sampling-spacing", type=float, default=1.5)
    parser.add_argument("--gamma-sections", type=int, default=24)
    parser.add_argument("--omega-ring-layers", type=int, default=5)
    parser.add_argument("--omega-ring-dr", type=float, default=0.5)
    parser.add_argument("--omega-ring-arc-dist", type=float, default=0.4)
    parser.add_argument("--sigma-slices", type=int, default=9)
    parser.add_argument("--sigma-slice-fraction", type=float, default=0.7)
    parser.add_argument("--sigma-ring-radius", type=float, default=2.0)
    parser.add_argument("--sigma-ring-samples", type=int, default=96)
    parser.add_argument("--sigma-min-coherence", type=float, default=0.6)
    parser.add_argument("--sigma-min-coherence-gap", type=float, default=0.15)
    parser.add_argument("--fourier-crop-margin", type=int, default=6)
    parser.add_argument("--k-nu-tolerance-deg", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_path = args.data_dir / f"Q_{args.step}.npy"
    if not q_path.exists():
        raise FileNotFoundError(q_path)
    output_dir = args.output_dir or args.data_dir / "loop_parameter_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    q5 = load_q(q_path)
    q_obj = build_q_object(q5, args.step)
    line, selected_loop_index, detected_loop_count = choose_single_loop(
        q_obj, args.loop_index
    )
    smooth, smooth_coords = smooth_loop_coords(line)
    raw_coords = np.asarray(line.calc_defect_coords, dtype=float)

    nu, nu_metrics = measure_nu(line)
    axes_nml, nml_diagnostics = measure_nml_frame(
        q_obj,
        raw_coords,
        spacing=float(args.nml_sampling_spacing),
    )
    gamma_rows, gamma_summary = measure_gamma(
        smooth,
        nu,
        sections=int(args.gamma_sections),
        ring_layers=int(args.omega_ring_layers),
        ring_dr=float(args.omega_ring_dr),
        ring_arc_dist=float(args.omega_ring_arc_dist),
    )
    sigma_rows, sigma_summary = measure_sigma(
        q_obj,
        smooth_coords,
        axes_nml,
        slices=int(args.sigma_slices),
        slice_fraction=float(args.sigma_slice_fraction),
        ring_radius=float(args.sigma_ring_radius),
        ring_samples=int(args.sigma_ring_samples),
        min_coherence=float(args.sigma_min_coherence),
        min_coherence_gap=float(args.sigma_min_coherence_gap),
    )
    k_result = measure_dominant_k(
        q5,
        raw_coords,
        axes_nml,
        nu,
        crop_margin=int(args.fourier_crop_margin),
        grid_spacing=tuple(float(value) for value in args.grid_spacing),
    )
    k_nu_consistent = bool(
        k_result["angle_k_nu_deg"] <= float(args.k_nu_tolerance_deg)
    )

    center = np.mean(raw_coords, axis=0)
    radius = float(np.sqrt(np.mean(np.sum((raw_coords - center) ** 2, axis=1))))
    summary = {
        "input": {
            "q_path": str(q_path.resolve()),
            "step": int(args.step),
            "grid_shape": q5.shape[:3],
            "grid_spacing": args.grid_spacing,
            "periodic_boundary": PERIODIC_BOUNDARY,
        },
        "loop": {
            "detected_loop_count": detected_loop_count,
            "selected_loop_index": selected_loop_index,
            "defect_points": int(line.calc_defect_num),
            "center_index": center,
            "rms_radius_index": radius,
        },
        "nu": {
            "vector_lab": nu,
            "definition": "normal of the best-fit common plane of the near-elliptical loop",
            "metrics": nu_metrics,
        },
        "NML": {
            "N_lab": axes_nml[:, 0],
            "M_lab": axes_nml[:, 1],
            "L_lab": axes_nml[:, 2],
            "definition": "decreasing-eigenvalue eigenvectors of q_box = <n n>_box",
            "principal_plane": "N-M",
            "principal_plane_normal": "L",
            "diagnostics": nml_diagnostics,
        },
        "gamma": {
            **gamma_summary,
            "definition": "mean local arccos(abs(nu dot Omega)) over loop sections",
            "from_nu_and_L_deg": unsigned_angle_deg(nu, axes_nml[:, 2]),
        },
        "sigma": {
            **sigma_summary,
            "definition": "principal angle from chi (+1/2 to -1/2) to p (+1/2 symmetry axis)",
        },
        "k": k_result,
        "comparisons": {
            "k_nu_tolerance_deg": float(args.k_nu_tolerance_deg),
            "k_nu_consistent": k_nu_consistent,
            "gamma_local_minus_k_deg": float(
                gamma_summary["mean_deg"] - k_result["gamma_from_k_deg"]
            ),
            "sigma_local_minus_k_deg": float(
                sigma_summary["mean_deg"] - k_result["sigma_from_k_deg"]
            ),
        },
        "loop_convention": {
            "name": CONVENTION_NAME,
            "source": CONVENTION_SOURCE,
            "ideal_loop_assumptions_tested": ["k parallel nu", "Omega parallel L"],
            "small_perturbation_mapping_not_used_for_loop_classification": True,
        },
        "local_gamma": gamma_rows,
        "local_sigma": sigma_rows,
    }

    json_path = output_dir / f"loop_parameters_step_{args.step:05d}.json"
    csv_path = output_dir / f"loop_parameters_step_{args.step:05d}_local.csv"
    with json_path.open("w") as handle:
        json.dump(json_value(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_local_csv(csv_path, gamma_rows, sigma_rows)

    print(f"frame={q_path}")
    print(
        f"loop_count={detected_loop_count} selected={selected_loop_index} "
        f"points={line.calc_defect_num} planarity={nu_metrics['planarity_score']:.6f}"
    )
    print(f"nu={np.round(nu, 8).tolist()}")
    print(
        f"gamma_mean_deg={gamma_summary['mean_deg']:.6f} "
        f"gamma_std_deg={gamma_summary['std_deg']:.6f} "
        f"accepted_sections={gamma_summary['accepted_sections']}/"
        f"{gamma_summary['total_sections']}"
    )
    print(
        f"sigma_mean_deg={sigma_summary['mean_deg']:.6f} "
        f"sigma_std_deg={sigma_summary['std_deg']:.6f} "
        f"valid_slices={sigma_summary['valid_slices']}"
    )
    print(f"k_hat={np.round(k_result['k_hat_lab'], 8).tolist()}")
    print(
        f"angle_k_nu_deg={k_result['angle_k_nu_deg']:.6f} "
        f"consistent={k_nu_consistent} tolerance_deg={args.k_nu_tolerance_deg:g}"
    )
    print(
        f"gamma_from_k_deg={k_result['gamma_from_k_deg']:.6f} "
        f"sigma_from_k_deg={k_result['sigma_from_k_deg']:.6f} "
        f"nearest_pure_type={k_result['nearest_pure_type']}"
    )
    print(f"json={json_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
