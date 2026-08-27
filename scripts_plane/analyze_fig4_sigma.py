#!/usr/bin/env python3
"""Measure the Fig. 4 observable <sigma>/H from benchmark Q snapshots.

For each open disclination line that touches both z walls, sigma is the
periodic minimum-image distance in the xy plane between its bottom- and
top-wall endpoints.  Closed loops, one-wall arches, and interior segments are
counted for diagnostics but excluded from the Fig. 4 average.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NEMATICS_CANDIDATES = (
    PROJECT_ROOT.parent / "Nematics3D" / "src",
    PROJECT_ROOT / "Nematics3D" / "src",
)
for candidate in NEMATICS_CANDIDATES:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import nematics3d as n3d
from pssolver.models.active_nematics import Q_convention_metadata
from pssolver.models.active_nematics.nematics3d_adapter import director_from_Q


Q_PATTERN = re.compile(r"^Q_(\d+)\.npy$")
PERIODIC = (True, True, False)
PLANES = (True, True, True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract <sigma>/H from one or more Fig. 4 benchmark runs."
    )
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--wall-tolerance-index",
        type=float,
        default=0.51,
        help="Distance from z index 0 or Nz-1 counted as touching a wall.",
    )
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--output-prefix",
        default="fig4_sigma",
        help="Basename used for per-run and combined output files.",
    )
    args = parser.parse_args()
    if args.wall_tolerance_index < 0:
        parser.error("--wall-tolerance-index must be non-negative")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark metadata: {path}")
    with path.open() as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError(f"{path} must contain schema_version=1 metadata")

    solver = metadata.get("solver")
    if not isinstance(solver, dict):
        raise ValueError(f"{path} is missing solver metadata")
    missing_solver = [
        name for name in ("shape", "lengths", "dt") if name not in solver
    ]
    if missing_solver:
        raise ValueError(f"{path} is missing solver keys {missing_solver}")
    shape = solver["shape"]
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError(f"{path} has invalid solver.shape={shape!r}")
    lengths = solver["lengths"]
    if (
        not isinstance(lengths, (list, tuple))
        or len(lengths) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
            for value in lengths
        )
    ):
        raise ValueError(f"{path} has invalid solver.lengths={lengths!r}")
    dt = solver["dt"]
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError(f"{path} has invalid solver.dt={dt!r}")

    model = metadata.get("model")
    if not isinstance(model, dict) or model.get("name") != "active_nematics":
        raise ValueError(f"{path} must declare model.name='active_nematics'")
    expected_convention = Q_convention_metadata()
    if model.get("Q_convention") != expected_convention:
        raise ValueError(
            f"{path} must declare the complete canonical Q convention "
            f"{expected_convention!r}"
        )
    parameters = model.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{path} is missing model.parameters")
    for name in ("S_initial", "S_bulk"):
        value = parameters.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{path} has invalid model.parameters.{name}={value!r}")

    for name in ("activity_number", "frank_k", "zeta"):
        value = metadata.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{path} has invalid {name}={value!r}")
    return metadata


def discover_runs(scan_root: Path, include_incomplete: bool) -> list[tuple[Path, dict]]:
    candidates = [scan_root] if (scan_root / "metadata.json").exists() else []
    candidates.extend(
        path.parent for path in scan_root.glob("*/metadata.json") if path.parent not in candidates
    )
    runs = []
    for run_dir in candidates:
        if not include_incomplete and not (run_dir / "COMPLETE").exists():
            print(f"Skipping incomplete run {run_dir}", file=sys.stderr)
            continue
        metadata = load_metadata(run_dir)
        runs.append((run_dir, metadata))
    runs.sort(
        key=lambda item: (
            float(item[1]["solver"]["lengths"][2]),
            float(item[1]["activity_number"]),
        )
    )
    if not runs:
        raise RuntimeError(f"No analyzable benchmark runs found under {scan_root}")
    return runs


def q_steps(run_dir: Path, args: argparse.Namespace) -> list[int]:
    steps = []
    for path in run_dir.glob("Q_*.npy"):
        match = Q_PATTERN.match(path.name)
        if match:
            steps.append(int(match.group(1)))
    steps.sort()
    if args.start_step is not None:
        steps = [step for step in steps if step >= args.start_step]
    if args.end_step is not None:
        steps = [step for step in steps if step <= args.end_step]
    steps = steps[:: args.frame_stride]
    if args.limit is not None:
        steps = steps[: args.limit]
    if not steps:
        raise RuntimeError(f"No selected Q snapshots in {run_dir}")
    return steps


def load_q(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    q = np.load(path, mmap_mode="r")
    if q.shape != (*expected_shape, 5):
        raise ValueError(
            f"{path} has shape {q.shape}; expected {(*expected_shape, 5)}"
        )
    return np.asarray(q)


def detect_lines(q: np.ndarray, threshold: float, lengths: tuple[float, float, float]):
    director = director_from_Q(q)
    defects = n3d.defect_detect(
        director,
        threshold=threshold,
        is_boundary_periodic=PERIODIC,
        planes=PLANES,
    )
    shape = q.shape[:3]
    spacing = np.asarray(lengths, dtype=float) / np.asarray(shape, dtype=float)
    lines = n3d.defect_classify_into_lines(
        defects,
        box_size_periodic=[shape[0], shape[1], np.inf],
        grid_offset=0.5 * spacing,
        grid_transform=np.diag(spacing),
    )
    return defects, lines


def periodic_circular_mean(values: np.ndarray, period: float) -> float:
    wrapped = np.mod(np.asarray(values, dtype=float), period)
    angles = 2.0 * np.pi * wrapped / period
    resultant = np.mean(np.exp(1j * angles))
    if abs(resultant) <= 1e-12:
        return float(wrapped[0])
    return float(np.mod(np.angle(resultant), 2.0 * np.pi) * period / (2.0 * np.pi))


def wall_xy_center(
    indices: np.ndarray,
    mask: np.ndarray,
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
) -> np.ndarray:
    dx = lengths[0] / shape[0]
    dy = lengths[1] / shape[1]
    x_index = periodic_circular_mean(indices[mask, 0], shape[0])
    y_index = periodic_circular_mean(indices[mask, 1], shape[1])
    # The spectral solver uses a cell-centered grid.  The common half-cell
    # offset cancels in sigma but is retained in reported endpoint positions.
    return np.asarray(((x_index + 0.5) * dx, (y_index + 0.5) * dy))


def minimum_image_delta(delta: np.ndarray, periods: np.ndarray) -> np.ndarray:
    return delta - periods * np.round(delta / periods)


def measure_frame(
    q: np.ndarray,
    *,
    step: int,
    threshold: float,
    wall_tolerance: float,
    lengths: tuple[float, float, float],
) -> tuple[dict, list[dict]]:
    shape = q.shape[:3]
    defects, lines = detect_lines(q, threshold, lengths)
    periods = np.asarray(lengths[:2], dtype=float)
    line_rows: list[dict] = []
    counts = {
        "line_count": len(lines),
        "seg_count": 0,
        "loop_count": 0,
        "cross_count": 0,
        "through_count": 0,
        "bottom_only_count": 0,
        "top_only_count": 0,
        "interior_count": 0,
    }

    for line_index, line in enumerate(lines):
        kind = str(getattr(line, "kind", "unknown"))
        if f"{kind}_count" in counts:
            counts[f"{kind}_count"] += 1
        indices = np.asarray(line.raw_defect_indices, dtype=float)
        bottom = indices[:, 2] <= wall_tolerance
        top = indices[:, 2] >= (shape[2] - 1.0 - wall_tolerance)
        touches_bottom = bool(np.any(bottom))
        touches_top = bool(np.any(top))

        if kind == "seg" and touches_bottom and touches_top:
            bottom_xy = wall_xy_center(indices, bottom, shape, lengths)
            top_xy = wall_xy_center(indices, top, shape, lengths)
            delta_xy = minimum_image_delta(top_xy - bottom_xy, periods)
            sigma = float(np.linalg.norm(delta_xy))
            counts["through_count"] += 1
            line_rows.append(
                {
                    "step": step,
                    "line_index": line_index,
                    "line_kind": kind,
                    "defect_points": len(indices),
                    "bottom_x": bottom_xy[0],
                    "bottom_y": bottom_xy[1],
                    "top_x": top_xy[0],
                    "top_y": top_xy[1],
                    "delta_x_minimum_image": delta_xy[0],
                    "delta_y_minimum_image": delta_xy[1],
                    "sigma": sigma,
                    "sigma_over_h": sigma / lengths[2],
                }
            )
        elif touches_bottom and not touches_top:
            counts["bottom_only_count"] += 1
        elif touches_top and not touches_bottom:
            counts["top_only_count"] += 1
        elif not touches_bottom and not touches_top:
            counts["interior_count"] += 1

    values = np.asarray([row["sigma_over_h"] for row in line_rows], dtype=float)
    frame = {
        "step": step,
        "defect_point_count": len(defects),
        **counts,
        "mean_sigma_over_h": float(np.mean(values)) if len(values) else math.nan,
        "std_sigma_over_h": float(np.std(values, ddof=1)) if len(values) > 1 else math.nan,
    }
    return frame, line_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    names = fieldnames if fieldnames is not None else list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarize_run(metadata: dict, frame_rows: list[dict], line_rows: list[dict]) -> dict:
    solver = metadata["solver"]
    parameters = metadata["model"]["parameters"]
    line_values = np.asarray([row["sigma_over_h"] for row in line_rows], dtype=float)
    frame_values = np.asarray(
        [row["mean_sigma_over_h"] for row in frame_rows if np.isfinite(row["mean_sigma_over_h"])],
        dtype=float,
    )
    frame_std = float(np.std(frame_values, ddof=1)) if len(frame_values) > 1 else math.nan
    return {
        "activity_number": float(metadata["activity_number"]),
        "height": float(solver["lengths"][2]),
        "frank_k": float(metadata["frank_k"]),
        "zeta": float(metadata["zeta"]),
        "S_initial": float(parameters["S_initial"]),
        "S_bulk": float(parameters["S_bulk"]),
        "frames": len(frame_rows),
        "frames_with_through_lines": len(frame_values),
        "through_line_observations": len(line_values),
        "mean_sigma_over_h_pooled": finite_or_none(np.mean(line_values)) if len(line_values) else None,
        "std_sigma_over_h_pooled": finite_or_none(np.std(line_values, ddof=1)) if len(line_values) > 1 else None,
        "mean_sigma_over_h_frame": finite_or_none(np.mean(frame_values)) if len(frame_values) else None,
        "std_sigma_over_h_frame": finite_or_none(frame_std),
        "sem_sigma_over_h_frame": finite_or_none(frame_std / np.sqrt(len(frame_values))),
    }


def analyze_run(run_dir: Path, metadata: dict, args: argparse.Namespace) -> dict:
    solver = metadata["solver"]
    shape = tuple(int(value) for value in solver["shape"])
    lengths = tuple(float(value) for value in solver["lengths"])
    steps = q_steps(run_dir, args)
    frame_rows: list[dict] = []
    line_rows: list[dict] = []
    for index, step in enumerate(steps, start=1):
        path = run_dir / f"Q_{step}.npy"
        try:
            frame, measured_lines = measure_frame(
                load_q(path, shape),
                step=step,
                threshold=args.threshold,
                wall_tolerance=args.wall_tolerance_index,
                lengths=lengths,
            )
        except Exception as error:
            if not args.continue_on_error:
                raise
            print(f"Failed {path}: {error}", file=sys.stderr)
            continue
        frame_rows.append(frame)
        line_rows.extend(measured_lines)
        print(
            f"[{run_dir.name} {index}/{len(steps)}] step={step} "
            f"lines={frame['line_count']} through={frame['through_count']} "
            f"mean_sigma/H={frame['mean_sigma_over_h']:.6g}"
        )

    if not frame_rows:
        raise RuntimeError(f"No frames were successfully analyzed in {run_dir}")
    prefix = args.output_prefix
    write_csv(run_dir / f"{prefix}_frames.csv", frame_rows)
    line_fields = [
        "step", "line_index", "line_kind", "defect_points",
        "bottom_x", "bottom_y", "top_x", "top_y",
        "delta_x_minimum_image", "delta_y_minimum_image", "sigma", "sigma_over_h",
    ]
    write_csv(run_dir / f"{prefix}_lines.csv", line_rows, line_fields)
    summary = summarize_run(metadata, frame_rows, line_rows)
    with (run_dir / f"{prefix}_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return {"run_dir": str(run_dir), **summary}


def plot_summary(path: Path, summaries: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping combined plot", file=sys.stderr)
        return

    valid = [row for row in summaries if row["mean_sigma_over_h_frame"] is not None]
    if not valid:
        print("No through-line measurements; skipping combined plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    marker_by_height = {10.0: "o", 15.0: "s", 20.0: "D"}
    for height in sorted({float(row["height"]) for row in valid}):
        group = sorted(
            (row for row in valid if float(row["height"]) == height),
            key=lambda row: float(row["activity_number"]),
        )
        activity = np.asarray([row["activity_number"] for row in group])
        mean = np.asarray([row["mean_sigma_over_h_frame"] for row in group])
        sem = np.asarray([
            0.0 if row["sem_sigma_over_h_frame"] is None else row["sem_sigma_over_h_frame"]
            for row in group
        ])
        ax.errorbar(
            activity,
            mean,
            yerr=sem,
            marker=marker_by_height.get(height, "o"),
            capsize=3,
            linewidth=1.4,
            label=rf"$H={height:g}$",
        )
    ax.set_xlabel(r"Activity number $A=H\sqrt{\zeta/K}$")
    ax.set_ylabel(r"$\langle\sigma\rangle/H$")
    ax.set_title("Shendruk Fig. 4 benchmark")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    scan_root = args.scan_root.resolve()
    summaries = [
        analyze_run(run_dir, metadata, args)
        for run_dir, metadata in discover_runs(scan_root, args.include_incomplete)
    ]
    combined_csv = scan_root / f"{args.output_prefix}_scan_summary.csv"
    write_csv(combined_csv, summaries)
    combined_json = scan_root / f"{args.output_prefix}_scan_summary.json"
    with combined_json.open("w") as handle:
        json.dump(summaries, handle, indent=2, allow_nan=False)
        handle.write("\n")
    plot_summary(scan_root / f"{args.output_prefix}_vs_activity.png", summaries)
    print(f"Wrote {combined_csv}")
    print(f"Wrote {combined_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
