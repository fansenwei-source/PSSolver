#!/usr/bin/env python3
"""Detect Nematics3D defects in saved Q snapshots and plot step-wise counts."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LOCAL_NEMATICS_SRC_CANDIDATES = (
    ROOT / "Nematics3D" / "src",
    ROOT.parent / "Nematics3D" / "src",
)
for candidate in LOCAL_NEMATICS_SRC_CANDIDATES:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("matplotlib is required to write histogram PNG files") from exc

N3D = None

from pssolver.models.active_nematics.nematics3d_adapter import director_from_Q
from pssolver.snapshots import load_q_snapshot


Q_PATTERN = re.compile(r"^Q_(\d+)\.npy$")


def parse_bool_triplet(values: list[int]) -> tuple[bool, bool, bool]:
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected exactly 3 integer flags")
    return tuple(bool(value) for value in values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Q_<step>.npy snapshots, detect defects with nematics3d, "
            "and save per-step diagnostics plus step-count figures."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data_channel")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output basename. Default: <data-dir>/defect_histogram",
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--periodic",
        type=int,
        nargs=3,
        default=(1, 0, 0),
        metavar=("PX", "PY", "PZ"),
        help="Periodic boundary flags for x y z, as 0/1 integers.",
    )
    parser.add_argument(
        "--planes",
        type=int,
        nargs=3,
        default=(1, 1, 1),
        metavar=("YZ", "XZ", "XY"),
        help=(
            "Plaquette-normal flags passed to nematics3d.defect_detect. "
            "For example 1 1 1 checks yz, xz, and xy plaquettes."
        ),
    )
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def q_steps(data_dir: Path) -> list[int]:
    steps = []
    for path in data_dir.glob("Q_*.npy"):
        match = Q_PATTERN.match(path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def require_q5(path: Path) -> np.ndarray:
    q = load_q_snapshot(path, require_S_initial=True).values
    return np.asarray(q, dtype=np.float64)


def detect_snapshot(
    q: np.ndarray,
    threshold: float,
    periodic: tuple[bool, bool, bool],
    planes: tuple[bool, bool, bool],
) -> tuple[np.ndarray, list]:
    if N3D is None:
        raise RuntimeError("nematics3d has not been imported")
    director = director_from_Q(q)
    defects = N3D.defect_detect(
        director,
        threshold=threshold,
        is_boundary_periodic=periodic,
        planes=planes,
    )
    box_size_periodic = [
        director.shape[axis] if periodic[axis] else np.inf
        for axis in range(3)
    ]
    lines = N3D.defect_classify_into_lines(
        defects,
        box_size_periodic=box_size_periodic,
        grid_offset=np.zeros(3),
        grid_transform=np.eye(3),
    )
    return defects, lines


def line_length(line: object) -> int:
    if hasattr(line, "calc_defect_num"):
        return int(line.calc_defect_num)
    if hasattr(line, "raw_defect_indices"):
        return int(len(line.raw_defect_indices))
    return int(len(np.asarray(line)))


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_npy(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = list(rows[0])
    dtype_fields = [("step", np.int64), ("grid_nx", np.int64), ("grid_ny", np.int64), ("grid_nz", np.int64)]
    dtype_fields.extend((name, np.float64) for name in fieldnames if name not in {"step", "grid_nx", "grid_ny", "grid_nz"})
    array = np.array([tuple(row[name] for name in fieldnames) for row in rows], dtype=dtype_fields)
    np.save(path, array)


def plot_step_curve(
    steps: np.ndarray,
    values: np.ndarray,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=180)
    if len(values) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
    else:
        ax.plot(steps, values, color="#1f5d7a", linewidth=1.7)
        ax.scatter(steps, values, color="#1f5d7a", s=9, linewidth=0, zorder=3)
        if len(steps) > 1:
            step_padding = 0.02 * (steps.max() - steps.min())
            ax.set_xlim(steps.min() - step_padding, steps.max() + step_padding)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    global N3D
    args = parse_args()
    try:
        import nematics3d as n3d
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "nematics3d is required for defect detection. Keep "
            "/home/fansenwei/Desktop/Develop/Nematics3D/src available and run with:\n"
            "  /home/fansenwei/anaconda3/envs/Nematics3D/bin/python "
            "scripts/defect_histogram.py\n"
            f"Original import error: {exc}"
        ) from exc
    N3D = n3d
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    periodic = parse_bool_triplet(list(args.periodic))
    planes = parse_bool_triplet(list(args.planes))

    steps = q_steps(data_dir)
    if args.steps is not None:
        requested = set(args.steps)
        steps = [step for step in steps if step in requested]
    if args.limit is not None:
        steps = steps[: args.limit]
    if not steps:
        raise RuntimeError(f"No Q_<step>.npy snapshots found in {data_dir}")

    rows: list[dict[str, float]] = []

    for index, step in enumerate(steps, start=1):
        q_path = data_dir / f"Q_{step}.npy"
        q = require_q5(q_path)
        defects, lines = detect_snapshot(
            q,
            threshold=args.threshold,
            periodic=periodic,
            planes=planes,
        )
        lengths = [line_length(line) for line in lines]
        row = {
            "step": step,
            "grid_nx": q.shape[0],
            "grid_ny": q.shape[1],
            "grid_nz": q.shape[2],
            "defect_points": len(defects),
            "defect_lines": len(lines),
            "line_length_min": min(lengths) if lengths else 0,
            "line_length_mean": float(np.mean(lengths)) if lengths else 0.0,
            "line_length_max": max(lengths) if lengths else 0,
        }
        rows.append(row)
        print(
            f"[{index}/{len(steps)}] step={step} "
            f"defect_points={row['defect_points']} "
            f"defect_lines={row['defect_lines']} "
            f"line_length_mean={row['line_length_mean']:.3f}",
            flush=True,
        )

    out_base = (args.out or data_dir / "defect_histogram").resolve()
    out_base.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_base.with_suffix(".csv")
    npy_path = out_base.with_suffix(".npy")
    defect_points_png = out_base.with_name(out_base.name + "_defect_points.png")
    defect_lines_png = out_base.with_name(out_base.name + "_defect_lines.png")
    line_lengths_png = out_base.with_name(out_base.name + "_line_length_mean.png")

    write_csv(csv_path, rows)
    write_npy(npy_path, rows)
    step_values = np.asarray([row["step"] for row in rows], dtype=float)
    plot_step_curve(
        step_values,
        np.asarray([row["defect_points"] for row in rows], dtype=float),
        "Defect points by step",
        "defect points",
        defect_points_png,
    )
    plot_step_curve(
        step_values,
        np.asarray([row["defect_lines"] for row in rows], dtype=float),
        "Classified defect lines by step",
        "defect lines",
        defect_lines_png,
    )
    plot_step_curve(
        step_values,
        np.asarray([row["line_length_mean"] for row in rows], dtype=float),
        "Mean defect points per classified line by step",
        "defect points per classified line",
        line_lengths_png,
    )

    defect_counts = np.asarray([row["defect_points"] for row in rows], dtype=float)
    line_counts = np.asarray([row["defect_lines"] for row in rows], dtype=float)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved NPY: {npy_path}")
    print(f"Saved defect-point step plot: {defect_points_png}")
    print(f"Saved defect-line step plot: {defect_lines_png}")
    print(f"Saved line-length step plot: {line_lengths_png}")
    print(
        "Summary: "
        f"snapshots={len(rows)}, "
        f"defect_points[min/mean/max]={defect_counts.min():.0f}/"
        f"{defect_counts.mean():.3f}/{defect_counts.max():.0f}, "
        f"defect_lines[min/mean/max]={line_counts.min():.0f}/"
        f"{line_counts.mean():.3f}/{line_counts.max():.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
