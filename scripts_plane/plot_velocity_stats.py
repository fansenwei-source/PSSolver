#!/usr/bin/env python3
"""Plot step-wise statistics of saved velocity-field snapshots."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("matplotlib is required to write velocity-stat PNG files") from exc


ROOT = Path(__file__).resolve().parents[1]
U_PATTERN = re.compile(r"^u_(\d+)\.npy$")
DEFAULT_DATA_DIR = ROOT / "data_plane_H=10"
DEFAULT_NX, DEFAULT_NY, DEFAULT_NZ = 512, 512, 40
DEFAULT_COMPONENT_AXIS = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan u_<step>.npy snapshots, compute min/max/mean/RMS of the "
            "velocity magnitude, and plot those quantities versus step."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path. Default: <data-dir>/velocity_stats_vs_step.png",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: same basename as --out with .csv suffix.",
    )
    parser.add_argument(
        "--npy",
        type=Path,
        default=None,
        help="Optional structured NPY output path for the computed statistics.",
    )
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--component-axis",
        type=int,
        default=DEFAULT_COMPONENT_AXIS,
        help=(
            "Axis containing velocity components. Default -1 matches "
            "PSSolver/data/u_<step>.npy shape (Nx, Ny, Nz, 3)."
        ),
    )
    return parser.parse_args()


def velocity_steps(data_dir: Path) -> list[int]:
    steps = []
    for path in data_dir.glob("u_*.npy"):
        match = U_PATTERN.match(path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def velocity_magnitude(path: Path, component_axis: int) -> np.ndarray:
    velocity = np.load(path, mmap_mode="r")
    expected_shape = (DEFAULT_NX, DEFAULT_NY, DEFAULT_NZ)
    if velocity.shape[:3] != expected_shape:
        raise ValueError(
            f"{path} grid shape {velocity.shape[:3]} does not match "
            f"Plane.py shape {expected_shape}"
        )
    axis = component_axis if component_axis >= 0 else velocity.ndim + component_axis
    if axis < 0 or axis >= velocity.ndim:
        raise ValueError(
            f"{path} has {velocity.ndim} dimensions; component axis {component_axis} is invalid"
        )
    if velocity.shape[axis] not in (2, 3):
        raise ValueError(
            f"{path} component axis {component_axis} must have length 2 or 3, "
            f"got shape {velocity.shape}"
        )
    return np.sqrt(np.sum(np.asarray(velocity, dtype=np.float64) ** 2, axis=axis))


def compute_stats(path: Path, component_axis: int) -> dict[str, float]:
    speed = velocity_magnitude(path, component_axis)
    return {
        "min": float(np.min(speed)),
        "max": float(np.max(speed)),
        "mean": float(np.mean(speed)),
        "rms": float(np.sqrt(np.mean(speed**2))),
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["step", "min", "max", "mean", "rms"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_npy(path: Path, rows: list[dict[str, float]]) -> None:
    dtype = [
        ("step", np.int64),
        ("min", np.float64),
        ("max", np.float64),
        ("mean", np.float64),
        ("rms", np.float64),
    ]
    array = np.array(
        [
            (
                int(row["step"]),
                float(row["min"]),
                float(row["max"]),
                float(row["mean"]),
                float(row["rms"]),
            )
            for row in rows
        ],
        dtype=dtype,
    )
    np.save(path, array)


def plot_stats(rows: list[dict[str, float]], path: Path) -> None:
    steps = np.array([row["step"] for row in rows], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=180)
    ax.plot(steps, [row["max"] for row in rows], label="Max", linewidth=1.8)
    ax.plot(steps, [row["min"] for row in rows], label="Min", linewidth=1.8)
    ax.plot(steps, [row["mean"] for row in rows], label="Mean", linewidth=1.8)
    ax.plot(steps, [row["rms"] for row in rows], label="RMS", linewidth=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("velocity magnitude")
    ax.set_title("Velocity Statistics vs Step")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    steps = velocity_steps(data_dir)
    if args.steps is not None:
        requested = set(args.steps)
        steps = [step for step in steps if step in requested]
    if args.limit is not None:
        steps = steps[: args.limit]
    if not steps:
        raise RuntimeError(f"No u_<step>.npy snapshots found in {data_dir}")

    out_path = (
        args.out.resolve()
        if args.out is not None
        else data_dir / "velocity_stats_vs_step.png"
    )
    csv_path = (
        args.csv.resolve()
        if args.csv is not None
        else out_path.with_suffix(".csv")
    )

    rows: list[dict[str, float]] = []
    for index, step in enumerate(steps, start=1):
        u_path = data_dir / f"u_{step}.npy"
        stats = compute_stats(u_path, args.component_axis)
        row = {"step": step, **stats}
        rows.append(row)
        print(
            f"[{index}/{len(steps)}] step={step} "
            f"min={row['min']:.6e} max={row['max']:.6e} "
            f"mean={row['mean']:.6e} rms={row['rms']:.6e}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    plot_stats(rows, out_path)
    write_csv(csv_path, rows)
    if args.npy is not None:
        npy_path = args.npy.resolve()
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        write_npy(npy_path, rows)
        print(f"Wrote {npy_path}")

    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
