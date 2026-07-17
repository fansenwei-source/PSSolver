#!/usr/bin/env python3
"""Plot loop radius over one alpha-control cycle."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NEMATICS_SRC_CANDIDATES = [
    REPO_ROOT.parent / "Nematics3D" / "src",
    REPO_ROOT / "Nematics3D" / "src",
]
for candidate in LOCAL_NEMATICS_SRC_CANDIDATES:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import matplotlib.pyplot as plt
import nematics3d as n3d


PERIODIC_BOUNDARY = (True, False, False)
DETECTION_PLANES = (True, True, True)
DEFAULT_TARGET_CENTER = np.array([256.5, 26.4, 21.0], dtype=float)


def read_control_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def alpha_state_by_step(rows: list[dict[str, str]]) -> dict[int, str]:
    return {int(row["step"]): row["control_state"] for row in rows}


def available_steps(data_dir: Path, start: int, end: int) -> list[int]:
    steps = []
    for q_path in data_dir.glob("Q_*.npy"):
        try:
            step = int(q_path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if start <= step <= end:
            steps.append(step)
    return sorted(steps)


def smooth_or_raw_coords(line: object) -> np.ndarray:
    if getattr(line, "calc_defect_num", 0) >= 5:
        try:
            smooth = line.act_smooth(
                window_length=5,
                min_line_length=5,
                is_window_warning=False,
            )
            coords = np.asarray(smooth.result, dtype=float)
            if coords.ndim == 2 and coords.shape[1] == 3:
                return coords
        except Exception:
            pass
    return np.asarray(line.calc_defect_coords, dtype=float)


def loop_radius(coords: np.ndarray) -> float:
    center = coords.mean(axis=0)
    delta = coords - center
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def detect_loops(q5: np.ndarray) -> list[object]:
    _, director = n3d.Q_diagonalize(q5)
    defects = n3d.defect_detect(
        director,
        threshold=0.0,
        is_boundary_periodic=PERIODIC_BOUNDARY,
        planes=DETECTION_PLANES,
    )
    lines = n3d.defect_classify_into_lines(
        defects,
        box_size_periodic=[
            director.shape[0],
            np.inf,
            np.inf,
        ],
        grid_offset=np.zeros(3),
        grid_transform=np.eye(3),
    )
    return [line for line in lines if getattr(line, "kind", None) == "loop"]


def measure_loop_radius(q_path: Path, target: np.ndarray) -> tuple[float, np.ndarray | None, int]:
    q5 = np.load(q_path)
    loops = detect_loops(q5)
    if not loops:
        return 0.0, None, 0

    candidates = []
    for line in loops:
        coords = smooth_or_raw_coords(line)
        center = coords.mean(axis=0)
        radius = loop_radius(coords)
        points = int(getattr(line, "calc_defect_num", len(coords)))
        dist = float(np.linalg.norm(center - target))
        candidates.append((dist, radius, center, points))
    _, radius, center, points = min(candidates, key=lambda item: item[0])
    return radius, center, points


def contiguous_state_spans(steps: list[int], states: dict[int, str]) -> list[tuple[int, int, str]]:
    spans = []
    if not steps:
        return spans
    start = steps[0]
    previous = steps[0]
    current_state = states.get(start, "unknown")
    for step in steps[1:]:
        state = states.get(step, current_state)
        if state != current_state:
            spans.append((start, step, current_state))
            start = step
            current_state = state
        previous = step
    spans.append((start, previous, current_state))
    return spans


def write_radius_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "alpha_state",
                "radius",
                "center_x",
                "center_y",
                "center_z",
                "line_points",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_radius(
    output: Path,
    rows: list[dict[str, object]],
    spans: list[tuple[int, int, str]],
) -> None:
    steps = np.asarray([row["step"] for row in rows], dtype=float)
    radii = np.asarray([row["radius"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.4, 5.2), dpi=180)
    for start, end, state in spans:
        color = "#dcefdc" if state == "on" else "#f3d7d7"
        ax.axvspan(start, end, color=color, alpha=0.75, linewidth=0)
        label_x = 0.5 * (start + end)
        ax.text(
            label_x,
            0.96,
            f"alpha {state}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=16,
            color="#303030",
        )

    ax.plot(steps, radii, color="black", linewidth=2.4, marker="o", markersize=3.8)
    ax.set_xlabel("time step", fontsize=22)
    ax.set_ylabel("loop radius", fontsize=22)
    ax.set_title("Loop radius over control cycle", fontsize=21)
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, color="#d9d9d9", linewidth=0.8, alpha=0.8)
    x_min = float(np.nanmin(steps))
    x_max = float(np.nanmax(steps))
    ax.set_xlim(x_min, x_max)
    middle_ticks = [
        tick
        for tick in range(
            int(np.ceil(x_min / 200.0) * 200),
            int(np.floor(x_max / 200.0) * 200) + 1,
            200,
        )
        if x_min < tick < x_max
    ]
    ax.set_xticks([int(x_min), *middle_ticks, int(x_max)])
    finite = radii[np.isfinite(radii)]
    if finite.size:
        ax.set_ylim(bottom=0.0, top=float(np.nanmax(finite)) * 1.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data_control_box")
    parser.add_argument("--start", type=int, default=6070)
    parser.add_argument("--end", type=int, default=7130)
    parser.add_argument("--target-center", type=float, nargs=3, default=DEFAULT_TARGET_CENTER.tolist())
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data_control_box" / "loop_growth_local_clean" / "loop_radius_control_cycle.png",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "data_control_box" / "loop_growth_local_clean" / "loop_radius_control_cycle.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    control_rows = read_control_history(args.data_dir / "control_history.csv")
    states = alpha_state_by_step(control_rows)
    steps = available_steps(args.data_dir, args.start, args.end)
    target = np.asarray(args.target_center, dtype=float)
    rows: list[dict[str, object]] = []

    for index, step in enumerate(steps, start=1):
        radius, center, points = measure_loop_radius(args.data_dir / f"Q_{step}.npy", target)
        if center is not None:
            target = center
            center_values = center.tolist()
        else:
            center_values = [np.nan, np.nan, np.nan]
        rows.append(
            {
                "step": step,
                "alpha_state": states.get(step, "unknown"),
                "radius": radius,
                "center_x": center_values[0],
                "center_y": center_values[1],
                "center_z": center_values[2],
                "line_points": points,
            }
        )
        print(f"[{index:03d}/{len(steps):03d}] step={step} radius={radius:.6g} state={states.get(step, 'unknown')}")

    write_radius_csv(args.csv, rows)
    plot_radius(args.output, rows, contiguous_state_spans(steps, states))
    print(f"csv={args.csv}")
    print(f"png={args.output}")


if __name__ == "__main__":
    main()
