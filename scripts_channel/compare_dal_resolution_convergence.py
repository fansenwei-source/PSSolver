#!/usr/bin/env python3
"""Compare loop translation as the DAL x resolution is refined."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    8: REPO_ROOT / "data_dal_smooth_trajectory_final",
    4: REPO_ROOT / "data_dal_resolution_xstride4",
    2: REPO_ROOT / "data_dal_resolution_xstride2",
}
FULL_DX = 128.0 / 512.0


def periodic_delta(final: float, initial: float, period: float = 128.0) -> float:
    return (final - initial + 0.5 * period) % period - 0.5 * period


def read_metrics(x_stride: int, run_dir: Path) -> dict:
    activity_summary = json.loads(
        (run_dir / "activity_visualization" / "summary.json").read_text()
    )
    with (run_dir / "history.csv").open(newline="") as stream:
        terminal = list(csv.DictReader(stream))[-1]
    return {
        "x_stride": x_stride,
        "dx": x_stride * FULL_DX,
        "q_disturbance_displacement": periodic_delta(
            activity_summary["final_actual_center_x"],
            activity_summary["initial_center_x"],
        ),
        "core_displacement": periodic_delta(
            activity_summary["final_coarse_core_center_x"],
            activity_summary["initial_coarse_core_center_x"],
        ),
        "target_displacement": periodic_delta(
            activity_summary["target_center_x"],
            activity_summary["initial_center_x"],
        ),
        "radius_rms": float(terminal["radius_rms"]),
        "radius_y_rms": float(terminal["radius_y_rms"]),
        "radius_z_rms": float(terminal["radius_z_rms"]),
        "x_thickness_rms": float(terminal["x_thickness_rms"]),
        "core_mass": float(terminal["core_mass"]),
        "profile_correlation": float(terminal["profile_correlation"]),
        "plaquettes": {
            axis: int(float(terminal[f"plaquettes_{axis}"]))
            for axis in ("xy", "xz", "yz")
        },
    }


def plot(metrics: list[dict], output: Path) -> None:
    dx = np.asarray([item["dx"] for item in metrics])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), dpi=190, constrained_layout=True)
    axes[0, 0].plot(
        dx,
        [item["q_disturbance_displacement"] for item in metrics],
        marker="o",
        linewidth=2,
        label="Q-disturbance",
    )
    axes[0, 0].plot(
        dx,
        [item["core_displacement"] for item in metrics],
        marker="s",
        linewidth=2,
        label="Low-order core",
    )
    axes[0, 0].plot(
        dx,
        [item["target_displacement"] for item in metrics],
        color="#343A40",
        linestyle="--",
        label="Target",
    )
    axes[0, 0].set_title("Resolved translation")
    axes[0, 0].set_ylabel("Displacement along x")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        dx,
        [item["radius_rms"] for item in metrics],
        marker="o",
        linewidth=2,
    )
    axes[0, 1].axhline(1.5667, color="#343A40", linestyle="--", label="Initial")
    axes[0, 1].set_title("Core radius")
    axes[0, 1].set_ylabel("Radius RMS")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(
        dx,
        [item["profile_correlation"] for item in metrics],
        marker="o",
        linewidth=2,
    )
    axes[1, 0].set_ylim(0.0, 1.02)
    axes[1, 0].set_title("Initial-profile correlation")
    axes[1, 0].set_ylabel("Correlation")

    totals = [
        sum(item["plaquettes"].values())
        for item in metrics
    ]
    contamination = [
        item["plaquettes"]["yz"] / max(total, 1)
        for item, total in zip(metrics, totals)
    ]
    imbalance = [
        abs(item["plaquettes"]["xy"] - item["plaquettes"]["xz"])
        / max(item["plaquettes"]["xy"] + item["plaquettes"]["xz"], 1)
        for item in metrics
    ]
    axes[1, 1].plot(
        dx, contamination, marker="o", linewidth=2, label="yz contamination"
    )
    axes[1, 1].plot(
        dx, imbalance, marker="s", linewidth=2, label="xy/xz imbalance"
    )
    axes[1, 1].axhline(0.0, color="#343A40", linestyle="--", linewidth=1.2)
    axes[1, 1].set_title("Normalized topology diagnostics")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.set_xlabel(r"$\Delta x$")
        axis.set_xticks(dx)
        axis.grid(alpha=0.22)
        axis.invert_xaxis()
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main() -> None:
    output_dir = REPO_ROOT / "data_dal_resolution_convergence"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [read_metrics(stride, run_dir) for stride, run_dir in RUNS.items()]
    metrics.sort(key=lambda item: item["dx"], reverse=True)
    (output_dir / "convergence_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot(metrics, output_dir / "resolution_convergence.png")
    print(json.dumps(metrics, indent=2))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
