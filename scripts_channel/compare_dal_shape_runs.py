#!/usr/bin/env python3
"""Compare baseline, trajectory-only, and comoving-Q DAL runs."""

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
    "B: original moment loss": REPO_ROOT / "data_dal_shape_comparison_B_baseline",
    "T: smooth trajectory": REPO_ROOT / "data_dal_shape_comparison_T_trajectory",
    "Q: trajectory + comoving Q": REPO_ROOT / "data_dal_shape_comparison_Q_comoving",
}
COLORS = ("#2474B5", "#D9781F", "#159570")
TARGET_SHIFT = -2.0
TARGET_RADIUS = 1.5667031958501174


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def values(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def main() -> None:
    histories = {label: read_history(path / "history.csv") for label, path in RUNS.items()}
    output = REPO_ROOT / "data_dal_shape_comparison"
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=180, constrained_layout=True)
    panels = (
        ("profile_shift", "Translation along x", TARGET_SHIFT),
        ("radius_rms", "Core radius", TARGET_RADIUS),
        ("profile_correlation", "Initial-profile correlation", 1.0),
        ("core_mass", "Core mass", None),
        ("plaquettes_yz", "yz defect plaquettes", 0.0),
        ("cost_after", "Normalized objective", None),
    )
    summary = {}
    for (label, rows), color in zip(histories.items(), COLORS):
        iteration = values(rows, "iteration") + 1
        for axis, (key, title, target) in zip(axes.flat, panels):
            y = values(rows, key)
            if key == "cost_after":
                y = y / float(rows[0]["cost_before"])
            axis.plot(iteration, y, marker="o", linewidth=2, color=color, label=label)
            axis.set_title(title)
            axis.set_xlabel("Accepted DAL update")
            axis.grid(alpha=0.22)
            if target is not None:
                axis.axhline(target, color="#343A40", linestyle="--", linewidth=1.2)
        last = rows[-1]
        summary[label] = {
            "translation_x": float(last["profile_shift"]),
            "radius_rms": float(last["radius_rms"]),
            "profile_correlation": float(last["profile_correlation"]),
            "core_mass": float(last["core_mass"]),
            "plaquettes": {
                axis: int(float(last[f"plaquettes_{axis}"]))
                for axis in ("xy", "xz", "yz")
            },
            "relative_objective": float(last["cost_after"])
            / float(rows[0]["cost_before"]),
        }
        for key in (
            "trajectory_max_comoving_q_shape_penalty",
            "trajectory_max_radius_relative_error",
            "trajectory_max_transverse_anisotropy",
            "trajectory_topology_violation_count",
        ):
            if key in last:
                summary[label][key] = float(last[key])
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.savefig(output / "iteration_comparison.png", facecolor="white")
    plt.close(fig)
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
