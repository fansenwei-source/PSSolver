#!/usr/bin/env python3
"""Compare the baseline and core-moment DAL experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pssolver.control import CoreAwareAlignedShapeObjective, CoreAwareMomentObjective


RUNS = {
    "A: baseline": REPO_ROOT / "data_dal_moment_comparison_A_baseline",
    "B: moment 0.05 / 1.0": REPO_ROOT / "data_dal_moment_comparison_B_moment1",
    "C: moment 0.1 / 2.0": REPO_ROOT / "data_dal_moment_comparison_C_moment2",
    "D: B loss + RBF 8x2x2": REPO_ROOT / "data_dal_moment_comparison_D_rbf8x2x2",
    "E: RBF 8x2x2 + step 2": REPO_ROOT / "data_dal_moment_comparison_E_rbf8x2x2_step2",
    "F: E + aligned core shape": REPO_ROOT / "data_dal_moment_comparison_F_aligned_shape",
}
COLORS = ("#2474B5", "#159570", "#D9781F", "#8C3B63", "#5B4B9A", "#B04759")
TARGET_RADIUS = 1.5667031958501174
TARGET_MASS = 18.92599105834961
TARGET_SHIFT = -2.0


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = rows[0]
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in keys
        if key not in {"accepted", "pure_splay_topology", "yz_plane_topology"}
    }


def plot_history(histories: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=180, constrained_layout=True)
    panels = (
        ("cost_after", "Relative objective", None),
        ("profile_shift", "Translation along x", TARGET_SHIFT),
        ("radius_rms", "Core radius", TARGET_RADIUS),
        ("core_mass", "Core mass", TARGET_MASS),
        ("plaquettes_yz", "yz defect plaquettes", 0.0),
        ("gradient_norm", "Gradient norm", None),
    )
    for axis, (key, title, target) in zip(axes.flat, panels):
        for (label, values), color in zip(histories.items(), COLORS):
            x = values["iteration"] + 1
            y = values[key]
            if key == "cost_after":
                y = y / values["cost_before"][0]
            axis.plot(x, y, marker="o", linewidth=2.0, markersize=4, label=label, color=color)
        if target is not None:
            axis.axhline(target, color="#343A40", linestyle="--", linewidth=1.3, label="Target")
        axis.set_title(title)
        axis.set_xlabel("Accepted DAL update")
        axis.grid(alpha=0.22)
    axes[0, 0].legend(frameon=False, fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def compose_loops(output: Path) -> None:
    entries = []
    for label, run_dir in RUNS.items():
        image = Image.open(run_dir / "visualization" / "loop_comparison.png").convert("RGB")
        entries.append((label, image))
    width = max(image.width for _, image in entries)
    label_height = 62
    title_height = 82
    canvas = Image.new(
        "RGB",
        (width, title_height + sum(label_height + image.height for _, image in entries)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36
        )
        label_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
        )
    except OSError:
        title_font = label_font = None
    draw.text((36, 20), "Core-moment objective: final-loop comparison", fill="#20252A", font=title_font)
    y = title_height
    for label, image in entries:
        draw.text((38, y + 12), label, fill="#30363B", font=label_font)
        y += label_height
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def common_moment_penalty(run_dir: Path) -> float:
    target_values = np.load(run_dir / "Q_target_downsampled.npy")
    final_values = np.load(run_dir / "Q_optimized_final.npy")
    target = torch.from_numpy(target_values).movedim(-1, 0).unsqueeze(1)
    final = torch.from_numpy(final_values).movedim(-1, 0).unsqueeze(1)
    objective = CoreAwareMomentObjective(
        target_q=target,
        dt=1e-3,
        domain_lengths=(128.0, 10.0, 10.0),
        spatial_mask=torch.from_numpy(np.load(run_dir / "control_mask.npy")),
        S_bulk=0.5393446629166317,
    )
    return float(objective.components(final)["core_moment_penalty"].item())


def common_aligned_shape_penalty(run_dir: Path) -> float:
    target_values = np.load(run_dir / "Q_target_downsampled.npy")
    final_values = np.load(run_dir / "Q_optimized_final.npy")
    target = torch.from_numpy(target_values).movedim(-1, 0).unsqueeze(1)
    final = torch.from_numpy(final_values).movedim(-1, 0).unsqueeze(1)
    objective = CoreAwareAlignedShapeObjective(
        target_q=target,
        dt=1e-3,
        domain_lengths=(128.0, 10.0, 10.0),
        spatial_mask=torch.from_numpy(np.load(run_dir / "control_mask.npy")),
        S_bulk=0.5393446629166317,
    )
    return float(objective.components(final)["aligned_core_shape_penalty"].item())


def final_summary(histories: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    summary = {}
    for label, values in histories.items():
        summary[label] = {
            "relative_cost": float(values["cost_after"][-1] / values["cost_before"][0]),
            "profile_shift": float(values["profile_shift"][-1]),
            "radius_rms": float(values["radius_rms"][-1]),
            "core_mass": float(values["core_mass"][-1]),
            "plaquettes_xy": int(values["plaquettes_xy"][-1]),
            "plaquettes_xz": int(values["plaquettes_xz"][-1]),
            "plaquettes_yz": int(values["plaquettes_yz"][-1]),
            "q_tracking_mse": float(values["terminal_q_tracking_mse"][-1]),
            "core_tracking_mse": float(values["terminal_core_tracking_mse"][-1]),
            "core_mass_penalty": float(values["terminal_core_mass_penalty"][-1]),
            "core_moment_penalty": common_moment_penalty(RUNS[label]),
            "aligned_core_shape_penalty": common_aligned_shape_penalty(RUNS[label]),
        }
    return summary


def main() -> None:
    output_dir = REPO_ROOT / "data_dal_moment_comparison"
    histories = {
        label: read_history(run_dir / "history.csv")
        for label, run_dir in RUNS.items()
    }
    plot_history(histories, output_dir / "iteration_comparison.png")
    compose_loops(output_dir / "final_loop_comparison.png")
    summary = final_summary(histories)
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"history_plot={output_dir / 'iteration_comparison.png'}")
    print(f"loop_plot={output_dir / 'final_loop_comparison.png'}")


if __name__ == "__main__":
    main()
