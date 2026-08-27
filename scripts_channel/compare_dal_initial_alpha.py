#!/usr/bin/env python3
"""Compare smooth-trajectory DAL runs initialized at alpha=4.5 and alpha=5."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "Initial alpha = 4.5": REPO_ROOT / "data_dal_smooth_trajectory_final",
    "Initial alpha = 5.0": REPO_ROOT / "data_dal_smooth_trajectory_alpha5",
}


def final_metrics(run_dir: Path) -> dict[str, float | dict[str, int]]:
    with (run_dir / "history.csv").open(newline="") as stream:
        row = list(csv.DictReader(stream))[-1]
    amplitudes = np.load(run_dir / "alpha_amplitudes.npy")
    return {
        "objective": float(row["cost_after"]),
        "translation_x": float(row["profile_shift"]),
        "profile_correlation": float(row["profile_correlation"]),
        "core_mass": float(row["core_mass"]),
        "radius_rms": float(row["radius_rms"]),
        "radius_y_rms": float(row["radius_y_rms"]),
        "radius_z_rms": float(row["radius_z_rms"]),
        "x_thickness_rms": float(row["x_thickness_rms"]),
        "trajectory_position_penalty": float(
            row["terminal_trajectory_position_penalty"]
        ),
        "core_moment_penalty": float(row["terminal_core_moment_penalty"]),
        "aligned_core_shape_penalty": float(
            row["terminal_aligned_core_shape_penalty"]
        ),
        "plaquettes": {
            axis: int(float(row[f"plaquettes_{axis}"]))
            for axis in ("xy", "xz", "yz")
        },
        "activity_min": float(amplitudes.min()),
        "activity_max": float(amplitudes.max()),
        "activity_mean": float(amplitudes.mean()),
    }


def compose(output: Path) -> None:
    entries = [
        (
            label,
            Image.open(run_dir / "visualization" / "dal_result_summary.png").convert(
                "RGB"
            ),
        )
        for label, run_dir in RUNS.items()
    ]
    image_width = 1500
    label_height = 76
    title_height = 100
    resized = []
    for label, image in entries:
        height = round(image.height * image_width / image.width)
        resized.append(
            (label, image.resize((image_width, height), Image.Resampling.LANCZOS))
        )
    canvas = Image.new(
        "RGB",
        (
            2 * image_width,
            title_height + label_height + max(image.height for _, image in resized),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42
        )
        label_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32
        )
    except OSError:
        title_font = label_font = None
    draw.text(
        (42, 24),
        "Smooth-trajectory DAL: initial activity comparison",
        fill="#20252A",
        font=title_font,
    )
    for index, (label, image) in enumerate(resized):
        x = index * image_width
        draw.text((x + 42, title_height + 16), label, fill="#30363B", font=label_font)
        canvas.paste(image, (x, title_height + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    output_dir = REPO_ROOT / "data_dal_initial_alpha_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {label: final_metrics(run_dir) for label, run_dir in RUNS.items()}
    (output_dir / "comparison_metrics.json").write_text(json.dumps(metrics, indent=2))
    compose(output_dir / "side_by_side.png")
    print(json.dumps(metrics, indent=2))
    print(f"saved={output_dir}")


if __name__ == "__main__":
    main()
