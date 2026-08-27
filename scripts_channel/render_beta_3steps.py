#!/usr/bin/env python3
"""Render the three beta snapshot panels used by the summary figure.

This wrapper fixes the camera, crop, director, and defect-line parameters that
were used for steps 6200, 6360, and 7000.  Color bars are hidden by default so
the three panels have a consistent appearance before they are composed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RENDER_SCRIPT = Path(__file__).with_name("render_beta_frame.py")
DEFAULT_STEPS = (6200, 6360, 7000)


def render_panel(
    step: int,
    output: Path,
    *,
    show_colorbar_at_7000: bool,
) -> None:
    xvfb_run = shutil.which("xvfb-run")
    if xvfb_run is None:
        raise RuntimeError("xvfb-run is required for off-screen PyVista rendering")

    command = [
        xvfb_run,
        "-a",
        sys.executable,
        str(RENDER_SCRIPT),
        "--step",
        str(step),
        "--real-beta",
        "--focus-loop",
        "--highlight-defect-directors",
        "--director-plane-z",
        "20",
        "--local-half-widths",
        "10",
        "10",
        "10",
        "--local-x-bounds",
        "246",
        "266",
        "--local-y-bounds",
        "4",
        "36",
        "--local-z-bounds",
        "4",
        "36",
        "--director-spacing",
        "1.8",
        "--director-length",
        "1.25",
        "--director-radius",
        "0.105",
        "--director-opacity",
        "0.40",
        "--highlight-director-radius",
        "0.105",
        "--highlight-director-opacity",
        "0.96",
        "--line-radius",
        "0.22",
        "--line-sides",
        "128",
        "--line-smooth-density",
        "24",
        "--beta-samples",
        "281",
        "--surface-stride",
        "2",
        "--shell-opacity",
        "0.06",
        "--bounds-radius",
        "0.014",
        "--bounds-opacity",
        "0.075",
        "--camera-distance",
        "86",
        "--camera-azimuth",
        "45",
        "--camera-elevation",
        "24",
        "--camera-roll",
        "0",
        "--camera-window-center",
        "-0.06",
        "-0.24",
        "--colorbar-position",
        "0.805",
        "0.275",
        "--colorbar-size",
        "0.040",
        "0.45",
        "--figure-size",
        "2200",
        "1500",
        "--save-scale",
        "3",
        "--output",
        str(output),
    ]

    # The historical step-7000 panel used the renderer's color-bar feature.
    # It can still be reproduced explicitly, while the new default is hidden.
    if not (step == 7000 and show_colorbar_at_7000):
        command.append("--hide-colorbar")

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data_control_box" / "beta_channel",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Text inserted after the step number, before .png.",
    )
    parser.add_argument(
        "--show-colorbar-at-7000",
        action="store_true",
        help="Reproduce the historical step-7000 panel with its color bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for step in (int(value) for value in args.steps):
        output = args.output_dir / f"beta_{step}{args.suffix}.png"
        render_panel(
            step,
            output,
            show_colorbar_at_7000=bool(args.show_colorbar_at_7000),
        )
        print(f"panel={output}")


if __name__ == "__main__":
    main()
