#!/usr/bin/env python3
"""Run the smooth-trajectory DAL x-resolution convergence study."""

from __future__ import annotations

import csv
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CONFIGS = {
    8: REPO_ROOT / "data_dal_smooth_trajectory_final",
    4: REPO_ROOT / "data_dal_resolution_xstride4",
    2: REPO_ROOT / "data_dal_resolution_xstride2",
}


def complete(run_dir: Path) -> bool:
    history = run_dir / "history.csv"
    if not history.is_file():
        return False
    with history.open(newline="") as stream:
        return len(list(csv.DictReader(stream))) >= 10


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def optimization_command(x_stride: int, output: Path) -> list[str]:
    return [
        PYTHON,
        "Channel_dal.py",
        "--steps",
        "1000",
        "--iterations",
        "10",
        "--block-size",
        "100",
        "--checkpoint-stride",
        "10",
        "--stride",
        str(x_stride),
        "2",
        "2",
        "--mask-type",
        "path_gaussian",
        "--num-masks",
        "8",
        "--armijo-step",
        "1",
        "--initial-alpha",
        "4.5",
        "--loss-function",
        "core_aware_comoving_q_trajectory",
        "--running-aligned-shape-weight",
        "0",
        "--terminal-aligned-shape-weight",
        "0",
        "--running-comoving-q-weight",
        "0",
        "--terminal-comoving-q-weight",
        "0",
        "--skip-gradient-check",
        "--skip-trajectory-diagnostics",
        "--output",
        str(output.relative_to(REPO_ROOT)),
    ]


def main() -> None:
    for x_stride, output in CONFIGS.items():
        if not complete(output):
            run(optimization_command(x_stride, output))
        else:
            print(f"skip completed optimization: {output.name}", flush=True)
        run(
            [
                PYTHON,
                "scripts_channel/visualize_dal_activity_mechanism.py",
                "--result-dir",
                str(output.relative_to(REPO_ROOT)),
            ]
        )
    run([PYTHON, "scripts_channel/compare_dal_resolution_convergence.py"])


if __name__ == "__main__":
    main()
