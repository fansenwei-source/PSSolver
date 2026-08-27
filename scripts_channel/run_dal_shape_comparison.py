#!/usr/bin/env python3
"""Run the three controlled comparisons for trajectory-aware shape preservation."""

from __future__ import annotations

import subprocess
import sys
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
COMMON = [
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
    "8",
    "2",
    "2",
    "--mask-type",
    "path_gaussian",
    "--num-masks",
    "8",
    "--armijo-step",
    "1",
    "--skip-gradient-check",
]
RUNS = (
    (
        "data_dal_shape_comparison_B_baseline",
        [
            "--loss-function",
            "core_aware_moment_tracking",
        ],
    ),
    (
        "data_dal_shape_comparison_T_trajectory",
        [
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
        ],
    ),
    (
        "data_dal_shape_comparison_Q_comoving",
        [
            "--loss-function",
            "core_aware_comoving_q_trajectory",
            "--running-aligned-shape-weight",
            "0",
            "--terminal-aligned-shape-weight",
            "0",
            "--running-comoving-q-weight",
            "0.1",
            "--terminal-comoving-q-weight",
            "2.0",
        ],
    ),
)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def optimization_complete(output: str) -> bool:
    history = REPO_ROOT / output / "history.csv"
    if not history.is_file():
        return False
    with history.open(newline="") as stream:
        return len(list(csv.DictReader(stream))) >= 10


def main() -> None:
    for output, arguments in RUNS:
        if not optimization_complete(output):
            run([*COMMON, *arguments, "--output", output])
        else:
            print(f"skip completed optimization: {output}", flush=True)
        run(
            [
                "xvfb-run",
                "-a",
                PYTHON,
                "scripts_channel/visualize_dal_result.py",
                "--result-dir",
                output,
            ]
        )
    run([PYTHON, "scripts_channel/compare_dal_shape_runs.py"])


if __name__ == "__main__":
    main()
