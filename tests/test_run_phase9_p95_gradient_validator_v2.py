"""Tests for the fresh-process P9.5 validator-v2 runner."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.run_phase9_p95_gradient_validator_v2 import build_commands


ROOT = Path(__file__).resolve().parents[1]


def test_runner_builds_two_unique_grid_processes(tmp_path):
    plan = json.loads(
        (
            ROOT
            / "notes"
            / "architecture_v0_2"
            / "phase_9_p95_gradient_validator_v2_diagnostic_plan.json"
        ).read_text(encoding="utf-8")
    )
    commands = build_commands(
        plan,
        initial_paths={
            "R128": Path("/inputs/R128/Q_0.npy"),
            "R320": Path("/inputs/R320/Q_0.npy"),
        },
        output_directory=tmp_path / "diagnostics",
        python_executable="/fixed/python",
    )

    assert [item["grid"] for item in commands] == ["R128", "R320"]
    assert len({item["output"] for item in commands}) == 2
    assert all(
        item["argv"][:3]
        == [
            "/fixed/python",
            "-m",
            "benchmarks.diagnose_phase9_p95_gradient_validator_v2",
        ]
        for item in commands
    )
