"""Tests for the fresh-process P9.5 gradient diagnostic runner."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.run_phase9_p95_gradient_diagnostic import build_commands


ROOT = Path(__file__).resolve().parents[1]


def test_runner_builds_six_unique_fresh_process_commands(tmp_path):
    plan = json.loads(
        (
            ROOT
            / "notes"
            / "architecture_v0_2"
            / "phase_9_p95_gradient_diagnostic_plan.json"
        ).read_text(encoding="utf-8")
    )
    paths = {
        "R128": Path("/inputs/R128/Q_0.npy"),
        "R320": Path("/inputs/R320/Q_0.npy"),
    }

    commands = build_commands(
        plan,
        initial_paths=paths,
        output_directory=tmp_path / "diagnostics",
        python_executable="/fixed/python",
    )

    assert [item["grid"] for item in commands] == [
        "R128",
        "R128",
        "R128",
        "R320",
        "R320",
        "R320",
    ]
    assert [item["trial"] for item in commands] == [1, 2, 3, 1, 2, 3]
    assert len({item["output"] for item in commands}) == 6
    assert all(
        item["argv"][:3]
        == [
            "/fixed/python",
            "-m",
            "benchmarks.diagnose_phase9_p95_gradient_epsilon",
        ]
        for item in commands
    )


def test_runner_rejects_a_noncanonical_trial_contract(tmp_path):
    plan = {
        "phase": "P9.5-gradient-diagnostic",
        "execution": {
            "trials_per_grid": 1,
            "diagnostic_process_count": 2,
            "analysis_only": True,
            "dtype": "float64",
            "dt": 0.001,
            "base_activity": 0.013,
            "tf32": False,
            "fresh_process_per_trial": True,
            "submission_count_max": 1,
            "automatic_retry": False,
        },
        "activity_epsilons": [
            0.0001,
            0.0003,
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.008,
            0.01,
        ],
        "frozen_relative_tolerance": 0.00002,
        "grids": [
            {"id": "R128", "shape": [128, 128, 32], "lengths": [100, 100, 20]},
            {"id": "R320", "shape": [320, 320, 80], "lengths": [100, 100, 20]},
        ],
    }

    import pytest

    with pytest.raises(ValueError, match="three trials"):
        build_commands(
            plan,
            initial_paths={
                "R128": Path("/inputs/R128/Q_0.npy"),
                "R320": Path("/inputs/R320/Q_0.npy"),
            },
            output_directory=tmp_path / "diagnostics",
            python_executable="/fixed/python",
        )
