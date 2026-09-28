"""Fail-closed tests for the P9.5 gradient diagnostic analyzer."""

from __future__ import annotations

import copy

import pytest

from benchmarks.analyze_phase9_p95_gradient_diagnostic import (
    GradientDiagnosticEvidenceError,
    analyze,
)
from benchmarks.diagnose_phase9_p95_gradient_epsilon import ACTIVITY_EPSILONS


COMMIT = "a" * 40


def _report(grid, trial):
    errors = {
        epsilon: (8.0e-6 if epsilon in {0.003, 0.004} else 1.0e-3)
        for epsilon in ACTIVITY_EPSILONS
    }
    return {
        "schema_version": 1,
        "kind": "p95_activity_gradient_epsilon_diagnostic",
        "phase": "P9.5-gradient-diagnostic",
        "config": {
            "grid_id": grid,
            "trial": trial,
            "shape": [128, 128, 32] if grid == "R128" else [320, 320, 80],
            "lengths": [100.0, 100.0, 20.0],
            "device": "cuda:0",
            "dtype": "float64",
            "dt": 0.001,
            "base_activity": 0.013,
            "initial_q_sha256": ("1" if grid == "R128" else "2") * 64,
            "activity_epsilons": list(ACTIVITY_EPSILONS),
            "relative_tolerance": 2.0e-5,
        },
        "environment": {
            "cuda_available": True,
            "device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "git": {"head": COMMIT, "status_porcelain": ""},
        },
        "frozen_validation": {
            "passed": False,
            "gradient_paths": [
                {
                    "path": f"path-{index}",
                    "finite": True,
                    "nonzero": True,
                    "passed": True,
                }
                for index in range(6)
            ],
            "directional_derivatives": [
                {"input": "state", "passed": True},
                {"input": "activity", "passed": False},
            ],
        },
        "activity_sweep": [
            {
                "epsilon": epsilon,
                "relative_error": errors[epsilon],
                "finite": True,
                "objective_replay_bitwise": True,
                "passes_frozen_relative_tolerance": errors[epsilon] <= 2.0e-5,
            }
            for epsilon in ACTIVITY_EPSILONS
        ],
        "qualification_changed": False,
        "p9_5_pass_claimed": False,
    }


def test_analyzer_finds_common_candidate_without_qualifying_p95():
    reports = [
        _report(grid, trial)
        for grid in ("R128", "R320")
        for trial in range(1, 4)
    ]
    result = analyze(reports, expected_commit=COMMIT)

    assert result["classification"] == (
        "DIAGNOSED_ACTIVITY_FD_SCALE_SENSITIVITY_WITH_COMMON_CANDIDATE"
    )
    assert result["common_passing_epsilons"] == [0.003, 0.004]
    assert result["qualification_complete"] is False
    assert result["p9_5_complete"] is False
    assert result["eligible_for_p9_6"] is False
    assert result["contract_change_authorized"] is False


def test_analyzer_can_report_no_common_candidate_without_false_pass():
    reports = [
        _report(grid, trial)
        for grid in ("R128", "R320")
        for trial in range(1, 4)
    ]
    for row in reports[-1]["activity_sweep"]:
        row["relative_error"] = 1.0e-3
        row["passes_frozen_relative_tolerance"] = False

    result = analyze(reports, expected_commit=COMMIT)

    assert result["classification"] == (
        "DIAGNOSTIC_COMPLETE_WITHOUT_COMMON_EPSILON_CANDIDATE"
    )
    assert result["common_passing_epsilons"] == []
    assert result["qualification_complete"] is False


def test_analyzer_rejects_wrong_commit_or_objective_replay():
    reports = [
        _report(grid, trial)
        for grid in ("R128", "R320")
        for trial in range(1, 4)
    ]
    wrong = copy.deepcopy(reports)
    wrong[0]["environment"]["git"]["head"] = "b" * 40
    with pytest.raises(GradientDiagnosticEvidenceError, match="commit differs"):
        analyze(wrong, expected_commit=COMMIT)

    reports[0]["activity_sweep"][0]["objective_replay_bitwise"] = False
    with pytest.raises(GradientDiagnosticEvidenceError, match="not bitwise"):
        analyze(reports, expected_commit=COMMIT)
