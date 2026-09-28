"""Fail-closed tests for the P9.5 validator-v2 analyzer."""

from __future__ import annotations

import copy

import pytest

from benchmarks.analyze_phase9_p95_gradient_validator_v2 import (
    GradientValidatorV2EvidenceError,
    analyze,
)
from benchmarks.diagnose_phase9_p95_gradient_validator_v2 import (
    ACTIVITY_EPSILONS,
    DIRECTION_KINDS,
    METHODS,
    REFERENCE_RELATIVE_TOLERANCE,
    STATE_EPSILONS,
)


COMMIT = "a" * 40


def _report(grid):
    sweeps = []
    for target in ("state", "activity"):
        epsilons = STATE_EPSILONS if target == "state" else ACTIVITY_EPSILONS
        passing = epsilons[3]
        for direction in DIRECTION_KINDS:
            rows = []
            for epsilon in epsilons:
                methods = {}
                for method in METHODS:
                    error = 1.0e-3
                    if direction == "low_mode" and method == "paired_quadratic":
                        error = 8.0e-6 if epsilon == passing else 4.0e-4
                    methods[method] = {
                        "finite_difference": 1.0,
                        "absolute_error": error,
                        "relative_error": error,
                        "finite": True,
                        "passes_reference_tolerance": (
                            error <= REFERENCE_RELATIVE_TOLERANCE
                        ),
                    }
                rows.append(
                    {
                        "epsilon": epsilon,
                        "objective_plus": 1.0 + epsilon,
                        "objective_minus": 1.0 - epsilon,
                        "methods": methods,
                        "one_sided_linearization_remainder": {
                            "plus": 1.0e-8,
                            "minus": 1.0e-8,
                            "normalized_max": 1.0e-4,
                        },
                    }
                )
            sweeps.append(
                {
                    "target": target,
                    "direction": direction,
                    "autograd": 1.0,
                    "autograd_nonzero": True,
                    "absolute_direction_cosine": 0.5,
                    "rows": rows,
                }
            )
    return {
        "schema_version": 1,
        "kind": "p95_gradient_validator_v2_diagnostic",
        "phase": "P9.5-gradient-validator-v2-diagnostic",
        "config": {
            "grid_id": grid,
            "shape": [128, 128, 32] if grid == "R128" else [320, 320, 80],
            "lengths": [100.0, 100.0, 20.0],
            "device": "cuda:0",
            "dtype": "float64",
            "dt": 0.001,
            "base_activity": 0.013,
            "initial_q_sha256": ("1" if grid == "R128" else "2") * 64,
            "state_epsilons": list(STATE_EPSILONS),
            "activity_epsilons": list(ACTIVITY_EPSILONS),
            "direction_kinds": list(DIRECTION_KINDS),
            "methods": list(METHODS),
            "reference_relative_tolerance": REFERENCE_RELATIVE_TOLERANCE,
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
            "gradient_paths": [
                {
                    "finite": True,
                    "nonzero": True,
                    "passed": True,
                }
                for _ in range(6)
            ],
            "directional_derivatives": [
                {"input": "state", "passed": False},
                {"input": "activity", "passed": False},
            ],
        },
        "base_objective": 1.0,
        "sweeps": sweeps,
        "all_finite": True,
        "objective_replay_bitwise": True,
        "qualification_changed": False,
        "p9_5_pass_claimed": False,
        "p9_6_authorized": False,
    }


def test_analyzer_finds_independent_paired_cross_grid_candidates():
    result = analyze([_report("R128"), _report("R320")], expected_commit=COMMIT)

    assert result["classification"] == (
        "PAIRED_FD_INDEPENDENT_CROSS_GRID_CANDIDATE_FOUND"
    )
    assert {
        item["target"] for item in result["common_cross_grid_candidates"]
    } == {"state", "activity"}
    assert result["diagnostic_complete"] is True
    assert result["qualification_complete"] is False
    assert result["p9_5_complete"] is False
    assert result["eligible_for_p9_6"] is False
    assert result["contract_change_authorized"] is False


def test_analyzer_rejects_dirty_or_disconnected_evidence():
    reports = [_report("R128"), _report("R320")]
    dirty = copy.deepcopy(reports)
    dirty[0]["environment"]["git"]["status_porcelain"] = "?? output.json"
    with pytest.raises(GradientValidatorV2EvidenceError, match="dirty"):
        analyze(dirty, expected_commit=COMMIT)

    reports[0]["frozen_validation"]["gradient_paths"][0]["nonzero"] = False
    reports[0]["frozen_validation"]["gradient_paths"][0]["passed"] = False
    with pytest.raises(GradientValidatorV2EvidenceError, match="paths failed"):
        analyze(reports, expected_commit=COMMIT)
