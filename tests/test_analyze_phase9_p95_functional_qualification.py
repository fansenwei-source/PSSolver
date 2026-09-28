"""Fail-closed tests for the frozen P9.5 H100 evidence analyzer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import statistics

import pytest

from benchmarks.analyze_phase9_p95_functional_qualification import (
    P95EvidenceError,
    PASS_CLASSIFICATION,
    analyze,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_9_p95_h100_qualification_plan.json"
)
COMMIT = "a" * 40


def _plan():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _report(plan, grid, role, trial):
    initial = ("1" if grid["id"] == "R128" else "2") * 64
    state = ("3" if grid["id"] == "R128" else "4") * 64
    gradient = ("5" if grid["id"] == "R128" else "6") * 64
    mean = {
        "production_forward": 0.010,
        "functional_forward": 0.011,
        "functional_vjp": 0.055,
    }[role]
    result = (
        {
            "finite": True,
            "nonzero": True,
            "gradient_norms": [1.0, 2.0, 3.0],
            "gradient_sha256": gradient,
            "replay_bitwise_equal": True,
        }
        if role == "functional_vjp"
        else {"finite": True, "state_sha256": state}
    )
    correctness = {}
    if trial == 1 and role != "production_forward":
        correctness["r12"] = {"passed": True}
    if trial == 1 and role == "functional_vjp":
        correctness["gradient"] = {"passed": True}
    return {
        "schema_version": 1,
        "kind": "p95_periodic_functional_profile",
        "phase": "P9.5",
        "config": {
            "role": role,
            "trial": trial,
            "shape": grid["shape"],
            "lengths": grid["lengths"],
            "device": "cuda:0",
            "dtype": "float64",
            "dt": plan["dt"],
            "base_activity": plan["base_activity"],
            "warmup_steps": plan["warmup_steps"],
            "profile_steps": plan["profile_steps"],
            "initial_q_sha256": initial,
        },
        "environment": {
            "cuda_available": True,
            "device": "cuda:0",
            "device_name": plan["gpu_model"],
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "git": {"head": COMMIT, "status_porcelain": ""},
        },
        "functional_identity": (
            None
            if role == "production_forward"
            else {
                "execution": {
                    "functional_runtime": {
                        "fallback_used": False,
                        "deterministic_replay": "not_qualified",
                    }
                }
            }
        ),
        "timing": {
            "samples_seconds": [mean] * plan["profile_steps"],
            "mean_seconds": mean,
            "median_seconds": mean,
            "coefficient_of_variation": 0.0,
        },
        "memory": {
            "peak_allocated_bytes": 100 if role != "functional_vjp" else 300,
            "peak_active_bytes": 90 if role != "functional_vjp" else 290,
            "peak_reserved_bytes": 200 if role != "functional_vjp" else 400,
            "device_total_bytes": 1000,
        },
        "result": result,
        "correctness": correctness,
    }


def _matrix():
    plan = _plan()
    reports = [
        _report(plan, grid, role, trial)
        for grid in plan["grids"]
        for role in plan["roles"]
        for trial in range(1, plan["trials"] + 1)
    ]
    return plan, reports


def test_complete_matrix_passes_and_qualifies_only_batch_one():
    plan, reports = _matrix()

    result = analyze(plan, reports, expected_commit=COMMIT)

    assert result["classification"] == PASS_CLASSIFICATION
    assert result["profile_count"] == 18
    assert result["batch_sizes_qualified"] == [1]
    assert result["larger_batch_qualified"] is False
    assert result["cuda_deterministic_replay"] == "bitwise"
    assert result["eligible_for_p9_6"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report: report["timing"].update(
                coefficient_of_variation=0.21
            ),
            "reported CV",
        ),
        (
            lambda report: report["environment"]["git"].update(head="b" * 40),
            "commit differs",
        ),
        (
            lambda report: report["result"].update(
                gradient_sha256="7" * 64
            ),
            "VJP replay differs",
        ),
    ],
)
def test_analyzer_rejects_invalid_evidence(mutation, message):
    plan, reports = _matrix()
    target = next(
        report
        for report in reports
        if report["config"]["role"] == "functional_vjp"
        and report["config"]["trial"] == 2
    )
    mutation(target)

    with pytest.raises(P95EvidenceError, match=message):
        analyze(plan, reports, expected_commit=COMMIT)


def test_analyzer_rejects_missing_or_duplicate_profile():
    plan, reports = _matrix()
    with pytest.raises(P95EvidenceError, match="expected 18"):
        analyze(plan, reports[:-1], expected_commit=COMMIT)

    duplicate = copy.deepcopy(reports)
    duplicate[-1] = copy.deepcopy(reports[0])
    with pytest.raises(P95EvidenceError, match="duplicate profile"):
        analyze(plan, duplicate, expected_commit=COMMIT)


def test_analyzer_rejects_relaxed_plan_or_fabricated_timing_summary():
    plan, reports = _matrix()
    relaxed = copy.deepcopy(plan)
    relaxed["gates"]["profile_cv_max"] = 1.0
    with pytest.raises(P95EvidenceError, match="plan gates changed"):
        analyze(relaxed, reports, expected_commit=COMMIT)

    reports[0]["timing"]["mean_seconds"] = 0.009
    with pytest.raises(P95EvidenceError, match="reported mean"):
        analyze(plan, reports, expected_commit=COMMIT)


def test_analyzer_rejects_measured_profile_variability_above_gate():
    plan, reports = _matrix()
    timing = reports[0]["timing"]
    samples = [0.005, 0.015] * 10
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples)
    timing.update(
        samples_seconds=samples,
        mean_seconds=mean,
        median_seconds=statistics.median(samples),
        coefficient_of_variation=deviation / mean,
    )

    with pytest.raises(P95EvidenceError, match="CV exceeds"):
        analyze(plan, reports, expected_commit=COMMIT)
