"""Synthetic, fail-closed tests for RC4.4.7 H100 qualification support."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.analyze_rc447_pressure_h100 import (
    COMMITS,
    CORRECTNESS_PASS,
    CORRECTNESS_SCHEMA,
    PASS,
    WHEELS,
    analyze,
)
from benchmarks.profile_rc447_channel_pressure import (
    SCHEMA as PROFILE_SCHEMA,
    _atomic_json as write_profile,
    _pressure_observation,
)
from benchmarks.plan_rc447_pressure_h100 import SUPPORT_FILES, build_plan
from benchmarks.qualify_rc447_pressure_cuda import (
    Gate,
    _atomic_json as write_correctness,
)
from benchmarks.run_rc447_cuda_tests import NODE_IDS, classify


def _correctness() -> dict[str, object]:
    names = {
        "public_identity",
        "scale_matrix",
        "transpose_identity",
        "custom_adjoint",
        "finite_difference",
        "resolution_aware_taylor",
        "checkpointed_full_history",
        "same_version_restart",
        "negative_identity_rejection",
    }
    return {
        "schema": CORRECTNESS_SCHEMA,
        "classification": CORRECTNESS_PASS,
        "qualification_complete": True,
        "environment": {
            "allocated_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_after_policy": {"matmul": False, "cudnn": False},
        },
        "identities": {
            "provider_commit": COMMITS["candidate"],
            "consumer_source_commit": (
                "0cf55b4fb699ff40433000f5b4efe52b1d33546b"
            ),
        },
        "gates": {name: {"passed": True} for name in names},
        "runtime_fallback_used": False,
        "compile_fallback_used": False,
        "graph_breaks": 0,
    }


def _cuda_tests() -> dict[str, object]:
    outcomes = {node: "passed" for node in NODE_IDS}
    return {
        "schema": "pssolver.rc4_4_7.cuda_only_tests.v1",
        "classification": "PASS_RC4_4_7_SEVEN_CUDA_ONLY_TESTS",
        "passed": True,
        "pytest_returncode": 0,
        "outcomes": outcomes,
        "environment": {
            "allocated_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_after_policy": {"matmul": False, "cudnn": False},
            "source_shadow_import": False,
        },
    }


def _profile(role: str, grid: str, trial: int) -> dict[str, object]:
    shape = {
        "R128x20x20": [128, 20, 20],
        "R512x40x40": [512, 40, 40],
    }[grid]
    seconds = 1.0 if role == "baseline" else 1.02
    if role == "baseline":
        item = {
            "schema": "legacy_public_flow_diagnostics.v1",
            "achieved_iteration_count": 7,
            "termination_reason": "not_exposed_by_legacy_schema",
            "termination_reason_observed": False,
            "acceptable": None,
            "acceptable_observed": False,
            "achieved_normalized_residual": 1.0e-7,
        }
    else:
        item = {
            "schema": "channel_pressure_diagnostics.v2",
            "achieved_iteration_count": 7,
            "termination_reason": "relative_tolerance_met",
            "termination_reason_observed": True,
            "acceptable": True,
            "acceptable_observed": True,
            "achieved_normalized_residual": 1.0e-7,
        }
    history = [copy.deepcopy(item) for _ in range(50)]
    digest = ("a" if grid == "R128x20x20" else "b") * 64
    return {
        "schema": PROFILE_SCHEMA,
        "role": role,
        "provider_identity": {
            "commit": COMMITS[role],
            "wheel_sha256": WHEELS[role],
        },
        "config": {
            "runtime_path": "legacy_channel",
            "trial": trial,
            "shape": shape,
            "lengths": [32.0, 5.0, 5.0],
            "device": "cuda",
            "dt": 0.01,
            "activity": 5.0,
            "warmup_steps": 10,
            "profile_steps": 50,
            "seed": 24,
        },
        "runtime_identity": {
            "requested": "legacy_channel",
            "effective": "legacy_channel",
            "fallback_used": False,
        },
        "environment": {
            "allocated_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_after_policy": {"matmul": False, "cudnn": False},
        },
        "timing": {
            "samples_seconds": [seconds] * 50,
            "mean_timestep_seconds": seconds,
            "median_timestep_seconds": seconds,
        },
        "memory": {
            "peak_allocated_bytes": 100 if role == "baseline" else 102,
            "peak_active_bytes": 100 if role == "baseline" else 102,
            "peak_reserved_bytes": 120 if role == "baseline" else 124,
        },
        "transform_calls": {
            "forward_per_step": 11.0,
            "inverse_per_step": 36.0,
        },
        "compiler": {
            "graph_breaks_during_profile": 0,
            "pointwise": {
                "requested": "compile",
                "effective": "compile",
                "fallback_allowed": False,
                "fallback_reason": None,
            },
        },
        "pressure_history": history,
        "initial_q_sha256": digest,
        "final_state_sha256": digest[::-1],
        "completed_steps": 60,
        "finite": True,
    }


def _matrix(role: str) -> list[dict[str, object]]:
    return [
        _profile(role, grid, trial)
        for grid in ("R128x20x20", "R512x40x40")
        for trial in (1, 2, 3)
    ]


def test_atomic_writers_round_trip_and_refuse_overwrite_or_nonfinite(tmp_path):
    first = tmp_path / "profile.json"
    write_profile(first, {"value": 1.0})
    assert json.loads(first.read_text()) == {"value": 1.0}
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_profile(first, {"value": 2.0})

    second = tmp_path / "correctness.json"
    write_correctness(second, {"value": [1, 2, 3]})
    assert json.loads(second.read_text()) == {"value": [1, 2, 3]}
    with pytest.raises(ValueError, match="non-finite"):
        write_correctness(tmp_path / "nan.json", {"value": float("nan")})


def test_gate_persists_a_failed_check_without_claiming_pass(tmp_path):
    gate = Gate(tmp_path, "synthetic")
    gate.check("positive", True, 1, 1)
    gate.check("negative", False, 2, 1)
    loaded = gate.finish()
    assert loaded["passed"] is False
    assert [item["passed"] for item in loaded["checks"]] == [True, False]


def test_profile_preserves_legacy_and_v2_pressure_schema_without_inference():
    class Legacy:
        @staticmethod
        def flow_diagnostics():
            return {
                "last_pressure_iterations": 9,
                "last_pressure_residual": 1.0e-8,
                "last_pressure_relative_residual": 2.0e-7,
            }

    class V2:
        @staticmethod
        def flow_diagnostics():
            return {
                "schema_version": 2,
                "pressure": {
                    "schema_version": 2,
                    "achieved_iteration_count": 8,
                    "termination_reason": "relative_tolerance_met",
                    "acceptable": True,
                    "achieved_normalized_residual": 1.0e-7,
                },
            }

    legacy = _pressure_observation(Legacy())
    current = _pressure_observation(V2())
    assert legacy["termination_reason"] == "not_exposed_by_legacy_schema"
    assert legacy["termination_reason_observed"] is False
    assert legacy["acceptable"] is None
    assert current["termination_reason"] == "relative_tolerance_met"
    assert current["termination_reason_observed"] is True
    assert current["acceptable"] is True


def test_analyzer_accepts_complete_matrix_and_preserves_nonclaims():
    report = analyze(
        _matrix("baseline"), _matrix("candidate"), _correctness(), _cuda_tests()
    )
    assert report["classification"] == PASS
    assert report["qualification_complete"] is True
    assert report["rc4_4_7_complete"] is True
    assert report["B25_closed_verified"] is True
    assert report["B28_closed_verified"] is True
    assert report["B29_closed_verified"] is True
    assert report["eligible_for_rc4_5_planning"] is True
    assert report["eligible_for_automatic_merge"] is False
    assert report["eligible_for_default_promotion"] is False
    assert report["production_default_changed"] is False
    assert report["larger_batch_qualified"] is False
    assert report["optimizer_campaign_authorized"] is False
    assert report["scientific_control_result"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_profile", "profile matrix incomplete"),
        ("graph_break", "graph break"),
        ("compile_fallback", "compile fallback contract differs"),
        ("performance", "timestep regression"),
        ("allocated", "allocated-memory regression"),
        ("correctness", "correctness subgate failed"),
    ],
)
def test_analyzer_rejects_incomplete_or_regressed_evidence(mutation, message):
    baseline = _matrix("baseline")
    candidate = _matrix("candidate")
    correctness = _correctness()
    if mutation == "missing_profile":
        candidate.pop()
    elif mutation == "graph_break":
        candidate[0]["compiler"]["graph_breaks_during_profile"] = 1
    elif mutation == "compile_fallback":
        candidate[0]["compiler"]["pointwise"]["fallback_reason"] = "eager"
    elif mutation == "performance":
        candidate[0]["timing"]["mean_timestep_seconds"] = 2.0
    elif mutation == "allocated":
        candidate[0]["memory"]["peak_allocated_bytes"] = 200
    else:
        correctness["gates"]["finite_difference"]["passed"] = False
    with pytest.raises(ValueError, match=message):
        analyze(baseline, candidate, correctness, _cuda_tests())


def test_final_state_identity_is_conditional_on_pressure_path_identity():
    baseline = _matrix("baseline")
    candidate = _matrix("candidate")
    candidate[0]["final_state_sha256"] = "c" * 64
    report = analyze(baseline, candidate, _correctness(), _cuda_tests())
    paired = report["grids"]["R128x20x20"]["paired"][0]
    assert paired["pressure_paths_match"] is False
    assert paired["final_state_sha_equal"] is False


def test_synthetic_payloads_are_json_strict():
    report = analyze(
        _matrix("baseline"), _matrix("candidate"), _correctness(), _cuda_tests()
    )
    encoded = json.dumps(report, allow_nan=False, sort_keys=True)
    assert json.loads(encoded) == report
    assert len(report["grids"]) == 2
    assert all(len(value["paired"]) == 3 for value in report["grids"].values())


def test_command_plan_has_balanced_unique_profiles_and_installed_interpreters(
    tmp_path,
):
    baseline_python = tmp_path / "venv_a" / "bin" / "python"
    candidate_python = tmp_path / "venv_b" / "bin" / "python"
    for path in (baseline_python, candidate_python):
        path.parent.mkdir(parents=True)
        path.write_text("#!/bin/sh\n", encoding="utf-8")
    support = tmp_path / "support"
    for relative in SUPPORT_FILES:
        path = support / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen support\n", encoding="utf-8")
    consumer = tmp_path / "consumer"
    provider = tmp_path / "provider"
    consumer.mkdir()
    provider.mkdir()
    plan = build_plan(
        baseline_python=baseline_python,
        candidate_python=candidate_python,
        support_root=support,
        output_root=tmp_path / "results",
        consumer_root=consumer,
        provider_test_root=provider,
    )
    commands = plan["profile_commands"]
    assert plan["profile_count"] == 12
    assert [item["sequence"] for item in commands] == list(range(1, 13))
    assert len({item["name"] for item in commands}) == 12
    assert [item["role"] for item in commands[:6]] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert [item["role"] for item in commands[6:]] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert len(
        {
            item["environment"]["TRITON_CACHE_DIR"]
            for item in commands
        }
    ) == 12
    assert len(
        {
            item["environment"]["TORCHINDUCTOR_CACHE_DIR"]
            for item in commands
        }
    ) == 12
    assert all(item["cwd"] == str(support.resolve()) for item in commands)
    assert all("--device" not in item["argv"] for item in commands)
    assert plan["execution_started"] is False
    assert plan["h100_submission_authorized"] is False
    assert plan["automatic_retry"] is False
    assert plan["execution_order"] == [
        "cuda_test_command",
        "correctness_command",
        "profile_commands",
        "analyzer_command",
    ]
    cuda = plan["cuda_test_command"]
    assert cuda["stage"] == 1
    assert cuda["argv"][0] == str(candidate_python.resolve())
    assert cuda["argv"][1:3] == ["-m", "benchmarks.run_rc447_cuda_tests"]
    assert plan["correctness_command"]["stage"] == 2
    assert plan["analyzer_command"]["stage"] == 4


def test_command_plan_refuses_support_root_that_can_shadow_pssolver(tmp_path):
    baseline_python = tmp_path / "a-python"
    candidate_python = tmp_path / "b-python"
    baseline_python.write_text("", encoding="utf-8")
    candidate_python.write_text("", encoding="utf-8")
    support = tmp_path / "support"
    (support / "pssolver").mkdir(parents=True)
    with pytest.raises(ValueError, match="must not contain a pssolver"):
        build_plan(
            baseline_python=baseline_python,
            candidate_python=candidate_python,
            support_root=support,
            output_root=tmp_path / "output",
            consumer_root=tmp_path / "consumer",
            provider_test_root=tmp_path / "provider",
        )


def test_cuda_test_classifier_requires_seven_unique_passes():
    collected = list(NODE_IDS)
    outcomes = {node: "passed" for node in collected}
    assert classify(0, collected, outcomes) == (True, [])
    outcomes[collected[-1]] = "skipped"
    passed, problems = classify(0, collected, outcomes)
    assert passed is False
    assert any("non-pass outcomes" in problem for problem in problems)
    passed, problems = classify(0, collected[:-1], {node: "passed" for node in collected[:-1]})
    assert passed is False
    assert any("collected 6 tests" in problem for problem in problems)
