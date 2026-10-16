"""Synthetic contracts for the rc5 G-1 environment and CUDA producers."""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from benchmarks.plan_rc5_g1_h100 import (
    CONSUMER_WHEEL_SHA256,
    PROVIDER_BASELINE_WHEEL_SHA256,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    build_plan,
)
from benchmarks.run_rc5_g1_cuda_only import SCHEMA as CUDA_SCHEMA, summarize
from benchmarks.run_rc5_g1_consumer import summarize as summarize_consumer
from benchmarks.run_rc5_g1_checkpoints import (
    CELL_SCHEMA as CHECKPOINT_CELL_SCHEMA,
    DIRECTIONS as CHECKPOINT_DIRECTIONS,
    FAMILIES as CHECKPOINT_FAMILIES,
    RELEASED_RECORD_SHA256,
    summarize as summarize_checkpoints,
    validate_released_record,
)
from benchmarks.run_rc5_g1_environment import validate_observations
from benchmarks.run_rc5_g1_numerical import (
    CHILD_SCHEMA as NUMERICAL_CHILD_SCHEMA,
    FAMILIES as NUMERICAL_FAMILIES,
    REPAIRED as NUMERICAL_REPAIRED,
    summarize as summarize_numerical,
)


def test_consumer_helper_imports_the_actual_control_error_module():
    helper = Path(__file__).resolve().parents[1] / "benchmarks/run_rc5_g1_consumer.py"
    tree = ast.parse(helper.read_text(encoding="utf-8"), filename=str(helper))
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (
        "pssolver_control.errors",
        "PSSolverIntegrationUnavailable",
    ) in imports
    assert not any(module == "pssolver_control.exceptions" for module, _ in imports)


def _environment_inputs():
    observations = {
        "baseline": {"pssolver_version": "0.2.0rc4"},
        "candidate": {"pssolver_version": "0.2.0rc5"},
        "consumer": {
            "pssolver_version": "0.2.0rc5",
            "pssolver_control_version": "0.0.1.dev0",
        },
    }
    identities = {
        "baseline": {"passed": True},
        "candidate": {"passed": True},
        "consumer_provider": {"passed": True},
        "consumer": {"passed": True},
    }
    pip_checks = {
        role: {"status": "PASS"}
        for role in ("baseline", "candidate", "consumer")
    }
    wheel_hashes = {
        "baseline": PROVIDER_BASELINE_WHEEL_SHA256,
        "candidate": PROVIDER_CANDIDATE_WHEEL_SHA256,
        "consumer": CONSUMER_WHEEL_SHA256,
    }
    cuda = {
        "available": True,
        "visible_device_count": 1,
        "allocated_device": "cuda:0",
        "device_name_contains_h100": True,
    }
    tf32 = {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    return observations, identities, pip_checks, wheel_hashes, cuda, tf32


def test_environment_validator_accepts_only_the_frozen_installed_h100_identity():
    observations, identities, checks, wheels, cuda, tf32 = _environment_inputs()
    validate_observations(
        observations=observations,
        identities=identities,
        pip_checks=checks,
        wheel_hashes=wheels,
        support_contains_pssolver=False,
        cuda=cuda,
        tf32_after=tf32,
    )


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    (
        ("observations", ("candidate", "pssolver_version", "0.2.0rc4"), "version"),
        ("identities", ("consumer", "passed", False), "identity"),
        ("checks", ("consumer", "status", "FAIL"), "pip check"),
        ("wheels", ("candidate", None, "0" * 64), "wheel"),
        ("cuda", ("visible_device_count", None, 2), "H100 identity"),
        ("tf32", ("cudnn_allow_tf32", None, True), "TF32-off"),
    ),
)
def test_environment_validator_rejects_identity_drift(
    target, mutation, message
):
    observations, identities, checks, wheels, cuda, tf32 = _environment_inputs()
    values = {
        "observations": observations,
        "identities": identities,
        "checks": checks,
        "wheels": wheels,
        "cuda": cuda,
        "tf32": tf32,
    }
    first, second, replacement = mutation
    if second is None:
        values[target][first] = replacement
    else:
        values[target][first][second] = replacement
    with pytest.raises(ValueError, match=message):
        validate_observations(
            observations=observations,
            identities=identities,
            pip_checks=checks,
            wheel_hashes=wheels,
            support_contains_pssolver=False,
            cuda=cuda,
            tf32_after=tf32,
        )


def _cuda_detail():
    nodes = list(build_plan()["cuda_only_tests"])
    return {
        "requested_nodeids": nodes,
        "pytest_exit_code": 0,
        "passed": True,
        "outcomes": {
            "passed": nodes,
            "failed": [],
            "skipped": [],
            "xfailed": [],
            "xpassed": [],
            "collection_errors": [],
        },
    }


def _policy():
    return {"enforcement": {"passed": True}}


def test_cuda_summary_is_the_exact_analyzer_schema():
    result = summarize(_cuda_detail(), _policy())
    assert result["schema"] == CUDA_SCHEMA
    assert result["passed"] is True
    assert result["passed_count"] == 7
    assert result["node_ids"] == build_plan()["cuda_only_tests"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (("pytest_exit_code", 1), "did not pass"),
        (("passed", False), "did not pass"),
        (("outcomes", "skipped", [{"nodeid": "x"}]), "non-pass"),
        (("outcomes", "passed", []), "passed node IDs"),
    ),
)
def test_cuda_summary_rejects_incomplete_or_nonpassing_results(mutation, message):
    detail = _cuda_detail()
    if len(mutation) == 2:
        detail[mutation[0]] = mutation[1]
    else:
        detail[mutation[0]][mutation[1]] = mutation[2]
    with pytest.raises(ValueError, match=message):
        summarize(detail, _policy())


def test_cuda_summary_rejects_tf32_policy_failure():
    policy = copy.deepcopy(_policy())
    policy["enforcement"]["passed"] = False
    with pytest.raises(ValueError, match="TF32"):
        summarize(_cuda_detail(), policy)


def _consumer_inputs():
    contract = build_plan()["consumer_h100_gate"]
    outcomes = {
        "passed": [
            f"test_{index}" for index in range(contract["expected_passed"])
        ],
        "failed": [],
        "skipped": [{"nodeid": contract["allowed_skip"]}],
        "xfailed": [],
        "xpassed": [],
        "collection_errors": [],
        "deselected": [],
    }
    identity_report = {
        "identities": {
            "periodic": contract["periodic_identity_sha256"],
            "channel_fixed": contract["channel_fixed_identity_sha256"],
            "channel_tolerance": contract["channel_tolerance_identity_sha256"],
        },
        "old_pin": {"rejected": True, "target_unchanged": True},
        "finite_smoke": True,
    }
    return outcomes, identity_report, {"enforcement": {"passed": True}}


def test_consumer_summary_accepts_the_exact_342_plus_one_contract():
    outcomes, identities, policy = _consumer_inputs()
    result = summarize_consumer(
        exit_code=0,
        outcomes=outcomes,
        identity_report=identities,
        tf32_policy=policy,
    )
    assert result["schema"] == "pssolver.rc5.g1.consumer_h100.v1"
    assert result["passed_count"] == 342
    assert result["skipped_count"] == 1
    assert result["old_periodic_pin_rejected_before_mutation"] is True


@pytest.mark.parametrize(
    ("target", "mutation", "message"),
    (
        ("outcomes", ("passed", []), "passed count"),
        ("outcomes", ("deselected", ["test_x"]), "forbidden outcome"),
        ("identities", ("finite_smoke", False), "finite smoke"),
        ("policy", ("enforcement", {"passed": False}), "TF32"),
    ),
)
def test_consumer_summary_rejects_incomplete_results(target, mutation, message):
    outcomes, identities, policy = _consumer_inputs()
    values = {
        "outcomes": outcomes,
        "identities": identities,
        "policy": policy,
    }
    values[target][mutation[0]] = mutation[1]
    with pytest.raises(ValueError, match=message):
        summarize_consumer(
            exit_code=0,
            outcomes=outcomes,
            identity_report=identities,
            tf32_policy=policy,
        )


def _numerical_children(tmp_path):
    result = {}
    for family in NUMERICAL_FAMILIES:
        for role in ("baseline", "candidate"):
            report = {
                "schema": NUMERICAL_CHILD_SCHEMA,
                "family": family,
                "provider_role": role,
                "initial_state_sha256": "a" * 64,
                "final_state_sha256": "b" * 64,
                "finite": True,
                "fallback_used": False,
                "completed_steps": 2,
                "physical_artifacts": {},
                "hermitian_violation": None,
            }
            if family in NUMERICAL_REPAIRED:
                artifacts = {}
                for name in ("Q", "velocity", "pressure"):
                    path = tmp_path / f"{role}_{family}_{name}.npy"
                    np.save(path, np.ones((2, 2), dtype=np.float64))
                    artifacts[name] = {"path": str(path), "finite": True}
                report["physical_artifacts"] = artifacts
                report["hermitian_violation"] = 0.0
                if role == "candidate":
                    report["final_state_sha256"] = "c" * 64
            result[(role, family)] = report
    return result


def test_numerical_summary_separates_repaired_and_unchanged_families(tmp_path):
    report = summarize_numerical(
        _numerical_children(tmp_path),
        baseline_commit="1" * 40,
        candidate_commit="2" * 40,
    )
    assert report["passed"] is True
    assert report["family_count"] == 8
    indexed = {entry["family"]: entry for entry in report["families"]}
    assert {indexed[name]["comparison"] for name in NUMERICAL_REPAIRED} == {
        "physical_tolerance"
    }
    assert all(
        indexed[name]["byte_identical"] is True
        for name in set(NUMERICAL_FAMILIES) - NUMERICAL_REPAIRED
    )


def test_numerical_summary_rejects_unchanged_family_drift(tmp_path):
    children = _numerical_children(tmp_path)
    children[("candidate", "channel_functional")]["final_state_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="channel_functional final state"):
        summarize_numerical(
            children,
            baseline_commit="1" * 40,
            candidate_commit="2" * 40,
        )


def test_numerical_summary_rejects_nonzero_candidate_hermitian_violation(tmp_path):
    children = _numerical_children(tmp_path)
    children[("candidate", "periodic_functional")]["hermitian_violation"] = 1.0e-30
    with pytest.raises(RuntimeError, match="Hermitian violation"):
        summarize_numerical(
            children,
            baseline_commit="1" * 40,
            candidate_commit="2" * 40,
        )


def _checkpoint_cells():
    cells = []
    for family in CHECKPOINT_FAMILIES:
        for source, target in CHECKPOINT_DIRECTIONS:
            cells.append(
                {
                    "schema": CHECKPOINT_CELL_SCHEMA,
                    "family": family,
                    "source_device": source,
                    "target_device": target,
                    "passed": True,
                    "continuous_split_restart_equal": True,
                    "source_checkpoint_unchanged": True,
                    "fallback_used": False,
                    "payload_tamper": {
                        "rejected": True,
                        "guard_reached": True,
                        "target_unchanged": True,
                    },
                    "identity_mismatch": {
                        "rejected": True,
                        "guard_reached": True,
                        "target_unchanged": True,
                    },
                }
            )
    return cells


def _released():
    return {
        "sha256": RELEASED_RECORD_SHA256,
        "primary_case_count": 13,
        "matched": True,
    }


def test_checkpoint_summary_requires_all_eight_transactional_cells():
    report = summarize_checkpoints(_checkpoint_cells(), _released())
    assert report["passed"] is True
    assert report["current_family_count"] == 4
    assert report["direction_count"] == 2
    assert report["cell_count"] == 8


@pytest.mark.parametrize(
    ("path", "message"),
    (
        ((0, "continuous_split_restart_equal"), "replay differs"),
        ((0, "source_checkpoint_unchanged"), "source changed"),
        ((0, "payload_tamper", "target_unchanged"), "mutated target"),
        ((0, "identity_mismatch", "rejected"), "accepted"),
    ),
)
def test_checkpoint_summary_rejects_incomplete_transactional_evidence(path, message):
    cells = _checkpoint_cells()
    if len(path) == 2:
        cells[path[0]][path[1]] = False
    else:
        cells[path[0]][path[1]][path[2]] = False
    with pytest.raises(RuntimeError, match=message):
        summarize_checkpoints(cells, _released())


def test_checkpoint_released_record_is_bound_to_the_qualified_cpu_matrix():
    path = Path(__file__).resolve().parents[1] / "notes/PSSolver_v0_2_0rc5_g1_checkpoint_matrix.json"
    result = validate_released_record(path)
    assert result == _released()
