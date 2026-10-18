"""Synthetic fail-closed checks for the RC6 H100 stage producers."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pytest

from benchmarks.build_rc6_single_h100_stage_plan import build_draft
from benchmarks.plan_rc6_single_h100_qualification import (
    CONSUMER_WHEEL_SHA256,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    build_plan,
)
from benchmarks.run_rc5_g1_checkpoints import CELL_SCHEMA, DIRECTIONS, FAMILIES
from benchmarks.run_rc5_g1_numerical import CHILD_SCHEMA, FAMILIES as NUMERICAL_FAMILIES
from benchmarks.run_rc6_consumer import GUARD_NODE, REQUIRED_NODES, summarize as summarize_consumer
from benchmarks.run_rc6_current_checkpoints import summarize as summarize_current
from benchmarks.run_rc6_numerical import summarize as summarize_numerical
from benchmarks.run_rc6_released_checkpoints import summarize as summarize_released
from benchmarks.run_rc6_single_h100_stage_plan import DRAFT_SCHEMA, STAGES


def _cell(family: str, source: str, target: str) -> dict[str, object]:
    gate = {"rejected": True, "guard_reached": True, "target_unchanged": True}
    return {
        "schema": CELL_SCHEMA,
        "family": family,
        "source_device": source,
        "target_device": target,
        "passed": True,
        "continuous_split_restart_equal": True,
        "source_checkpoint_unchanged": True,
        "fallback_used": False,
        "payload_tamper": dict(gate),
        "identity_mismatch": dict(gate),
    }


def test_current_checkpoint_summary_requires_all_cells_and_negative_gates():
    cells = [_cell(family, source, target) for family in FAMILIES for source, target in DIRECTIONS]
    result = summarize_current(cells, True)
    assert result["passed"] is True and result["cell_count"] == 8
    broken = copy.deepcopy(cells)
    broken[0]["payload_tamper"]["target_unchanged"] = False
    with pytest.raises(ValueError, match="negative gate"):
        summarize_current(broken, True)
    with pytest.raises(ValueError, match="materialized output"):
        summarize_current(cells, False)


def test_released_checkpoint_summary_requires_twenty_clean_tests_and_immutable_sources():
    outcomes = {
        "passed": [f"test_{index}" for index in range(20)],
        "failed": [], "skipped": [], "xfailed": [], "xpassed": [], "collection_errors": [],
    }
    record = {"released_checkpoint_matrix": {"complete": True, "primary_case_count": 13}}
    assert summarize_released(outcomes, record, True)["primary_case_count"] == 13
    with pytest.raises(ValueError, match="pytest result"):
        summarize_released({**outcomes, "skipped": [{"nodeid": "x"}]}, record, True)
    with pytest.raises(ValueError, match="fixture tree changed"):
        summarize_released(outcomes, record, False)


def _child(role: str, family: str) -> dict[str, object]:
    return {
        "schema": CHILD_SCHEMA,
        "provider_role": role,
        "family": family,
        "completed_steps": 2,
        "finite": True,
        "fallback_used": False,
        "initial_state_sha256": "a" * 64,
        "final_state_sha256": "b" * 64,
        "hermitian_violation": 0.0 if family.startswith("periodic_") else None,
        "pssolver_version": "0.2.0rc5" if role == "baseline" else "0.2.0rc6",
        "pssolver_import": f"/venv/{role}/lib/python3.10/site-packages/pssolver/__init__.py",
    }


def test_numerical_summary_requires_byte_identity_and_periodic_zero_hermitian_violation():
    children = {(role, family): _child(role, family) for role in ("baseline", "candidate") for family in NUMERICAL_FAMILIES}
    result = summarize_numerical(children)
    assert result["family_count"] == 8 and result["passed"] is True
    broken = copy.deepcopy(children)
    broken[("candidate", NUMERICAL_FAMILIES[0])]["final_state_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="final state differs"):
        summarize_numerical(broken)
    broken = copy.deepcopy(children)
    broken[("candidate", "periodic_functional")]["hermitian_violation"] = 1.0e-16
    with pytest.raises(ValueError, match="Hermitian violation"):
        summarize_numerical(broken)


def _consumer_outcomes() -> dict[str, object]:
    plan = build_plan()["consumer_h100_gate"]
    passed = [GUARD_NODE, *REQUIRED_NODES]
    passed.extend(f"tests/test_synthetic.py::test_{index}" for index in range(plan["expected_passed"] - len(passed)))
    return {
        "passed": passed,
        "failed": [],
        "skipped": [{"nodeid": plan["allowed_skip"]}],
        "xfailed": [], "xpassed": [], "collection_errors": [], "deselected": [],
    }


def test_consumer_summary_binds_counts_identities_and_semantic_nodes():
    plan = build_plan()
    identities = {
        "periodic": plan["consumer"]["periodic_identity_sha256"],
        "channel_fixed": plan["consumer"]["channel_fixed_identity_sha256"],
        "channel_tolerance": plan["consumer"]["channel_tolerance_identity_sha256"],
    }
    report = {
        "identities": identities,
        "old_pin": {"rejected": True, "target_unchanged": True},
        "finite_smoke": True,
        "cuda_finite_smokes": {"periodic": True, "channel": True},
    }
    policy = {"enforcement": {"passed": True}}
    result = summarize_consumer(exit_code=0, outcomes=_consumer_outcomes(), identity_report=report, tf32_policy=policy)
    assert result["source_commit"] == plan["consumer"]["qualified_source_commit"]
    assert result["wheel_sha256"] == CONSUMER_WHEEL_SHA256
    broken = _consumer_outcomes()
    broken["passed"].remove(GUARD_NODE)
    broken["passed"].append("tests/test_synthetic.py::replacement")
    with pytest.raises(ValueError, match="semantic gate"):
        summarize_consumer(exit_code=0, outcomes=broken, identity_report=report, tf32_policy=policy)


def test_stage_plan_builder_emits_exact_shell_free_stage_order(tmp_path):
    paths = {name: tmp_path / name for name in (
        "baseline_python", "candidate_python", "consumer_python", "baseline_wheel",
        "candidate_wheel", "consumer_wheel", "provider_support", "consumer_support",
        "provider_reference", "released_record", "tmp_preflight", "output_root",
    )}
    args = argparse.Namespace(**paths, candidate_wheel_sha256=PROVIDER_CANDIDATE_WHEEL_SHA256)
    draft = build_draft(args)
    assert draft["schema"] == DRAFT_SCHEMA
    assert [row["name"] for row in draft["stages"]] == list(STAGES)
    assert all("-c" not in row["command"] for row in draft["stages"])
    assert all(Path(row["command"][0]).is_absolute() for row in draft["stages"])
    wrong = argparse.Namespace(**paths, candidate_wheel_sha256="0" * 64)
    with pytest.raises(ValueError, match="wheel SHA-256"):
        build_draft(wrong)
