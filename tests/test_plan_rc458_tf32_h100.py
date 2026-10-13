"""Frozen RC4.5.8 single-H100 qualification-plan tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.plan_rc458_tf32_h100 import (
    CALLER_STATES,
    CUDA_ONLY_NODE_IDS,
    FAMILIES,
    PROVIDER_BASELINE,
    PROVIDER_CANDIDATE,
    build_plan,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc458_h100_closure_plan.json"
RC457 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_recovery.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc458_profile_matrix_is_exact_complete_and_unique():
    plan = build_plan()
    profiles = plan["profile_matrix"]
    assert len(FAMILIES) == 8
    assert len(profiles) == 96
    assert [entry["sequence"] for entry in profiles] == list(range(1, 97))
    assert len({entry["key"] for entry in profiles}) == 96
    assert {entry["provider_commit"] for entry in profiles} == {
        PROVIDER_BASELINE,
        PROVIDER_CANDIDATE,
    }
    for family in FAMILIES:
        selected = [entry for entry in profiles if entry["family"] == family.key]
        assert len(selected) == 12
        assert {entry["grid"] for entry in selected} == set(family.grids)
        assert {entry["trial"] for entry in selected} == {1, 2, 3}
        assert {entry["provider_role"] for entry in selected} == {
            "baseline",
            "candidate",
        }
        assert {entry["forward_transforms_per_step"] for entry in selected} == {
            family.forward_transforms_per_step
        }
        assert {entry["inverse_transforms_per_step"] for entry in selected} == {
            family.inverse_transforms_per_step
        }


def test_rc458_tf32_gate_covers_all_families_and_caller_states():
    gate = build_plan()["tf32_scope_gate"]
    assert gate["runtime_families"] == [family.key for family in FAMILIES]
    assert gate["caller_states"] == list(CALLER_STATES)
    assert {state["key"] for state in CALLER_STATES} == {
        "caller_off",
        "caller_on",
        "caller_mixed",
    }
    assert gate["construction_must_not_mutate_caller_state"] is True
    assert gate["successful_execution_must_restore_exact_caller_state"] is True
    assert gate["exceptional_execution_must_restore_exact_caller_state"] is True
    assert gate["nested_scope_must_restore_lifo"] is True
    assert gate["invalid_policy_must_not_mutate_caller_state"] is True


def test_rc458_plan_runs_every_frozen_cuda_only_test_on_h100():
    plan = build_plan()
    assert len(CUDA_ONLY_NODE_IDS) == 7
    assert plan["cuda_only_tests"] == list(CUDA_ONLY_NODE_IDS)
    assert len(set(CUDA_ONLY_NODE_IDS)) == 7
    assert all("cuda" in node.lower() for node in CUDA_ONLY_NODE_IDS)


def test_rc458_thresholds_and_submission_contract_are_fail_closed():
    plan = build_plan()
    thresholds = plan["thresholds"]
    assert thresholds["mean_timestep_ratio_max"] == 1.03
    assert thresholds["median_timestep_ratio_max"] == 1.03
    assert thresholds["paired_timestep_ratio_max"] == 1.05
    assert thresholds["peak_allocated_ratio_max"] == 1.03
    assert thresholds["peak_reserved_ratio_max"] == 1.03
    assert thresholds["paired_final_state_sha_equal"] is True
    assert thresholds["transform_counts_exact"] is True
    assert plan["submission"] == {
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "second_job_without_new_authorization": False,
        "fail_fast": True,
    }


def test_rc458_plan_cli_writes_strict_atomic_json(tmp_path):
    output = tmp_path / "plan.json"
    assert main(["--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == build_plan()
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_rc458_record_binds_prerequisite_sources_and_no_authorization():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == "READY_RC4_5_8_SINGLE_H100_PLAN"
    assert record["qualification_complete"] is False
    assert record["baseline"]["rc4_5_7_record_sha256"] == _sha256(RC457)
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    assert record["authorization"] == {
        "push_authorized": False,
        "h100_submission_authorized": False,
        "automatic_retry": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
