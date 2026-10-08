"""Frozen planning contract for RC4.2.6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "notes" / "PSSolver_v0_2_0rc4_rc426_cross_device_checkpoint_plan.json"
RECOVERY = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc426_functional_checkpoint_recovery.json"
)


def _plan() -> dict[str, object]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc426_plan_binds_rc425_and_reviewed_sources():
    plan = _plan()
    recovery = json.loads(RECOVERY.read_text(encoding="utf-8"))
    assert plan["status"] == "ready_for_implementation"
    assert plan["classification"] == (
        "READY_RC4_2_6_SINGLE_H100_CROSS_DEVICE_CHECKPOINT_PORTABILITY"
    )
    source = plan["source_identity"]
    assert source["planning_parent"] == "6f9bc92539a016fee8bcc39acf2cf717ff5e14d8"
    for key in ("rc4_2_0_plan", "rc4_2_5_record"):
        assert _sha256(source[key]) == source[f"{key}_sha256"]
    assert source["rc4_2_5_manifest_entries"] == 26
    for relative, expected in plan["reviewed_source_sha256"].items():
        actual = _sha256(relative)
        if actual == expected:
            continue
        registration = recovery["historical_successor_sources"][relative]
        assert registration["historical_sha256"] == expected
        assert registration["current_sha256"] == actual
        assert registration["introduced_by_phase"] == "RC4.2.6 recovery"


def test_rc426_matrix_is_complete_ordered_and_bidirectional():
    matrix = _plan()["matrix"]
    assert [cell["id"] for cell in matrix] == [f"X{index:02d}" for index in range(1, 15)]
    grouped: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for cell in matrix:
        key = (cell["family"], cell["runtime_path"])
        grouped.setdefault(key, set()).add((cell["source"], cell["target"]))
    assert grouped == {
        ("plane_production", "legacy_production"): {("cpu", "cuda"), ("cuda", "cpu")},
        ("plane_production", "compiled_v2"): {("cpu", "cuda"), ("cuda", "cpu")},
        ("plane_production", "separated_canary"): {("cpu", "cuda"), ("cuda", "cpu")},
        ("periodic_production", "periodic_spectral"): {("cpu", "cuda"), ("cuda", "cpu")},
        ("channel_production", "channel_complete_stress"): {("cpu", "cuda"), ("cuda", "cpu")},
        ("periodic_functional", "periodic_activity_batch_one"): {("cpu", "cuda"), ("cuda", "cpu")},
        ("channel_functional", "channel_activity_batch_one"): {("cpu", "cuda"), ("cuda", "cpu")},
    }


def test_rc426_claim_is_exact_restore_not_cross_device_trajectory_identity():
    plan = _plan()
    boundary = plan["claim_boundary"]
    assert boundary["device_token_is_run_provenance_not_compatibility_identity"] is True
    for key in (
        "cross_runtime_restart_claimed",
        "cross_dtype_restart_claimed",
        "cross_shape_restart_claimed",
        "cross_release_legacy_portability_claimed",
        "cross_device_post_restore_byte_identical_trajectory_claimed",
        "performance_or_memory_claimed",
        "scientific_long_run_claimed",
    ):
        assert boundary[key] is False
    comparison = plan["comparison_semantics"]
    assert comparison["serialized_state_comparison"] == (
        "torch.equal_after_detach_contiguous_CPU_transfer"
    )
    assert comparison["post_restore_step"] == "finite_and_successful_only"
    assert comparison["post_restore_cross_device_numerical_equality_gate"] is False


def test_rc426_per_cell_gate_is_fail_closed_and_pre_mutation():
    gates = _plan()["per_cell_required_gates"]
    assert len(gates) == 13
    assert "checkpoint metadata and every tensor checksum validate before restore" in gates
    assert "checkpoint source tree hash is unchanged by restore" in gates
    assert "every restored serialized tensor is torch.equal after transfer to CPU" in gates
    assert "persistent backend state is exact after transfer to CPU" in gates
    assert "no target mutation occurs before complete compatibility and integrity acceptance" in gates


def test_rc426_h100_job_is_single_bounded_and_non_performance():
    plan = _plan()
    assert len(plan["cuda_only_node_ids"]) == 7
    h100 = plan["h100_environment"]
    assert h100["gpu_name_contains"] == "H100"
    assert h100["visible_cuda_device_count"] == 1
    assert h100["allocated_device"] == "cuda:0"
    assert h100["tf32_matmul"] is False
    assert h100["tf32_cudnn"] is False
    slurm = plan["slurm"]
    assert slurm["partition"] == "hagan-gpu"
    assert slurm["account"] == "hagan-lab"
    assert slurm["qos"] == "medium"
    assert slurm["formal_h100_submission_limit"] == 1
    assert slurm["automatic_retry"] is False
    assert plan["claim_boundary"]["performance_or_memory_claimed"] is False


def test_rc426_planning_scope_does_not_authorize_execution_or_release():
    plan = _plan()
    assert plan["successful_classification"] == (
        "PASS_RC4_2_6_SINGLE_H100_CROSS_DEVICE_CHECKPOINT_PORTABILITY"
    )
    success = plan["success_state"]
    assert success["rc4_2_6_complete"] is True
    assert success["rc4_2_complete"] is True
    assert success["eligible_for_rc4_3_planning"] is True
    assert success["eligible_for_automatic_merge"] is False
    assert success["eligible_for_default_promotion"] is False
    scope = plan["scope"]
    assert scope["planning_only"] is True
    assert all(value is False for key, value in scope.items() if key != "planning_only")
    authorization = plan["authorization"]
    assert authorization["rc4_2_6_planning_authorized"] is True
    assert authorization["rc4_2_6_planning_complete"] is True
    assert all(
        value is False
        for key, value in authorization.items()
        if key not in {"rc4_2_6_planning_authorized", "rc4_2_6_planning_complete"}
    )
