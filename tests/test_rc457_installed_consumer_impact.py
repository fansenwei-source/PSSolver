"""Frozen RC4.5.7 installed-wheel Consumer-impact adjudication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_impact.json"
)
MATRIX = ROOT / "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json"
RECOVERY = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc457_installed_cpu_consumer_recovery.json"
)
RC458 = ROOT / "notes/PSSolver_v0_2_0rc4_rc458_h100_closure.json"
RELEASE = ROOT / "notes/pssolver_v0_2_0_rc4.json"
POST_RELEASE = ROOT / "notes/PSSolver_v0_2_0rc4_post_release_findings.json"
RC5_W1 = ROOT / "notes/PSSolver_v0_2_0rc5_w1_documentation.json"
RC5_R51 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
RC5_R52 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R53 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
)
RC5_R55 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
RC5_R56 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC5_R57 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RC5_R58 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
RC5_R59 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
RC5_R510 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
RC5_R511 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc457_records_provider_pass_and_exact_consumer_stop():
    record = _read(RECORD)
    assert record["classification"] == (
        "STOPPED_RC4_5_7_CONSUMER_IDENTITY_SCHEMA_MIGRATION_REQUIRED"
    )
    assert record["qualification_complete"] is False
    provider = record["provider_installed_wheel_gate"]
    assert provider["passed"] == 6
    assert provider["failed"] == 0
    assert provider["skipped"] == 0
    assert len(provider["gates"]) == 6
    consumer = record["consumer_installed_wheel_gate"]
    assert consumer["passed"] == 299
    assert consumer["failed"] == 1
    assert consumer["skipped"] == 1
    assert consumer["failed_node_id"] == (
        "tests/test_pssolver_adapter.py::"
        "test_identity_carries_the_whole_provider_identity"
    )
    assert consumer["failure_kind"] == (
        "consumer_exact_identity_schema_expectation_missing_tf32_policy"
    )
    assert record["scope"]["solver_failure"] is False
    assert record["scope"]["scientific_failure"] is False


def test_rc457_freezes_expected_tf32_identity_and_migration_boundary():
    record = _read(RECORD)
    assert record["identity_delta"]["tf32_policy"] == {
        "tf32_requested": "off",
        "tf32_effective": False,
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    migration = record["consumer_migration"]
    assert migration["required"] is True
    assert migration["PSSolver_Control_source_modified_by_RC4_5_7"] is False
    assert migration["adapter_runtime_rejected_provider"] is False
    assert migration["consumer_test_contract_rejected_identity_delta"] is True
    assert migration["adapter_version_bump_required"] is False
    assert migration["functional_api_version_bump_required"] is False


def test_rc457_preserves_the_stop_and_resolves_through_the_recovery():
    record = _read(RECORD)
    assert _sha256(ROOT / record["baseline"]["preceding_stage_record"]) == (
        record["baseline"]["preceding_stage_record_sha256"]
    )
    recovery = _read(RECOVERY)
    recovery_successors = recovery["historical_successor_sources"]
    closure_successors = _read(RC458)["historical_successor_sources"]
    release_successors = _read(RELEASE)["historical_successor_sources"]
    post_release_successors = _read(POST_RELEASE)["historical_successor_sources"]
    rc5_w1_successors = _read(RC5_W1)["historical_successor_sources"]
    rc5_r51_successors = _read(RC5_R51)["historical_successor_sources"]
    rc5_r52_successors = _read(RC5_R52)["historical_successor_sources"]
    rc5_r53_successors = _read(RC5_R53)["historical_successor_sources"]
    rc5_r55_successors = _read(RC5_R55)["historical_successor_sources"]
    rc5_r56_successors = _read(RC5_R56)["historical_successor_sources"]
    rc5_r57_successors = _read(RC5_R57)["historical_successor_sources"]
    rc5_r58_successors = _read(RC5_R58)["historical_successor_sources"]
    rc5_r59_successors = _read(RC5_R59)["historical_successor_sources"]
    rc5_r510_successors = _read(RC5_R510)["historical_successor_sources"]
    rc5_r511_successors = _read(RC5_R511)["historical_successor_sources"]
    for relative, expected in record["source_sha256"].items():
        successor = recovery_successors.get(relative)
        resolved = expected
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = closure_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = post_release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_w1_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r51_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r52_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r53_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r55_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r56_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r57_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r58_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r59_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r510_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r511_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert _sha256(ROOT / relative) == resolved
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        resolved = registration["current_sha256"]
        successor = recovery_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = closure_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = post_release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_w1_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r51_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r52_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r53_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r55_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r56_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r57_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r58_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r59_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r510_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        successor = rc5_r511_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == resolved:
            resolved = successor["current_sha256"]
        assert _sha256(ROOT / relative) == resolved

    matrix = _read(MATRIX)["progress"]
    assert matrix["rc4_5_7_executed"] is True
    assert matrix["rc4_5_7_initial_record_sha256"] == _sha256(RECORD)
    assert matrix["rc4_5_7_complete"] is True
    assert matrix["rc4_5_7_consumer_migration_required"] is False
    assert matrix["eligible_for_rc4_5_8_h100_closure"] is True
    assert matrix["rc4_5_7_record_sha256"] == _sha256(RECOVERY)
    assert matrix["rc4_5_8_h100_complete"] is True
