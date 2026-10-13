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


def test_rc457_binds_sources_and_blocks_h100_until_recovery():
    record = _read(RECORD)
    assert _sha256(ROOT / record["baseline"]["preceding_stage_record"]) == (
        record["baseline"]["preceding_stage_record_sha256"]
    )
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        assert _sha256(ROOT / relative) == registration["current_sha256"]

    matrix = _read(MATRIX)["progress"]
    assert matrix["rc4_5_7_executed"] is True
    assert matrix["rc4_5_7_complete"] is False
    assert matrix["rc4_5_7_consumer_migration_required"] is True
    assert matrix["eligible_for_rc4_5_8_h100_closure"] is False
    assert matrix["rc4_5_7_record_sha256"] == _sha256(RECORD)
