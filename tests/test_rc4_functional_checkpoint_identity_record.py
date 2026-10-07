"""Frozen implementation record for RC4.2.4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc424_functional_checkpoint_identity.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc424_record_binds_parent_plans_and_implementation_files():
    record = _record()
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_2_4_FUNCTIONAL_DERIVATIVE_IDENTITY_"
        "AND_EXPLICIT_STATE_MIGRATION"
    )
    assert record["git_identity"]["implementation_parent"] == (
        "18f255fef74721c3d6c62725d32af096e4bdb2c7"
    )
    for key in ("planning_record", "predecessor_record"):
        relative = record["git_identity"][key]
        assert _sha256(relative) == record["git_identity"][f"{key}_sha256"]
    for section in ("implementation_files", "adapted_regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == expected


def test_rc424_record_freezes_four_layer_functional_identity():
    identity = _record()["functional_identity_contract"]
    assert identity["identity_layers"] == [
        "forward_dynamics",
        "derivative_dynamics",
        "state_layout",
        "backend_restart",
    ]
    assert identity["run_provenance_in_compatibility_digest"] is False
    assert identity["device_token_in_compatibility_digest"] is False
    assert identity["snapshot_identity_in_compatibility_digest"] is False
    assert identity["unknown_identity_rejected_before_tensor_load"] is True
    assert identity["migration_provenance_validated_before_tensor_load"] is True


def test_rc424_record_closes_m13_and_m14_without_false_exact_restart():
    record = _record()
    periodic = record["periodic_functional"]
    channel = record["channel_functional"]
    assert periodic["current_bridge_format_version"] == 3
    assert periodic["rc1_direct_restore_allowed"] is False
    assert periodic["qualified_post_repair_v2_direct_restore_allowed"] is True
    assert periodic["rc1_trajectory_equivalent"] is False
    assert periodic["current_v3_production_exact_restore_allowed"] is False
    assert channel["current_bridge_format_version"] == 3
    assert channel["legacy_direct_restore_allowed"] is False
    assert channel["rc1_rc2_even_periodic_axis_trajectory_equivalent"] is False
    matrix = record["cross_version_matrix"]
    assert set(matrix) == {
        "M13_channel_functional_even_periodic_grid_pre_rc3",
        "M14_periodic_functional_provisional_pre_Hermitian_repair",
        "M15_unknown_or_unregistered_schema",
        "M16_different_runtime_path",
    }
    assert all(value.startswith("PASS_") for value in matrix.values())


def test_rc424_registers_exact_rc423_historical_successors():
    record = _record()
    assert _sha256(record["git_identity"]["predecessor_record"]) == (
        "211330513cb1f4aed0d563f2e0b1d32cfdeca8701d199e29f64b2d2f28d31363"
    )
    successors = record["historical_successor_sources"]
    assert set(successors) == {
        "pssolver/workflows/periodic_checkpoint.py",
        "pssolver/functional/periodic_checkpoint.py",
        "pssolver/functional/channel_checkpoint.py",
    }
    for relative, registration in successors.items():
        assert registration["introduced_by_phase"] == "RC4.2.4"
        assert _sha256(relative) == registration["current_sha256"]
        assert registration["historical_sha256"] != registration["current_sha256"]


def test_rc424_scope_requires_later_consumer_requalification():
    record = _record()
    scope = record["scope"]
    assert scope["functional_checkpoint_format_changed"] is True
    assert scope["functional_derivative_identity_changed"] is True
    assert scope["public_legacy_migration_entry_added"] is True
    assert scope["PSSolver_Control_requalification_required"] is True
    for key in (
        "stable_functional_API_version_changed",
        "production_checkpoint_exact_acceptance_broadened",
        "cross_runtime_restart_expanded",
        "runtime_hot_path_changed",
        "PSSolver_Control_modified",
        "PSSolver_Control_requalification_executed",
        "nematics3d_modified",
        "production_default_changed",
        "H100_executed",
    ):
        assert scope[key] is False
    authorization = record["authorization"]
    assert authorization["rc4_2_4_complete"] is True
    assert authorization["eligible_for_rc4_2_5_planning"] is True
    for key, value in authorization.items():
        if key not in {"rc4_2_4_complete", "eligible_for_rc4_2_5_planning"}:
            assert value is False
