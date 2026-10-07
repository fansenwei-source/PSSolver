"""Frozen implementation record for RC4.2.3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc423_periodic_channel_checkpoint_identity.json"
)
SUCCESSOR_RECORD = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc424_functional_checkpoint_identity.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc423_record_binds_plan_parent_and_implementation_files():
    record = _record()
    successor = json.loads(SUCCESSOR_RECORD.read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_2_3_PERIODIC_CHANNEL_PRODUCTION_LAYERED_"
        "CHECKPOINT_IDENTITY_AND_LEGACY_UPGRADE"
    )
    assert record["git_identity"]["implementation_parent"] == (
        "f4a5aa344a98eec40ff8e3be2194cd7ba6098ccb"
    )
    assert _sha256(record["git_identity"]["planning_record"]) == (
        record["git_identity"]["planning_record_sha256"]
    )
    for relative, expected in record["implementation_files"].items():
        actual = _sha256(relative)
        if actual == expected:
            continue
        registered = successor["historical_successor_sources"][relative]
        assert registered["historical_sha256"] == expected
        assert registered["current_sha256"] == actual
        assert registered["introduced_by_phase"] == "RC4.2.4"


def test_rc423_record_freezes_periodic_and_channel_v3_contracts():
    record = _record()
    periodic = record["periodic_production"]
    channel = record["channel_complete_stress_production"]

    assert periodic["current_format_version"] == 3
    assert periodic["legacy_format_versions"] == [1, 2]
    assert periodic["legacy_direct_production_restore_allowed"] is False
    assert periodic["functional_bridge_v2_direct_restore_preserved"] is True
    assert periodic["functional_bridge_upgrade_deferred_to"] == "RC4.2.4"
    assert channel["runtime_path"] == "channel_complete_stress"
    assert channel["current_format_version"] == 3
    assert channel["legacy_rc1_format_version"] == 1
    assert channel["legacy_rc2_rc3_format_version"] == 2
    assert channel["legacy_direct_production_restore_allowed"] is False


def test_rc423_record_closes_exactly_m07_through_m12():
    matrix = _record()["cross_version_matrix"]
    assert set(matrix) == {
        "M07_periodic_full_complex_rc1_pre_repair",
        "M08_periodic_hermitian_half_rc1",
        "M09_periodic_rc2_rc3_post_repair",
        "M10_channel_rc1_even_periodic_grid",
        "M11_channel_rc1_odd_periodic_grid",
        "M12_channel_rc2_rc3",
    }
    assert all(value.startswith("PASS_") for value in matrix.values())


def test_rc423_record_freezes_fail_closed_pre_mutation_contract():
    identity = _record()["identity_contract"]
    assert identity["identity_layers"] == [
        "forward_dynamics",
        "state_layout",
        "backend_restart",
    ]
    assert identity["source_opaque_digest_authenticated_before_tensor_load"] is True
    assert identity["current_identity_digest_checked_before_tensor_load"] is True
    assert identity["unknown_format_rejected_before_tensor_load"] is True
    assert identity["target_mutation_before_compatibility_acceptance"] is False
    assert identity["device_token_in_compatibility_digest"] is False
    assert identity["in_place_rewrite"] is False


def test_rc423_scope_and_next_authorization_remain_narrow():
    record = _record()
    scope = record["scope"]
    assert scope["periodic_production_checkpoint_modified"] is True
    assert scope["channel_complete_stress_production_checkpoint_modified"] is True
    assert scope["periodic_functional_integration_reader_modified"] is True
    assert scope["channel_functional_integration_reader_modified"] is True
    for key in (
        "functional_checkpoint_format_changed",
        "functional_derivative_identity_changed",
        "p7_channel_checkpoint_modified",
        "runtime_hot_path_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "production_default_changed",
        "H100_executed",
    ):
        assert scope[key] is False
    authorization = record["authorization"]
    assert authorization["rc4_2_3_complete"] is True
    assert authorization["eligible_for_rc4_2_4_planning"] is True
    for key, value in authorization.items():
        if key not in {"rc4_2_3_complete", "eligible_for_rc4_2_4_planning"}:
            assert value is False
