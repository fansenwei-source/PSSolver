"""Static record checks for RC4.4.4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc444_pressure_metadata_identity_connection.json"
)
RC453 = ROOT / "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
RC455 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc444_record_binds_frozen_contract_predecessor_and_current_sources():
    record = _record()
    assert record["phase"] == "RC4.4.4"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC4_4_4_PROVIDER_PRESSURE_METADATA_AND_IDENTITY_CONNECTION"
    )
    assert record["implementation_parent"] == (
        "cf7431c6336bb700183d6b9862d1e32cd18f171a"
    )
    assert _sha256(record["frozen_contract"]["path"]) == record[
        "frozen_contract"
    ]["sha256"]
    assert _sha256(record["predecessor"]["path"]) == record["predecessor"][
        "sha256"
    ]
    successor_records = [
        json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ]
        for path in (RC453, RC455)
    ]
    for relative, expected in record["source_sha256"].items():
        actual = _sha256(relative)
        resolved = expected
        for successors in successor_records:
            registration = successors.get(relative)
            if (
                registration is not None
                and registration["historical_sha256"] == resolved
            ):
                resolved = registration["current_sha256"]
        assert resolved == actual


def test_rc444_connects_policy_diagnostics_and_truthful_capabilities():
    connection = _record()["connection"]
    assert connection["tensor_free_declaration_policy_metadata"] is True
    assert connection["public_compiler_normalization_policy_metadata"] is True
    assert connection["public_user_input_keys_unchanged"] is True
    assert connection["unknown_derived_input_keys_rejected"] is True
    assert connection["production_runtime_public_diagnostics_schema"] == 2
    assert connection["functional_runtime_public_diagnostics_schema"] == 2
    assert connection["declaration_runtime_capabilities_equal"] is True
    assert connection["explicit_vjp"] is False
    assert connection["inner_solve_gradient"] == (
        "custom_implicit_pressure_adjoint_v2"
    )
    assert connection["ambiguous_component_field_removed"] == (
        "functional_runtime_executable"
    )


def test_rc444_versions_only_forward_derivative_semantics():
    identity = _record()["identity"]
    assert identity["production_forward_dynamics"] == (
        "channel.production.forward.v4"
    )
    assert identity["functional_forward_dynamics"] == (
        "channel.functional.forward.v4"
    )
    assert identity["functional_derivative_dynamics"] == (
        "channel.functional.derivative.v4"
    )
    assert identity["production_state_layout"] == "channel.production.state.v3"
    assert identity["production_backend_restart"] == (
        "channel.production.backend.v1"
    )
    assert identity["functional_state_layout"] == "channel.functional.state.v2"
    assert identity["functional_backend_restart"] == (
        "channel.functional.backend.v2"
    )
    assert identity["functional_bridge_format"] == 3
    assert identity["checkpoint_envelope_changed"] is False


def test_rc444_restart_is_fail_closed_before_tensor_or_target_mutation():
    restart = _record()["restart"]
    for key in (
        "production_v3_to_v4_rejected_before_tensor_load",
        "functional_forward_v3_to_v4_rejected_before_tensor_load",
        "functional_derivative_v3_to_v4_rejected_before_tensor_load",
        "production_v3_to_functional_v4_rejected_before_tensor_load",
    ):
        assert restart[key] is True
    assert restart["target_mutation_before_identity_acceptance"] is False
    assert restart["old_checkpoint_rewritten"] is False
    assert restart["silent_state_migration_added"] is False


def test_rc444_scope_stops_before_installed_wheel_consumer_and_h100():
    record = _record()
    assert all(value is False for value in record["nonclaims"].values())
    assert record["authorization"] == {
        "rc4_4_4_complete": True,
        "rc4_4_5_planning_eligible": True,
        "rc4_4_5_execution_authorized": False,
        "PSSolver_Control_change_authorized": False,
        "h100_submission_authorized": False,
        "push_authorized": False,
        "production_default_changed": False,
    }
