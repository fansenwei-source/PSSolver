"""Static closure record for R5-8 checkpoint integrity contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
R5_7_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
R5_9_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
R5_10_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
R5_11_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
R5_12_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)
R5_13_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
G1_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r58_record_binds_implementation_and_exact_sources():
    record = _record()
    successors = json.loads(R5_9_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_10_successors = json.loads(R5_10_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_11_successors = json.loads(R5_11_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_12_successors = json.loads(R5_12_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    r5_13_successors = json.loads(R5_13_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    g1_successors = json.loads(G1_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ]
    assert record["git_identity"] == {
        "branch": "fix/v0.2.0rc5",
        "implementation_commit": "52e57f1f4f406efcd92bb9245c13e9c635ba721c",
        "qualified_source_commit": "52e57f1f4f406efcd92bb9245c13e9c635ba721c",
        "record_commit_is_qualified_source": False,
        "pushed": False,
    }
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            registration = successors.get(relative)
            if (
                registration is not None
                and registration["historical_sha256"] == expected
            ):
                expected = registration["current_sha256"]
            registration = r5_10_successors.get(relative)
            if (
                registration is not None
                and registration["historical_sha256"] == expected
            ):
                expected = registration["current_sha256"]
            registration = r5_11_successors.get(relative)
            if (
                registration is not None
                and registration["historical_sha256"] == expected
            ):
                expected = registration["current_sha256"]
            registration = r5_12_successors.get(relative)
            if (
                registration is not None
                and registration["historical_sha256"] == expected
            ):
                expected = registration["current_sha256"]
            registration = r5_13_successors.get(relative)
            if (
                registration is not None
                and registration["historical_sha256"] == expected
            ):
                expected = registration["current_sha256"]
            registration = g1_successors.get(relative)
            if (
                registration is not None
                and registration["historical_sha256"] == expected
            ):
                expected = registration["current_sha256"]
            assert _sha256(ROOT / relative) == expected
    for relative, registration in record["historical_successor_sources"].items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        expected = registration["current_sha256"]
        successor = successors.get(relative)
        if (
            successor is not None
            and successor["historical_sha256"] == expected
        ):
            expected = successor["current_sha256"]
        successor = r5_10_successors.get(relative)
        if (
            successor is not None
            and successor["historical_sha256"] == expected
        ):
            expected = successor["current_sha256"]
        successor = r5_11_successors.get(relative)
        if (
            successor is not None
            and successor["historical_sha256"] == expected
        ):
            expected = successor["current_sha256"]
        successor = r5_12_successors.get(relative)
        if (
            successor is not None
            and successor["historical_sha256"] == expected
        ):
            expected = successor["current_sha256"]
        successor = r5_13_successors.get(relative)
        if (
            successor is not None
            and successor["historical_sha256"] == expected
        ):
            expected = successor["current_sha256"]
        successor = g1_successors.get(relative)
        if (
            successor is not None
            and successor["historical_sha256"] == expected
        ):
            expected = successor["current_sha256"]
        assert _sha256(ROOT / relative) == expected
        assert registration["reason"]


def test_r58_contract_is_fail_closed_and_pre_mutation():
    contract = _record()["contract"]
    for key in (
        "channel_current_tensor_file_keys_exact",
        "channel_current_backend_file_keys_exact",
        "unknown_manifest_key_is_integrity_error",
        "periodic_refresh_interval_bound_to_target_runtime",
        "channel_refresh_interval_bound_to_target_runtime",
        "refresh_policy_mismatch_is_compatibility_error",
        "refresh_policy_mismatch_rejected_before_target_mutation",
        "periodic_rc1_without_functional_bridge_rejected_by_functional_import",
        "periodic_rc1_functional_explicit_state_only_migration_preserved",
        "periodic_rc2_rc3_functional_direct_read_preserved",
        "metadata_digest_missing_invalid_or_mismatched_is_integrity_error",
        "functional_provenance_downgrade_is_integrity_error",
        "generic_value_error_backward_compatibility_preserved",
    ):
        assert contract[key] is True


def test_r58_does_not_rewrite_r57_or_claim_out_of_scope_work():
    record = _record()
    assert _sha256(R5_7_RECORD) == (
        "d216a566fb1bc43ae70da88ac0134ab60597c86960423d283a1fb060299251ed"
    )
    scope = record["scope"]
    assert scope["checkpoint_acceptance_semantics_changed"] is True
    for key in (
        "scientific_equations_changed",
        "numerical_algorithms_changed",
        "checkpoint_file_schema_changed",
        "runtime_scientific_identity_changed",
        "production_default_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "H100_required",
        "H100_executed",
        "push_executed",
    ):
        assert scope[key] is False


def test_r58_final_status_requires_completed_cpu_closure():
    record = _record()
    completed = record["status"] == "implementation_complete_cpu_verified"
    if completed:
        assert record["classification"] == (
            "PASS_R5_8_CHECKPOINT_INTEGRITY_CPU_CLOSURE"
        )
        assert record["installed_wheel_evidence"] is not None
        assert record["authorization"]["r5_8_cpu_closure_complete"] is True
        assert record["authorization"]["eligible_for_next_work_item"] is True
    else:
        assert record["status"] == "implementation_complete_validation_pending"
        assert record["classification"].startswith("PENDING_")
        assert record["authorization"]["r5_8_cpu_closure_complete"] is False
