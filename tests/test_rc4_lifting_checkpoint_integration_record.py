"""Frozen RC4.3.5 lifted runtime/checkpoint integration record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pssolver.workflows.plane_checkpoint import (
    PLANE_FORWARD_DYNAMICS_IDENTITY_VERSION,
    PLANE_STATIC_LIFTING_RESTART_SCHEMA_VERSION,
    PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
)
from pssolver.runtime.static_lifting import (
    FRESH_CONSTRUCTION_PROVENANCE_SCHEMA_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
RC5_R57 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc435_runtime_checkpoint_integration.json"
)
RC453 = ROOT / "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
RC455 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
)
RC456 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
)
RC5_R52 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R511 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
RC5_R512 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)
RC5_R513 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
RC6_R1 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
RC6_R2 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
RC6_R3 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
RC6_R4 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc435_versions_fresh_provenance_without_dynamics_bump():
    record = _record()
    versions = record["identity_versions"]

    assert record["classification"] == (
        "PASS_RC4_3_5_LIFTING_RUNTIME_CHECKPOINT_INTEGRATION_CPU_CANDIDATE"
    )
    assert versions["plane_workflow_checkpoint_format_version"] == (
        PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    )
    assert versions["plane_forward_dynamics_identity_version"] == (
        PLANE_FORWARD_DYNAMICS_IDENTITY_VERSION
    )
    assert versions["static_lifting_restart_identity_schema_version"] == (
        PLANE_STATIC_LIFTING_RESTART_SCHEMA_VERSION
    )
    assert versions["fresh_construction_provenance_schema_version"] == (
        FRESH_CONSTRUCTION_PROVENANCE_SCHEMA_VERSION
    )
    assert versions["checkpoint_format_bumped"] is False
    assert versions["forward_dynamics_identity_bumped"] is False
    assert versions["static_lifting_restart_identity_bumped"] is False


def test_rc435_separates_provenance_from_restart_compatibility():
    separation = _record()["metadata_separation"]

    assert separation["runtime_metadata_records_fresh_construction"] is True
    assert separation[
        "checkpoint_run_provenance_records_fresh_construction"
    ] is True
    assert separation["conditioning_applied_is_explicit"] is True
    assert separation["fresh_construction_in_compatibility_digest"] is False
    assert separation["fresh_construction_in_forward_dynamics_identity"] is False
    assert separation[
        "static_lift_plan_and_linear_correction_remain_in_identity"
    ] is True
    assert separation["restored_state_reconditioned"] is False


def test_rc435_freezes_acceptance_and_fail_closed_boundaries():
    behavior = _record()["checkpoint_behavior"]
    matrix = _record()["regression_matrix"]

    assert behavior["current_reader_accepts_format_v2_without_fresh_provenance"]
    assert behavior["unknown_static_lifting_restart_schema"] == (
        "reject_before_tensor_file_read"
    )
    assert behavior["unknown_plane_forward_dynamics_identity_version"] == (
        "reject_before_tensor_file_read"
    )
    assert behavior["changed_lift_or_linear_correction"] == (
        "reject_before_target_mutation"
    )
    assert behavior["historical_checkpoint_rewritten_in_place"] is False
    assert matrix == {
        "same_version_continuous_vs_split_restart": "PASS_BYTE_IDENTICAL",
        "different_fresh_policy_after_restore": "PASS_BYTE_IDENTICAL",
        "historical_conditioning_metadata_file_round_trip": (
            "PASS_ACCEPT_EXACT_RESTART"
        ),
        "missing_fresh_provenance_on_historical_v2": (
            "PASS_ACCEPT_EXACT_RESTART"
        ),
        "unknown_lifting_schema": "PASS_REJECT_PRE_TENSOR_LOAD",
        "different_lift_plan": "PASS_REJECT_PRE_TARGET_MUTATION",
    }


def test_rc435_record_binds_implementation_and_scope():
    record = _record()
    successor_records = [
        json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ]
        for path in (
            RC453,
            RC455,
            RC456,
            RC5_R52,
            RC5_R57,
            RC5_R511,
            RC5_R512,
            RC5_R513,
            RC6_R1,
            RC6_R2,
            RC6_R3,
            RC6_R4,
        )
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

    scope = record["scope"]
    assert scope["runtime_metadata_modified"] is True
    assert scope["plane_checkpoint_reader_modified"] is True
    assert scope["plane_checkpoint_writer_modified"] is True
    for key in (
        "supported_forward_numerical_dynamics_changed",
        "hot_path_kernel_modified",
        "public_lifting_restriction_changed",
        "checkpoint_tensor_layout_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "production_default_changed",
    ):
        assert scope[key] is False
    assert record["authorization"]["h100_submission_authorized"] is False
    assert record["authorization"]["push_authorized"] is False
    assert record["eligibility"][
        "eligible_for_rc4_3_6_installed_wheel_cpu_closure"
    ] is True
    assert record["eligibility"]["localized_policy_publicly_qualified"] is False
    assert record["verification"]["focused_tests"] == {
        "passed": 103,
        "failed": 0,
    }
    assert record["verification"]["full_cpu_suite"] == {
        "executed": True,
        "passed": 3004,
        "skipped": 0,
        "deselected": 7,
        "subtests_passed": 8,
        "failed": 0,
    }
