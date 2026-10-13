"""Static contracts for the RC4.5.2 public robustness freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc452_public_robustness_contract.json"
MATRIX = ROOT / "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha256(commit: str, relative: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def test_rc452_binds_the_frozen_plan_analysis_and_parent_matrix():
    baseline = _read(RECORD)["baseline"]
    assert baseline["contract_parent_commit"] == (
        "b396a8a0999d86fc4e7555a4f6aacbef51903014"
    )
    assert _sha256(ROOT / baseline["plan_record"]) == baseline[
        "plan_record_sha256"
    ]
    assert _sha256(ROOT / baseline["adjudication_record"]) == baseline[
        "adjudication_record_sha256"
    ]
    assert _git_blob_sha256(
        baseline["contract_parent_commit"],
        "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json",
    ) == baseline["known_issues_record_at_contract_parent_sha256"]


def test_rc452_freezes_all_nine_residuals_into_the_planned_cohorts():
    record = _read(RECORD)
    assert record["schema_version"] == 1
    assert record["phase"] == "RC4.5.2"
    assert record["classification"] == (
        "PASS_RC4_5_2_PUBLIC_CROSS_RUNTIME_ROBUSTNESS_CONTRACT"
    )
    assert record["status"] == "contract_complete_implementation_not_started"
    issues = record["issues"]
    assert set(issues) == {
        "B7", "B9", "B11", "B13", "B15", "B17", "B21", "B24", "B27"
    }
    assert {key for key, value in issues.items() if value["implementation_stage"] == "RC4.5.3"} == {
        "B7", "B24", "B27"
    }
    assert {key for key, value in issues.items() if value["implementation_stage"] == "RC4.5.4"} == {
        "B9", "B21"
    }
    assert {key for key, value in issues.items() if value["implementation_stage"] == "RC4.5.5"} == {
        "B13", "B15", "B17"
    }
    assert issues["B11"]["implementation_stage"] == "RC4.5.6"


def test_checkpoint_schema_and_payload_paths_are_fail_closed():
    contract = _read(RECORD)["checkpoint_contract"]
    schema = contract["schema"]
    assert schema["current_format_unknown_top_level_keys"] == "reject"
    assert schema["current_format_unknown_nested_identity_keys"] == "reject"
    assert schema["current_format_unknown_nested_bridge_keys"] == "reject"
    assert schema["current_format_unknown_tensor_record_keys"] == "reject"
    assert schema["silent_schema_downgrade"] is False
    assert schema["functional_origin_marker_required_for_current_functional_formats"] is True
    assert schema["cryptographic_authentication_claimed"] is False
    assert schema["adversarial_resealing_prevented_without_an_external_trust_root"] is False

    paths = contract["payload_paths"]
    assert paths["stored_form"] == "single_relative_basename"
    for key in (
        "absolute_paths_allowed",
        "parent_components_allowed",
        "path_separators_allowed",
        "empty_or_dot_names_allowed",
        "payload_symlink_allowed",
    ):
        assert paths[key] is False
    assert paths["payload_must_be_regular_file"] is True
    assert paths["resolved_payload_must_remain_within_checkpoint_root"] is True


def test_checkpoint_restore_and_writer_are_transactional_and_umask_aware():
    contract = _read(RECORD)["checkpoint_contract"]
    restore = contract["restore_transaction"]
    assert restore["ordered_preflight"][-2:] == [
        "fresh_restored_state_construction",
        "single_target_commit",
    ]
    assert restore["target_mutation_before_all_preflight_passes"] is False
    assert restore["partial_restored_state_visible_on_failure"] is False
    assert restore["completed_step_or_workspace_mutation_on_failure"] is False
    assert restore["source_checkpoint_mutation"] is False

    writer = contract["writer_transaction"]
    assert writer["target_overwrite_allowed"] is False
    assert writer["staging_location"] == (
        "unique_sibling_under_target_parent_on_same_filesystem"
    )
    assert writer["publish_operation"] == (
        "single_atomic_directory_replace_after_all_files_and_metadata_succeed"
    )
    assert writer["directory_creation_mode"] == "0o777 filtered by caller umask"
    assert writer["regular_file_creation_mode"] == "0o666 filtered by caller umask"
    assert writer["post_creation_chmod_that_widens_permissions_allowed"] is False
    assert writer["minimum_umask_matrix"] == ["002", "022", "077"]
    assert writer["failed_target_exists"] is False
    assert writer["failed_staging_residue_exists"] is False


def test_checkpoint_public_errors_cover_semantics_without_message_parsing():
    errors = _read(RECORD)["checkpoint_contract"]["public_error_taxonomy"]
    assert errors["generic_categories"] == [
        "CheckpointNotFoundError",
        "CheckpointExistsError",
        "CheckpointIntegrityError",
        "CheckpointCompatibilityError",
        "CheckpointIOError",
    ]
    assert errors["functional_compatibility_names_retained"] is True
    assert errors["functional_classes_remain_subclasses_of_FunctionalCheckpointError"] is True
    assert errors["io_error_must_preserve_original_exception_as_cause"] is True
    assert errors["message_text_used_for_classification"] is False
    assert errors["machine_metadata_required"] == [
        "code", "operation", "failure_domain", "legacy_exception_type"
    ]


def test_boundary_contract_encodes_q_sign_equivalence_and_finite_robin_domain():
    boundary = _read(RECORD)["boundary_contract"]
    anchoring = boundary["finite_q_anchoring"]
    assert anchoring["director_equivalence"] == "n_is_equivalent_to_minus_n"
    assert anchoring["parallel_or_antiparallel_to_oriented_face_normal_accepted"] is True
    assert anchoring["canonical_metadata_normal"] == "oriented_outward_face_normal"
    assert anchoring["canonical_identity_equal_for_sign_equivalent_inputs"] is True
    assert anchoring["existing_outward_input_numerics_changed"] is False

    robin = boundary["robin"]
    assert robin["lower_bound_inclusive"] == 0.0
    assert robin["upper_bound_inclusive"] == 1.0e14
    assert robin["above_bound_rejected_before_operator_or_output_materialization"] is True
    assert robin["independent_reference_precision_decimal_digits"] == 80
    assert "normalized_characteristic_residual_at_most_1e-12" in robin[
        "qualified_interval_requirements"
    ]
    assert robin["existing_qualified_inputs_byte_identical"] is True
    assert robin["scientific_model_or_boundary_equation_changed"] is False


def test_public_validation_is_shared_early_and_has_one_capability_truth_table():
    validation = _read(RECORD)["public_validation_contract"]
    devices = validation["device_tokens"]
    assert devices["accepted_requested_tokens"] == ["auto", "cpu", "cuda", "cuda:N"]
    assert devices["cpu_indexed_tokens_allowed"] is False
    assert devices["leading_or_trailing_whitespace_allowed"] is False
    assert devices["validation_before_output_creation"] is True
    assert devices["validation_before_runtime_or_cuda_allocation"] is True
    assert devices["requested_and_resolved_device_both_persisted"] is True

    pressure = validation["pressure_options"]
    assert pressure["max_iterations"] == "positive_non_boolean_integer"
    assert pressure["unknown_keys"] == "reject"
    assert pressure["not_applicable_declaration"] == "reject"

    capabilities = validation["capability_truth_table"]
    assert capabilities["catalogued_supported_combination_must_compile"] is True
    assert capabilities["uncatalogued_combination_must_reject"] is True
    assert capabilities["implicit_capability_fallback_allowed"] is False

    errors = validation["public_compilation_errors"]
    assert errors["public_type"] == "PublicSimulationCompilationError"
    assert errors["original_exception_preserved_as_cause"] is True
    assert errors["machine_rejection_code_required"] is True
    assert errors["raw_internal_exception_escape_allowed"] is False


def test_diagnostics_add_one_envelope_without_changing_existing_payloads():
    diagnostics = _read(RECORD)["diagnostics_contract"]
    assert diagnostics["metadata_location"] == "metadata.json::diagnostics"
    assert diagnostics["envelope_schema_version"] == 1
    assert diagnostics["required_fields"] == [
        "schema_version",
        "application_family",
        "runtime_path",
        "enabled",
        "count",
        "steps",
        "artifact_paths",
        "payload_schema",
        "payload_units",
    ]
    assert diagnostics["payload_schema_entry_fields"] == ["name", "dtype"]
    assert diagnostics["payload_units_covers_every_payload_field_exactly"] is True
    disabled = diagnostics["disabled_contract"]
    assert disabled == {
        "enabled": False,
        "count": 0,
        "steps": [],
        "artifacts_are_null": True,
        "payload_schema_and_units_still_declared": True,
    }
    compatibility = diagnostics["compatibility"]
    assert compatibility["existing_npy_filename_retained"] == "diagnostics.npy"
    assert compatibility["existing_csv_filename_retained"] == "diagnostics.csv"
    assert compatibility["existing_payload_bytes_retained_for_identical_run"] is True
    assert compatibility["legacy_npy_and_csv_metadata_aliases_retained_for_0_2_series"] is True


def test_tf32_scope_restores_all_caller_owned_state_on_every_exit():
    tf32 = _read(RECORD)["tf32_contract"]
    assert tf32["caller_state_fields"] == [
        "torch_float32_matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    ]
    assert tf32["lifecycle"] == [
        "validate_requested_policy_without_mutation",
        "snapshot_all_caller_state",
        "apply_effective_policy",
        "execute_owned_region",
        "restore_exact_snapshot_in_finally",
    ]
    assert tf32["restore_on_success"] is True
    assert tf32["restore_on_exception"] is True
    assert tf32["nested_scopes_restore_lifo"] is True
    assert tf32["construction_only_call_leaves_global_state_unchanged"] is True
    assert tf32["invalid_policy_leaves_global_state_unchanged"] is True
    assert tf32["CPU_path_covered"] is True
    assert tf32["CUDA_path_covered_by_final_h100"] is True
    assert len(tf32["runtime_families_covered"]) == 8
    assert tf32["metadata"]["process_global_ownership_claimed"] is False


def test_implementation_order_consumer_boundary_and_authorization_remain_explicit():
    record = _read(RECORD)
    assert [entry["stage"] for entry in record["implementation_order"]] == [
        "RC4.5.3", "RC4.5.4", "RC4.5.5", "RC4.5.6", "RC4.5.7", "RC4.5.8"
    ]
    assert record["implementation_order"][-1]["requires_separate_user_authorization"] is True
    assert record["implementation_order"][-1]["automatic_retry"] is False
    consumer = record["consumer_impact"]
    assert consumer["PSSolver_Control_source_changes_authorized"] is False
    assert consumer["future_consumer_requalification_required"] is True
    authorization = record["authorization"]
    assert authorization == {
        "rc4_5_2_contract_complete": True,
        "eligible_for_rc4_5_3_implementation": True,
        "rc4_5_implementation_authorized": False,
        "PSSolver_Control_modification_authorized": False,
        "h100_submission_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
    assert all(value is False for value in record["scope"].values())


def test_known_issue_matrix_registers_the_contract_without_authorizing_implementation():
    progress = _read(MATRIX)["progress"]
    assert progress["rc4_5_2_contract_authorized"] is True
    assert progress["rc4_5_2_contract_complete"] is True
    assert progress["rc4_5_2_classification"] == (
        "PASS_RC4_5_2_PUBLIC_CROSS_RUNTIME_ROBUSTNESS_CONTRACT"
    )
    assert progress["rc4_5_2_contract_record"] == RECORD.relative_to(ROOT).as_posix()
    assert progress["rc4_5_2_contract_record_sha256"] == _sha256(RECORD)
    assert progress["eligible_for_rc4_5_3_implementation"] is True
    assert progress["rc4_5_implementation_authorized"] is False
