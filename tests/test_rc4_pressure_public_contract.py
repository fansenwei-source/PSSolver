"""Static RC4.4.2 public Channel pressure contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc442_pressure_contract.json"
RC443 = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc443_pressure_policy_implementation.json"
)
RC444 = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc444_pressure_metadata_identity_connection.json"
)
RC447_RECOVERY = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_python310_compatibility_recovery.json"
)
RC447_PERFORMANCE_REPAIR = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_pressure_performance_repair.json"
)
RC447_PERFORMANCE_RECOVERY_V10 = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_pressure_performance_recovery_v10.json"
)
RC447_BOUNDED_OPERATOR_RECOVERY_V11 = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_pressure_bounded_operator_recovery_v11.json"
)
RC453 = ROOT / "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
RC455 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
)
RC456 = ROOT / "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
RC5_R52 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R53 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
RC5_R56 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC5_R57 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RC5_R58 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
RC5_R59 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
RC5_R511 = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
RC5_R513 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
RC6_R2 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
RC6_R3 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
RC6_R4 = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc442_binds_plan_and_adjudication_without_starting_implementation():
    record = _record()
    baseline = record["baseline"]

    assert record["phase"] == "RC4.4.2"
    assert record["status"] == "contract_complete_implementation_not_started"
    assert record["classification"] == (
        "PASS_RC4_4_2_PUBLIC_CHANNEL_PRESSURE_CONTRACT"
    )
    assert baseline["contract_parent_commit"] == (
        "0ce8828508982d3d156f31fecd2533328acc095c"
    )
    assert _sha256(ROOT / baseline["rc4_4_0_plan"]) == baseline[
        "rc4_4_0_plan_sha256"
    ]
    assert _sha256(ROOT / baseline["rc4_4_1_adjudication"]) == baseline[
        "rc4_4_1_adjudication_sha256"
    ]
    assert baseline["required_decision"] == "ACTIVE_MODE_RMS_REQUIRED"
    assert baseline["required_capability_finding"] == (
        "METADATA_REPAIR_REQUIRED"
    )


def test_rc442_keeps_user_input_shape_and_derives_one_mode():
    contract = _record()["public_configuration_contract"]

    assert set(contract["user_input_keys_unchanged"]) == {
        "algorithm",
        "relative_tolerance",
        "max_iterations",
        "fixed_iterations",
        "warm_start",
    }
    assert contract["new_required_user_input_fields"] == []
    assert contract["mode_is_derived_not_user_supplied"] is True
    assert contract["unknown_derived_input_keys_are_rejected"] is True
    assert contract["constructor_breaking_change_allowed"] is False
    fixed, tolerance = contract["mode_truth_table"]
    assert fixed == {
        "fixed_iterations": "positive_integer",
        "convergence_mode": "fixed_work",
        "authoritative_iteration_limit_source": "fixed_iterations",
        "max_iterations_role": "inactive_compatibility_field",
        "relative_tolerance_role": "diagnostic_only_not_early_exit",
    }
    assert tolerance == {
        "fixed_iterations": None,
        "convergence_mode": "relative_tolerance",
        "authoritative_iteration_limit_source": "max_iterations",
        "max_iterations_role": "authoritative_iteration_cap",
        "relative_tolerance_role": "required_postcondition",
    }


def test_rc442_scale_safe_magnitude_cannot_confuse_underflow_with_zero():
    magnitude = _record()["scale_safe_magnitude_schema"]

    assert magnitude["name"] == "channel_pressure_active_rms_magnitude.v1"
    assert magnitude["fields"] == [
        "is_zero",
        "float64_value",
        "representable_as_positive_float64",
        "log2_value",
        "binary_mantissa",
        "binary_exponent",
    ]
    assert "abs(real)" in magnitude["component_amax_definition"]
    assert "abs(imag)" in magnitude["component_amax_definition"]
    assert magnitude["zero_encoding"] == {
        "is_zero": True,
        "float64_value": 0.0,
        "representable_as_positive_float64": False,
        "log2_value": None,
        "binary_mantissa": None,
        "binary_exponent": None,
    }
    assert magnitude["zero_is_determined_by"].startswith(
        "exact_active_coefficient_zero_test"
    )
    assert magnitude["subnormal_nonzero_must_not_be_mislabeled_zero"] is True
    assert magnitude["NaN_or_infinity_allowed"] is False


def test_rc442_freezes_one_scale_and_breakdown_definition_for_every_path():
    scale = _record()["numerical_scale_contract"]

    assert scale["public_absolute_scale"] == (
        "channel_pressure_active_rms_magnitude.v1"
    )
    assert scale["normalized_residual_denominator"] == (
        "projected_rhs_not_initial_warm_start_residual"
    )
    assert "power of two" in scale["working_vector_scale"]
    assert "mean_active" in scale["operator_breakdown"]
    assert "mean_active" in scale["preconditioned_residual_breakdown"]
    assert scale["absolute_breakdown_threshold_allowed"] is False
    assert set(scale["same_definition_required_for"]) == {
        "production_primal",
        "functional_primal",
        "functional_transpose",
        "standalone_transpose_qualification",
        "small_grid_unrolled_oracle",
    }


def test_rc442_diagnostics_schema_v2_is_complete_and_phase_isolated():
    diagnostics = _record()["diagnostics_schema_v2"]

    assert diagnostics["schema_version"] == 2
    assert diagnostics["public_and_json_safe"] is True
    assert diagnostics["immutable_snapshot"] is True
    assert diagnostics["tensor_values_allowed"] is False
    assert len(diagnostics["required_fields"]) == len(
        set(diagnostics["required_fields"])
    )
    assert set(diagnostics["required_fields"]) == set(
        diagnostics["field_types"]
    )
    assert set(diagnostics["termination_reasons"]) == {
        "not_run",
        "zero_rhs",
        "exact_zero_residual",
        "fixed_work_completed",
        "relative_tolerance_met",
        "iteration_limit_reached_without_postcondition",
        "operator_breakdown",
        "preconditioned_residual_breakdown",
        "nonfinite_input",
        "nonfinite_intermediate",
    }
    aggregate = diagnostics["implicit_adjoint_aggregate"]
    assert aggregate["new_primal_increments_sequence"] is True
    assert aggregate[
        "new_primal_atomically_replaces_transpose_with_not_run_for_the_same_sequence"
    ] is True
    assert aggregate["stale_transpose_from_an_older_primal_allowed"] is False
    assert diagnostics["failure_persistence"].startswith(
        "complete failure diagnostics"
    )


def test_rc442_stopping_and_failure_contracts_are_fail_closed():
    record = _record()
    stopping = record["stopping_and_state_contract"]
    failures = record["failure_contract"]

    assert stopping["fixed_work"]["ordinary_tolerance_early_exit"] is False
    assert stopping["fixed_work"]["requested_work_required"] is True
    assert stopping["fixed_work"]["breakdown"] == "fail_closed"
    assert stopping["relative_tolerance"][
        "tolerance_is_required_postcondition"
    ] is True
    assert stopping["relative_tolerance"]["cap_without_postcondition"] == (
        "fail_closed"
    )
    assert stopping["production_warm_start"][
        "committed_only_after_acceptable_solve"
    ] is True
    assert stopping["production_warm_start"]["unchanged_on_failure"] is True
    assert stopping["functional_pressure_state"]["persistent_pressure_guess"] is False
    assert stopping["partial_target_mutation_on_failure"] is False
    assert failures["production_execution"]["base"] == "RuntimeError"
    assert failures["functional_execution"]["code"] == (
        "functional_convergence_error"
    )
    assert failures["path_specific_exception_families_remain_until_rc4_5"]
    assert failures["message_text_is_not_a_machine_contract"] is True


def test_rc442_capabilities_distinguish_runtime_from_connected_component():
    capability = _record()["capability_metadata_v2"]

    assert capability["declaration_and_runtime_capability_sets_must_be_equal"]
    assert capability["declaration_executable"] is True
    assert capability["runtime_executable"] is True
    assert capability["explicit_vjp"] is False
    assert capability["differentiability"] == "validated_custom_adjoint"
    assert capability["inner_solve_gradient"] == (
        "custom_implicit_pressure_adjoint_v2"
    )
    implicit = capability["implicit_adjoint_component"]
    transpose = capability["standalone_transpose_component"]
    assert implicit["component_is_runtime"] is False
    assert implicit["component_connected_to_executable_runtime"] is True
    assert implicit["custom_autograd_rule"] is True
    assert transpose["component_is_runtime"] is False
    assert transpose["component_connected_to_executable_runtime"] is True
    assert transpose["custom_autograd_rule"] is False
    assert capability["ambiguous_component_field_removed"] == (
        "functional_runtime_executable"
    )


def test_rc442_versions_dynamics_but_not_state_or_envelope():
    identity = _record()["identity_and_restart_contract"]

    assert identity["pressure_convergence_schema"] == {"from": 1, "to": 2}
    assert identity["pressure_transpose_protocol"] == {"from": "1", "to": "2"}
    assert identity["implicit_pressure_adjoint"] == {"from": "1", "to": "2"}
    assert identity["production_forward_dynamics"]["to"].endswith(".v4")
    assert identity["functional_forward_dynamics"]["to"].endswith(".v4")
    assert identity["functional_derivative_dynamics"]["to"].endswith(".v4")
    assert identity["production_state_layout"]["from"] == identity[
        "production_state_layout"
    ]["to"]
    assert identity["functional_state_layout"]["from"] == identity[
        "functional_state_layout"
    ]["to"]
    assert identity["functional_bridge_format"] == {"from": 3, "to": 3}
    assert identity["checkpoint_envelope_format_changes"] is False
    for case in identity["compatibility_matrix"]:
        if "v3" in case["source"]:
            assert case["expected"] == (
                "reject_before_tensor_load_and_target_mutation"
            )
    assert identity["old_checkpoint_rewritten"] is False
    assert identity["silent_state_only_migration"] is False


def test_rc442_freezes_cpu_and_h100_thresholds_before_implementation():
    record = _record()
    cpu = record["provider_cpu_acceptance"]
    h100 = record["single_h100_contract"]

    assert cpu["thresholds"] == {
        "scale_covariance_relative_l2_max": 2e-11,
        "primal_transpose_inner_product_relative_error_max": 2e-13,
        "pcg_vs_dense_relative_l2_max": 5e-12,
        "normwise_backward_error_max": 5e-14,
        "custom_vjp_vs_unrolled_relative_l2_max": 3e-13,
        "central_finite_difference_relative_error_max": 2e-7,
        "minimum_Taylor_remainder_order": 1.95,
        "ordinary_unchanged_path_byte_identity": True,
    }
    assert cpu["full_source_suite_required"] is True
    assert cpu["installed_wheel_suite_required"] is True
    assert h100["separate_authorization_required"] is True
    assert h100["formal_submission_count"] == 1
    assert h100["automatic_retry"] is False
    assert h100["TF32_matmul"] is False
    assert h100["TF32_cudnn"] is False
    assert h100["pressure_scale_exponents"] == [-600, 0, 600]
    assert h100["profiles"]["shapes"] == [[128, 20, 20], [512, 40, 40]]
    assert h100["profiles"]["mean_timestep_ratio_max"] == 1.05
    assert h100["profiles"]["median_timestep_ratio_max"] == 1.05
    assert h100["profiles"]["peak_allocated_ratio_max"] == 1.05
    assert h100["threshold_change_after_formal_observation_forbidden"] is True


def test_rc442_stops_before_provider_consumer_h100_or_release_work():
    record = _record()
    stages = record["implementation_slices"]

    assert [item["stage"] for item in stages] == [
        "RC4.4.2",
        "RC4.4.3",
        "RC4.4.4",
        "RC4.4.5",
        "RC4.4.6",
        "RC4.4.7",
    ]
    assert stages[0]["authorized"] is True and stages[0]["complete"] is True
    assert all(item["authorized"] is False for item in stages[1:])
    assert all(item["complete"] is False for item in stages[1:])
    assert record["scope"] == {
        "runtime_source_modified": False,
        "public_API_modified": False,
        "checkpoint_reader_or_writer_modified": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "H100_executed": False,
    }
    assert record["completion"] == {
        "rc4_4_2_complete": True,
        "eligible_for_rc4_4_3_implementation": True,
        "B25_closed": False,
        "B28_closed": False,
        "B29_closed": False,
        "provider_implementation_complete": False,
        "consumer_migration_complete": False,
        "installed_wheel_qualified": False,
        "H100_qualified": False,
        "rc4_4_complete": False,
    }
    assert record["authorization"] == {
        "rc4_4_2_authorized_by_user": True,
        "rc4_4_2_complete": True,
        "rc4_4_3_implementation_authorized": False,
        "rc4_4_4_implementation_authorized": False,
        "PSSolver_Control_change_authorized": False,
        "H100_submission_authorized": False,
        "push_authorized": False,
        "automatic_merge_authorized": False,
        "default_promotion_authorized": False,
        "release_authorized": False,
    }
    assert record["verification"] == {
        "targeted_pressure_contract_and_matrix_tests": {
            "passed": 122,
            "failed": 0,
            "skipped": 0,
            "xfailed": 0,
        },
        "full_cpu_suite": {
            "passed": 3078,
            "failed": 0,
            "skipped": 0,
            "xfailed": 0,
            "deselected": 7,
            "subtests_passed": 8,
            "deselected_scope": (
                "the seven pre-existing CUDA-only node IDs frozen by the RC4 "
                "CPU contract"
            ),
            "optional_nematics3d_available": True,
        },
        "git_diff_check": "PASS",
    }


def test_rc442_reviewed_sources_are_content_addressed():
    successor_records = []
    for path in (
        RC443,
        RC444,
        RC447_RECOVERY,
        RC447_PERFORMANCE_REPAIR,
        RC447_PERFORMANCE_RECOVERY_V10,
        RC447_BOUNDED_OPERATOR_RECOVERY_V11,
        RC453,
        RC455,
        RC456,
        RC5_R52,
        RC5_R53,
        RC5_R56,
        RC5_R57,
        RC5_R58,
            RC5_R59,
            RC5_R511,
            RC5_R513,
            RC6_R2,
            RC6_R3,
            RC6_R4,
    ):
        record = json.loads(path.read_text(encoding="utf-8"))
        successor_records.append(
            record.get("historical_successor_sources", record.get("successor_sources", {}))
        )
    for relative, expected in _record()["reviewed_source_sha256"].items():
        actual = _sha256(ROOT / relative)
        resolved = expected
        for successors in successor_records:
            registration = successors.get(relative)
            if (
                registration is not None
                and registration["historical_sha256"] == resolved
            ):
                resolved = registration["current_sha256"]
        assert resolved == actual


def test_rc442_does_not_authorize_the_post_0_2_pcg_refactor():
    files = _record()["expected_implementation_files"]

    assert files["all_four_pcg_loops_may_be_consolidated_now"] is False
    assert files["R5_refactor_remains_post_0_2"] is True
    assert files["new_runtime_file_requires_scope_review"] is True
    assert _record()["consumer_contract"][
        "consumer_change_authorized_by_rc4_4_2"
    ] is False
