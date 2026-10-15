"""Static contracts for RC4.4 closure and planning-only RC4.5.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = ROOT / "notes/PSSolver_v0_2_0rc4_rc447_h100_closure.json"
PLAN = ROOT / "notes/PSSolver_v0_2_0rc4_rc450_cross_runtime_robustness_plan.json"
CONTRACT = ROOT / "notes/PSSolver_v0_2_0rc4_rc452_public_robustness_contract.json"
MATRIX = ROOT / "notes/PSSolver_v0_2_0rc3_known_issues_matrix.json"
RC453 = ROOT / "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
RC454 = ROOT / "notes/PSSolver_v0_2_0rc4_rc454_boundary_semantics.json"
RC455 = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc455_public_validation_observability.json"
)
RC456 = ROOT / "notes/PSSolver_v0_2_0rc4_rc456_tf32_execution_scope.json"
POST_RELEASE = ROOT / "notes/PSSolver_v0_2_0rc4_post_release_findings.json"
RC5_W1 = ROOT / "notes/PSSolver_v0_2_0rc5_w1_documentation.json"
RC5_R51 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_1_legacy_functional_checkpoint.json"
RC5_R52 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
RC5_R53 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
RC5_R55 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
RC5_R56 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
RC5_R57 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
RC5_R58 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
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


def _git_blob_sha256(commit: str, relative: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def _resolve_successor_sha256(relative: str, historical: str) -> str:
    resolved = historical
    for record_path in (
        RC453,
        RC454,
        RC455,
        RC456,
        POST_RELEASE,
        RC5_W1,
        RC5_R51,
        RC5_R52,
        RC5_R53,
        RC5_R55,
        RC5_R56,
        RC5_R57,
        RC5_R58,
        RC5_R510,
        RC5_R511,
    ):
        registration = _read(record_path).get(
            "historical_successor_sources", {}
        ).get(relative)
        if registration is None:
            continue
        if registration["historical_sha256"] == resolved:
            resolved = registration["current_sha256"]
    return resolved


def test_rc447_closure_binds_local_records_and_external_manifest():
    closure = _read(CLOSURE)
    assert closure["classification"] == (
        "PASS_RC4_4_7_CHANNEL_PRESSURE_SINGLE_H100_CUMULATIVE_CLOSURE"
    )
    assert closure["status"] == "complete"
    assert closure["external_evidence"]["manifest_entries"] == 1534
    assert closure["external_evidence"]["manifest_sha256"] == (
        "69de4b004f178e3d0427227cf5aa83b301899b4cce1ea8f54169eb7ffdbfbaf4"
    )
    for record in closure["local_records"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_rc447_closure_records_the_single_successful_h100_job_and_boundaries():
    closure = _read(CLOSURE)
    job = closure["h100_job"]
    assert job["job_id"] == 10878767
    assert job["state"] == "COMPLETED"
    assert job["exit_code"] == "0:0"
    assert job["formal_h100_submission_count"] == 1
    assert job["automatic_retry"] is False
    assert job["second_job_submitted"] is False
    authorization = closure["authorization"]
    assert authorization["qualification_complete"] is True
    assert authorization["rc4_4_complete"] is True
    assert authorization["eligible_for_rc4_5_planning"] is True
    assert authorization["rc4_5_implementation_authorized"] is False
    assert authorization["eligible_for_automatic_merge"] is False
    assert authorization["eligible_for_default_promotion"] is False
    assert closure["scope"]["PSSolver_Control_modified_by_closure"] is False
    assert closure["scope"]["nematics3d_modified"] is False


def test_rc447_closure_marks_only_the_three_pressure_items_closed():
    closure = _read(CLOSURE)
    assert [item["id"] for item in closure["known_issues"]] == [
        "B25",
        "B28",
        "B29",
    ]
    assert all(
        item["state_after"] == "closed_verified"
        for item in closure["known_issues"]
    )
    qualification = closure["qualification"]
    assert qualification["cuda_only_tests_passed"] == 7
    assert qualification["correctness_gates_passed"] == 9
    assert qualification["profile_count_passed"] == 12
    assert qualification["runtime_fallback_used"] is False
    assert qualification["compile_fallback_used"] is False


def test_rc447_closure_retains_the_preregistered_performance_contract():
    performance = _read(CLOSURE)["performance"]
    thresholds = performance["thresholds"]
    for grid in ("R128x20x20", "R512x40x40"):
        result = performance[grid]
        assert result["mean_candidate_over_baseline"] <= thresholds[
            "mean_timestep_ratio_max"
        ]
        assert result["median_candidate_over_baseline"] <= thresholds[
            "median_timestep_ratio_max"
        ]
        assert result["maximum_paired_candidate_over_baseline"] <= thresholds[
            "every_paired_timestep_ratio_max"
        ]
        assert result["peak_allocated_candidate_over_baseline"] <= thresholds[
            "peak_allocated_ratio_max"
        ]


def test_rc450_binds_the_rc44_closure_and_preplanning_matrix():
    baseline = _read(PLAN)["baseline"]
    assert baseline["planning_parent_commit"] == (
        "51f47e9f06bd3e019d785305bff47c80f5519b10"
    )
    assert _sha256(ROOT / baseline["rc4_4_closure_record"]) == baseline[
        "rc4_4_closure_record_sha256"
    ]
    assert _git_blob_sha256(
        baseline["planning_parent_commit"],
        baseline["known_issues_record"],
    ) == baseline["known_issues_record_at_planning_parent_sha256"]


def test_rc450_handles_exactly_the_nine_cross_runtime_residuals():
    plan = _read(PLAN)
    assert plan["schema_version"] == 1
    assert plan["phase"] == "RC4.5.0"
    assert plan["classification"] == (
        "READY_RC4_5_1_CROSS_RUNTIME_RESIDUAL_ADJUDICATION_NOT_EXECUTED"
    )
    assert [item["id"] for item in plan["issues"]] == [
        "B7",
        "B9",
        "B11",
        "B13",
        "B15",
        "B17",
        "B21",
        "B24",
        "B27",
    ]
    cohorts = {item["id"]: item["issues"] for item in plan["cohorts"]}
    assert cohorts == {
        "checkpoint_transactionality": ["B7", "B24", "B27"],
        "boundary_semantics": ["B9", "B21"],
        "public_validation_and_observability": ["B13", "B15", "B17"],
        "accelerator_global_state": ["B11"],
    }


def test_rc450_analysis_is_evidence_first_and_fail_closed():
    analysis = _read(PLAN)["rc4_5_1_adjudication"]
    assert analysis["kind"] == "analysis_only"
    assert analysis["source_changes_allowed"] is False
    assert len(analysis["required_probe_ids"]) == 10
    assert analysis["required_outcomes_per_issue"] == [
        "REPRODUCED_REPAIR_REQUIRED",
        "ALREADY_CLOSED_WITH_NEW_EVIDENCE",
        "PUBLIC_BOUND_REQUIRED",
        "ACCEPTED_LIMITATION_WITH_EXPLICIT_CONTRACT",
    ]
    assert any("source inspection alone" in rule for rule in analysis["fail_closed_rules"])
    assert any("CPU" in rule and "CUDA" in rule for rule in analysis["fail_closed_rules"])


def test_rc450_checkpoint_contract_covers_downgrade_paths_atomicity_and_umask():
    contract = _read(PLAN)["checkpoint_contract_to_freeze_in_rc4_5_2"]
    assert contract["integrity_downgrade_requires_explicit_versioned_reader"] is True
    assert contract[
        "stored_payload_paths_must_be_relative_normalized_and_confined_to_checkpoint_root"
    ] is True
    assert contract["all_validation_before_target_mutation"] is True
    assert contract["all_writers_stage_then_atomically_publish"] is True
    assert contract[
        "published_modes_follow_creation_mode_and_caller_umask_not_hard_coded_world_readability"
    ] is True
    assert contract["failed_publication_leaves_no_target_or_staging_residue"] is True


def test_rc450_boundary_contract_requires_equivalence_or_an_explicit_bound():
    contract = _read(PLAN)["boundary_contract_to_freeze_in_rc4_5_2"]
    assert contract["anchoring_Q_normal_sign_equivalence_must_be_adjudicated"] is True
    assert contract["robin_extreme_scale_decision"] == (
        "support_with_stable_oracle_or_publish_and_enforce_a_finite_bound"
    )
    assert contract["independent_high_precision_reference_required"] is True
    assert contract["existing_qualified_range_byte_identity_required"] is True


def test_rc450_validation_retains_payloads_but_adds_one_envelope():
    contract = _read(PLAN)[
        "validation_and_diagnostics_contract_to_freeze_in_rc4_5_2"
    ]
    assert contract["canonical_device_parser_shared_semantics_required"] is True
    assert contract["invalid_configuration_rejects_before_output_creation"] is True
    assert contract["capability_catalog_and_compiler_truth_table_equal"] is True
    assert set(contract["diagnostics_envelope_fields"]) == {
        "schema_version",
        "application_family",
        "runtime_path",
        "enabled",
        "count",
        "steps",
        "artifact_paths",
        "payload_schema",
        "payload_units",
    }
    assert contract["legacy_diagnostic_payload_removal_allowed"] is False
    assert contract["legacy_diagnostic_filename_change_allowed"] is False


def test_rc450_tf32_contract_restores_all_caller_state_on_every_exit():
    contract = _read(PLAN)["tf32_contract_to_freeze_in_rc4_5_2"]
    assert contract["caller_state_fields"] == [
        "torch_float32_matmul_precision",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
    ]
    assert contract["restore_on_success"] is True
    assert contract["restore_on_exception"] is True
    assert contract["nested_calls_restore_stack_order"] is True
    assert contract["CPU_and_CUDA_paths_both_covered"] is True
    assert contract["runtime_default_changed"] is False


def test_rc450_orders_analysis_contract_four_cohorts_and_two_qualifications():
    plan = _read(PLAN)
    assert [stage["stage"] for stage in plan["stages"]] == [
        f"RC4.5.{index}" for index in range(1, 9)
    ]
    assert plan["stages"][0]["kind"] == "analysis_only"
    assert plan["stages"][1]["kind"] == "contract"
    assert [stage["kind"] for stage in plan["stages"][2:6]] == [
        "implementation",
        "implementation",
        "implementation",
        "implementation",
    ]
    assert plan["stages"][-2]["kind"] == "installed_wheel_qualification"
    assert plan["stages"][-1]["kind"] == "single_h100_qualification"


def test_rc450_consumer_and_h100_work_remain_separately_authorized():
    plan = _read(PLAN)
    consumer = plan["consumer_contract"]
    assert consumer["requalification_decision"] == "case_by_case_after_rc4_5_1"
    assert consumer["PSSolver_Control_changes_authorized_by_rc4_5_0"] is False
    h100 = plan["h100_contract"]
    assert h100["authorization_required_separately"] is True
    assert h100["single_formal_submission"] is True
    assert h100["automatic_retry"] is False
    assert h100["installed_wheels_only"] is True
    assert h100["threshold_changes_after_observing_formal_results_forbidden"] is True
    authorization = plan["authorization"]
    assert authorization["planning_complete"] is True
    assert authorization["rc4_5_1_analysis_authorized"] is False
    assert authorization["rc4_5_implementation_authorized"] is False
    assert authorization["h100_submission_authorized"] is False


def test_rc450_reviewed_sources_are_content_addressed():
    for relative, expected in _read(PLAN)["reviewed_source_sha256"].items():
        actual = _sha256(ROOT / relative)
        assert _resolve_successor_sha256(relative, expected) == actual


def test_known_issue_matrix_preserves_rc44_and_tracks_rc45_closure():
    matrix = _read(MATRIX)
    assert matrix["summary"]["closed_verified"] == 28
    assert matrix["summary"]["partial_residual"] == 9
    assert matrix["summary"]["regressed_after_closure"] == 1
    assert matrix["summary"]["next_implementation_batch"] == "rc5_must"
    issues = {item["id"]: item for item in matrix["items"]}
    assert all(issues[item]["state"] == "closed_verified" for item in ("B25", "B28", "B29"))
    assert all(
        issues[item]["state"] == "partial_residual"
        for item in ("B7", "B24")
    )
    assert issues["B27"]["state"] == "closed_verified"
    assert all(
        issues[item]["state"] == "closed_verified"
        for item in ("B9", "B21")
    )
    assert all(
        issues[item]["state"] == "partial_residual"
        for item in ("B13", "B17")
    )
    assert issues["B15"]["state"] == "closed_verified"
    assert issues["B11"]["state"] == "partial_residual"
    assert matrix["post_release"]["classification"] == "FAIL"
    progress = matrix["progress"]
    assert progress["rc4_4_complete"] is True
    assert progress["rc4_4_closure_record_sha256"] == _sha256(CLOSURE)
    assert progress["rc4_5_planning_complete"] is True
    assert progress["rc4_5_plan_record_sha256"] == _sha256(PLAN)
    assert progress["rc4_5_1_analysis_authorized"] is True
    assert progress["rc4_5_1_analysis_complete"] is True
    analysis = ROOT / progress["rc4_5_1_analysis_record"]
    assert progress["rc4_5_1_analysis_record_sha256"] == _sha256(analysis)
    assert progress["eligible_for_rc4_5_2_contract"] is True
    assert progress["rc4_5_2_contract_complete"] is True
    assert progress["rc4_5_2_contract_record_sha256"] == _sha256(CONTRACT)
    assert progress["eligible_for_rc4_5_3_implementation"] is True
    assert progress["rc4_5_4_complete"] is True
    assert progress["eligible_for_rc4_5_5_implementation"] is True
    assert progress["rc4_5_5_implementation_authorized"] is True
    assert progress["rc4_5_5_complete"] is True
    assert progress["eligible_for_rc4_5_6_implementation"] is True
    assert progress["rc4_5_6_implementation_authorized"] is True
    assert progress["rc4_5_6_complete"] is True
    assert progress["eligible_for_rc4_5_7_installed_cpu_closure"] is True
    assert progress["rc4_5_8_h100_complete"] is True
    assert progress["rc4_5_complete"] is True
    assert progress["rc4_complete"] is True
    assert progress["eligible_for_rc4_release_preparation"] is True
    assert progress["rc4_5_implementation_authorized"] is True


def test_rc450_scope_does_not_smuggle_architecture_or_default_changes():
    scope = _read(PLAN)["scope"]
    for key in (
        "numerical_equations_changed",
        "time_integrator_changed",
        "spectral_transform_algorithm_changed",
        "pressure_algorithm_changed",
        "checkpoint_tensor_layout_changed_by_plan",
        "post_0_2_R1_to_R9_refactor_authorized",
        "new_geometry_or_model_added",
        "PSSolver_Control_modified_by_rc4_5_0",
        "nematics3d_modified",
        "production_default_selection_changed",
    ):
        assert scope[key] is False
