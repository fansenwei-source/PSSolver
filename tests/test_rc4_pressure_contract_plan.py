"""Static contracts for the planning-only RC4.4 pressure batch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "notes/PSSolver_v0_2_0rc4_rc440_pressure_contract_plan.json"


def _plan() -> dict[str, object]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


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


def test_rc440_binds_the_rc43_closure_and_preplanning_matrix():
    baseline = _plan()["baseline"]
    assert baseline["planning_parent_commit"] == (
        "b469d908c25fe66a45915f5654f50696f5648a10"
    )
    assert _sha256(ROOT / baseline["rc4_3_closure_record"]) == baseline[
        "rc4_3_closure_record_sha256"
    ]
    assert _git_blob_sha256(
        baseline["planning_parent_commit"],
        baseline["known_issues_record"],
    ) == baseline["known_issues_record_at_planning_parent_sha256"]


def test_rc440_handles_only_the_three_pressure_residuals():
    plan = _plan()
    assert plan["schema_version"] == 1
    assert plan["phase"] == "RC4.4.0"
    assert plan["classification"] == (
        "READY_RC4_4_1_PRESSURE_CONTRACT_ADJUDICATION_NOT_EXECUTED"
    )
    assert [item["id"] for item in plan["issues"]] == [
        "B25",
        "B28",
        "B29",
    ]
    scope = plan["scope"]
    assert scope["geometry"] == "rectangular_channel"
    assert scope["all_pcg_loops_consolidated"] is False
    assert scope["post_0_2_R5_refactor_authorized"] is False
    for key in (
        "plane_pressure_modified",
        "periodic_pressure_modified",
        "new_geometry_or_model_added",
        "PSSolver_Control_modified_by_rc4_4_0",
        "nematics3d_modified",
        "production_default_selection_changed",
    ):
        assert scope[key] is False


def test_rc440_inventory_names_all_four_existing_pressure_loops_and_b28_drift():
    inventory = _plan()["review_inventory"]
    loops = inventory["pressure_loop_implementations"]
    assert [item["id"] for item in loops] == ["L1", "L2", "L3", "L4"]
    assert len({item["symbol"] for item in loops}) == 4
    finding = inventory["b28_source_finding"]
    assert finding["declaration_executable"] is True
    assert finding["runtime_executable"] is True
    assert finding[
        "implicit_adjoint_component_metadata_claims_functional_runtime_executable"
    ] is False
    assert finding["standalone_transpose_operator_is_not_itself_a_runtime"] is True


def test_rc440_fixed_and_tolerance_modes_have_unambiguous_roles():
    contract = _plan()["frozen_policy_contract"]
    assert contract["public_constructor_compatibility"]["fields_retained"] == [
        "relative_tolerance",
        "max_iterations",
        "fixed_iterations",
    ]
    fixed = contract["modes"]["fixed_work"]
    assert fixed["authoritative_iteration_limit"] == "fixed_iterations"
    assert fixed["max_iterations_role"] == "inactive_compatibility_field"
    assert fixed["relative_tolerance_role"] == (
        "diagnostic_only_not_an_early_exit"
    )
    assert fixed["ordinary_tolerance_early_exit_allowed"] is False
    tolerance = contract["modes"]["relative_tolerance"]
    assert tolerance["authoritative_iteration_limit"] == "max_iterations"
    assert tolerance["relative_tolerance_role"] == "required_postcondition"
    assert tolerance["iteration_cap_without_postcondition"] == "fail_closed"


def test_rc440_does_not_guess_the_scale_formula_before_analysis():
    adjudication = _plan()["scale_adjudication"]
    assert adjudication["kind"] == "analysis_only"
    assert adjudication["source_changes_allowed"] is False
    assert adjudication["formula_selected_by_rc4_4_0"] is False
    assert adjudication["candidates"] == [
        "overflow_safe_global_l2_relative_residual",
        "active_mode_rms_relative_residual",
        "normwise_backward_error",
        "a_preregistered_hybrid_if_no_single_candidate_passes",
    ]
    assert list(adjudication["required_cases"]) == [
        f"P{index:02d}" for index in range(1, 9)
    ]
    assert any(
        "1e-30" in requirement
        for requirement in adjudication["selection_requirements"]
    )


def test_rc440_diagnostics_are_phase_complete_and_not_stale():
    diagnostics = _plan()["diagnostics_schema_v2"]
    required = set(diagnostics["required_fields"])
    assert {
        "solve_phase",
        "initial_guess_policy",
        "convergence_mode",
        "authoritative_iteration_limit_source",
        "active_pressure_degrees_of_freedom",
        "rhs_norm_definition",
        "normalized_residual_definition",
        "required_postcondition",
        "postcondition_satisfied",
        "acceptable",
    } <= required
    assert diagnostics["atomic_snapshot_per_solve_phase"] is True
    assert diagnostics["no_stale_transpose_result_after_a_new_primal_phase"] is True
    assert diagnostics[
        "production_exposes_warm_start_without_mislabeling_it_zero_start"
    ] is True


def test_rc440_versions_forward_and_derivative_changes_without_layout_churn():
    identity = _plan()["identity_and_restart_contract"]
    assert identity["pressure_convergence_schema_target"] == 2
    assert identity["pressure_transpose_protocol_target"] == "2"
    assert identity["implicit_pressure_adjoint_target"] == "2"
    assert identity["production_forward_dynamics_target"] == (
        "channel.production.forward.v4"
    )
    assert identity["functional_forward_dynamics_target"] == (
        "channel.functional.forward.v4"
    )
    assert identity["functional_derivative_dynamics_target"] == (
        "channel.functional.derivative.v4"
    )
    assert identity["production_state_layout_version_changes"] is False
    assert identity["functional_state_layout_version_changes"] is False
    assert identity["backend_restart_layout_version_changes"] is False
    assert identity["old_exact_restart_into_new_pressure_dynamics"] == (
        "reject_before_tensor_load_or_target_mutation"
    )


def test_rc440_orders_provider_consumer_and_h100_closure():
    plan = _plan()
    assert [stage["stage"] for stage in plan["stages"]] == [
        f"RC4.4.{index}" for index in range(1, 8)
    ]
    assert plan["stages"][0]["kind"] == "analysis_only"
    assert plan["stages"][-2]["kind"] == "consumer_qualification"
    assert plan["stages"][-1]["name"] == (
        "single-H100 cumulative pressure closure"
    )
    consumer = plan["consumer_contract"]
    assert consumer["requalification_required"] is True
    assert consumer["qualified_source_commit"] == (
        "03dbc39975978d9d0b7f84965015da24f9d3c206"
    )
    assert consumer[
        "provider_must_publish_policy_and_identity_record_before_consumer_edits"
    ] is True
    assert consumer["consumer_changes_authorized_by_rc4_4_0"] is False


def test_rc440_h100_contract_requires_separate_single_submission():
    contract = _plan()["h100_contract"]
    assert contract["authorization_required_separately"] is True
    assert contract["single_formal_submission"] is True
    assert contract["automatic_retry"] is False
    assert contract["installed_wheels_only"] is True
    assert contract["fixed_work_and_tolerance_modes_required"] is True
    assert contract["custom_adjoint_FD_and_Taylor_required"] is True
    assert contract["profile_shapes"] == [[128, 20, 20], [512, 40, 40]]
    assert contract["threshold_changes_after_observing_formal_results_forbidden"]


def test_rc440_reviewed_provider_sources_are_content_addressed():
    for relative, expected in _plan()["reviewed_source_sha256"].items():
        assert _sha256(ROOT / relative) == expected, relative


def test_rc440_only_authorizes_planning():
    assert _plan()["authorization"] == {
        "rc4_4_planning_authorized": True,
        "rc4_4_planning_complete": True,
        "rc4_4_1_analysis_authorized": False,
        "rc4_4_implementation_authorized": False,
        "PSSolver_Control_change_authorized": False,
        "h100_submission_authorized": False,
        "nematics3d_change_authorized": False,
        "production_default_changed": False,
    }
    assert all(value is False for value in _plan()["non_claims"].values())
