"""Phase 3.0/3.1 ownership and disconnected-state gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import pssolver
import pssolver.execution as execution
from pssolver.execution.state import (
    CurrentRepresentation,
    IntegratorProgress,
    RepresentationLedger,
    RuntimeState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_state_inventory.json"
)
P35_QUALIFICATION_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p35_h100_qualification.json"
)
P36_LOCAL_QUALIFICATION_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p36_local_qualification.json"
)
P36_PERFORMANCE_QUALIFICATION_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p36_performance_adjudication.json"
)
P36_CLOSURE_CONTINUATION_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p36_closure_continuation.json"
)
P36_RESET_REBIND_RECOVERY_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p36_reset_rebind_recovery.json"
)
P36_RESET_REBIND_RECOVERY_V3_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p36_reset_rebind_recovery_v3.json"
)
P36_RESET_REBIND_RECOVERY_V4_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p36_reset_rebind_recovery_v4.json"
)
P36_RESET_REBIND_RECOVERY_V5_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p36_reset_rebind_recovery_v5.json"
)
P36_RESET_REBIND_RECOVERY_V6_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p36_reset_rebind_recovery_v6.json"
)
P36_FINAL_CLOSURE_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_p36_final_closure.json"
)
RUNTIME_MODULES = (
    PROJECT_ROOT / "pssolver" / "runtime" / "plane_legacy.py",
    PROJECT_ROOT / "pssolver" / "runtime" / "plane_beris_edwards.py",
)
STATE_BACKED_CORE = (
    PROJECT_ROOT / "pssolver" / "integrators" / "state_backed.py"
)


def test_phase3_inventory_freezes_ownership_and_connection_order():
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert inventory["schema_version"] == 1
    assert inventory["phase"] == 3
    assert inventory["status"] == (
        "PASS_PHASE3_CLOSURE_NON_REGRESSION_WITH_AUTHORIZED_COMPOSITE_"
        "RECOVERY_V7"
    )
    assert inventory["checkpoint"] == {
        "format_version": 1,
        "must_remain_readable": True,
        "persistent_payload": [
            "evolved_spatial_Q",
            "evolved_spectral_Q",
            "completed_steps",
            "spectral_refresh_interval",
            "integrator_step_count",
            "integrator_refresh_count",
            "declared_backend_restart_state",
        ],
    }
    assert inventory["connection_policy"] == {
        "legacy_production_touched_before_p3_5": False,
        "p3_4_h100_qualified": True,
        "production_default_changed": False,
        "p3_1_runtime_imports_state": False,
        "p3_2_runtime_imports_workspace": False,
        "p3_3_step_program_connected": False,
        "p3_4_separated_canary_connected": True,
        "p3_5_legacy_production_connected": True,
        "p3_5_h100_qualified": True,
        "p3_6_authorized": True,
        "p3_6_closure_continuation_authorized": True,
        "p3_6_cumulative_continuous_qualified": True,
        "p3_6_cumulative_restart_qualified": True,
        "p3_6_closure_complete": True,
        "p3_6_h100_closure_pending": False,
        "p3_6_h100_performance_qualified": True,
        "p3_6_local_qualified": True,
        "p3_6_negative_gates_pending": False,
        "p3_6_reset_rebind_harness_identity_recovery_pending": False,
        "p3_6_reset_rebind_roundoff_contract_pending": False,
        "p3_6_reset_rebind_relative_l2_serialization_pending": False,
        "p3_6_runtime_applicability_negative_control_pending": False,
        "p3_6_reset_rebind_static_sync_recovery_pending": False,
        "p3_6_reset_rebind_pending": False,
        "phase_3_complete": True,
        "eligible_for_phase_4_planning": True,
        "phase_4_authorized": False,
        "separated_canary_touched_before_p3_4": False,
    }
    assert inventory["timestep_oracle"] == [
        "synchronize_algebraic_if_needed",
        "pre_update_callback",
        "explicit_rhs",
        "spectral_add_dt_rhs",
        "spectral_divide_by_denominator",
        "project_dynamic_spectra",
        "inverse_dynamic_spectra",
        "scheduled_spectral_refresh",
        "commit_progress",
    ]


def test_phase3_inventory_records_p35_h100_qualification():
    qualification = json.loads(
        P35_QUALIFICATION_PATH.read_text(encoding="utf-8")
    )
    assert qualification["classification"] == (
        "PASS_P3_5_PRODUCTION_FACADE_CONNECTION_NON_REGRESSION_WITH_"
        "VALIDATOR_RECOVERY_V2"
    )
    assert qualification["source_identity"] == {
        "parent_commit": "54cffec80ef8aab63988f8cd65dc66600b48452c",
        "qualified_commit": "5bb796fa75a6d30afc37251f86dbd1b1723efa51",
        "branch": "next/pssolver-v0.2.0-architecture",
    }
    assert qualification["eligibility"] == {
        "p3_5_complete": True,
        "p3_6": True,
        "default_promotion": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
        "scientific_failure": False,
    }
    assert qualification["identity_gate"] == {
        "canonical_hashes_self_consistent": True,
        "canonical_hash_equality_required": False,
        "runtime_identity_equal_per_pair": True,
        "resolved_run_spec_allowed_differences": ["$.output_dir"],
        "unauthorized_differences": 0,
        "validation_config_sha256_difference": False,
        "legacy_separated_architecture_is_null": True,
        "connection_identity_smoke_passed": 14,
    }
    assert qualification["evidence"]["complete"] is True
    assert qualification["evidence"]["manifest_entries"] == 727
    assert qualification["evidence"]["manifest_sha256"] == (
        "59f92f35f5f8ad36b0a0dc81954d3bce01049e5c4f4b82b507fe09c4b8d8ddf4"
    )


def test_phase3_inventory_freezes_p36_local_closure_contract():
    qualification = json.loads(
        P36_LOCAL_QUALIFICATION_PATH.read_text(encoding="utf-8")
    )
    assert qualification["status"] == (
        "PASS_P3_6_LOCAL_H100_CLOSURE_PENDING"
    )
    assert qualification["source_identity"] == {
        "phase_2_closure_baseline": (
            "0d5b7186b7a104354db173e04a43e68c15daa515"
        ),
        "phase_3_cumulative_candidate": (
            "227b095c12e494ca74756228a30962692474c516"
        ),
        "candidate_branch": "next/pssolver-v0.2.0-architecture",
        "baseline_is_candidate_ancestor": True,
        "cumulative_changed_file_count": 23,
        "git_diff_check_passed": True,
    }
    assert qualification["eligibility"] == {
        "p3_6_local_complete": True,
        "phase_3_complete": False,
        "h100_closure_required": True,
        "default_promotion": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
    gate = qualification["final_h100_gate"]
    assert gate["baseline_commit"] == (
        "0d5b7186b7a104354db173e04a43e68c15daa515"
    )
    assert gate["candidate_commit"] == (
        "227b095c12e494ca74756228a30962692474c516"
    )
    assert gate["maximum_mean_timestep_ratio"] == 1.02
    assert gate["maximum_median_timestep_ratio"] == 1.02
    assert gate["maximum_peak_allocated_ratio"] == 1.02
    assert gate["maximum_peak_reserved_ratio"] == 1.02
    assert gate["forward_transforms_per_step"] == 7
    assert gate["inverse_transforms_per_step"] == 32
    assert gate["same_runtime_byte_identity_required"] is True
    assert gate["eligible_classification"] == (
        "PASS_PHASE3_CLOSURE_NON_REGRESSION"
    )


def test_phase3_inventory_records_p36_performance_adjudication():
    qualification = json.loads(
        P36_PERFORMANCE_QUALIFICATION_PATH.read_text(encoding="utf-8")
    )
    assert qualification["classification"] == (
        "PASS_SIMULATION_TIMER_NON_REGRESSION"
    )
    assert qualification["eligibility"] == {
        "performance_non_regression_resolved": True,
        "p3_6_closure_continuation": True,
        "performance_optimization_required": False,
        "additional_performance_measurement_required": False,
        "phase_3_complete": False,
        "phase_4_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
    assert qualification["measurement_contract"][
        "primary_metric"
    ] == (
        "metadata.elapsed_seconds / "
        "metadata.workflow.requested_additional_steps"
    )
    assert qualification["measurement_contract"][
        "non_regression_threshold"
    ] == 1.02
    for grid in ("R128", "R320"):
        result = qualification["results"][grid]
        assert result["classification"] == "PASS_NON_REGRESSION"
        assert result["paired_geometric_candidate_over_baseline"] <= 1.02
        assert result["confidence_interval_95"][1] <= 1.02
    assert qualification["execution"]["job_id"] == 10836439
    assert qualification["execution"]["formal_processes_completed"] == 48
    assert qualification["execution"]["formal_processes_expected"] == 48
    assert qualification["validator"] == {
        "synthetic_tests_passed": 33,
        "core_validation_passed": True,
        "derived_value_contract_passed": True,
        "real_fixture_contract_passed": True,
        "command_record_round_trip_passed": True,
    }
    assert qualification["evidence"]["manifest_entries"] == 4228
    assert qualification["evidence"]["manifest_sha256"] == (
        "ce19ad4a1d3dbc8840a3bda59896a7ba9246ed145ef5ae2dae33c0317b95df46"
    )


def test_phase3_inventory_records_p36_closure_continuation():
    qualification = json.loads(
        P36_CLOSURE_CONTINUATION_PATH.read_text(encoding="utf-8")
    )
    assert qualification["classification"] == (
        "FAIL_P3_6_CLOSURE_HARNESS_RECOVERY_PENDING"
    )
    assert qualification["failure"] == {
        "gate": "reset_rebind_gpu_smoke",
        "layer": "reset_rebind_harness_production_metadata_contract",
        "exception": (
            "ValueError: production metadata lacks the Stage K comparison "
            "contract"
        ),
        "qualification_harness_failure": True,
        "solver_failure": False,
        "scientific_failure": False,
        "numerical_failure": False,
        "performance_failure": False,
    }
    eligibility = qualification["eligibility"]
    assert eligibility["performance_non_regression_resolved"] is True
    assert eligibility["cumulative_continuous_trajectory_gate_complete"] is True
    assert eligibility["bidirectional_restart_gate_complete"] is True
    assert eligibility["reset_rebind_gate_complete"] is False
    assert eligibility["negative_gate_complete"] is False
    assert eligibility["phase_3_complete"] is False
    assert qualification["continuous_trajectory_gate"]["commands_completed"] == 8
    assert qualification["continuous_trajectory_gate"][
        "same_runtime_parent_candidate_byte_identical"
    ] is True
    assert qualification["restart_gate"]["passed"] == 4
    assert qualification["restart_gate"][
        "all_q_u_p_byte_identical_to_continuous"
    ] is True
    assert qualification["remaining_gates"] == {
        "reset_rebind_gpu_smoke": "pending_harness_recovery",
        "negative_gates_completed": 0,
        "negative_gates_expected": 11,
        "repeat_continuous_trajectories": False,
        "repeat_restart_trajectories": False,
        "repeat_performance_measurement": False,
    }
    assert qualification["evidence"]["manifest_entries"] == 1298
    assert qualification["evidence"]["manifest_sha256"] == (
        "98207dd3561e182d5da0af7364388bc94a5bf9d5a37ee45ccec761db9993fbab"
    )


def test_phase3_inventory_records_p36_reset_rebind_recovery():
    qualification = json.loads(
        P36_RESET_REBIND_RECOVERY_PATH.read_text(encoding="utf-8")
    )
    assert qualification["classification"] == (
        "FAIL_P3_6_RESET_REBIND_HARNESS_IDENTITY_RECOVERY_PENDING"
    )
    assert qualification["failure"]["reset_call_returned"] is True
    assert qualification["failure"]["qualification_harness_failure"] is True
    assert qualification["failure"]["solver_failure"] is False
    assert qualification["harness_findings"] == {
        "old_objects_retained_by_strong_reference": False,
        "integer_id_or_address_inequality_was_used_after_release": True,
        "allocator_address_reuse_is_legal": True,
        "initial_tensors_shared_across_evolving_reset_and_fresh_roles": True,
        "fresh_control_had_independent_initial_storage": False,
    }
    correction = qualification["required_correction"]
    assert correction["retain_old_state_workspace_tensor_and_storage_objects"] is True
    assert correction["compare_live_objects_with_python_is"] is True
    assert correction["independent_initial_clones_per_role"] is True
    assert correction["runtime_state_replaced"] is True
    assert correction["bounded_workspace_identity_preserved"] is True
    assert correction["workspace_reallocation_required"] is False
    assert correction["progress_preserved_across_rebind"] is True
    assert correction["fresh_control_progress_must_be_aligned"] is True
    assert correction["verify_no_storage_overlap_between_initial_clones"] is True
    assert correction["verify_pristine_source_hash_before_and_after"] is True
    assert correction["repeat_previous_continuous_trajectories"] is False
    assert correction["repeat_previous_restart_trajectories"] is False
    assert correction["repeat_performance_measurement"] is False
    assert qualification["remaining_gates"] == {
        "legacy_production_reset_rebind": "pending_corrected_harness",
        "separated_canary_reset_rebind": "not_started",
        "negative_gates_passed": 0,
        "negative_gates_expected": 11,
    }
    assert qualification["evidence"]["manifest_entries"] == 120
    assert qualification["evidence"]["manifest_sha256"] == (
        "1500e64f8def530a21a9dc9bda328518b6c019977d494089d8553db3386e6901"
    )


def test_phase3_inventory_records_p36_reset_rebind_recovery_v3():
    qualification = json.loads(
        P36_RESET_REBIND_RECOVERY_V3_PATH.read_text(encoding="utf-8")
    )
    assert qualification["classification"] == (
        "FAIL_P3_6_LEGACY_POST_RESET_HARNESS_DIAGNOSTIC_COMPLETE"
    )
    assert qualification["failure"]["runtime_path"] == "legacy_production"
    assert qualification["failure"]["per_predicate_results_persisted_before_raise"] is False
    assert qualification["failure"]["numerical_failure"] == (
        "undetermined_from_persisted_subgate_detail"
    )
    diagnostic = qualification["local_diagnostic"]
    assert diagnostic["legacy_without_static_sync"]["spatial_byte_identical"] is False
    assert diagnostic["legacy_without_static_sync"]["spectral_byte_identical"] is False
    assert diagnostic["legacy_with_production_restore_order"] == {
        "spatial_byte_identical": True,
        "spectral_byte_identical": True,
        "spatial_linf": 0.0,
    }
    assert diagnostic["required_order"] == [
        "restore_or_reset_independent_dynamic_fields",
        "synchronize_for_observation",
        "restore_progress",
        "advance_equal_steps",
        "compare_state_and_progress",
    ]
    correction = qualification["required_harness_correction"]
    assert correction["persist_each_subgate_before_aggregate"] is True
    assert correction["synchronize_before_restore_progress"] is True
    assert correction["repeat_previous_continuous_trajectories"] is False
    assert correction["repeat_previous_restart_trajectories"] is False
    assert correction["repeat_performance_measurement"] is False
    assert qualification["evidence"]["manifest_entries"] == 51
    assert qualification["evidence"]["manifest_sha256"] == (
        "b4f2d1eba64e8945747281097062d250acbfaa73bc6afb9e8fe319682cd833a5"
    )


def test_phase3_inventory_records_p36_reset_rebind_recovery_v4():
    qualification = json.loads(
        P36_RESET_REBIND_RECOVERY_V4_PATH.read_text(encoding="utf-8")
    )
    assert qualification["status"] == (
        "P3_6_RESET_REBIND_ROUNDOFF_CONTRACT_ADJUDICATION_PENDING"
    )
    assert qualification["execution"]["h100_job_submitted"] is False
    assert all(qualification["passed_contracts"].values())
    roundoff = qualification["production_order_roundoff"]
    assert roundoff["physical_linf"] < 1.0e-15
    assert roundoff["spectral_linf"] < 1.0e-15
    assert roundoff["byte_identity_is_part_of_frozen_reset_contract"] is False
    assert qualification["negative_control"]["spectral_linf"] > 1.0e-5
    adjudication = qualification["contract_adjudication"]
    assert adjudication["continuous_trajectory_byte_identity_remains_required"] is True
    assert adjudication["checkpoint_restart_byte_identity_remains_required"] is True
    assert adjudication["immediate_reset_binding_contract_remains_exact"] is True
    assert adjudication["independent_reconstruction_post_reset_uses_float64_tolerance"] is True
    assert adjudication["proposed_epsilon_multiplier"] == 128
    assert adjudication["solver_source_change_required"] is False
    assert qualification["evidence"]["manifest_entries"] == 47
    assert qualification["evidence"]["manifest_sha256"] == (
        "155fbf22053a6cc3a0cda39dabfbd81e77147f014a2520f849c67959732eeaa8"
    )


def test_phase3_inventory_records_p36_reset_rebind_recovery_v5():
    qualification = json.loads(
        P36_RESET_REBIND_RECOVERY_V5_PATH.read_text(encoding="utf-8")
    )
    assert qualification["status"] == (
        "P3_6_RELATIVE_L2_SERIALIZATION_RECOVERY_PENDING"
    )
    assert qualification["failure"]["layer"] == (
        "qualification_harness_serialization"
    )
    assert qualification["failure"]["mathematical_relative_l2"] == (
        "positive_infinity"
    )
    assert qualification["failure"]["harness_failure"] is True
    assert qualification["failure"]["numerical_failure"] is False
    contract = qualification["strict_float64_contract"]
    assert contract["multiplier"] == 128
    assert contract["qualification_authority"] == (
        "elementwise_violation_count"
    )
    assert contract["relative_l2_role"] == "diagnostic_only"
    schema = qualification["required_schema"]
    assert schema["zero_reference_zero_difference"]["relative_l2"] == 0.0
    assert schema["zero_reference_nonzero_difference"]["relative_l2"] is None
    assert schema["zero_reference_nonzero_difference"][
        "relative_l2_is_mathematically_infinite"
    ] is True
    assert schema["allow_nan"] is False
    assert schema["arbitrary_finite_sentinel_for_infinity"] is False
    assert qualification["execution"]["h100_job_submitted"] is False
    assert qualification["evidence"]["manifest_entries"] == 48
    assert qualification["evidence"]["manifest_sha256"] == (
        "b9c5a2a96eb29c38b551e7401d3e7bc571a3ecfa2e66a4c4d7c8b06296cf7e14"
    )


def test_phase3_inventory_records_p36_reset_rebind_recovery_v6():
    qualification = json.loads(
        P36_RESET_REBIND_RECOVERY_V6_PATH.read_text(encoding="utf-8")
    )
    assert qualification["status"] == (
        "P3_6_RUNTIME_APPLICABILITY_NEGATIVE_CONTROL_RECOVERY_PENDING"
    )
    assert qualification["relative_l2_schema"]["recovery_passed"] is True
    assert qualification["relative_l2_schema"][
        "zero_reference_nonzero_difference_relative_l2"
    ] is None
    cpu = qualification["cpu_adjudication"]
    assert cpu["legacy_production"]["formal_trials_passed"] == 3
    assert cpu["legacy_production"][
        "negative_control_discrimination_applicable"
    ] is True
    assert cpu["separated_canary"][
        "negative_control_discrimination_applicable"
    ] is False
    assert cpu["separated_canary"][
        "unsynchronized_control_violating_elements"
    ] == 0
    correction = qualification["contract_correction"]
    assert correction["legacy_unsynchronized_control_must_violate_tolerance"] is True
    assert correction["canary_unsynchronized_control_must_violate_tolerance"] is False
    assert correction["canary_unsynchronized_control_is_diagnostic_only"] is True
    assert correction["canary_formal_sync_call_order_must_be_instrumented"] is True
    assert correction["canary_formal_production_order_must_pass_tolerance"] is True
    assert correction["repeat_legacy_cpu_trials"] is False
    assert correction["solver_source_change_required"] is False
    assert qualification["execution"]["h100_job_submitted"] is False
    assert qualification["evidence"]["manifest_entries"] == 44
    assert qualification["evidence"]["manifest_sha256"] == (
        "4d0beb30deaf505eb0d9dc49d880cc12289c6968853ddfa0a0bd6448d606e9f7"
    )


def test_phase3_inventory_records_p36_final_closure():
    qualification = json.loads(
        P36_FINAL_CLOSURE_PATH.read_text(encoding="utf-8")
    )
    assert qualification["classification"] == (
        "PASS_PHASE3_CLOSURE_NON_REGRESSION_WITH_AUTHORIZED_COMPOSITE_"
        "RECOVERY_V7"
    )
    assert qualification["eligibility"] == {
        "qualification_complete": True,
        "phase_3_complete": True,
        "eligible_for_phase_4_planning": True,
        "phase_4_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
        "eligible_for_default_promotion": False,
    }
    gates = qualification["closure_gates"]
    assert all(gates.values())
    assert qualification["reset_rebind"]["cpu_canary_trials_passed"] == 3
    assert qualification["reset_rebind"]["cpu_canary_trials_expected"] == 3
    assert qualification["reset_rebind"]["h100_runtimes_passed"] == 2
    assert qualification["reset_rebind"]["h100_runtimes_expected"] == 2
    assert qualification["reset_rebind"]["formal_event_trace"] == [
        "reset",
        "synchronize_for_observation",
        "restore_progress",
        "advance",
    ]
    assert qualification["negative_checkpoint_gates"] == {
        "passed": 11,
        "expected": 11,
        "rejected_before_timestep_output_or_complete": True,
        "fallback_used": False,
    }
    assert qualification["execution"]["h100_job_id"] == 10837073
    assert qualification["execution"]["h100_state"] == "COMPLETED"
    assert qualification["execution"]["h100_exit_code"] == "0:0"
    assert qualification["evidence"]["complete_marker_present"] is True
    assert qualification["evidence"]["manifest_entries"] == 302
    assert qualification["evidence"]["manifest_sha256"] == (
        "567bf448cfb9d0cefc70c41aa156c7196f30ed76933e8a52a45044569a80a582"
    )


def test_state_is_not_promoted_and_shared_core_owns_runtime_imports():
    provisional = {
        "CurrentRepresentation",
        "IntegratorProgress",
        "RepresentationLedger",
        "RuntimeState",
    }
    assert provisional.isdisjoint(execution.__all__)
    assert provisional.isdisjoint(pssolver.__all__)
    legacy_source = RUNTIME_MODULES[0].read_text(encoding="utf-8")
    selector_source = RUNTIME_MODULES[1].read_text(encoding="utf-8")
    core_source = STATE_BACKED_CORE.read_text(encoding="utf-8")
    assert "pssolver.execution.state" not in legacy_source
    assert "RuntimeState" not in legacy_source
    assert "from pssolver.execution.state import RuntimeState" in selector_source
    assert "from pssolver.execution.state import (" in core_source
    assert "StateBackedProjectedIntegratorMixin" in legacy_source


def test_representation_ledger_models_updates_and_synchronization():
    ledger = RepresentationLedger()
    assert ledger.current is CurrentRepresentation.SYNCHRONIZED
    assert ledger.to_metadata() == {
        "generation": 0,
        "physical_generation": 0,
        "spectral_generation": 0,
        "current": "synchronized",
    }

    assert ledger.mark_spectral_updated() == 1
    assert ledger.current is CurrentRepresentation.SPECTRAL
    with pytest.raises(RuntimeError, match="physical representation is stale"):
        ledger.require_physical_current()
    ledger.mark_physical_synchronized()
    assert ledger.current is CurrentRepresentation.SYNCHRONIZED

    assert ledger.mark_physical_updated() == 2
    assert ledger.current is CurrentRepresentation.PHYSICAL
    with pytest.raises(RuntimeError, match="spectral representation is stale"):
        ledger.require_spectral_current()
    ledger.mark_spectral_synchronized()
    assert ledger.current is CurrentRepresentation.SYNCHRONIZED


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "generation": 2,
                "physical_generation": 1,
                "spectral_generation": 1,
            },
            "at least one representation",
        ),
        (
            {
                "generation": 1,
                "physical_generation": 2,
                "spectral_generation": 1,
            },
            "cannot lead state",
        ),
    ],
)
def test_representation_ledger_rejects_invalid_generation_states(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RepresentationLedger(**kwargs)


def test_integrator_progress_matches_checkpoint_v1_counter_rules():
    progress = IntegratorProgress(
        dt=0.005,
        completed_steps=7,
        refresh_interval=3,
        refresh_step_count=1,
        refresh_count=2,
    )
    assert progress.time == pytest.approx(0.035)
    assert progress.refresh_due_after_next_step is False
    progress.commit_step(refreshed=False)
    assert progress.completed_steps == 8
    assert progress.refresh_due_after_next_step is True
    with pytest.raises(RuntimeError, match="refresh outcome"):
        progress.commit_step(refreshed=False)
    progress.commit_step(refreshed=True)
    assert progress.to_metadata()["spectral_refresh"] == {
        "interval": 3,
        "step_count": 0,
        "refresh_count": 3,
    }

    disabled = IntegratorProgress(
        dt=0.01,
        completed_steps=4,
        refresh_interval=None,
        refresh_step_count=4,
        refresh_count=0,
    )
    disabled.commit_step(refreshed=False)
    assert disabled.completed_steps == 5
    assert disabled.refresh_step_count == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "dt": 0.01,
            "completed_steps": 4,
            "refresh_interval": 3,
            "refresh_step_count": 0,
            "refresh_count": 1,
        },
        {
            "dt": 0.01,
            "completed_steps": 4,
            "refresh_interval": None,
            "refresh_step_count": 4,
            "refresh_count": 1,
        },
    ],
)
def test_integrator_progress_rejects_inconsistent_counters(kwargs):
    with pytest.raises(ValueError, match="counters are inconsistent"):
        IntegratorProgress(**kwargs)


def test_runtime_state_adopts_storage_without_copy_or_conversion():
    physical = torch.arange(30, dtype=torch.float64).reshape(3, 1, 2, 5)
    spectral = torch.complex(physical, torch.zeros_like(physical))
    warm_start = torch.ones((1, 2, 5), dtype=torch.float64)
    progress = IntegratorProgress(dt=0.01, refresh_interval=20)
    ledger = RepresentationLedger()
    state = RuntimeState(
        component_names=("Qxx", "Qxy", "Qxz"),
        physical=physical,
        spectral=spectral,
        progress=progress,
        representations=ledger,
        persistent_algebraic={"pressure_guess": warm_start},
    )

    assert state.physical is physical
    assert state.spectral is spectral
    assert state.progress is progress
    assert state.representations is ledger
    assert state.physical_component("Qxy").untyped_storage().data_ptr() == (
        physical.untyped_storage().data_ptr()
    )
    assert state.spectral_component("Qxz").untyped_storage().data_ptr() == (
        spectral.untyped_storage().data_ptr()
    )
    assert state.persistent_algebraic["pressure_guess"] is warm_start

    physical[1, 0, 0, 0] = -7.0
    assert state.physical_component("Qxy")[0, 0, 0].item() == -7.0
    metadata = state.to_metadata()
    assert metadata["component_names"] == ["Qxx", "Qxy", "Qxz"]
    assert metadata["persistent_algebraic_names"] == ["pressure_guess"]


def test_runtime_state_validates_layout_device_and_names():
    physical = torch.zeros((2, 1, 4), dtype=torch.float64)
    spectral = torch.zeros((2, 1, 3), dtype=torch.complex128)
    progress = IntegratorProgress(dt=0.01)

    with pytest.raises(ValueError, match="unique"):
        RuntimeState(
            component_names=("q", "q"),
            physical=physical,
            spectral=spectral,
            progress=progress,
        )
    with pytest.raises(ValueError, match="leading dimension"):
        RuntimeState(
            component_names=("q",),
            physical=physical,
            spectral=spectral,
            progress=progress,
        )
    with pytest.raises(ValueError, match="batch dimensions"):
        RuntimeState(
            component_names=("q", "r"),
            physical=physical,
            spectral=torch.zeros((2, 2, 3), dtype=torch.complex128),
            progress=progress,
        )
