"""Frozen local checks for the pre-authorization v0.3 S-09 plan."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.plan_v0_3_s09_h100 import (
    BASELINE_COMMIT,
    BASELINE_WHEEL_SHA256,
    CANDIDATE_COMMIT,
    CANDIDATE_WHEEL_SHA256,
    CLASSIFICATION,
    CONSUMER_BASE,
    CONSUMER_EVIDENCE_COMMIT,
    CONSUMER_RUNTIME_COMMIT,
    CONSUMER_WHEEL_SHA256,
    H100_BASE_ENV,
    H100_CONTROL_ROOT,
    H100_OUTPUT_ROOT,
    PLANNING_PARENT,
    build_plan,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s09_h100_plan.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_plan_record_is_generated_from_the_versioned_builder() -> None:
    assert _record() == build_plan()


def test_plan_binds_released_baseline_and_pending_candidate() -> None:
    plan = _record()
    assert plan["classification"] == CLASSIFICATION
    assert plan["plan_step"] == "S-09"
    assert plan["issues"] == ["A3-18", "NEW-14"]
    provider = plan["provider"]
    assert provider["baseline_commit"] == BASELINE_COMMIT
    assert provider["baseline_wheel_sha256"] == BASELINE_WHEEL_SHA256
    assert provider["candidate_planning_parent"] == PLANNING_PARENT
    assert provider["candidate_implementation_commit"] == CANDIDATE_COMMIT
    assert provider["candidate_wheel_sha256"] == CANDIDATE_WHEEL_SHA256
    assert plan["consumer"]["stable_qualification_base_commit"] == CONSUMER_BASE
    assert plan["consumer"]["migration_commit"] == CONSUMER_RUNTIME_COMMIT
    assert plan["consumer"]["evidence_commit"] == CONSUMER_EVIDENCE_COMMIT
    assert plan["consumer"]["wheel_sha256"] == CONSUMER_WHEEL_SHA256
    assert plan["consumer"]["functional_api_version"] == "1.1"


def test_new14_gap_blocks_h100_until_a_contract_is_selected() -> None:
    plan = _record()
    diagnosis = plan["diagnosis"]
    assert diagnosis == {
        "channel_backward_observed_caller_tf32_on": True,
        "current_runtime_owns_forward_calls_only": True,
        "h100_must_not_be_submitted_before_resolution": True,
        "new_14_is_a_confirmed_scope_gap": True,
        "periodic_backward_observed_caller_tf32_on": True,
    }
    assert plan["selected_answers"] == {
        "S09-Q01": "A",
        "S09-Q02": "A",
        "S09-Q03": "A",
        "S09-Q04": "A",
    }
    questions = {value["id"]: value for value in plan["decision_questions"]}
    assert set(questions) == {"S09-Q01", "S09-Q02", "S09-Q03", "S09-Q04"}
    assert all(value["recommended"] == "A" for value in questions.values())
    assert plan["authorization"]["h100_submission_authorized"] is False
    assert plan["functional_api_impact"] == {
        "api_1_0_current_format_checkpoint_reader_required": True,
        "checkpoint_forward_compatibility_identity_changed": False,
        "outside_execution_scope_backward_uses_caller_tf32_state": True,
        "provider_identity_changed": True,
        "version_after": "1.1",
        "version_before": "1.0",
        "version_changed": True,
    }


def test_plan_freezes_long_horizon_tf32_and_single_job_boundaries() -> None:
    plan = _record()
    assert plan["long_horizon"]["cases"] == [
        {"case": "loop3d", "horizon": 200.0, "storage": "hermitian_half"},
        {"case": "loop3d", "horizon": 200.0, "storage": "full_complex"},
        {"case": "r1", "horizon": 200.0, "storage": "hermitian_half"},
    ]
    assert plan["long_horizon"]["tolerance_factor"] == 64.0
    assert plan["tf32_backward"]["functional_families"] == [
        "periodic_functional",
        "channel_functional",
    ]
    assert len(plan["tf32_backward"]["caller_states"]) == 6
    assert plan["job_stage_order"].index(
        "independent_control_consumer"
    ) < plan["job_stage_order"].index(
        "paired_performance_memory_profiles"
    )
    assert plan["submission"] == {
        "automatic_retry": False,
        "complete_only_after_all_gates_and_checksums": True,
        "fail_fast": True,
        "formal_h100_submission_count": 1,
    }


def test_plan_reuses_the_full_accepted_paired_profile_contract() -> None:
    plan = _record()
    profiles = plan["profile_matrix"]
    assert len(profiles) == 192
    assert [value["sequence"] for value in profiles] == list(range(1, 193))
    assert len({value["key"] for value in profiles}) == 192
    assert {value["provider_role"] for value in profiles} == {
        "baseline",
        "candidate",
    }
    assert {value["device"] for value in profiles} == {"cuda"}
    assert {value["samples_per_profile"] for value in profiles} == {60}
    contract = plan["profile_contract"]
    assert contract["runtime_families"] == [
        "plane_legacy_production",
        "plane_compiled_v2",
        "periodic_complete_stress",
        "channel_legacy",
        "channel_compiled_v2",
        "channel_complete_stress",
        "periodic_functional",
        "channel_functional",
    ]
    assert contract["profile_count"] == 192
    assert contract["outlier_removal_allowed"] is False
    assert contract["trial_replacement_allowed"] is False


def test_plan_does_not_relax_accepted_numerical_or_profile_limits() -> None:
    plan = _record()
    assert plan["thresholds"] == {
        "aggregate_all_sample_mean_ratio_max": 1.03,
        "median_of_profile_means_ratio_max": 1.03,
        "median_of_six_trial_ratios_max": 1.03,
        "baseline_then_candidate_stratum_median_ratio_max": 1.05,
        "candidate_then_baseline_stratum_median_ratio_max": 1.05,
        "order_stratum_imbalance_ratio_max": 1.05,
        "individual_trial_ratio_reference_max": 1.05,
        "compile_fallback": False,
        "finite_required": True,
        "graph_breaks": 0,
        "minimum_trials_at_or_below_individual_reference": 4,
        "paired_initial_sha_equal": True,
        "peak_allocated_group_ratio_max": 1.03,
        "peak_reserved_group_ratio_max": 1.03,
        "runtime_fallback": False,
        "transform_counts_exact": True,
    }
    numerical = plan["numerical"]
    assert numerical["unchanged_runtime_families"] == [
        "plane_legacy_production",
        "plane_compiled_v2",
        "periodic_complete_stress",
        "periodic_functional",
    ]
    assert numerical["changed_runtime_families"] == [
        "channel_legacy",
        "channel_compiled_v2",
        "channel_complete_stress",
        "channel_functional",
    ]
    assert numerical["changed_float64_paths_relative_l2_max"] == 1.0e-12
    assert numerical["changed_float64_paths_linf_max"] == 1.0e-12
    assert numerical["gradient_relative_l2_max"] == 2.0e-5
    assert numerical["finite_difference_relative_error_max"] == 2.0e-5
    assert numerical["threshold_relaxation_allowed"] is False

    cpu = plan["s06_cpu_preflight"]
    assert cpu["grid"] == "channel_R128x20x20"
    assert cpu["shape"] == [128, 20, 20]
    assert cpu["lengths"] == [40.0, 6.0, 6.0]
    assert cpu["paired_trials"] == 6
    assert cpu["warmup_steps"] == 10
    assert cpu["measured_steps_per_trial"] == 60
    assert cpu["fixed_block_size"] == 10
    assert cpu["fixed_block_count"] == 6
    assert cpu["center"] == "median_of_six_fixed_block_means"
    assert cpu["provider_process_isolation"] is True
    assert cpu["alternating_pair_order"] is True
    assert cpu["telemetry_classification"] == "NO_PAIRED_LIMIT_EXCEEDANCE"
    assert cpu["focused_median_ratio"] <= 1.03
    assert cpu["frozen_function_tolerance_median_ratio"] <= 1.03
    assert cpu["maximum_paired_ratio"] <= 1.05
    assert cpu["candidate_optimization_required"] is False


def test_plan_states_the_exact_performance_adjudication_without_overclaiming() -> None:
    rules = _record()["performance_adjudication"]
    assert rules == {
        "aggregate_all_sample_mean_ratio_max": 1.03,
        "all_samples_retained": True,
        "baseline_then_candidate_order_stratum_median_ratio_max": 1.05,
        "between_order_strata_imbalance_ratio_max": 1.05,
        "candidate_then_baseline_order_stratum_median_ratio_max": 1.05,
        "every_individual_trial_must_be_at_or_below_1_05": False,
        "median_of_profile_means_ratio_max": 1.03,
        "median_of_six_trial_ratios_max": 1.03,
        "peak_allocated_group_ratio_max": 1.03,
        "peak_reserved_group_ratio_max": 1.03,
        "per_cell_minimum_trials_at_or_below_1_05": "at least 4 of 6",
    }


def test_plan_freezes_upload_roots_venv_origin_and_every_job_variable() -> None:
    deployment = _record()["deployment"]
    assert deployment["upload_root"] == H100_CONTROL_ROOT
    assert deployment["control_root"] == H100_CONTROL_ROOT
    assert deployment["output_root"] == H100_OUTPUT_ROOT
    assert deployment["output_root_must_initially_not_exist"] is True
    assert deployment["python_environment_source"] == H100_BASE_ENV
    variables = deployment["variables"]
    assert set(variables) == {
        "S09_BASELINE_PYTHON",
        "S09_CANDIDATE_PYTHON",
        "S09_CONSUMER_PYTHON",
        "S09_BASELINE_WHEEL",
        "S09_CANDIDATE_WHEEL",
        "S09_CONSUMER_WHEEL",
        "S09_PROVIDER_SUPPORT",
        "S09_CONSUMER_SUPPORT",
        "S09_PROVIDER_REFERENCE",
        "S09_CONTROL_ROOT",
        "S09_OUTPUT_ROOT",
    }
    assert variables["S09_CONTROL_ROOT"] == H100_CONTROL_ROOT
    assert variables["S09_OUTPUT_ROOT"] == H100_OUTPUT_ROOT
    assert all(value.startswith("/") for value in variables.values())


def test_atomic_writer_refuses_overwrite_and_round_trips(tmp_path) -> None:
    from benchmarks.plan_v0_3_s09_h100 import _atomic_json

    output = tmp_path / "plan.json"
    value = build_plan()
    _atomic_json(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    with pytest.raises(FileExistsError, match="overwrite"):
        _atomic_json(output, value)
