"""Frozen RC4.4.7 cumulative Channel-pressure H100 plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "notes/PSSolver_v0_2_0rc4_rc447_h100_qualification_plan.json"
RECOVERY = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_python310_h100_recovery_v5.json"
)
FIELDS_RECOVERY = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_fields_iteration_harness_recovery_v6.json"
)
IMPORT_RECOVERY = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_support_import_bootstrap_recovery_v7.json"
)
PROFILE_RECOVERY = (
    ROOT
    / "notes"
    / "PSSolver_v0_2_0rc4_rc447_profile_diagnostics_recovery_v8.json"
)


def _record() -> dict[str, object]:
    return json.loads(PLAN.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc447_plan_binds_exact_provider_consumer_sources_and_records():
    record = _record()
    assert record["phase"] == "RC4.4.7"
    assert record["status"] == (
        "LOCAL_QUALIFICATION_SUPPORT_READY_H100_NOT_AUTHORIZED"
    )
    identity = record["source_identity"]
    assert identity["provider_baseline_commit"] == (
        "5071d73a00e0d19be62ebd39edf9918818d1267a"
    )
    assert identity["provider_candidate_runtime_commit"] == (
        "dc8c87abc7b216231d09ca12fcc7e76b2249a7ab"
    )
    assert identity["consumer_baseline_source_commit"] == (
        "03dbc39975978d9d0b7f84965015da24f9d3c206"
    )
    assert identity["consumer_candidate_source_commit"] == (
        "0cf55b4fb699ff40433000f5b4efe52b1d33546b"
    )
    assert identity["consumer_cpu_closure_record_commit"] == (
        "c7e7f8631d94c98bd6fa92192e0b806ac25c912c"
    )

    records = record["predecessor_records"]
    for key in ("master_plan", "pressure_contract", "provider_cpu_closure"):
        assert _sha256(records[key]["path"]) == records[key]["sha256"]


def test_rc447_plan_requires_only_installed_artifacts_and_preserves_scope():
    record = _record()
    artifacts = record["artifact_contract"]
    assert artifacts == {
        "installed_wheels_only": True,
        "provider_baseline_wheel_sha256": (
            "9c99ebcdc2e03b2c61b5bc3a5d84d97e99775f488409e2e61f5c06e009695e61"
        ),
        "provider_candidate_wheel_sha256": (
            "814c8a22252b290f7ebd3a249ebaa63152ec978d2726a7eaf82946ec64901ca7"
        ),
        "consumer_baseline_wheel_sha256": (
            "48a452a53dabff90eae50da74b7beb46d5a3fee90c0337fba1341e13cbf0da5b"
        ),
        "consumer_candidate_wheel_sha256": (
            "068dc2f3d3be5b8eee96169711878a71bb4e513b452cdcc2fa54dd67c124b96d"
        ),
        "fresh_isolated_venvs": True,
        "install_with_no_deps": True,
        "pip_check_required": True,
        "source_shadow_import_forbidden": True,
        "nematics3d_required": False,
        "nematics3d_modification_forbidden": True,
    }
    success = record["success_state"]
    assert success["eligible_for_automatic_merge"] is False
    assert success["eligible_for_default_promotion"] is False
    assert success["production_default_changed"] is False
    assert success["larger_batch_qualified"] is False
    assert success["optimizer_campaign_authorized"] is False
    assert success["scientific_control_result"] is False


def test_rc447_plan_freezes_pressure_identity_and_correctness_matrix():
    record = _record()
    pressure = record["candidate_pressure_contract"]
    assert pressure["functional_api_version"] == "1.0"
    assert pressure["consumer_adapter_version"] == 3
    assert pressure["convergence_schema_version"] == 2
    assert pressure["diagnostics_schema_version"] == 2
    assert pressure["pressure_transpose_protocol"] == "2"
    assert pressure["implicit_pressure_adjoint"] == "2"
    assert pressure["inner_solve_gradient"] == (
        "custom_implicit_pressure_adjoint_v2"
    )
    assert pressure["functional_warm_start"] is False
    assert pressure["production_warm_start_preserved"] is True

    matrix = record["candidate_correctness_matrix"]
    assert matrix["convergence_modes"] == ["fixed_work", "relative_tolerance"]
    assert matrix["pressure_scale_exponents"] == [-600, 0, 600]
    assert matrix["pressure_scale_shapes"] == [[7, 6, 5], [8, 6, 4], [9, 8, 6]]
    assert set(matrix["solve_actions"]) == {
        "production_primal",
        "functional_primal",
        "functional_transpose",
    }
    assert len(matrix["restart_gates"]) == 4
    assert len(matrix["negative_gates"]) == 4
    assert matrix["runtime_fallback_allowed"] is False
    assert matrix["compile_fallback_allowed"] is False
    assert matrix["graph_breaks_allowed"] == 0


def test_rc447_plan_freezes_profile_matrix_and_thresholds_before_h100():
    record = _record()
    profiles = record["performance_matrix"]
    assert profiles["roles"] == ["baseline", "candidate"]
    assert [item["shape"] for item in profiles["grids"]] == [
        [128, 20, 20],
        [512, 40, 40],
    ]
    assert profiles["trials_per_role_and_grid"] == 3
    assert profiles["balanced_order"] == ["A/B", "B/A", "A/B"]
    assert profiles["completed_steps"] == 60
    assert profiles["forward_transforms_per_step"] == 11.0
    assert profiles["inverse_transforms_per_step"] == 36.0

    thresholds = record["thresholds"]
    assert thresholds["checkpointed_full_history_gradient_relative_l2_max"] == 1e-12
    assert thresholds["central_finite_difference_max_relative_error"] == 2e-5
    assert thresholds["taylor_min_resolved_levels"] == 3
    assert thresholds["taylor_r0_slope_range"] == [0.8, 1.2]
    assert thresholds["taylor_r1_slope_range"] == [1.8, 2.2]
    assert thresholds["mean_candidate_over_baseline_timestep_max"] == 1.05
    assert thresholds["median_candidate_over_baseline_timestep_max"] == 1.05
    assert thresholds["peak_allocated_candidate_over_baseline_max"] == 1.05
    assert thresholds["peak_reserved_role"] == "diagnostic_only"


def test_rc447_plan_freezes_single_job_and_requires_later_authorization():
    record = _record()
    slurm = record["slurm"]
    assert slurm == {
        "partition": "hagan-gpu",
        "account": "hagan-lab",
        "qos": "medium",
        "gres": "gpu:H100:1",
        "cpus_per_task": 8,
        "memory": "64G",
        "time_limit": "01:00:00",
        "requeue": False,
        "formal_submission_limit": 1,
        "automatic_retry": False,
    }
    authorization = record["authorization"]
    assert authorization["local_planning_authorized_by_user"] is True
    assert authorization["qualification_support_implementation_authorized"] is True
    for key, value in authorization.items():
        if key not in {
            "local_planning_authorized_by_user",
            "qualification_support_implementation_authorized",
        }:
            assert value is False, key


def test_rc447_plan_binds_the_local_qualification_support_by_hash():
    support = _record()["qualification_support"]
    successor_chain = [
        (
            json.loads(RECOVERY.read_text(encoding="utf-8"))[
                "qualification_support_successors"
            ],
            "candidate_identity_only",
        ),
        (
            json.loads(FIELDS_RECOVERY.read_text(encoding="utf-8"))[
                "qualification_support_successors"
            ],
            "external_qualification_harness_fields_iteration_only",
        ),
        (
            json.loads(IMPORT_RECOVERY.read_text(encoding="utf-8"))[
                "qualification_support_successors"
            ],
            "external_qualification_support_import_bootstrap_only",
        ),
        (
            json.loads(PROFILE_RECOVERY.read_text(encoding="utf-8"))[
                "qualification_support_successors"
            ],
            "external_profile_helper_diagnostics_compatibility_only",
        ),
    ]
    assert support["implementation_scope"] == (
        "qualification_only_no_solver_or_runtime_change"
    )
    assert support["profile_count"] == 12
    assert support["balanced_profile_order"] == ["A/B", "B/A", "A/B"]
    assert support["isolated_compiler_cache_per_profile"] is True
    assert support["atomic_json_no_overwrite"] is True
    assert support["fail_closed_analyzer"] is True
    assert support["execution_started"] is False
    assert support["h100_job_submitted"] is False
    for relative, expected in support["files"].items():
        actual = _sha256(relative)
        observed = expected
        for successors, change_scope in successor_chain:
            successor = successors.get(relative)
            if successor is None:
                continue
            assert successor["historical_sha256"] == observed
            assert successor["change_scope"] == change_scope
            observed = successor["current_sha256"]
        assert actual == observed


def test_rc447_python310_recovery_binds_candidate_without_changing_thresholds():
    recovery = json.loads(RECOVERY.read_text(encoding="utf-8"))

    assert recovery["phase"] == "RC4.4.7-recovery-v5"
    assert recovery["candidate_provider_commit"] == (
        "5b4ee16c81a3ae452ffcf8992a84a738a76ff629"
    )
    assert recovery["candidate_provider_wheel_sha256"] == (
        "6f38996c31586ff7b2a0637de5ff479410bbd01bb53926828071affaf5d4cf26"
    )
    assert recovery["consumer_source_commit"] == (
        "0cf55b4fb699ff40433000f5b4efe52b1d33546b"
    )
    assert recovery["thresholds_changed"] is False
    assert recovery["scientific_contract_changed"] is False
    assert recovery["formal_h100_submission_limit"] == 1
    assert recovery["automatic_retry"] is False


def test_rc447_profile_diagnostics_recovery_is_qualification_only_and_fail_closed():
    recovery = json.loads(PROFILE_RECOVERY.read_text(encoding="utf-8"))

    assert recovery["phase"] == "RC4.4.7-recovery-v8"
    assert recovery["failed_v7_evidence"]["job_id"] == "10878109"
    assert recovery["failed_v7_evidence"]["profile_json_persisted"] == 0
    contract = recovery["diagnostics_compatibility_contract"]
    assert contract["diagnostics_owner"] == "adapter.solver.model.static_model"
    assert contract["role_selection_explicit"] is True
    assert contract["candidate_silent_legacy_downgrade_allowed"] is False
    assert contract["missing_or_malformed_diagnostics_fail_closed"] is True
    assert recovery["thresholds_changed"] is False
    assert recovery["scientific_contract_changed"] is False
    assert recovery["profile_matrix_changed"] is False
    assert recovery["correctness_matrix_changed"] is False
    assert recovery["provider_runtime_modified"] is False
    assert recovery["PSSolver_Control_modified"] is False
    assert recovery["nematics3d_modified"] is False
    assert recovery["production_default_changed"] is False
    assert recovery["authorization"] == {
        "push_authorized": False,
        "single_h100_submission_authorized": False,
        "automatic_retry_authorized": False,
    }


def test_rc447_fields_iteration_recovery_is_local_and_preserves_science_contract():
    recovery = json.loads(FIELDS_RECOVERY.read_text(encoding="utf-8"))

    assert recovery["phase"] == "RC4.4.7-recovery-v6"
    assert recovery["status"] == (
        "LOCAL_QUALIFICATION_SUPPORT_READY_H100_NOT_AUTHORIZED"
    )
    assert recovery["failed_v5_evidence"]["job_id"] == "10875768"
    assert recovery["failure_classification"] == (
        "external_qualification_harness_fields_iteration_contract"
    )
    assert recovery["field_comparison_contract"] == {
        "public_keys_only": True,
        "both_key_sets_must_match": True,
        "values_compared_by_field_name": True,
        "private_fields_storage_used": False,
        "bare_fields_iteration_used": False,
    }
    assert recovery["thresholds_changed"] is False
    assert recovery["scientific_contract_changed"] is False
    assert recovery["profile_matrix_changed"] is False
    assert recovery["correctness_matrix_changed"] is False
    assert recovery["production_default_changed"] is False
    assert recovery["authorization"] == {
        "push_authorized": False,
        "single_h100_submission_authorized": False,
        "automatic_retry_authorized": False,
    }


def test_rc447_support_import_recovery_freezes_complete_cuda_import_closure():
    recovery = json.loads(IMPORT_RECOVERY.read_text(encoding="utf-8"))

    assert recovery["phase"] == "RC4.4.7-recovery-v7"
    assert recovery["status"] == (
        "LOCAL_QUALIFICATION_SUPPORT_READY_H100_NOT_AUTHORIZED"
    )
    assert recovery["failed_v6_evidence"]["job_id"] == "10878065"
    assert recovery["failure_classification"] == (
        "external_qualification_support_incomplete_benchmarks_import_bootstrap"
    )
    support = recovery["support_import_contract"]
    assert support["cuda_node_count"] == 7
    assert support["support_file_count"] == len(support["support_files"])
    assert support["collection_only_bootstrap_passed"] is True
    assert support["source_shadow_import"] is False
    assert recovery["thresholds_changed"] is False
    assert recovery["scientific_contract_changed"] is False
    assert recovery["profile_matrix_changed"] is False
    assert recovery["correctness_matrix_changed"] is False
    assert recovery["production_default_changed"] is False
    assert recovery["authorization"] == {
        "push_authorized": False,
        "single_h100_submission_authorized": False,
        "automatic_retry_authorized": False,
    }


def test_rc447_plan_stops_on_incomplete_or_inferred_evidence():
    record = _record()
    execution = " ".join(record["execution_order"])
    stops = " ".join(record["stop_conditions"])
    assert "decomposed H100" in execution
    assert "twelve balanced" in execution
    assert "old identity" in stops
    assert "inferred unpersisted value" in stops
    assert "automatic retry" in stops
    assert record["successful_classification"] == (
        "PASS_RC4_4_7_CHANNEL_PRESSURE_SINGLE_H100_CUMULATIVE_CLOSURE"
    )
