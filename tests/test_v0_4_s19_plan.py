"""S-19 must preserve the frozen matrix, every threshold, and D-0019."""
from benchmarks.plan_v0_3_s09_h100 import build_plan as frozen_plan
from benchmarks.plan_v0_4_s19_h100 import build_plan


def test_s19_preserves_every_frozen_performance_contract_field():
    old, new = frozen_plan(), build_plan()
    for key in ("scheduler", "grid_contracts", "profile_contract", "thresholds", "performance_adjudication",
                "cuda_only_tests", "long_horizon", "tf32_backward", "job_stage_order"):
        assert new[key] == old[key], key
    assert len(new["profile_matrix"]) == 192
    for a, b in zip(old["profile_matrix"], new["profile_matrix"]):
        assert {k: v for k, v in a.items() if k not in ("provider_commit", "provider_wheel_sha256")} == {
            k: v for k, v in b.items() if k not in ("provider_commit", "provider_wheel_sha256")}
        assert b["provider_commit"] == new["provider"][b["provider_role"]]["commit"]
        assert b["provider_wheel_sha256"] == new["provider"][b["provider_role"]]["wheel_sha256"]
    assert new["d0013_exception_applies"] is False


def test_s19_self_comparison_and_failure_policy_do_not_authorize_tolerance():
    plan = build_plan()
    numerical = plan["numerical"]
    assert numerical["roles"] == ["baseline_a", "baseline_b", "candidate"]
    assert numerical["sequential_not_concurrent"] is True
    assert numerical["same_absolute_calculation_output_restart_path"] is True
    assert numerical["compile_cache_per_process"] == "unique, initially empty, never reused"
    assert len(numerical["families"]) * len(numerical["refresh_intervals"]) * numerical["steps"] == 800
    assert numerical["pcg_paths"] == ["production", "transpose", "implicit", "oracle"]
    assert numerical["nonzero_tolerance_authorized"] is False
    assert plan["failure_policy"]["H"]["continue"] is False
    assert plan["failure_policy"]["N"]["continue_control_and_performance"] is True
    assert plan["failure_policy"]["N"]["qualification"] == "FAIL"
    assert plan["job_stage_order"].index("independent_control_consumer") < plan["job_stage_order"].index("paired_performance_memory_profiles")


def test_s19_plan_is_not_submission_authority_or_completed_qualification():
    plan = build_plan()
    assert plan["classification"] == "S19_PREPARATION_DRAFT_NOT_SUBMISSION_AUTHORITY"
    assert plan["submission"] == {"formal_jobs": 1, "automatic_retry": False, "diagnostic_jobs_max": 1,
                                  "separate_user_confirmation_required": True}
    assert plan["preparation_review_classification"] == "INTERIM_PASS_*"
