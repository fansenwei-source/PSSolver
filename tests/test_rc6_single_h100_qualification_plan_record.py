"""Repository record checks for the frozen RC6 single-H100 plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.plan_rc6_single_h100_qualification import build_plan


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_single_h100_qualification_plan.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def test_rc6_h100_record_binds_the_frozen_sources_and_prerequisites():
    record = _record()
    assert record["classification"] == (
        "READY_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION_SUPPORT"
    )
    assert record["qualified_sources"]["provider_candidate"]["commit"] == (
        "ea801ec08f5b2465661334e1c126133e55b68e96"
    )
    assert record["qualified_sources"]["consumer_candidate"]["commit"] == (
        "208edd03e6b26a0e788c07d295fddd7a67237189"
    )
    for value in record["prerequisite_records"].values():
        assert _sha256(ROOT / value["path"]) == value["sha256"]


def test_rc6_h100_record_binds_the_canonical_planner_output():
    record = _record()["planner"]
    assert _sha256(ROOT / record["path"]) == record["sha256"]
    assert _canonical_sha256(build_plan()) == record["canonical_output_sha256"]
    assert record["atomic_json"] is True
    assert record["refuse_overwrite"] is True
    assert record["output_root_created_during_planning"] is False


def test_rc6_h100_record_freezes_the_full_balanced_crossover_matrix():
    contract = _record()["profile_contract"]
    assert contract["runtime_family_count"] == 8
    assert contract["grid_count"] == 16
    assert contract["trials_per_cell"] == 6
    assert contract["provider_roles"] == 2
    assert contract["profile_count"] == 192
    assert contract["baseline_then_candidate_trials"] == [1, 3, 5]
    assert contract["candidate_then_baseline_trials"] == [2, 4, 6]
    assert contract["all_raw_samples_retained"] is True
    assert contract["outlier_removal_allowed"] is False
    assert contract["sample_deletion_allowed"] is False
    assert contract["trial_replacement_allowed"] is False


def test_rc6_h100_record_requires_all_cumulative_gates():
    record = _record()
    assert record["provider_cuda_only_gate"] == {
        "test_count": 7,
        "all_must_pass": True,
        "skip_xfail_deselect_allowed": 0,
    }
    assert record["tf32_contract"]["case_count"] == 48
    assert record["checkpoint_contract"]["current_family_count"] == 4
    assert record["checkpoint_contract"]["released_rc1_through_rc5_primary_case_count"] == 13
    assert record["numerical_contract"]["runtime_family_count"] == 8
    assert record["consumer_h100_contract"]["expected_passed"] == 344
    assert record["consumer_h100_contract"]["expected_skipped"] == 1


def test_rc6_h100_record_is_fail_closed_and_not_authorized():
    record = _record()
    assert record["qualification_complete"] is False
    assert record["execution_status"] == {
        "plan_frozen": True,
        "qualification_support_implemented": False,
        "h100_job_submitted": False,
        "formal_h100_submission_count": 0,
        "qualification_complete": False,
    }
    preconditions = record["pre_submission_conditions"]
    assert preconditions["ready"] is False
    assert all(value is False for value in preconditions.values())
    authorization = record["authorization"]
    assert authorization["local_plan_implementation_complete"] is True
    assert authorization["qualification_support_implementation_authorized"] is True
    for key, value in authorization.items():
        if key not in {
            "local_plan_implementation_complete",
            "qualification_support_implementation_authorized",
        }:
            assert value is False, key


def test_rc6_h100_success_never_implies_automatic_release_actions():
    success = _record()["success_contract"]
    assert success["classification"] == "PASS_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION"
    assert success["qualification_complete"] is True
    assert success["eligible_for_release_integration_review"] is True
    assert success["independent_post_release_verification_required"] is True
    assert success["eligible_for_automatic_merge"] is False
    assert success["eligible_for_default_promotion"] is False
    assert success["production_default_changed"] is False
