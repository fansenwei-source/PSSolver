"""Frozen contract for the rc5 G-1 single-H100 cumulative qualification."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_single_h100_qualification_plan.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_plan_binds_provider_and_consumer_qualified_sources():
    record = _record()
    assert record["provider_identities"]["baseline"]["commit"] == (
        "658869456d762c917ca4409b091539c3733df2fb"
    )
    assert record["provider_identities"]["candidate"]["qualified_source_commit"] == (
        "674a8121779be72c95aa5dc9e2babca0ce7b93a9"
    )
    consumer = record["consumer_identities"]
    assert consumer["qualified_source_commit"] == (
        "c14e6b56c88081e7f9a5642e65ffd722eec7445b"
    )
    assert consumer["evidence_only_commit"] == (
        "2474592fa64d1b3243277c7439a16104f2cdeb9f"
    )
    assert consumer["remote_branch_present"] is False


def test_consumer_cpu_review_is_complete_and_did_not_change_runtime_source():
    review = _record()["consumer_cpu_review"]
    assert review["classification"] == (
        "PASS_RC5_CONTROL_IDENTITY_MIGRATION_AND_INSTALLED_WHEEL_CPU_REVIEW"
    )
    assert review["source_runtime_files_changed"] is False
    assert review["external_evidence"]["checksum_entries"] == 469
    assert review["external_evidence"]["checksum_passed"] == 469
    assert review["external_evidence"]["checksum_failed"] == 0
    assert review["independent_targeted_review"] == {
        "passed": 36,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert review["provider_contract_discovery"]["adapter_change_required"] is False


def test_plan_freezes_fixture_specific_identity_values():
    identity = _record()["identity_contract"]
    assert identity["periodic"]["rc4_canonical_sha256"] == (
        "f83d7b368bda23aacdf1c70e77de974df1c05a3e8b0c7886191dd82c14ade416"
    )
    assert identity["periodic"]["rc5_canonical_sha256"] == (
        "d85f6540494696fddb4549cd4d431d362fe4a55bc53cb9cfd8158fae7c09530d"
    )
    assert identity["periodic"]["only_changed_path"] == (
        "provider_identity.execution.functional_runtime.hermitian_state_projection"
    )
    assert identity["channel_identities_unchanged"] is True
    assert identity["old_periodic_pin_rejects_before_mutation"] is True


def test_plan_requires_one_non_retrying_h100_job_after_all_preconditions():
    record = _record()
    scheduler = record["scheduler_contract"]
    assert scheduler["formal_job_count"] == 1
    assert scheduler["automatic_retry"] is False
    assert scheduler["partition"] == "hagan-gpu"
    assert scheduler["account"] == "hagan-lab"
    assert scheduler["qos"] == "medium"
    preflight = record["pre_submission_conditions"]
    assert preflight["ready"] is False
    assert not any(preflight.values())


def test_plan_covers_cuda_tf32_checkpoint_profile_and_consumer_gates():
    record = _record()
    assert len(record["provider_cuda_only_tests"]) == 7
    assert len(record["tf32_matrix"]["caller_states"]) == 6
    assert record["tf32_matrix"]["runtime_family_count"] == 8
    assert record["tf32_matrix"]["case_count"] == 48
    assert record["checkpoint_contract"]["released_matrix_primary_case_count"] == 13
    assert record["profile_matrix"]["profile_count"] == 96
    assert len(record["profile_matrix"]["runtime_families"]) == 8
    consumer = record["consumer_h100_contract"]
    assert consumer["complete_suite_expected_passed"] == 336
    assert consumer["allowed_skip_count"] == 1
    assert consumer["failed_allowed"] == 0
    assert consumer["xfailed_allowed"] == 0
    assert consumer["deselected_allowed"] == 0


def test_periodic_repair_uses_physical_tolerance_and_exact_hermitian_gate():
    numerical = _record()["profile_matrix"]["numerical_comparison"]
    assert numerical["no_global_final_state_sha_equality_assumption"] is True
    repaired = numerical["periodic_repaired_paths"]
    assert "physical Q/u/p" in repaired
    assert "1e-12" in repaired
    assert "exactly zero" in repaired


def test_plan_does_not_authorize_implementation_push_or_h100():
    record = _record()
    authorization = record["authorization"]
    assert authorization["consumer_cpu_review_complete"] is True
    assert authorization["h100_plan_frozen"] is True
    for key, value in authorization.items():
        if key not in {"consumer_cpu_review_complete", "h100_plan_frozen"}:
            assert value is False
    assert record["success_semantics"] == {
        "h100_pass_makes_candidate_eligible_for_release_integration_review": True,
        "h100_pass_does_not_automatically_merge_or_publish": True,
        "g1_complete_after_h100_alone": False,
        "post_release_verification_still_required": True,
        "automatic_merge": False,
        "default_promotion": False,
    }
    assert record["next_gate"] == (
        "implement_and_cpu_validate_versioned_rc5_qualification_support_without_h100_submission"
    )
