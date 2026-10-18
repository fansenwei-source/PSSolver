"""Evidence contract for the RC6 checkpoint compatibility CPU gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_checkpoint_cpu.json"
INSTALLED = ROOT / "notes/PSSolver_v0_2_0rc6_installed_wheel_cpu.json"
HISTORICAL = ROOT / "notes/PSSolver_v0_2_0rc5_g1_checkpoint_matrix.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rc6_checkpoint_record_binds_the_qualified_installed_wheel():
    record = _record()
    assert record["phase"] == "RC6-10"
    assert record["classification"] == (
        "PASS_RC6_RELEASED_CHECKPOINT_COMPATIBILITY_AND_TAMPER_CPU"
    )
    assert record["qualified_source"]["commit"] == (
        "ea801ec08f5b2465661334e1c126133e55b68e96"
    )
    assert record["qualified_source"]["package_version"] == "0.2.0rc6"
    assert record["qualified_source"]["record_commit_is_qualified_source"] is False
    assert _sha256(INSTALLED) == (
        "b18da63ade78adc5fa5bbc5e3e77615facfcb054937ddefdf042257c48108980"
    )


def test_rc6_checkpoint_record_preserves_the_historical_release_matrix():
    released = _record()["released_checkpoint_matrix"]
    assert _sha256(HISTORICAL) == (
        "a83a8286212d450223278fb8492c69f0a41c0c546aa303da7f55d850d14ce572"
    )
    assert released["historical_record_rewritten"] is False
    assert released["primary_case_count"] == 13
    assert released["installed_wheel_tests"] == {
        "passed": 20,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xml_sha256": (
            "5c86d52e6eda80bf5917e5606057d44a52847d6e67c7492f967223ff0c5f1f43"
        ),
    }
    assert released["external_fixture_crosscheck"]["passed"] == 10
    assert released["external_fixture_crosscheck"]["failed"] == 0


def test_rc6_checkpoint_record_preserves_fail_closed_dispositions():
    assert _record()["released_checkpoint_matrix"][
        "fail_closed_dispositions_preserved"
    ] == {
        "rc1_periodic_functional": "explicit_state_only_migration",
        "rc1_channel_functional": "explicit_state_only_migration",
        "rc2_channel_functional": "explicit_state_only_migration",
        "rc1_legacy_channel_even": (
            "reject_pre_rc4_1_forward_nyquist_before_mutation"
        ),
    }


def test_rc6_current_checkpoint_matrix_is_transactional_and_exact():
    current = _record()["current_rc6_matrix"]
    assert current["family_count"] == 4
    assert current["passed"] == 4
    assert current["failed"] == 0
    for key in (
        "serialized_state_equal",
        "continuous_split_restart_replay_exact",
        "restored_state_and_observation_finite",
        "payload_tamper_rejected_by_integrity_guard",
        "identity_tamper_rejected_by_provenance_or_compatibility_guard",
        "negative_gates_rejected_before_target_mutation",
        "source_checkpoints_unchanged",
    ):
        assert current[key] is True
    assert current["fallback_used"] is False


def test_rc6_checkpoint_runner_and_external_manifest_are_bound():
    record = _record()
    runner = record["runner_contract"]
    assert _sha256(ROOT / runner["path"]) == runner["sha256"]
    assert runner["sha256"] == (
        "3fbd7759caf8c5335a4f3d4e39a6db3db4abd62be8213cf44ffc316ca692ca88"
    )
    assert runner["python_isolated_mode"] is True
    assert runner["installed_package_imported_before_support_path"] is True
    evidence = record["evidence"]
    assert evidence["checksum_entries"] == 143
    assert evidence["checksum_passed"] == 143
    assert evidence["checksum_failed"] == 0
    assert evidence["checksums_sha256"] == (
        "5f1235eeb626422842eb129f47bb09164bd67af4ca07f88f02888cbdc449ab46"
    )


def test_rc6_checkpoint_record_preserves_version_boundaries():
    contract = _record()["version_contract"]
    assert contract == {
        "checkpoint_schema_changed_in_rc6": False,
        "functional_api_version": "1.0",
        "functional_api_version_changed_in_rc6": False,
        "production_default_changed": False,
    }


def test_rc6_checkpoint_record_remains_fail_closed_for_later_gates():
    record = _record()
    assert record["pending_rc6_gates"] == [
        "provider_identity_impact_review",
        "pssolver_control_migration_and_installed_wheel_qualification_if_required",
        "single_h100_cumulative_qualification",
        "independent_post_release_verification",
    ]
    authorization = record["authorization"]
    assert (
        authorization["released_checkpoint_compatibility_and_tamper_complete"]
        is True
    )
    assert authorization["provider_identity_impact_review_complete"] is False
    assert authorization["rc6_cumulative_qualification_complete"] is False
    assert authorization["eligible_for_push"] is False
    assert authorization["eligible_for_h100_submission"] is False
    assert authorization["h100_submission_count"] == 0
    assert authorization["automatic_retry"] is False
    assert authorization["tag_created"] is False
    assert authorization["release_created"] is False
    assert authorization["pypi_published"] is False
    assert authorization["production_default_changed"] is False
    assert authorization["PSSolver_Control_modified"] is False
    assert authorization["nematics3d_modified"] is False
