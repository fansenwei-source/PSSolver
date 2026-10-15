"""Evidence contract for the rc5 released-checkpoint matrix."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_checkpoint_matrix.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_matrix_record_binds_the_qualified_source_and_final_classification():
    record = _record()
    assert record["phase"] == "G-1.2"
    assert record["classification"] == (
        "PASS_RC5_G1_RELEASED_CHECKPOINT_MATRIX_WITH_PROVIDER_IDENTITY_MIGRATION_REQUIRED"
    )
    assert record["qualified_source"] == {
        "commit": "674a8121779be72c95aa5dc9e2babca0ce7b93a9",
        "package_version": "0.2.0rc5",
        "record_commit_is_qualified_source": False,
        "requires_checkpoint_matrix_requalification": False,
    }


def test_matrix_record_requires_all_released_fixture_and_wheel_gates():
    matrix = _record()["released_checkpoint_matrix"]
    assert matrix["complete"] is True
    assert matrix["primary_case_count"] == 13
    assert matrix["external_fixture_crosscheck"] == {"passed": 10, "failed": 0}
    assert matrix["rc1_rc3_installed_wheel_tests"]["passed"] == 20
    assert matrix["rc1_rc3_installed_wheel_tests"]["failed"] == 0
    assert matrix["rc1_rc3_installed_wheel_tests"]["skipped"] == 0
    for key in (
        "periodic_functional_exact",
        "channel_functional_exact",
        "legacy_channel_exact",
        "source_checkpoints_unchanged",
    ):
        assert matrix["rc4_writer"][key] is True
        assert matrix["rc5_self_round_trip"][key] is True


def test_matrix_record_preserves_fail_closed_released_dispositions():
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


def test_provider_identity_review_is_component_resolved():
    impact = _record()["provider_identity_impact"]
    assert impact["review_complete"] is True
    assert impact["periodic"] == {
        "changed": True,
        "rc4_sha256": "abd09b19055cfbac809fa67445a5166d4ee3338e9c439d54da08bd99e105428f",
        "rc5_sha256": "8185c6d4a82d796c9f91857f64d38a9ed7cf6d5641e26423b774c1e561052058",
        "only_changed_path": (
            "execution.functional_runtime.hermitian_state_projection"
        ),
        "old_value": "self_conjugate_planes_each_step",
        "new_value": "self_conjugate_stored_planes_projection_each_step.v2",
    }
    assert impact["channel"]["changed"] is False
    assert impact["channel"]["rc4_sha256"] == impact["channel"]["rc5_sha256"]


def test_consumer_impact_requires_periodic_migration_and_requalification():
    impact = _record()["consumer_impact"]
    assert impact["unrestricted_rc5_cpu_construction_passed"] is True
    assert impact["periodic_rc4_identity_pin_rejects_rc5"] is True
    assert impact["channel_rc4_identity_pin_accepts_rc5"] is True
    assert impact["PSSolver_Control_migration_required"] is True
    assert impact["PSSolver_Control_installed_wheel_requalification_required"] is True


def test_matrix_record_rejects_the_invalid_harness_preflight_as_evidence():
    rejected = _record()["rejected_harness_preflight"]
    assert rejected == {
        "classification": "EXTERNAL_HARNESS_DISCRETIZATION_NONE_ERROR",
        "used_as_qualification_evidence": False,
        "rc4_and_rc5_failed_identically": True,
        "corrected_to_empty_mapping": True,
    }


def test_matrix_record_binds_the_external_archive_and_remains_fail_closed():
    record = _record()
    evidence = record["external_evidence"]
    assert evidence["checksum_entries"] == 123
    assert evidence["checksum_passed"] == 123
    assert evidence["checksum_failed"] == 0
    assert evidence["checksums_sha256"] == (
        "6bc84435b3dab444156a87ba7ddc5981f93989c38098dfa3f9e56d8dcff9d627"
    )
    assert record["authorization"] == {
        "checkpoint_matrix_complete": True,
        "provider_identity_impact_review_complete": True,
        "g1_complete": False,
        "eligible_for_h100_submission": False,
        "h100_submission_count": 0,
        "automatic_retry": False,
        "push_executed": False,
        "tag_created": False,
        "release_created": False,
        "pypi_published": False,
        "production_default_changed": False,
        "nematics3d_modified": False,
    }
    assert record["next_gate"] == (
        "pssolver_control_periodic_identity_migration_and_installed_wheel_qualification"
    )
