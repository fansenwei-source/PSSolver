"""Immutable evidence contract for the v0.2.0rc6 H100 release smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc6_release_h100_smoke.json"
PACKAGE_RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc6_final_package_preparation.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_rc6_release_smoke_binds_the_exact_package_source_and_artifacts():
    record = _record()
    assert record["classification"] == (
        "PASS_RC6_INSTALLED_WHEEL_SINGLE_H100_RELEASE_SMOKE"
    )
    assert record["qualification_complete"] is True
    source = record["qualified_source"]
    expected = "fd7323d5e9ef87c537ed07476f8bd50b982af443"
    assert source["package_source_commit"] == expected
    assert source["prospective_annotated_tag_target_commit"] == expected
    assert source["record_commit_is_package_source"] is False
    assert source["evidence_only_successor_requires_h100_requalification"] is False

    artifacts = record["artifacts"]
    assert artifacts["source_archive"]["sha256"] == (
        "c2f3be60f09c47b0aa45064e9ec36a7ad76fde666f207d8595f622057d5b804c"
    )
    assert artifacts["sdist"]["sha256"] == (
        "e6f2013d7d16c3f265759017bae8b5c8d0c1f6a43b54ff52476c9da5c296b907"
    )
    assert artifacts["wheel"]["sha256"] == (
        "9fe8bd47fc2cd194a0263bb9e45cc37e4a29fffec329fd5356f39296b2efdad5"
    )
    assert artifacts["qualification_support"]["contains_pssolver_package"] is False


def test_rc6_release_smoke_preserves_the_package_preparation_record():
    frozen = _record()["frozen_package_preparation"]
    assert hashlib.sha256(PACKAGE_RECORD.read_bytes()).hexdigest() == frozen["sha256"]
    assert frozen["rewritten_by_registration"] is False
    assert frozen["pending_h100_fields_preserved"] is True
    package = json.loads(PACKAGE_RECORD.read_text(encoding="utf-8"))
    assert package["release_decision"]["single_h100_release_smoke_complete"] is False
    assert package["release_decision"]["tag_created"] is False


def test_rc6_release_smoke_freezes_the_single_h100_job_and_archives():
    record = _record()
    job = record["h100_job"]
    assert job["job_id"] == 10901816
    assert job["state"] == "COMPLETED"
    assert job["exit_code"] == "0:0"
    assert job["formal_h100_submission_count"] == 1
    assert job["automatic_retry"] is False
    assert job["second_job_submitted"] is False
    assert job["result_sha256"] == (
        "5dc252592fec4a00717147392f0efff635102f5d616d4bb7a238593525f08b9d"
    )

    archive = record["archive_integrity"]
    assert archive["control_manifest_entries"] == 716
    assert archive["control_manifest_passed"] == 716
    assert archive["control_manifest_sha256"] == (
        "f7e39759c6e1d62990d6eeeaf9fc18c8d68979889a17f1c5c0580996566bdda6"
    )
    assert archive["control_complete_present"] is True
    assert archive["scratch_manifest_entries"] == 14
    assert archive["scratch_manifest_passed"] == 14
    assert archive["scratch_manifest_sha256"] == (
        "1f9bd052266d794f9eef23785bc88ebb76c82301324702d5937b3e60c18acb2f"
    )
    assert archive["scratch_complete_present"] is True


def test_rc6_release_smoke_records_h100_tf32_and_cuda_only_gates():
    record = _record()
    environment = record["environment"]
    assert environment["gpu"] == "NVIDIA H100 PCIe"
    assert environment["source_shadow_import"] is False
    assert environment["functional_api_version"] == "1.0"
    assert environment["tf32_runner_and_pytest_same_process"] is True
    assert environment["tf32_before_policy"] == "highest/false/true"
    assert environment["tf32_before_pytest"] == "highest/false/false"
    assert environment["tf32_after_pytest"] == "highest/false/false"
    assert environment["tf32_after_plane"] == "highest/false/false"

    gates = record["gates"]
    assert gates["installed_wheel_identity"] is True
    assert gates["cuda_only_tests"] == (
        "7 passed, 0 failed, 0 skipped, 0 xfailed"
    )


def test_rc6_release_smoke_records_the_plane_one_step_gate():
    gates = _record()["gates"]
    assert gates["plane_installed_wheel_cuda_one_step"] is True
    assert gates["plane_runtime_path"] == "legacy_production"
    assert gates["plane_pointwise_execution"] == "eager"
    assert gates["plane_runtime_fallback_used"] is False
    assert gates["plane_completed_steps"] == 1
    assert gates["plane_complete_marker"] is True
    assert gates["plane_q_finite"] is True
    assert gates["plane_velocity_finite"] is True
    assert gates["plane_pressure_finite"] is True


def test_rc6_release_smoke_does_not_overclaim_release_authority():
    record = _record()
    assert record["post_registration_cpu"] == {
        "passed": 4080,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
        "deselected_cuda_tests_ran_on_h100": True,
    }
    decision = record["release_decision"]
    assert decision["h100_release_smoke_complete"] is True
    assert decision["eligible_for_annotated_tag_review"] is True
    assert decision["eligible_for_github_prerelease_review"] is True
    for key in (
        "eligible_for_pypi_publication",
        "eligible_for_automatic_merge",
        "eligible_for_default_promotion",
        "production_default_changed",
        "functional_api_version_changed",
        "checkpoint_compatibility_removed",
        "larger_batch_qualified",
        "plane_control_qualified",
        "optimizer_campaign_authorized",
        "scientific_control_result",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "tag_authorized",
        "github_prerelease_authorized",
        "pypi_publication_authorized",
        "tag_created",
        "github_prerelease_created",
        "pypi_published",
        "package_published",
    ):
        assert decision[key] is False
