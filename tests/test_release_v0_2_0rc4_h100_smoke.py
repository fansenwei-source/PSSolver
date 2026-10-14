"""Immutable registration for the v0.2.0rc4 release H100 smoke."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes" / "PSSolver_v0_2_0rc4_release_h100_smoke.json"
SOURCE_RECORD = ROOT / "notes" / "pssolver_v0_2_0_rc4.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_rc4_release_smoke_binds_the_exact_qualified_source_and_artifacts():
    record = _record()
    assert record["classification"] == (
        "PASS_V0_2_0RC4_INSTALLED_WHEEL_SINGLE_H100_RELEASE_SMOKE_"
        "WITH_AUTHORIZED_TF32_CHILD_BOOTSTRAP_RECOVERY_V2"
    )
    assert record["qualification_complete"] is True
    source = record["qualified_source"]
    assert source["commit"] == "658869456d762c917ca4409b091539c3733df2fb"
    assert source["annotated_tag_target_commit"] == source["commit"]
    assert source["record_commit_is_qualified_source"] is False
    assert source["evidence_only_successor_requires_h100_requalification"] is False
    assert record["artifacts"] == {
        "source_archive": {
            "name": "PSSolver-0.2.0rc4.tar",
            "sha256": (
                "6e5f94991b02fcc9d5d437aa348205e5c1eeb79035f371a119ab5844f17fcbdb"
            ),
        },
        "sdist": {
            "name": "pssolver-0.2.0rc4.tar.gz",
            "sha256": (
                "c7f5f2be2881cd3377cd936f8596ce2faf1b2a35fd96a955e71d92b3a292964c"
            ),
        },
        "wheel": {
            "name": "pssolver-0.2.0rc4-py3-none-any.whl",
            "sha256": (
                "1bf31f324180dac58777a23785fb4e1424d3cc589099cf8c6328e2f6c74d8cc5"
            ),
        },
    }


def test_rc4_release_smoke_preserves_the_failed_attempt_and_binds_recovery():
    record = _record()
    failed = record["failed_attempt_preserved"]
    assert failed["job_id"] == 10881417
    assert failed["classification"].startswith("FAIL_")
    assert failed["manifest_entries"] == 878
    assert failed["manifest_sha256"] == (
        "f9668a6056ed59d70e6af88e293e1d4f21c8e50bd05c27a52f57fbb13b11ac2a"
    )
    assert failed["scientific_failure"] is False
    assert failed["rewritten"] is False

    recovery = record["h100_recovery"]
    assert recovery["job_id"] == 10881426
    assert recovery["state"] == "COMPLETED"
    assert recovery["exit_code"] == "0:0"
    assert recovery["manifest_entries"] == 895
    assert recovery["manifest_sha256"] == (
        "636e56b92170e99a3928818a2761f65e3e47a8dd9cebdf490311cd6bbaf1e9bd"
    )
    assert recovery["complete_present"] is True
    assert recovery["automatic_retry"] is False
    assert recovery["second_recovery_job_submitted"] is False


def test_rc4_release_smoke_records_tf32_cuda_and_plane_gates():
    record = _record()
    environment = record["environment"]
    assert environment["gpu"] == "NVIDIA H100 PCIe"
    assert environment["tf32_runner_and_pytest_same_process"] is True
    assert environment["tf32_policy_applied_before_pytest"] is True
    assert environment["tf32_policy_preserved_after_pytest"] is True
    assert environment["cuda_matmul_allow_tf32"] is False
    assert environment["cudnn_allow_tf32"] is False
    assert environment["float32_matmul_precision"] == "highest"

    gates = record["gates"]
    assert gates["cuda_only_tests"] == "7 passed, 0 failed, 0 skipped"
    assert gates["plane_installed_wheel_cuda_one_step"] is True
    assert gates["plane_completed_steps"] == 1
    assert gates["plane_complete_marker"] is True
    assert gates["plane_q_finite"] is True
    assert gates["plane_velocity_finite"] is True
    assert gates["plane_pressure_finite"] is True


def test_rc4_release_smoke_did_not_rewrite_the_qualified_source_record():
    frozen = _record()["frozen_release_metadata"]
    assert frozen["rewritten_by_registration"] is False
    assert frozen["source_preparation_status_preserved"] is True
    assert hashlib.sha256(SOURCE_RECORD.read_bytes()).hexdigest() == frozen["sha256"]
    source_record = json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))
    assert source_record["status"] == "release_preparation_source_candidate"
    assert source_record["artifacts"]["tag_created"] is False


def test_rc4_release_smoke_freezes_submission_accounting_without_overclaim():
    record = _record()
    accounting = record["submission_accounting"]
    assert accounting == {
        "failed_attempt_formal_h100_submission_count": 1,
        "recovery_v2_formal_h100_submission_count": 1,
        "release_smoke_formal_h100_submission_count_total": 2,
        "every_submission_separately_authorized": True,
        "automatic_retry": False,
    }
    decision = record["release_decision"]
    assert decision["h100_release_smoke_complete"] is True
    assert decision["eligible_for_annotated_tag"] is True
    assert decision["eligible_for_github_prerelease"] is True
    assert decision["eligible_for_pypi_publication"] is False
    for key in (
        "production_default_changed",
        "functional_api_version_changed",
        "checkpoint_compatibility_removed",
        "larger_batch_qualified",
        "plane_control_qualified",
        "optimizer_campaign_authorized",
        "scientific_control_result",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "tag_created",
        "github_prerelease_created",
        "package_published",
    ):
        assert decision[key] is False
