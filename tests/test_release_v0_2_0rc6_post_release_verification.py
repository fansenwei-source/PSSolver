"""Registration contract for the independent v0.2.0rc6 release audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_post_release_verification.json"
H100_RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_release_h100_smoke.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_rc6_post_release_record_binds_the_exact_release_identity():
    record = _record()
    assert record["classification"] == (
        "PASS_RC6_INDEPENDENT_POST_RELEASE_CPU_VERIFICATION"
    )
    assert record["verification_complete"] is True
    source = record["qualified_source"]
    expected = "fd7323d5e9ef87c537ed07476f8bd50b982af443"
    assert source["package_source_commit"] == expected
    assert source["record_commit_is_package_source"] is False
    assert source["evidence_only_successor_requires_h100_requalification"] is False
    release = record["release_identity"]
    assert release["release_id"] == 417781605
    assert release["tag"] == "v0.2.0rc6"
    assert release["tag_object_sha"] == (
        "533e3fb4254319d1a6a9a9a0e463447339c18b36"
    )
    assert release["tag_target_commit"] == expected
    assert release["tag_object_type"] == "tag"
    assert release["draft"] is False
    assert release["prerelease"] is True


def test_rc6_post_release_record_binds_downloaded_artifacts():
    artifacts = _record()["published_artifacts"]
    assert artifacts["sdist"] == {
        "asset_id": 657502345,
        "name": "pssolver-0.2.0rc6.tar.gz",
        "size": 1383734,
        "state": "uploaded",
        "sha256": (
            "e6f2013d7d16c3f265759017bae8b5c8d0c1f6a43b54ff52476c9da5c296b907"
        ),
        "github_digest_matched": True,
    }
    assert artifacts["wheel"] == {
        "asset_id": 657502344,
        "name": "pssolver-0.2.0rc6-py3-none-any.whl",
        "size": 799354,
        "state": "uploaded",
        "sha256": (
            "9fe8bd47fc2cd194a0263bb9e45cc37e4a29fffec329fd5356f39296b2efdad5"
        ),
        "github_digest_matched": True,
    }


def test_rc6_post_release_record_preserves_h100_evidence():
    frozen = _record()["release_h100_smoke"]
    assert hashlib.sha256(H100_RECORD.read_bytes()).hexdigest() == frozen["sha256"]
    assert frozen["job_id"] == 10901816
    assert frozen["classification"] == (
        "PASS_RC6_INSTALLED_WHEEL_SINGLE_H100_RELEASE_SMOKE"
    )
    assert frozen["rewritten_by_registration"] is False


def test_rc6_post_release_content_and_installation_gates_are_complete():
    record = _record()
    content = record["content_identity"]
    assert content["fresh_github_download"] is True
    assert content["local_build_artifacts_reused"] is False
    assert content["python_source_files_compared"] == 258
    assert content["sdist_python_files_match_tag"] is True
    assert content["wheel_python_files_match_tag"] is True
    assert content["sdist_compileall"] == "PASS"
    assert content["wheel_compileall"] == "PASS"

    installation = record["fresh_installation"]
    for kind in ("wheel", "sdist"):
        assert installation[kind]["pip_check"] == "PASS"
        assert installation[kind]["package_version"] == "0.2.0rc6"
        assert installation[kind]["functional_api_version"] == "1.0"
        assert installation[kind]["source_shadow_import"] is False
    assert installation["environment"]["nematics3d_modified"] is False


def test_rc6_post_release_runtime_and_source_gates_pass():
    record = _record()
    smoke = record["installed_runtime_smoke"]
    assert smoke["classification"] == (
        "PASS_RC6_POST_RELEASE_INSTALLED_WHEEL_CPU_SMOKE"
    )
    for family in ("periodic", "channel"):
        assert smoke[family]["completed_steps"] == 1
        assert smoke[family]["complete_present"] is True
        assert smoke[family]["finite_q_u_p"] is True

    suite = record["source_cpu_suite"]
    assert suite["head"] == "fd7323d5e9ef87c537ed07476f8bd50b982af443"
    assert suite["detached"] is True
    assert suite["clean"] is True
    assert {
        key: suite[key]
        for key in (
            "passed",
            "failed",
            "errors",
            "skipped",
            "xfailed",
            "deselected",
            "subtests_passed",
        )
    } == {
        "passed": 4068,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
    }


def test_rc6_post_release_record_preserves_the_discarded_harness_attempt():
    attempt = _record()["discarded_archive_only_suite_attempt"]
    assert attempt["used_as_product_qualification_evidence"] is False
    assert attempt["classification"] == (
        "EXTERNAL_TEST_HARNESS_MISSING_GIT_PROVENANCE"
    )
    assert attempt["passed"] == 4029
    assert attempt["failed"] == 34
    assert attempt["errors"] == 5
    assert attempt["runtime_smoke_failure"] is False
    assert "clean detached Git worktree" in attempt["corrected_by"]


def test_rc6_post_release_record_does_not_overclaim_scope():
    record = _record()
    assert record["registration_verification"] == {
        "targeted_contracts": "19 passed, 0 failed",
        "full_cpu_passed": 4087,
        "full_cpu_failed": 0,
        "full_cpu_skipped": 0,
        "full_cpu_xfailed": 0,
        "full_cpu_deselected": 7,
        "subtests_passed": 8,
        "git_diff_check": "PASS",
    }
    scope = record["scope"]
    for key in (
        "tag_or_release_modified_by_verification",
        "pssolver_runtime_modified",
        "tests_relaxed",
        "thresholds_changed",
        "skip_or_xfail_added",
        "production_default_changed",
        "functional_api_version_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "h100_job_submitted",
        "pypi_published",
    ):
        assert scope[key] is False

    decision = record["decision"]
    assert decision["post_release_cpu_verification_passed"] is True
    assert decision["release_remains_prerelease"] is True
    assert decision["eligible_for_independent_second_party_review"] is True
    assert decision["independent_second_party_review_complete"] is False
    assert decision["eligible_for_pypi_publication"] is False
    assert decision["release_mutation_recommended"] is False
