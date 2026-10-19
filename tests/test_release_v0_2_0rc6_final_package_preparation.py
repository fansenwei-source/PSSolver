"""Evidence contract for the final PSSolver 0.2.0rc6 package preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pssolver.functional.api import functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_final_package_preparation.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc6_final_package_binds_the_pushed_reviewed_source():
    record = _record()
    assert record["classification"] == (
        "PASS_RC6_FINAL_SDIST_WHEEL_AND_INSTALLED_CPU_RELEASE_PREPARATION"
    )
    source = record["source_identity"]
    expected = "fd7323d5e9ef87c537ed07476f8bd50b982af443"
    assert source["source_commit"] == expected
    assert source["remote_tip_at_build"] == expected
    assert source["release_integration_review_commit"] == expected
    assert source["archive_embedded_commit"] == expected
    assert source["record_commit_is_package_source"] is False
    assert source["source_exported_with_git_archive"] is True


def test_rc6_final_package_preserves_the_integration_review():
    review = _record()["release_integration_review"]
    assert _sha256(review["path"]) == review["sha256"]
    persisted = json.loads((ROOT / review["path"]).read_text(encoding="utf-8"))
    assert persisted["classification"] == review["classification"]
    assert review["release_integration_review_complete"] is True
    assert review["eligible_for_final_release_package_preparation"] is True


def test_rc6_final_artifact_identities_are_frozen():
    artifacts = _record()["artifacts"]
    assert artifacts["source_archive"] == {
        "name": "PSSolver-0.2.0rc6-fd7323d.tar",
        "size": 24002560,
        "sha256": "c2f3be60f09c47b0aa45064e9ec36a7ad76fde666f207d8595f622057d5b804c",
        "archive_integrity": "PASS",
    }
    assert artifacts["sdist"] == {
        "name": "pssolver-0.2.0rc6.tar.gz",
        "size": 1383734,
        "sha256": "e6f2013d7d16c3f265759017bae8b5c8d0c1f6a43b54ff52476c9da5c296b907",
        "archive_integrity": "PASS",
    }
    wheel = artifacts["wheel"]
    assert wheel["name"] == "pssolver-0.2.0rc6-py3-none-any.whl"
    assert wheel["size"] == 799354
    assert wheel["sha256"] == (
        "9fe8bd47fc2cd194a0263bb9e45cc37e4a29fffec329fd5356f39296b2efdad5"
    )
    assert wheel["archive_integrity"] == "PASS"
    assert wheel["metadata_version"] == "0.2.0rc6"
    assert wheel["top_level_entries"] == [
        "Plane_beris_edwards_stokes.py",
        "pssolver",
        "pssolver-0.2.0rc6.dist-info",
    ]


def test_rc6_final_installed_wheel_cpu_smoke_is_complete():
    smoke = _record()["installed_wheel_cpu_smoke"]
    assert smoke["installed_with_no_deps"] is True
    assert smoke["venv_local_numpy_version"] == "2.3.2"
    assert smoke["pip_check"] == "PASS"
    assert smoke["source_shadow_import"] is False
    assert smoke["package_version"] == "0.2.0rc6"
    assert smoke["functional_api_version"] == "1.0"
    assert smoke["console_entry_point_help"] == "PASS"
    assert smoke["real_cpu_smokes"] == {
        "plane_public_one_step": "PASS",
        "periodic_continuous_and_restart": "PASS",
        "channel_continuous_and_restart": "PASS",
        "pytest": "3 passed, 0 failed, 0 skipped, 0 xfailed",
    }
    assert smoke["environment"]["nematics3d_modified"] is False


def test_rc6_final_package_keeps_the_public_contract_stable():
    contract = _record()["public_contract"]
    assert contract["package_version"] == "0.2.0rc6"
    assert functional_protocol_provenance()["functional_protocol"]["current"] == (
        contract["functional_api_version"]
    )
    assert contract["functional_batch_sizes"] == [1]
    for key in (
        "production_default_changed",
        "compiled_runtime_promoted",
        "checkpoint_schema_changed",
        "PSSolver_Control_bundled",
        "nematics3d_modified",
    ):
        assert contract[key] is False


def test_rc6_final_package_requires_h100_before_any_release_action():
    rejected = _record()["rejected_environment_attempt"]
    assert rejected["used_as_qualification_evidence"] is False
    assert rejected["nematics3d_modified"] is False

    decision = _record()["release_decision"]
    assert decision["final_package_preparation_complete"] is True
    assert decision["eligible_for_single_h100_release_smoke"] is True
    for key in (
        "single_h100_release_smoke_authorized",
        "single_h100_release_smoke_complete",
        "eligible_for_annotated_tag",
        "eligible_for_github_prerelease",
        "eligible_for_pypi_publication",
        "eligible_for_automatic_merge",
        "eligible_for_default_promotion",
        "tag_created",
        "github_prerelease_created",
        "pypi_published",
        "production_default_changed",
    ):
        assert decision[key] is False
