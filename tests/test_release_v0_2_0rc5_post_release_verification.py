"""Immutable registration for the v0.2.0rc5 GitHub prerelease verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_release_and_post_release_verification.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc5_release_binds_the_qualified_source_tag_and_evidence_successor():
    record = _record()
    assert record["classification"] == (
        "PASS_V0_2_0RC5_GITHUB_PRERELEASE_POST_RELEASE_VERIFICATION"
    )
    assert record["release_complete"] is True
    assert record["post_release_verification_complete"] is True
    source = record["qualified_source"]
    assert source["commit"] == "e2e9d05bfac858d8e1ae1653ac86a2187c9917ef"
    assert source["evidence_only_successor"] == (
        "724686cf890585e059081a289a8dfc6e55f54334"
    )
    assert source["record_commit_is_qualified_source"] is False
    assert source["evidence_only_successor_requires_requalification"] is False

    tag = record["annotated_tag"]
    assert tag["name"] == "v0.2.0rc5"
    assert tag["object_type"] == "tag"
    assert tag["object_sha"] == "d11d4f3c26e851baeb034a3995aa821bfe280fb1"
    assert tag["peeled_target_commit"] == source["commit"]
    assert tag["peeled_target_type"] == "commit"
    assert tag["target_matches_qualified_source"] is True


def test_rc5_release_preserves_and_hashes_frozen_prerequisites():
    for prerequisite in _record()["frozen_prerequisites"].values():
        assert _sha256(prerequisite["path"]) == prerequisite["sha256"]
        assert prerequisite["classification"].startswith("PASS_")
    package = _record()["frozen_prerequisites"]["release_package_preparation"]
    assert package["rewritten_by_release_registration"] is False


def test_rc5_github_prerelease_binds_the_exact_assets():
    release = _record()["github_prerelease"]
    assert release["repository"] == "fansenwei-source/PSSolver"
    assert release["release_id"] == 415487879
    assert release["url"] == (
        "https://github.com/fansenwei-source/PSSolver/releases/tag/v0.2.0rc5"
    )
    assert release["draft"] is False
    assert release["prerelease"] is True
    assert release["tag_object_is_authoritative_target_identity"] is True
    assert release["assets"] == [
        {
            "id": 649054506,
            "name": "pssolver-0.2.0rc5.tar.gz",
            "bytes": 1334558,
            "state": "uploaded",
            "sha256": (
                "4477b6d8b0fa03b4e97390729623322e2b1d6ff453845004598bce4ed30f63dc"
            ),
        },
        {
            "id": 649054508,
            "name": "pssolver-0.2.0rc5-py3-none-any.whl",
            "bytes": 794227,
            "state": "uploaded",
            "sha256": (
                "29333022f799483e177a49d549a8db41d5741ee3c61c24023929f409c8941c39"
            ),
        },
    ]


def test_rc5_post_release_download_and_installed_wheel_passed():
    verification = _record()["post_release_verification"]
    assert verification["classification"] == (
        "PASS_V0_2_0RC5_GITHUB_PRERELEASE_POST_RELEASE_VERIFICATION"
    )
    assert verification["github_assets_downloaded_fresh"] is True
    assert verification["downloaded_asset_hashes_match_release_preparation"] is True
    assert set(verification["package_metadata_versions"].values()) == {"0.2.0rc5"}
    assert verification["pip_check"] == "PASS"
    assert verification["source_shadow_import"] is False
    assert verification["formal_h100_submission_count"] == 0

    smoke = verification["installed_wheel_cpu_one_step"]
    assert smoke["completed_steps"] == 1
    assert smoke["runtime_path"] == "periodic_spectral"
    assert smoke["finite"] == {"Q": True, "velocity": True, "pressure": True}
    assert smoke["matches_release_preparation_state_oracle"] is True


def test_rc5_post_release_archive_is_complete_and_failures_are_not_overclaimed():
    verification = _record()["post_release_verification"]
    assert verification["external_harness_attempts_preserved"] is True
    assert verification["solver_failure"] is False
    assert verification["numerical_failure"] is False
    assert verification["scientific_failure"] is False
    assert verification["checksum_entries"] == 38
    assert verification["checksum_passed"] == 38
    assert verification["checksum_failed"] == 0
    assert verification["checksums_sha256"] == (
        "10ca82b9d7c66d0f7cb35680166c9183067ceedb8b543d8c15525858deef78a0"
    )
    assert verification["complete_present"] is True


def test_rc5_release_scope_does_not_overclaim_unqualified_work():
    decision = _record()["release_decision"]
    for key in (
        "release_complete",
        "post_release_verification_complete",
        "tag_created",
        "github_prerelease_created",
        "github_assets_uploaded",
    ):
        assert decision[key] is True
    for key in (
        "eligible_for_pypi_publication",
        "pypi_published",
        "production_default_changed",
        "functional_api_version_changed",
        "checkpoint_compatibility_removed",
        "larger_batch_qualified",
        "plane_control_qualified",
        "optimizer_campaign_authorized",
        "scientific_control_result",
        "PSSolver_Control_modified",
        "nematics3d_modified",
    ):
        assert decision[key] is False
