"""Immutable registration for the rc5 final package preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_release_package_preparation.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_rc5_package_preparation_binds_the_pushed_qualified_source():
    record = _record()
    assert record["classification"] == (
        "PASS_V0_2_0RC5_SOURCE_AND_INSTALLED_WHEEL_RELEASE_PREPARATION_"
        "POST_RELEASE_VERIFICATION_PENDING"
    )
    assert record["qualification_complete"] is True
    source = record["qualified_source"]
    assert source["commit"] == "e2e9d05bfac858d8e1ae1653ac86a2187c9917ef"
    assert source["remote_tip_matched_at_build"] is True
    assert source["record_commit_is_qualified_source"] is False
    assert source["evidence_only_successor_requires_requalification"] is False
    assert source["annotated_tag_target_candidate"] == source["commit"]


def test_rc5_package_preparation_preserves_and_hashes_prerequisite_records():
    for value in _record()["prerequisites"].values():
        assert _sha256(value["path"]) == value["sha256"]
        assert value["classification"].startswith("PASS_")


def test_rc5_package_preparation_binds_exact_release_artifacts():
    assert _record()["artifacts"] == {
        "source_archive": {
            "name": "PSSolver-0.2.0rc5.tar",
            "bytes": 23173120,
            "members": 1631,
            "sha256": "e04d9aef7384dd8d271f59827d3c8b2b9637f50d3050d519c69690496d950299",
        },
        "sdist": {
            "name": "pssolver-0.2.0rc5.tar.gz",
            "bytes": 1334558,
            "members": 687,
            "sha256": "4477b6d8b0fa03b4e97390729623322e2b1d6ff453845004598bce4ed30f63dc",
        },
        "wheel": {
            "name": "pssolver-0.2.0rc5-py3-none-any.whl",
            "bytes": 794227,
            "members": 260,
            "sha256": "29333022f799483e177a49d549a8db41d5741ee3c61c24023929f409c8941c39",
            "package_files_byte_identical_to_source": True,
            "authorized_top_level_compatibility_module": "Plane_beris_edwards_stokes.py",
            "authorized_source_only_document": "pssolver/control/README.md",
            "unexpected_top_level_packages": 0,
        },
    }


def test_rc5_package_preparation_records_installed_wheel_execution():
    installed = _record()["installed_wheel_cpu"]
    assert installed["pip_check"] == "PASS"
    assert installed["source_shadow_import"] is False
    assert installed["package_version"] == "0.2.0rc5"
    assert installed["runtime_path"] == "periodic_spectral"
    assert installed["completed_steps"] == 1
    assert installed["finite"] == {"Q": True, "velocity": True, "pressure": True}
    assert "/venv/lib/python3.10/site-packages/pssolver/__init__.py" in (
        installed["import_path"]
    )


def test_rc5_package_preparation_separates_external_helper_findings():
    findings = _record()["preserved_external_harness_findings"]
    assert findings["first_smoke"].startswith("EXTERNAL_")
    assert findings["second_smoke"].startswith("EXTERNAL_")
    assert findings["finalizer"].startswith("EXTERNAL_")
    assert findings["solver_failure"] is False
    assert findings["numerical_failure"] is False
    assert findings["scientific_failure"] is False
    assert findings["wheel_rebuilt_after_smoke_failures"] is False

    local = _record()["local_workstation_environment_diagnostic"]
    assert local["classification"] == (
        "LOCAL_BASE_ENVIRONMENT_NEMATICS3D_NUMPY_METADATA_CONFLICT"
    )
    assert local["used_as_release_qualification_evidence"] is False
    assert local["nematics3d_modified"] is False
    assert local["numpy_base_environment_modified"] is False


def test_rc5_package_preparation_binds_complete_archive():
    archive = _record()["archive"]
    assert archive["checksum_entries"] == 17
    assert archive["checksum_passed"] == 17
    assert archive["checksum_failed"] == 0
    assert archive["checksums_sha256"] == (
        "e465f50e33aa3bd8415782d587838ed018f978c58d012ae5f580de4a53b36335"
    )
    assert archive["complete_present"] is True


def test_rc5_package_preparation_allows_review_but_not_release_actions():
    decision = _record()["release_decision"]
    assert decision["branch_pushed"] is True
    assert decision["release_package_preparation_complete"] is True
    assert decision["eligible_for_annotated_tag_review"] is True
    assert decision["eligible_for_github_prerelease_review"] is True
    assert decision["post_release_verification_required"] is True
    for key in (
        "eligible_for_pypi_publication",
        "eligible_for_automatic_merge",
        "eligible_for_default_promotion",
        "tag_authorized",
        "github_prerelease_authorized",
        "tag_created",
        "github_prerelease_created",
        "pypi_published",
        "production_default_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
    ):
        assert decision[key] is False
