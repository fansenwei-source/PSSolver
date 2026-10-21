"""Evidence contract for the v0.2.2 V-17 real-NFS qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_2_v17_real_nfs_qualification.json"
PACKAGE = ROOT / "notes/pssolver_v0_2_2_package_preparation.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_v022_v17_real_nfs_record_binds_source_package_and_wheel():
    record = _record()
    assert record["classification"] == (
        "PASS_V0_2_2_V17_INSTALLED_WHEEL_AND_REAL_NFS_QUALIFICATION"
    )
    assert record["qualification_complete"] is True
    source = record["source"]
    assert source["release_source_commit"] == (
        "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
    )
    assert source["package_record_commit"] == (
        "d607c31eda2de4662c5d775ca5fb233721c17d82"
    )
    assert source["package_record_sha256"] == _sha256(PACKAGE)
    assert source["record_commit_is_release_source"] is False
    assert record["installed_wheel"]["sha256"] == (
        "a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e"
    )


def test_v022_v17_real_nfs_record_proves_an_unforced_cross_device_fallback():
    gate = _record()["hpcc_login_node"]
    assert gate["home_filesystem_type"] == "0x6969"
    assert gate["scratch_filesystem_type"] == "0x6969"
    assert gate["link_lstat_device"] != gate["link_stat_device"]
    assert gate["cross_filesystem_parent_symlink"] is True
    assert gate["fallback_forced"] is False
    assert gate["formal_test_execution_count"] == 1
    assert gate["automatic_retry"] is False
    assert gate["slurm_submission_count"] == 0
    assert gate["formal_h100_submission_count"] == 0


def test_v022_v17_real_nfs_record_binds_periodic_and_channel_restart():
    record = _record()
    for family in ("periodic", "channel"):
        result = record[family]
        assert result["fallback_calls"] == 1
        assert result["completed_steps"] == 1
        assert result["checkpoint_import_byte_identical"] is True
        assert result["one_step_restart_byte_identical"] is True
        assert result["written_path_preserves_symlink_spelling"] is True
        assert result["temporary_residue"] == []
        assert len(result["checkpoint_tree_sha256"]) == 64


def test_v022_v17_real_nfs_record_binds_archive_and_cleanup():
    record = _record()
    evidence = record["evidence"]
    assert evidence["result_sha256"] == (
        "02e143018c5a54c2bedf3769f5a650df219220c8dd1a0893da9ab7b144309b0e"
    )
    assert evidence["helper_sha256"] == (
        "65e545e697a81f6d61aba8c7b73d3d9a3ce35da298a6f622abafbe395108cda1"
    )
    assert evidence["helper_exit_status"] == 0
    assert evidence["checksum_entries"] == evidence["checksum_passed"] == 11
    assert evidence["checksum_failed"] == 0
    assert evidence["checksum_manifest_sha256"] == (
        "62bdd3cbb0acb9ba2bfb62b0ca920d730fa0b7f420240b3fa4a1bdc5bc4f2731"
    )
    cleanup = record["cleanup"]
    assert cleanup["home_test_directory_removed"] is True
    assert cleanup["scratch_test_directory_removed"] is True


def test_v022_v17_real_nfs_record_preserves_scope():
    scope = _record()["scope"]
    for key in (
        "solver_equations_changed",
        "numerical_algorithms_changed",
        "checkpoint_bytes_changed",
        "checkpoint_schema_changed",
        "functional_api_version_changed",
        "provider_identity_changed",
        "production_default_changed",
        "pssolver_control_modified",
        "nematics3d_modified",
        "h100_rerun_required",
    ):
        assert scope[key] is False


def test_v022_v17_real_nfs_record_stops_before_release_actions():
    authorization = _record()["authorization"]
    assert authorization["eligible_for_release_integration_review"] is True
    for key in (
        "push_authorized",
        "tag_authorized",
        "github_release_authorized",
        "pypi_authorized",
        "additional_hpcc_write_authorized",
        "slurm_or_h100_authorized",
    ):
        assert authorization[key] is False
