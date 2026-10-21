"""Evidence contract for local PSSolver v0.2.2 package preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/pssolver_v0_2_2_package_preparation.json"
SOURCE_RECORD = ROOT / "notes/pssolver_v0_2_2.json"
V17_RECORD = ROOT / "notes/PSSolver_v0_2_2_v17_checkpoint_symlink_parent.json"
R0002_RECORD = ROOT / "notes/PSSolver_v0_2_2_v17_r0002_import.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_v022_package_record_binds_the_exact_release_source():
    record = _record()
    assert record["classification"] == "PASS_V0_2_2_LOCAL_PACKAGE_PREPARATION"
    source = record["source"]
    assert source["release_source_commit"] == (
        "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
    )
    assert source["release_source_parent"] == (
        "701991a0af6cab1cf0952f76e16c430f2c2613ac"
    )
    assert source["record_commit_is_release_source"] is False
    assert source["release_source_record_sha256"] == _sha256(SOURCE_RECORD)
    assert json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))["source"][
        "record_commit_is_release_source"
    ] is True


def test_v022_package_record_preserves_runtime_and_public_identity_scope():
    source = _record()["source"]
    assert source["runtime_delta_from_v0_2_1"] == [
        "pssolver/io/checkpoint.py: follow the immediate destination-parent symlink for filesystem identity",
        "pssolver/_version.py: 0.2.1 -> 0.2.2",
    ]
    for key in (
        "solver_equations_changed",
        "numerical_algorithms_changed",
        "checkpoint_bytes_changed",
        "checkpoint_schema_changed",
        "functional_api_version_changed",
        "provider_identity_changed",
        "production_default_changed",
    ):
        assert source[key] is False


def test_v022_package_record_binds_cpu_and_installed_wheel_gates():
    record = _record()
    cpu = record["cpu_gate"]
    assert cpu["passed"] == 4179
    assert cpu["failed"] == cpu["skipped"] == cpu["xfailed"] == 0
    assert cpu["deselected"] == 7
    assert len(cpu["cuda_only_deselects"]) == 7
    installed = record["installed_wheel_gate"]
    assert installed["system_site_packages"] is False
    assert installed["old_qualification_environment_unchanged"] is True
    assert installed["install_mode"] == "--no-deps"
    assert installed["pip_check"] == "PASS"
    assert installed["pssolver_version"] == "0.2.2"
    assert installed["functional_api_version"] == "1.0"
    assert installed["source_shadow_import"] is False
    assert installed["console_help"] == "PASS"
    assert installed["checkpoint_tests"] == {
        "n1_passed": 13,
        "v17_passed": 3,
        "passed": 16,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert installed["nematics3d_installed"] is False
    assert installed["nematics3d_modified"] is False


def test_v022_package_record_binds_local_artifacts_and_rebuild():
    build = _record()["build"]
    assert build["source_archive"]["sha256"] == (
        "b441ba1a1a943899da81fb294a67c8e2e69e00117aebc2a26af9d72efa5e41db"
    )
    assert build["sdist"]["sha256"] == (
        "ad8237e67478f7328c5152981ea4d888f1df64e1af7227f07ba0a0f070b02b39"
    )
    assert build["wheel"]["sha256"] == (
        "a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e"
    )
    assert build["wheel_member_count"] == 263
    assert build["python_file_count"] == 258
    assert build["sdist_rebuilt_wheel_member_content_identical"] is True
    assert build["wheel_python_content_matches_clean_archive"] is True


def test_v022_package_record_binds_independent_v17_evidence():
    qualification = _record()["qualification"]
    assert qualification["v17_candidate_record_sha256"] == _sha256(V17_RECORD)
    assert qualification["r0002_import_record_sha256"] == _sha256(R0002_RECORD)
    assert qualification["independent_classification"] == "PASS"
    assert qualification["real_nfs_login_node_retest"] == (
        "pending_separate_authorization"
    )
    assert qualification["h100_rerun_required"] is False


def test_v022_package_record_stops_before_external_actions():
    authorization = _record()["authorization"]
    assert authorization["eligible_for_real_nfs_retest"] is True
    assert authorization["eligible_for_release_review_after_real_nfs_retest"] is True
    for key in (
        "push_authorized",
        "tag_authorized",
        "github_release_authorized",
        "pypi_authorized",
        "hpcc_test_directory_authorized",
        "slurm_or_h100_authorized",
    ):
        assert authorization[key] is False
