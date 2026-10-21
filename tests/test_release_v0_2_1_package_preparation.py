"""Evidence contract for local PSSolver v0.2.1 package preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/pssolver_v0_2_1_package_preparation.json"
SOURCE_RECORD = ROOT / "notes/pssolver_v0_2_1.json"
N1_RECORD = ROOT / "notes/PSSolver_v0_2_1_n1_h100_qualification.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_v021_package_record_binds_the_exact_release_source():
    record = _record()
    assert record["classification"] == "PASS_V0_2_1_LOCAL_PACKAGE_PREPARATION"
    source = record["source"]
    assert source["release_source_commit"] == (
        "ca9c6729670f07639e7877881c777d1653f4a592"
    )
    assert source["release_source_parent"] == (
        "98f9a7d8579c23a7b86cf7011ce0a45bb7328193"
    )
    assert source["record_commit_is_release_source"] is False
    assert json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))[
        "source"
    ]["record_commit_is_release_source"] is True


def test_v021_package_record_preserves_runtime_and_public_identity_scope():
    source = _record()["source"]
    assert source["runtime_delta_from_qualified_source"] == [
        "pssolver/_version.py: 0.2.0 -> 0.2.1"
    ]
    for key in (
        "solver_equations_changed",
        "numerical_algorithms_changed",
        "checkpoint_schema_changed",
        "functional_api_version_changed",
        "provider_identity_changed",
        "production_default_changed",
    ):
        assert source[key] is False


def test_v021_package_record_binds_cpu_and_installed_wheel_gates():
    record = _record()
    cpu = record["cpu_gate"]
    assert cpu["passed"] == 4134
    assert cpu["failed"] == cpu["xfailed"] == 0
    assert cpu["skipped"] == 1
    assert cpu["deselected"] == 7
    assert len(cpu["cuda_only_deselects"]) == 7
    installed = record["installed_wheel_gate"]
    assert installed["system_site_packages"] is False
    assert installed["install_mode"] == "--no-deps"
    assert installed["pip_check"] == "PASS"
    assert installed["pssolver_version"] == "0.2.1"
    assert installed["functional_api_version"] == "1.0"
    assert installed["source_shadow_import"] is False
    assert installed["console_help"] == "PASS"
    assert installed["n1_checkpoint_tests"] == {
        "passed": 13,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
    }
    assert installed["nematics3d_installed"] is False
    assert installed["nematics3d_modified"] is False


def test_v021_package_record_binds_local_artifacts():
    build = _record()["build"]
    assert build["source_archive"]["sha256"] == (
        "2541432b9c5f8febdff68e913f005e2b1cc223a8234f0f06c87d2bca60cbc585"
    )
    assert build["sdist"]["sha256"] == (
        "cf795a976494dd552f3db843ba88185f77a0b9510ef3e60d4a9b9582413b6eba"
    )
    assert build["wheel"]["sha256"] == (
        "0f0f1c43a920582d3d707d285fe06502ff1852e6cca24f37872b06669201da8a"
    )
    assert build["wheel_member_count"] == 263
    assert build["python_file_count"] == 258
    assert build["sdist_rebuilt_wheel_member_content_identical"] is True
    assert build["sdist_wheel_python_content_identical"] is True


def test_v021_package_record_reuses_the_completed_h100_gate_without_rerun():
    qualification = _record()["qualification_reuse"]
    assert qualification["n1_cumulative_record_sha256"] == _sha256(N1_RECORD)
    assert qualification["h100_job_id"] == 10902665
    assert qualification["h100_rerun_required"] is False


def test_v021_package_record_stops_before_remote_release_actions():
    authorization = _record()["authorization"]
    assert authorization["eligible_for_release_review"] is True
    for key in (
        "push_authorized",
        "tag_authorized",
        "github_release_authorized",
        "pypi_authorized",
        "additional_h100_authorized",
    ):
        assert authorization[key] is False
