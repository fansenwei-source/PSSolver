"""Local release-source preparation contract for PSSolver v0.2.2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor
import subprocess

import pssolver
from pssolver.functional.api import functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/pssolver_v0_2_2.json"
ISSUES = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
DEV_INTEGRATION_RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_n1_v17_checkpoint_integration.json"
)
S01_RECORD = ROOT / "notes/PSSolver_v0_3_0_s01_registry_migration.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v022_has_a_registered_v03_development_successor() -> None:
    assert pssolver.__version__ == "0.3.0.dev0"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.3.0.dev0",
    }
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "PSSolver 0.2.2 is" in readme
    assert "qualified executable combinations in 0.2.2 are" in readme
    assert "## 0.2.2 — 2026-10-20" in changelog
    assert "only the verified V-17" in changelog
    integration = _read(DEV_INTEGRATION_RECORD)
    assert integration["source"]["v0_2_2_release_source"] == (
        "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
    )


def test_v022_record_binds_the_exact_linear_source_chain() -> None:
    source = _read(RECORD)["source"]
    assert source["base_release"] == "v0.2.1"
    assert source["base_release_commit"] == (
        "ca9c6729670f07639e7877881c777d1653f4a592"
    )
    assert source["release_preparation_parent"] == (
        "701991a0af6cab1cf0952f76e16c430f2c2613ac"
    )
    assert source["v17_implementation_commit"] == (
        "611d167d1b2c2d634222f608f4c9cd4dc671636d"
    )
    assert source["v17_candidate_record_commit"] == (
        "6bd8e491dde4a72dd1dc29035c2310d565990070"
    )


def test_v022_record_binds_candidate_and_independent_review() -> None:
    qualification = _read(RECORD)["qualification"]
    for key in ("v17_candidate_record", "r0002_import_record"):
        assert _sha256(ROOT / qualification[key]) == qualification[f"{key}_sha256"]
    assert qualification["sync_result_sha256"] == (
        "8837d79a34997a762cca26f967df3396ea87b5208afc39a744fe2c6f9f32c93d"
    )
    assert qualification["independent_classification"] == "PASS"
    assert qualification["independent_archive_checksums"] == 32253
    assert qualification["independent_archive_complete"] is True
    assert qualification["release_source_cpu_gate"] == {
        "passed": 4179,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 7,
        "warnings": 6,
        "subtests_passed": 8,
        "cuda_visible_devices": "",
        "pytest_cache_disabled": True,
        "bytecode_writes_disabled": True,
    }
    assert qualification["h100_rerun_required"] is False


def test_v022_release_delta_is_strictly_bounded() -> None:
    source = _read(RECORD)["source"]
    assert source["runtime_delta_from_v0_2_1"] == [
        (
            "pssolver/io/checkpoint.py: follow the immediate destination-parent "
            "symlink for filesystem identity"
        ),
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
        "v0_2_1_tag_rewritten",
        "v0_2_1_assets_replaced",
    ):
        assert source[key] is False


def test_v022_forward_successors_match_every_changed_historical_source() -> None:
    record = _read(RECORD)
    parent = record["source"]["release_preparation_parent"]
    expected_paths = {
        "CHANGELOG.md",
        "README.md",
        "pssolver/_version.py",
        "tests/test_release_v0_1.py",
        "tests/test_release_v0_2_rc6.py",
        "tests/test_release_v0_2_0_final.py",
        "tests/test_release_v0_2_1.py",
        "tests/test_v0_2_2_v17_checkpoint_symlink_parent_record.py",
        "tests/test_v0_2_2_v17_r0002_import.py",
        "tests/test_rc5_w1_documentation.py",
        "tests/test_rc6_2_pressure_floor_record.py",
        "tests/test_rc6_3_compile_preflight_record.py",
        "tests/test_rc6_4_channel_failure_record.py",
        "tests/test_rc6_5_checkpoint_control_flow_record.py",
        "tests/test_rc6_6_restart_output_atomicity_record.py",
        "tests/test_v0_2_1_n1_checkpoint_nfs_record.py",
        "tests/test_rc5_g1_checkpoint_glibc_compatibility_record.py",
        "tests/test_rc5_legacy_functional_checkpoint_record.py",
        "tests/test_rc5_r510_channel_checkpoint_observation_record.py",
        "tests/test_rc5_r511_metadata_strict_record.py",
        "tests/test_rc5_r512_compiled_simulation_integrity_record.py",
        "tests/test_rc5_r513_optional_hygiene_record.py",
        "tests/test_rc5_r52_finite_workflow_record.py",
        "tests/test_rc5_r53_channel_checkpoint_record.py",
        "tests/test_rc5_r55_tf32_state_restore.py",
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_r57_device_normalization_record.py",
        "tests/test_rc5_r58_checkpoint_integrity_record.py",
        "tests/test_rc5_r59_pressure_postcondition_record.py",
        "tests/test_rc6_1_plane_diagnostic_error_record.py",
        "tests/test_rc5_g1_historical_audit_portability.py",
    }
    successors = record["historical_successor_sources"]
    development_successors = _read(DEV_INTEGRATION_RECORD)[
        "historical_successor_sources"
    ]
    s01_successors = _read(S01_RECORD)["historical_successor_sources"]
    assert set(successors) == expected_paths
    for relative, identity in successors.items():
        historical = subprocess.run(
            ["git", "show", f"{parent}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == identity[
            "historical_sha256"
        ]
        expected = identity["current_sha256"]
        successor = development_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == expected:
            expected = successor["current_sha256"]
        successor = s01_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == expected:
            expected = successor["current_sha256"]
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, expected
        )
        assert identity["reason"]


def test_v022_keeps_new_low_severity_findings_deferred() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(ISSUES)["issues"]}
    assert record["known_deferred_findings"] == {
        "V-22": "registered for later 0.2.x work",
        "V-23": "registered for later 0.2.x work",
    }
    # The release record remains historical.  Later successor layers may advance
    # these issues without rewriting the v0.2.2 evidence.
    expected_status = {"V-22": "verified", "V-23": "verified"}
    for issue_id in record["known_deferred_findings"]:
        assert issues[issue_id]["status"] == expected_status[issue_id]
        assert issues[issue_id]["severity"] == "low"
    assert issues["V-23"]["verified_by"] == "R-0008"
    assert issues["V-22"]["fixed_in"] == (
        "468e06710e1b4881a557f0228dcb16be38791094"
    )
    assert issues["V-22"]["verified_by"] == "R-0012"


def test_v022_remote_and_hpcc_actions_remain_unauthorized() -> None:
    record = _read(RECORD)
    assert record["release_artifacts"] == {
        "source_archive": "pending_clean_build",
        "sdist": "pending_clean_build",
        "wheel": "pending_clean_build",
        "pypi_published": False,
    }
    authorization = record["authorization"]
    assert authorization["local_release_preparation_authorized"] is True
    for key in (
        "push_authorized",
        "tag_authorized",
        "github_release_authorized",
        "pypi_authorized",
        "hpcc_test_directory_authorized",
        "slurm_or_h100_authorized",
    ):
        assert authorization[key] is False
