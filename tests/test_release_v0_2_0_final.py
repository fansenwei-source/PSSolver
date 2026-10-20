"""Minimal final-release delta from the qualified v0.2.0rc6 source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver
from pssolver.functional.api import functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/pssolver_v0_2_0.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_v020_has_the_final_public_version():
    assert pssolver.__version__ == "0.2.0"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.2.0",
    }


def test_v020_changelog_records_the_rc6_source_boundary():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## 0.2.0 — 2026-10-19" in changelog
    assert "qualified as\n`v0.2.0rc6`" in changelog
    assert "it does not change the\nsolver implementation" in changelog
    assert "functional\nAPI version 1.0" in changelog
    assert "production defaults" in changelog


def test_v020_documents_nfs_checkpoint_limit_and_patch_policy():
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "### Known limitation" in changelog
    assert "cannot publish a\n  checkpoint directly" in changelog
    assert "node-local `/tmp`" in changelog
    assert "operational workaround, not a PSSolver public API" in changelog
    assert "### Maintenance policy" in changelog
    assert "unavailable on a supported target cluster" in changelog
    assert "NFS-compatible no-clobber\n  checkpoint publication in v0.2.1" in changelog


def test_v020_readme_uses_the_final_version():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "PSSolver 0.2.0 is" in readme
    assert "qualified executable combinations in 0.2.0 are" in readme
    assert "0.2.0rc6 is" not in readme


def test_v020_machine_record_limits_the_release_delta():
    record = _record()
    assert record["release"] == "0.2.0"
    assert record["classification"] == (
        "PASS_V0_2_0_RELEASE_SOURCE_PREPARATION_PENDING_REMOTE_AUTHORIZATION"
    )
    source = record["source"]
    assert source["qualified_release_candidate_commit"] == (
        "fd7323d5e9ef87c537ed07476f8bd50b982af443"
    )
    for key in (
        "solver_implementation_changed",
        "numerical_algorithm_changed",
        "checkpoint_schema_changed",
        "functional_api_version_changed",
        "adapter_version_changed",
        "production_default_changed",
    ):
        assert source[key] is False


def test_v020_successor_record_matches_every_registered_source():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "CHANGELOG.md",
        "README.md",
        "pssolver/_version.py",
        "tests/test_release_v0_1.py",
        "tests/test_release_v0_2_rc6.py",
        "tests/test_rc5_g1_checkpoint_glibc_compatibility_record.py",
        "tests/test_rc5_g1_historical_audit_portability.py",
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
        "tests/test_rc5_w1_documentation.py",
        "tests/test_rc6_1_plane_diagnostic_error_record.py",
        "tests/test_rc6_2_pressure_floor_record.py",
        "tests/test_rc6_3_compile_preflight_record.py",
        "tests/test_rc6_4_channel_failure_record.py",
        "tests/test_rc6_5_checkpoint_control_flow_record.py",
        "tests/test_rc6_6_restart_output_atomicity_record.py",
    }
    for relative, identity in successors.items():
        assert identity["historical_sha256"] != identity["current_sha256"]
        assert _sha256(ROOT / relative) == identity["current_sha256"]
        assert identity["reason"]


def test_v020_stops_before_remote_release_actions():
    authorization = _record()["authorization"]
    assert authorization["local_preparation_authorized"] is True
    for key in (
        "push_authorized",
        "tag_authorized",
        "github_release_authorized",
        "pypi_publication_authorized",
        "h100_rerun_authorized",
    ):
        assert authorization[key] is False
