"""Local release-preparation contract for PSSolver v0.2.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver
from pssolver.functional.api import functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/pssolver_v0_2_1.json"
V022_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_2.json"
V022_V17_RECORD = ROOT / (
    "notes/PSSolver_v0_2_2_v17_checkpoint_symlink_parent.json"
)
DEV_INTEGRATION_RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_n1_v17_checkpoint_integration.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_v021_has_a_registered_v03_development_successor():
    assert pssolver.__version__ == "0.3.0.dev0"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.3.0.dev0",
    }
    successor = json.loads(V022_RELEASE_RECORD.read_text(encoding="utf-8"))
    assert successor["source"]["base_release"] == "v0.2.1"
    assert successor["source"]["v0_2_1_tag_rewritten"] is False
    assert successor["source"]["v0_2_1_assets_replaced"] is False


def test_v021_release_documents_describe_the_narrow_patch_scope():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "PSSolver 0.2.2 is" in readme
    assert "qualified executable combinations in 0.2.2 are" in readme
    assert "## 0.2.2 — 2026-10-20" in changelog
    assert "## 0.2.1 — 2026-10-20" in changelog
    assert "only the N-1 checkpoint" in changelog
    assert "Functional API version 1.0" in changelog
    assert "provider\nidentity" in changelog
    assert "production defaults are\nunchanged" in changelog


def test_v021_record_binds_the_exact_qualified_sources():
    record = _record()
    source = record["source"]
    assert record["classification"] == "PASS_V0_2_1_RELEASE_SOURCE_PREPARATION"
    assert source["base_release"] == "v0.2.0"
    assert source["base_release_commit"] == (
        "6d2934e73b543f6652d112733657775333e0f396"
    )
    assert source["qualified_runtime_source"] == (
        "15670eb264a44c6707aeb1695b74e6cdbb34f611"
    )
    assert source["n1_implementation_commit"] == (
        "927dc89aa63bcee9b03248f52662389bc947c47a"
    )
    assert source["cpu_evidence_commit"] == (
        "80cac090dadc1b205a7ecaf655d0bf027caf0b1d"
    )
    assert source["h100_evidence_commit"] == (
        "98f9a7d8579c23a7b86cf7011ce0a45bb7328193"
    )


def test_v021_record_preserves_public_and_numerical_contracts():
    source = _record()["source"]
    for key in (
        "solver_equations_changed",
        "numerical_algorithms_changed",
        "checkpoint_schema_changed",
        "functional_api_version_changed",
        "provider_identity_changed",
        "production_default_changed",
        "v0_2_0_tag_rewritten",
        "v0_2_0_assets_replaced",
    ):
        assert source[key] is False


def test_v021_record_binds_cumulative_cpu_and_h100_qualification():
    qualification = _record()["qualification"]
    assert qualification["implementation_record_sha256"] == _sha256(
        ROOT / qualification["implementation_record"]
    )
    assert qualification["cpu_record_sha256"] == _sha256(
        ROOT / qualification["cpu_record"]
    )
    assert qualification["h100_record_sha256"] == _sha256(
        ROOT / qualification["h100_record"]
    )
    assert qualification["classification"] == (
        "PASS_V0_2_1_N1_CUMULATIVE_QUALIFICATION"
    )
    assert qualification["h100_job_id"] == 10902665
    assert qualification["h100_state"] == "COMPLETED"
    assert qualification["h100_exit_code"] == "0:0"
    assert qualification["h100_rerun_required_for_version_only_release_step"] is False


def test_v021_forward_successors_match_every_changed_historical_source():
    successors = _record()["historical_successor_sources"]
    v17_successors = json.loads(
        V022_V17_RECORD.read_text(encoding="utf-8")
    )["historical_successor_sources"]
    release_successors = json.loads(
        V022_RELEASE_RECORD.read_text(encoding="utf-8")
    )["historical_successor_sources"]
    development_successors = json.loads(
        DEV_INTEGRATION_RECORD.read_text(encoding="utf-8")
    )["historical_successor_sources"]
    assert set(successors) == {
        "CHANGELOG.md",
        "README.md",
        "pssolver/_version.py",
        "tests/test_release_v0_1.py",
        "tests/test_release_v0_2_rc6.py",
        "tests/test_release_v0_2_0_final.py",
        "tests/test_v0_2_1_n1_checkpoint_nfs_record.py",
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
        "tests/test_rc5_g1_checkpoint_glibc_compatibility_record.py",
        "tests/test_rc5_g1_historical_audit_portability.py",
    }
    for relative, identity in successors.items():
        assert identity["historical_sha256"] != identity["current_sha256"]
        expected = identity["current_sha256"]
        successor = v17_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == expected:
            expected = successor["current_sha256"]
        successor = release_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == expected:
            expected = successor["current_sha256"]
        successor = development_successors.get(relative)
        if successor is not None and successor["historical_sha256"] == expected:
            expected = successor["current_sha256"]
        assert _sha256(ROOT / relative) == expected
        assert identity["reason"]


def test_v021_remote_release_actions_remain_unauthorized():
    record = _record()
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
        "h100_rerun_authorized",
    ):
        assert authorization[key] is False
