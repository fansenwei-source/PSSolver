"""Source-freeze contracts for PSSolver 0.2.0rc6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver
from pssolver import capability_catalog
from pssolver.functional.api import functional_protocol_provenance


ROOT = Path(__file__).resolve().parents[1]
RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_0_rc6.json"
FREEZE_RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_source_freeze.json"


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_development_version_and_public_provenance_agree():
    development = _json(ROOT / "notes/pssolver_v0_3_0_dev.json")
    assert development["base_release"] == "v0.2.0"
    assert pssolver.__version__ == development["version"] == "0.3.0.dev0"
    assert functional_protocol_provenance()["package"] == {
        "name": "pssolver",
        "version": "0.3.0.dev0",
    }


def test_rc6_documents_and_machine_record_agree():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "notes/pssolver_v0_2_0_rc6.md").read_text(
        encoding="utf-8"
    )
    record = _json(RELEASE_RECORD)

    assert "PSSolver 0.2.2" in readme
    assert "## 0.2.2 — 2026-10-20" in changelog
    assert "## 0.2.1 — 2026-10-20" in changelog
    assert "## 0.2.0rc6" in changelog
    assert "## 0.2.0 — 2026-10-19" in changelog
    assert "PSSolver 0.2.0rc6 release candidate" in release_notes
    assert record["release"] == "0.2.0rc6"
    assert record["status"] == "source_freeze_preparation"
    assert record["source"]["freeze_commit"] == (
        "90ef40f328347c71d8baabeb187e18cd31e4b6bc"
    )
    assert record["qualification"]["source_freeze_complete"] is True
    assert record["qualification"]["single_h100_cumulative"] == "not_authorized"


def test_rc6_capability_counts_match_live_catalog():
    expected = _json(RELEASE_RECORD)["public_contract"]
    catalog = capability_catalog()

    assert len(catalog.models) == expected["models"]
    assert len(catalog.geometries) == expected["geometries"]
    assert len(catalog.boundary_policies) == expected["boundary_policies"]
    assert len(catalog.qualified_combinations) == expected["qualified_combinations"]
    assert sum(
        len(combination.runtime_paths)
        for combination in catalog.qualified_combinations
    ) == expected["runtime_paths"]


def test_rc6_provider_work_evidence_is_complete_and_content_addressed():
    record = _json(RELEASE_RECORD)
    assert set(record["completed_work"]) == {
        *(f"RC6-{index}" for index in range(1, 8)),
    }
    for relative, expected in record["provider_work_evidence"].items():
        assert _sha256(ROOT / relative) == expected


def test_rc6_independent_consumer_evidence_is_precisely_bounded():
    evidence = _json(RELEASE_RECORD)["independent_consumer_evidence"]
    assert evidence == {
        "repository": "PSSolver-Control",
        "branch": "fix/pssolver-v0.2.0rc6-guard-order",
        "qualified_source_commit": "79815cb77cf7f542a0da4d303e20c45a929876df",
        "record_commit": "208edd03e6b26a0e788c07d295fddd7a67237189",
        "record_path": (
            "docs/pssolver_v0_2_0rc6_periodic_guard_order_regression.json"
        ),
        "record_sha256": (
            "aa02a1332a6f96ccf769f08e51bfadf3fa7769453d6a040c98d0dede24816c60"
        ),
        "runtime_changed": False,
        "provider_identity_review_pending_after_freeze": True,
    }


def test_rc6_freeze_preserves_public_compatibility_boundaries():
    record = _json(RELEASE_RECORD)
    contract = record["public_contract"]
    assert contract["functional_api_version"] == "1.0"
    assert contract["functional_batch_sizes"] == [1]
    assert contract["production_default_changed"] is False
    assert contract["compiled_runtime_promoted"] is False
    assert contract["checkpoint_schema_changed_by_freeze"] is False
    assert contract["lifting_reopened"] is False

    assert all(value is False for value in record["release_boundary"].values())


def test_rc6_freeze_plan_is_fail_closed_and_h100_is_not_authorized():
    record = _json(FREEZE_RECORD)
    assert record["phase"] == "RC6-8"
    assert record["git_identity"]["freeze_commit"] == (
        "90ef40f328347c71d8baabeb187e18cd31e4b6bc"
    )
    assert record["classification"] == (
        "PASS_RC6_SOURCE_FREEZE_AND_FROZEN_SOURCE_CPU_GATE"
    )
    assert record["authorization"] == {
        "source_freeze_prepared": True,
        "frozen_source_cpu_passed": True,
        "eligible_for_installed_wheel_cpu": True,
        "cumulative_qualification_complete": False,
        "eligible_for_push": False,
        "eligible_for_h100_submission": False,
        "h100_submission_count": 0,
        "automatic_retry": False,
    }
    assert record["cpu_gate"] == {
        "python": "3.12.13",
        "pytorch": "2.6.0",
        "cuda_visible_devices": "",
        "pytest_cache_disabled": True,
        "bytecode_writes_disabled": True,
        "passed": 3936,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 7,
        "subtests_passed": 8,
        "deselected_scope": "the seven pre-existing CUDA-only node IDs",
    }
    assert record["gate_order"] == [
        "frozen_source_full_cpu",
        "frozen_installed_wheel_cpu",
        "released_checkpoint_compatibility_and_tamper",
        "provider_identity_impact_review",
        "pssolver_control_migration_and_installed_wheel_qualification_if_required",
        "single_h100_cumulative_qualification",
        "independent_post_release_verification",
    ]


def test_rc6_frozen_hashes_only_change_at_the_final_release_boundary():
    record = _json(FREEZE_RECORD)
    for relative, expected in record["new_sources"].items():
        if relative == "tests/test_release_v0_2_rc6.py":
            assert _sha256(ROOT / relative) != expected
        else:
            assert _sha256(ROOT / relative) == expected
    allowed_final_release_successors = {
        "CHANGELOG.md",
        "README.md",
        "pssolver/_version.py",
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
        "tests/test_release_v0_1.py",
    }
    for section in ("versioned_sources", "historical_successor_sources"):
        for relative, identity in record[section].items():
            current = _sha256(ROOT / relative)
            if relative in allowed_final_release_successors:
                assert current != identity["current_sha256"]
            else:
                assert current == identity["current_sha256"]


def test_rc6_release_documents_are_in_source_manifest():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for relative in (
        "notes/pssolver_v0_2_0_rc6.md",
        "notes/pssolver_v0_2_0_rc6.json",
    ):
        assert (ROOT / relative).is_file()
        assert f"include {relative}" in manifest
