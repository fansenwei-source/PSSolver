"""Fail-closed tests for the rc5 G-1 CPU-only cumulative closure."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import benchmarks.analyze_rc5_g1_cumulative_closure as closure
from benchmarks.verify_rc5_g1_cumulative_archive import main as verify_main


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _archives() -> dict[str, object]:
    return {
        name: {
            "passed": True,
            "entries": expected["entries"],
            "verified": expected["entries"],
            "failed": 0,
            "manifest_sha256": expected["manifest_sha256"],
        }
        for name, expected in closure.EXPECTED_ARCHIVES.items()
    }


def _inputs() -> dict[str, object]:
    focused = [
        {"family": family, "grid": grid, "passed": False}
        for family, grid in sorted(closure.FOCUSED_CELLS)
    ]
    reused = [
        {"family": f"reused_{index}", "grid": f"grid_{index}", "passed": True}
        for index in range(12)
    ]
    v16_groups = [
        {"family": family, "grid": grid, "passed": True}
        for family, grid in sorted(closure.FOCUSED_CELLS)
    ]
    return {
        "cpu_record": {
            "classification": "PASS_RC5_G1_FROZEN_SOURCE_AND_INSTALLED_WHEEL_CPU",
            "source_cpu": {"passed": 3623, "failed": 0},
            "installed_wheel": {
                "applicable_full": {"passed": 3609, "failed": 0, "xfailed": 0},
                "source_shadow_import": False,
                "pip_check": "PASS",
            },
        },
        "checkpoint_record": {
            "classification": "PASS_RC5_G1_RELEASED_CHECKPOINT_MATRIX_WITH_PROVIDER_IDENTITY_MIGRATION_REQUIRED",
            "released_checkpoint_matrix": {
                "complete": True,
                "primary_case_count": 13,
                "external_fixture_crosscheck": {"passed": 10, "failed": 0},
                "rc1_rc3_installed_wheel_tests": {"passed": 20, "failed": 0},
            },
            "provider_identity_impact": {"review_complete": True},
        },
        "v5_summary": {
            "classification": "PASS_RC5_G1_CONSUMER_H100_RECOVERY_V5_STOPPED_AT_CUMULATIVE_ANALYZER_TRANSFORM_WINDOW_CONTRACT",
            "consumer_recovery_complete": True,
            "qualification_complete": False,
            "g1_complete": False,
            "scientific_failure": False,
        },
        "v5_auxiliary": {"schema": "synthetic-validated-elsewhere"},
        "v6_summary": {
            "classification": "STOPPED_RC5_G1_TRANSFORM_WINDOW_RECOVERY_V6_AT_FROZEN_PROFILE_PERFORMANCE_MEMORY_GATE",
            "analysis_only": True,
            "qualification_complete": False,
            "g1_complete": False,
            "source_profiles_modified": False,
            "thresholds_changed": False,
            "new_h100_submission_count": 0,
        },
        "v6_performance": {
            "profile_count": 96,
            "group_count": 16,
            "groups": reused + focused,
            "failed_groups": focused,
        },
        "v6_transform": {
            "passed": True,
            "profile_count": 96,
            "source_profiles_modified": False,
        },
        "v6_profile_sources": {
            "passed": True,
            "verified": 96,
            "source_profiles_modified": False,
        },
        "v16_summary": {
            "classification": "PASS_RC5_G1_BALANCED_CROSSOVER_V16_SINGLE_H100_QUALIFICATION",
            "v16_qualification_complete": True,
            "eligible_for_g1_cumulative_closure": True,
            "g1_complete": False,
            "formal_h100_submission_count": 1,
            "automatic_retry": False,
            "second_job_submitted": False,
            "profile_count": 48,
        },
        "v16_analysis": {
            "classification": "PASS_RC5_G1_BALANCED_CROSSOVER_PERFORMANCE_MEMORY_EVIDENCE",
            "analysis_only": False,
            "formal_requalification_performed": True,
            "proposed_contract_passed": True,
            "groups": v16_groups,
        },
        "v16_validation": {
            "profile_count": 48,
            "checks": {"all_profiles_valid": True, "analyzer_byte_identical": True},
        },
        "v16_slurm": {
            "job_id": "10886152",
            "state": "COMPLETED",
            "exit_code": "0:0",
            "formal_h100_submission_count": 1,
            "automatic_retry": False,
            "second_job_submitted": False,
            "requeue": 0,
            "restarts": 0,
        },
        "archive_verification": _archives(),
        "input_sha256": {"input": _hash("input")},
    }


@pytest.fixture(autouse=True)
def _synthetic_auxiliary(monkeypatch):
    monkeypatch.setattr(closure, "_validate_auxiliary", lambda value, plan: None)


def test_cumulative_closure_combines_exactly_twelve_reused_and_four_requalified_cells():
    result = closure.analyze(**_inputs())
    assert result["schema"] == closure.SCHEMA
    assert result["classification"] == closure.PASS_CLASSIFICATION
    assert result["qualification_complete"] is True
    assert result["g1_complete"] is True
    assert result["new_h100_submission_count"] == 0
    assert result["post_release_verification_required"] is True
    performance = result["evidence"]["performance_memory"]
    assert performance["cell_count"] == 16
    assert performance["reused_v6_cell_count"] == 12
    assert performance["requalified_v16_cell_count"] == 4


def test_cumulative_closure_rejects_historical_reclassification():
    values = _inputs()
    values["v6_summary"]["qualification_complete"] = True
    with pytest.raises(ValueError, match="historical failure was rewritten"):
        closure.analyze(**values)


def test_cumulative_closure_rejects_a_missing_or_failed_focused_cell():
    values = _inputs()
    values["v16_analysis"]["groups"].pop()
    with pytest.raises(ValueError, match="focused cell set differs"):
        closure.analyze(**values)

    values = _inputs()
    values["v16_analysis"]["groups"][0]["passed"] = False
    with pytest.raises(ValueError, match="focused cell failed"):
        closure.analyze(**values)


def test_cumulative_closure_rejects_retry_or_manifest_drift():
    values = _inputs()
    values["v16_slurm"]["automatic_retry"] = True
    with pytest.raises(ValueError, match="retry used"):
        closure.analyze(**values)

    values = _inputs()
    values["archive_verification"]["v16"]["manifest_sha256"] = _hash("wrong")
    with pytest.raises(ValueError, match="v16 manifest identity differs"):
        closure.analyze(**values)


def test_cumulative_closure_rejects_v6_threshold_or_profile_mutation():
    values = _inputs()
    values["v6_summary"]["thresholds_changed"] = True
    with pytest.raises(ValueError, match="thresholds changed"):
        closure.analyze(**values)

    values = _inputs()
    values["v6_profile_sources"]["source_profiles_modified"] = True
    with pytest.raises(ValueError, match="source profiles changed"):
        closure.analyze(**values)


def test_archive_verifier_is_fail_closed(tmp_path: Path):
    payload = tmp_path / "payload.json"
    payload.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = tmp_path / "checksums.sha256"
    manifest.write_text(f"{digest}  payload.json\n", encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    report = closure.verify_archive(
        tmp_path,
        expected_entries=1,
        expected_manifest_sha256=manifest_digest,
    )
    assert report["passed"] is True
    assert report["verified"] == 1

    payload.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        closure.verify_archive(
            tmp_path,
            expected_entries=1,
            expected_manifest_sha256=manifest_digest,
        )


def test_versioned_archive_verifier_rejects_unknown_names(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown archive name"):
        verify_main(
            [
                "--archive",
                "unknown",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )
