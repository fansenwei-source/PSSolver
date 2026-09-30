"""Record contracts for the P9.8.4 local cumulative closure."""

from __future__ import annotations

import json
from pathlib import Path

import pssolver.functional.api as stable


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_9_p984_local_cumulative_closure.json"
)
ARCHIVE = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "build_verbatim_archive_pdf.py"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_p984_binds_clean_source_archives_and_installed_wheels():
    record = _record()
    assert record["phase"] == "P9.8.4"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_P9_8_4_LOCAL_CUMULATIVE_CPU_INSTALLED_WHEEL_CLOSURE"
    )
    for section in ("provider", "consumer"):
        assert len(record[section]["qualification_source_commit"]) == 40
        assert len(record[section]["source_archive_sha256"]) == 64
        assert len(record[section]["wheel_sha256"]) == 64
    assert record["installed_identity"] == {
        "stable_module": "pssolver.functional.api",
        "functional_api_version": stable.FUNCTIONAL_API_VERSION,
        "source_shadow_import": False,
        "provider_console_help": "pass",
        "periodic_consumer": "pass",
        "channel_consumer": "pass",
    }


def test_p984_records_cumulative_cpu_restart_gradient_and_boundary_gates():
    record = _record()
    assert record["provider"]["source_cpu_result"] == (
        "2607 passed, 8 subtests passed"
    )
    assert record["consumer"]["source_cpu_result"] == (
        "235 passed, 1 skipped"
    )
    assert record["installed_cpu_evidence"] == {
        "result": "92 passed",
        "checkpoint_read": "pass",
        "replay_and_restart": "pass",
        "gradient": "pass",
        "negative_gates": "pass",
        "import_boundary": "pass",
    }
    audit = record["dependency_audit"]
    assert audit["provider_and_consumer_declared_requirements"] == "pass"
    assert audit["global_pip_check"] == (
        "pre_existing_unrelated_nematics3d_numpy_pin_conflict"
    )
    assert audit["shared_environment_modified"] is False


def test_p984_authorizes_only_p985_planning():
    assert _record()["authorization"] == {
        "p9_8_4_complete": True,
        "p9_8_5_planning_eligible": True,
        "h100_authorized": False,
        "h100_executed": False,
        "production_default_changed": False,
        "larger_batch_authorized": False,
        "optimizer_campaign_authorized": False,
        "scientific_control_result": False,
        "phase_9_complete": False,
        "verbatim_pdf_regenerated": False,
    }


def test_p984_sources_are_classified_for_later_archive_regeneration():
    source = ARCHIVE.read_text(encoding="utf-8")
    assert '"phase_9_p984_local_cumulative_closure.md"' in source
    assert '"phase_9_p984_local_cumulative_closure.json"' in source
