"""Record and authorization contracts for P9.8.3 consumer migration."""

from __future__ import annotations

import json
from pathlib import Path

import pssolver.functional.api as stable


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_9_p983_consumer_migration.json"
)
ARCHIVE = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "build_verbatim_archive_pdf.py"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_p983_binds_stable_provider_and_two_independent_consumers():
    record = _record()
    assert record["phase"] == "P9.8.3"
    assert record["status"] == "complete"
    provider = record["provider"]
    assert provider["stable_api_baseline_commit"] == (
        "1ebafc22c6ed75ea5ba70a508724eddd8a090ef2"
    )
    assert provider["module"] == "pssolver.functional.api"
    assert provider["construction_protocol_version"] == (
        stable.FUNCTIONAL_API_VERSION
    )
    adapters = record["consumer"]["adapters"]
    assert set(adapters) == {"periodic", "channel"}
    assert {entry["adapter_version"] for entry in adapters.values()} == {2}


def test_p983_preserves_old_evidence_and_provider_owned_checkpoint_readers():
    record = _record()
    history = record["historical_evidence"]
    assert history["records_rewritten"] is False
    assert history["relabelled_as_stable_installed_wheel_evidence"] is False
    checkpoint = record["checkpoint_compatibility"]
    assert checkpoint["new_construction_accepts_legacy_protocol"] is False
    assert checkpoint["stable_checkpoint_read_versions"] == list(
        stable.FUNCTIONAL_CHECKPOINT_READ_API_VERSIONS
    )
    assert checkpoint["compatibility_reader_owned_by"] == "PSSolver"
    assert checkpoint["consumer_schema_repair"] is False
    assert checkpoint["silent_fallback"] is False


def test_p983_authorizes_only_local_p984_closure():
    record = _record()
    assert record["consumer_local_cpu_evidence"]["complete"] == (
        "235 passed, 1 skipped"
    )
    assert record["provider_local_cpu_evidence"]["complete"] == (
        "2607 passed, 8 subtests passed"
    )
    authorization = record["authorization"]
    assert authorization == {
        "p9_8_3_complete": True,
        "p9_8_4_planning_eligible": True,
        "installed_wheel_closure_complete": False,
        "h100_authorized": False,
        "production_default_changed": False,
        "larger_batch_authorized": False,
        "optimizer_campaign_authorized": False,
        "phase_9_complete": False,
        "verbatim_pdf_regenerated": False,
    }


def test_p983_sources_are_classified_for_later_verbatim_archive_regeneration():
    source = ARCHIVE.read_text(encoding="utf-8")
    assert '"phase_9_p983_consumer_migration.md"' in source
    assert '"phase_9_p983_consumer_migration.json"' in source
