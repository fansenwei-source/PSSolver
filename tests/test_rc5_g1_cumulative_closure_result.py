"""Immutable repository record for the rc5 G-1 cumulative closure."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_cumulative_closure_v17.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_g1_cumulative_result_binds_runtime_and_tooling_roles():
    record = _record()
    assert record["classification"] == (
        "PASS_RC5_G1_CUMULATIVE_CLOSURE_WITH_BALANCED_CROSSOVER_PERFORMANCE_QUALIFICATION"
    )
    assert record["source_identity"] == {
        "branch": "fix/v0.2.0rc5",
        "runtime_candidate_commit": "dabeb9a487a82700ddd364c4233941a175808a3a",
        "cumulative_analyzer_commit": "8a243fa286657361ca5021cf47c39d81e87964c6",
        "record_commit_is_qualified_source": False,
    }


def test_g1_cumulative_result_combines_exactly_twelve_plus_four_cells():
    evidence = _record()["cumulative_evidence"]
    assert evidence["performance_memory_cell_count"] == 16
    assert evidence["reused_v6_cell_count"] == 12
    assert evidence["requalified_v16_cell_count"] == 4
    assert {tuple(cell) for cell in evidence["focused_cells"]} == {
        ("channel_compiled_v2", "channel_R128x20x20"),
        ("channel_functional", "channel_R512x40x40"),
        ("periodic_functional", "periodic_R128"),
        ("plane_compiled_v2", "plane_R128"),
    }


def test_g1_cumulative_result_binds_all_five_immutable_archives():
    assert _record()["archive_identities"] == {
        "cpu": {
            "entries": 1712,
            "checksums_sha256": "6a10414a20d3c5f7c12e21e98afb594fe3008be3010388355a16e6fdb9fbe832",
        },
        "checkpoint": {
            "entries": 123,
            "checksums_sha256": "6bc84435b3dab444156a87ba7ddc5981f93989c38098dfa3f9e56d8dcff9d627",
        },
        "v5": {
            "entries": 26,
            "checksums_sha256": "049f7422b62618416745ca665f82ab2370f6a5f2e662ce296f332f461e3b7f71",
        },
        "v6": {
            "entries": 15,
            "checksums_sha256": "74ef98002f8bd6b37f0d9e6cc001d17d9653cd6cb274afbd726273bd02eaeac2",
        },
        "v16": {
            "entries": 1814,
            "checksums_sha256": "da50cbda196ace1e7a60c6974dfea40533575493096ed1971a91ef959e002de0",
        },
    }


def test_g1_cumulative_result_excludes_only_the_adjudicated_harness_attempt():
    cpu = _record()["cpu_adjudication"]
    assert cpu["frozen_cpu_gate_remains_authoritative"] is True
    assert cpu["additional_hpcc_full_suite"] == {
        "passed": 3833,
        "failed": 4,
        "skipped": 1,
        "deselected": 7,
        "subtests_passed": 8,
        "used_as_final_qualification_evidence": False,
        "classification": "EXCLUDED_SPLIT_HOST_AND_HISTORICAL_AUDIT_HARNESS_ATTEMPT",
    }
    assert cpu["corrected_local_record_tests"] == "21 passed, 0 failed"
    assert cpu["hpcc_portable_record_tests"].startswith("19 passed")
    for key in ("runtime_failure", "numerical_failure", "scientific_failure", "h100_failure"):
        assert cpu[key] is False


def test_g1_cumulative_result_binds_complete_external_archive():
    archive = _record()["closure_archive"]
    assert archive["complete_present"] is True
    assert archive["checksum_entries"] == 30
    assert archive["checksum_passed"] == 30
    assert archive["checksum_failed"] == 0
    assert archive["checksums_sha256"] == (
        "8c586ae80ddad9a49083116287b40cfff4a9ad630090aad3c2b3bd44a33dd4ae"
    )


def test_g1_cumulative_result_authorizes_review_but_not_release_actions():
    record = _record()
    assert record["scope"]["new_h100_submission_count"] == 0
    assert record["scope"]["production_source_modified"] is False
    assert record["scope"]["production_default_changed"] is False
    assert record["authorization"] == {
        "qualification_complete": True,
        "g1_complete": True,
        "eligible_for_release_integration_review": True,
        "post_release_verification_required": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "push_executed": False,
        "tag_created": False,
        "release_created": False,
        "pypi_published": False,
        "production_default_changed": False,
    }
