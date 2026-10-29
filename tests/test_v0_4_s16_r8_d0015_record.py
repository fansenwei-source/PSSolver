"""Forward-only contract for the D-0015 R8 implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_73_s16_r8_d0015.json"
CATALOG = ROOT / "notes/PSSolver_v0_4_0_s16_r8_local_evidence_catalog.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_73_binds_the_d0015_option_c_implementation():
    record = _read(RECORD)
    assert record["successor"] == 73
    assert record["classification"] == "S16_R8_D0015_IMPLEMENTED_PENDING_R0044"
    assert record["decision"] == {
        "id": "D-0015",
        "commit": "3477e9f3ee1d8f51a86300cd7153903d142fa869",
        "sha256": "69eaded6bff3a9ddcea173f1924f3d272199e00893b8fa483936bd3be3af56b6",
        "selection": "option C plus mandatory local verifier",
    }
    assert record["git_identity"]["implementation"] == (
        "d5c604a780f9e60c52f00135f163e207a0a0140a"
    )
    assert record["forward_corrections"]["successor_72"]["immutable"] is True


def test_r8_catalog_is_complete_and_repository_self_contained():
    record = _read(RECORD)
    catalog = _read(CATALOG)
    assert len(catalog["pytest_migrations"]) == 35
    assert len({node.split("::", 1)[0] for node in catalog["pytest_migrations"]}) == 27
    assert len(catalog["checks"]) == 103
    assert catalog["policy"] == {
        "external_content_copied_into_repository": False,
        "historical_records_rewritten": False,
        "missing_or_mismatched_evidence_is_failure": True,
        "pytest_external_path_reads": False,
        "standard_library_only": True,
    }
    assert record["r8"] == {
        "pytest_test_files": 27,
        "pytest_tests": 35,
        "local_verifier_checks": 103,
        "pytest_external_path_reads": False,
        "standard_library_verifier": True,
        "missing_or_mismatched_evidence_fails": True,
        "machine_readable_report": True,
        "crosswalk_complete": True,
        "external_content_copied": False,
        "historical_records_rewritten": False,
    }


def test_successor_73_binds_every_changed_and_introduced_source():
    record = _read(RECORD)
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
        assert identity["historical_sha256"] != identity["current_sha256"]
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_v51_is_claimed_at_the_r8_implementation_source():
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert issues["V-51"]["status"] in {"claimed_fixed", "verified"}
    assert issues["V-51"]["fixed_in"] == (
        "d5c604a780f9e60c52f00135f163e207a0a0140a"
    )
    assert issues["V-51"].get("verified_by") in {None, "R-0044"}


def test_r8_implementation_authorizes_no_external_or_gpu_action():
    record = _read(RECORD)
    assert record["scope"] == {
        "runtime_changed": False,
        "public_interface_changed": False,
        "numerical_algorithm_changed": False,
        "gpu_specific_path_changed": False,
        "checkpoint_schema_changed": False,
        "threshold_changed": False,
        "skip_or_xfail_added": False,
        "production_default_changed": False,
        "historical_record_rewritten": False,
        "archive_or_sync_content_copied": False,
        "archive_or_sync_content_moved_or_deleted": False,
    }
    assert record["authorization"] == {
        "local_commits": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
