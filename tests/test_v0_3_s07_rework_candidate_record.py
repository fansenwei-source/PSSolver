"""Bindings for the R-0017 S-07 rework candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_s03_successor_19_s07_rework_candidate.json"
)
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
IMPLEMENTATION = "8aceaab446e57a13866090ec06ecdf30e33ef490"
CLAIMS = {"A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30", "V-26", "V-30"}


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s07_rework_binds_the_runtime_and_v30_contract() -> None:
    record = _read(RECORD)
    assert record["git_identity"]["implementation_commit"] == IMPLEMENTATION
    assert record["roadmap"]["plan_step"] == "S-07"
    assert set(record["roadmap"]["claimed_issue_ids"]) == CLAIMS
    contract = record["v30_contract"]
    assert contract["compatibility_reexport_is_same_object"] is True
    assert contract["retained_modules_depend_on_excluded_control"] is False
    assert contract["installed_module_import_inventory_required"] is True
    assert set(contract["channel_runtime_builds"]) == {"legacy", "compiled_v2"}
    assert set(contract["public_dry_run_families"]) == {
        "plane", "periodic", "legacy_channel", "complete_stress_channel"
    }
    assert set(contract["public_compilation_fields"]) == {
        "schema_version", "compilation_identity_sha256", "source_simulation_sha256",
        "application", "application_simulation_sha256", "application_request_sha256",
        "normalization", "fallback_allowed"
    }
    for relative, expected in record["runtime_sources"].items():
        assert _sha256(ROOT / relative) == expected


def test_s07_rework_successors_are_forward_only() -> None:
    for relative, registration in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, registration["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_s07_rework_is_later_verified_by_r0018() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert all(issues[issue_id]["status"] == "verified" for issue_id in CLAIMS)
    assert all(issues[issue_id]["verified_by"] == "R-0018" for issue_id in CLAIMS)
    assert issues["V-30"]["fixed_in"] == IMPLEMENTATION
    assert _read(RECORD)["validation"]["independent_review_pending"] is True


def test_s07_rework_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["obs4_hardening"]["logging_filter_failure_cannot_mask_successful_publication"] is True
    scope = record["scope"]
    assert scope["checkpoint_schema_changed"] is False
    assert scope["provider_identity_changed"] is False
    assert scope["numerical_algorithm_changed"] is False
    assert scope["skip_or_xfail_added"] is False
    assert scope["production_default_changed"] is False
    assert scope["push_executed"] is False
    assert scope["HPCC_executed"] is False
    assert scope["GPU_executed"] is False
    assert scope["H100_executed"] is False
