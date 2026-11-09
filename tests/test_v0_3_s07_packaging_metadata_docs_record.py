"""Bindings for the S-07 packaging, metadata, and documentation candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / ("notes/PSSolver_v0_3_0_s03_successor_17_s07_packaging_metadata_docs.json")
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
IMPLEMENTATION = "b7cf4ca02095e8b41ed720809880628d2375cfbb"
ISSUES = {"A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30", "V-26"}
V30_IMPLEMENTATION = "8aceaab446e57a13866090ec06ecdf30e33ef490"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s07_registry_claims_are_verified_by_r0018():
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    for issue_id in ISSUES:
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["fixed_in"] == IMPLEMENTATION
        assert issues[issue_id]["verified_by"] == "R-0018"
    assert issues["V-30"]["status"] == "verified"
    assert issues["V-30"]["fixed_in"] == V30_IMPLEMENTATION
    assert issues["V-30"]["verified_by"] == "R-0018"


def test_s07_candidate_authorizes_no_external_action():
    record = _read(RECORD)
    assert record["scope"]["numerical_algorithm_changed"] is False
    assert record["scope"]["skip_or_xfail_added"] is False
    assert record["scope"]["production_default_changed"] is False
    assert record["authorization"]["local_implementation_authorized"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_implementation_authorized"
    )
