"""Bindings for the R-0017 S-07 rework candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
