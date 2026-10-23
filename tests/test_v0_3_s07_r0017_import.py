"""Evidence-only import contract for the R-0017 S-07 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_18_r0017_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
CLAIMS = {"A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30", "V-26"}


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0017_import_binds_the_exact_independent_review() -> None:
    record = _read(RECORD)
    assert record["classification"] == "REWORK_REQUIRED_R0017_S07_REVIEW_IMPORTED"
    assert record["git_identity"]["reviewed_candidate"] == (
        "2de78bc4becdc8d9b279b25d0a533306dc302149"
    )
    result = record["sync_result"]
    assert result["id"] == "R-0017"
    assert result["sha256"] == (
        "459b2a5e0f1ea580210242134323c30b72afb9a25684c712bf0574dfdec2f6db"
    )
    assert result["classification"] == "REWORK_REQUIRED"
    assert result["new_findings"] == ["V-30"]
    archive = record["independent_archive"]
    assert archive["complete_present"] is True
    assert archive["checksums_entries"] == 35


def test_r0017_registry_import_is_exact_and_does_not_close_s07() -> None:
    record = _read(RECORD)["registry_import"]
    assert set(record["kept_claimed_fixed"]) == CLAIMS
    assert record["registered"] == ["V-30"]
    assert record["s07_complete"] is False
    assert record["eligible_for_push_draft"] is False
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert {key for key, item in issues.items() if item["status"] == "claimed_fixed"} == CLAIMS
    assert issues["V-30"]["status"] == "registered"
    assert issues["V-30"]["severity"] == "high"


def test_r0017_import_is_a_forward_only_successor_layer() -> None:
    for relative, registration in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, registration["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0017_import_changes_no_runtime_or_external_state() -> None:
    scope = _read(RECORD)["scope"]
    assert scope["evidence_only"] is True
    assert all(value is False for key, value in scope.items() if key != "evidence_only")
