"""Evidence-only import contract for the R-0019 S-08 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_24_r0019_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0019_import_binds_the_exact_rework_result() -> None:
    record = _read(RECORD)
    assert record["classification"] == "REWORK_REQUIRED_R0019_S08_REVIEW_IMPORTED"
    assert record["git_identity"]["reviewed_candidate"] == (
        "4fbe33b004b31448618bf36fa0430d38da51c460"
    )
    result = record["sync_result"]
    assert result["id"] == "R-0019"
    assert result["sha256"] == (
        "1432f7437b5eb9654b11c2292b441cd19449b31ac362172a7fcccbd4af71fa82"
    )
    assert result["classification"] == "REWORK_REQUIRED"
    assert result["rework_claims"] == ["S08-RC2-ODD-AXIS-EXACT"]
    assert result["new_findings"] == ["V-31", "V-32"]
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 34
    assert archive["checksums_sha256"] == (
        "f2cc238a6ef4599fac62252e34a1954552f487831b610b5467b7f4221b297681"
    )
    assert archive["complete_present"] is True


def test_r0019_registers_findings_without_advancing_original_claims() -> None:
    imported = _read(RECORD)["registry_import"]
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert imported == {
        "kept_claimed_fixed": ["A2-25", "N5"],
        "registered": ["V-31", "V-32"],
        "s08_complete": False,
        "eligible_for_push_draft": False,
        "next_request": "R-0020",
    }
    assert all(issues[key]["status"] == "verified" for key in imported["kept_claimed_fixed"])
    assert all(issues[key]["verified_by"] == "R-0021" for key in imported["kept_claimed_fixed"])
    # R-0019 records the initial registered state.  The S-08 rework successor
    # advances both findings through R-0021 verification.
    assert all(
        issues[key]["status"] == "verified"
        and issues[key]["fixed_in"]
        == "e2343d19301de146ca632a1a12f7c9659e598f73"
        and issues[key]["verified_by"] == "R-0021"
        for key in imported["registered"]
    )


def test_r0019_import_is_a_forward_only_successor_layer() -> None:
    for relative, registration in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, registration["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_r0019_import_changes_no_runtime_or_external_state() -> None:
    scope = _read(RECORD)["scope"]
    assert scope["evidence_only"] is True
    assert all(value is False for key, value in scope.items() if key != "evidence_only")
