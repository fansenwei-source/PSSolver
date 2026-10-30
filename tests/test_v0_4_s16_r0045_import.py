"""Forward-only import contract for the R-0045 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_78_s16_r0045_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_78_binds_r0045_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 78
    assert record["classification"] == "R0045_IMPORTED_REWORK_PENDING_V54"
    assert record["r0045"] == {
        "classification": "REWORK_REQUIRED",
        "result_commit": "19de9af37b4dcf433ff45e138027e78b0ac9be90",
        "result_sha256": (
            "7d4d93166372eb2fbd4b1b76b3a6f274479f42b559a364838a6dc8cb1faa1764"
        ),
        "archive": {
            "path": "/home/fansenwei/pssolver_v0_4_claude_s16_r0045_review_20261029_v1",
            "checksums_entries": 228,
            "checksums_sha256": (
                "6bbf660bead2442dda1a214beaef689fefedde8171af225c4d2bb4a964f67f21"
            ),
            "complete": True,
        },
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0045_import_verifies_v53_and_registers_v54() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert issues["V-53"]["status"] == "verified"
    assert issues["V-53"]["fixed_in"] == (
        "05513a4132b1eb38e9bdff825f98a31b276c2932"
    )
    assert issues["V-53"]["verified_by"] == "R-0045"
    assert issues["V-54"]["status"] in {"registered", "claimed_fixed", "verified"}
    assert issues["V-54"]["source_report"] == (
        "PSSolver_sync/results/R-0045.result.json"
    )
    assert issues["V-54"]["target_release"] == "0.4 S-16"


def test_r0045_import_changes_only_records() -> None:
    record = _read(RECORD)
    assert all(value is False for value in record["scope"].values())
    assert record["authorization"] == {
        "local_commits": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
