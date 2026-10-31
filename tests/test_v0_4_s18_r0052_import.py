"""Forward contract for importing the R-0052 S-18 design review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_91_r0052_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_91_binds_r0052_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 91
    assert record["classification"] == "R0052_REWORK_REQUIRED_IMPORTED"
    assert record["review"]["request"] == "R-0052"
    assert record["review"]["classification"] == "REWORK_REQUIRED"
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0052_registers_v60_without_implementing_s18() -> None:
    record = _read(RECORD)
    assert record["new_findings"] == {
        "V-60": {
            "severity": "low",
            "status": "registered",
            "target_release": "0.4 S-18",
            "title": "S-18 design batch scopes are incomplete",
        }
    }
    issue = next(item for item in _read(REGISTRY)["issues"] if item["id"] == "V-60")
    assert issue["status"] in {"registered", "claimed_fixed", "verified"}
    assert issue["source_report"] == "PSSolver_sync/results/R-0052.result.json"
    assert issue["target_release"] == "0.4 S-18"


def test_r0052_import_changes_no_runtime_or_authorization() -> None:
    record = _read(RECORD)
    assert all(value is False for value in record["scope"].values())
    assert record["authorization"] == {
        "local_registration": True,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
