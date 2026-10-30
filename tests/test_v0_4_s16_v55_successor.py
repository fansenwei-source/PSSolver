"""Forward record contract for the S-16 V-55 rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_81_s16_v55_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_81_binds_the_v55_fix_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 81
    assert record["classification"] == "V55_CLAIMED_FIXED_PENDING_R0047"
    assert record["git_identity"] == {
        "implementation_parent": "da755a2b23d0e105f3735d12b20dbad669707eb1",
        "implementation": "625c3ca6b2c64bb674d43430d39ec3409f78c736",
        "record_commit_is_qualified_source": False,
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_v55_is_claimed_at_the_exact_implementation() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert issues["V-55"]["status"] in {"claimed_fixed", "verified"}
    assert issues["V-55"]["fixed_in"] == (
        "625c3ca6b2c64bb674d43430d39ec3409f78c736"
    )
    assert issues["V-55"]["target_release"] == "0.4 S-16"


def test_v55_rework_changes_no_runtime_or_gate() -> None:
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
