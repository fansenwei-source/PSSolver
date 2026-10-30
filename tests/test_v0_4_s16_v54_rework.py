"""Forward contract for the S-16 V-54 semantic-ratchet rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_79_s16_v54_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_79_binds_the_v54_implementation_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 79
    assert record["classification"] == "V54_CLAIMED_FIXED_PENDING_R0046"
    assert record["git_identity"] == {
        "implementation_parent": "bbd9709059d41cf59af40638c65dd3ab9d15242a",
        "implementation": "9cf4b34c37d5fa266ad1900eae09f143c5c98b63",
        "record_commit_is_qualified_source": False,
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_v54_is_claimed_at_the_exact_implementation() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert issues["V-54"]["status"] in {"claimed_fixed", "verified"}
    assert issues["V-54"]["fixed_in"] == (
        "9cf4b34c37d5fa266ad1900eae09f143c5c98b63"
    )
    assert issues["V-54"]["target_release"] == "0.4 S-16"


def test_v54_rework_changes_no_runtime_or_gate() -> None:
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
