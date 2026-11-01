"""Forward binding for the S-18 R-0052 scope rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_92_s18_scope_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_92_binds_the_forward_scope_rework() -> None:
    record = _read(RECORD)
    assert record["successor"] == 92
    assert record["classification"] == "V60_CLAIMED_FIXED_PENDING_R0053"
    assert record["git_identity"]["implementation"] == (
        "f5dbdbef43d1aa16e9c7638093df5c5c5ae58170"
    )
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_v60_is_claimed_at_the_implementation_commit() -> None:
    issue = next(item for item in _read(REGISTRY)["issues"] if item["id"] == "V-60")
    assert issue["status"] in {"claimed_fixed", "verified"}
    assert issue["fixed_in"] == "f5dbdbef43d1aa16e9c7638093df5c5c5ae58170"
    assert issue["target_release"] == "0.4 S-18"


def test_scope_rework_is_design_only_and_unauthorized_for_remote_work() -> None:
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
