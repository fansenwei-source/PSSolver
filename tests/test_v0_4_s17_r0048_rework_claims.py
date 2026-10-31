"""Forward-only bindings for the R-0048 S-17 rework claims."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_84_s17_r0048_rework.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_84_binds_every_forward_source() -> None:
    record = _read(RECORD)
    assert record["successor"] == 84
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_successor_84_claims_only_the_three_r0048_findings() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["registry_ids_claimed_fixed"] == ["V-56", "V-57", "V-58"]
    assert record["fix_sources"] == {
        "V-56": {
            "evidence_commit": "7e719bbbcd961f54532756ca8929fac3d441b995",
            "external_repository": "PSSolver-Control",
            "external_fix_commit": "215e35c5504a305d20a28ad3987fd5ed2f94683a",
        },
        "V-57": {
            "evidence_commit": "7e719bbbcd961f54532756ca8929fac3d441b995",
            "external_repository": "PSSolver-Control",
            "external_fix_commit": "140bd273693c341545ceb023bfd2de2e36458e07",
        },
        "V-58": {
            "evidence_commit": "d98472aa975776fcd78f95b133d2abb747bce83b",
            "external_repository": None,
            "external_fix_commit": None,
        },
    }
    for issue_id in record["registry_ids_claimed_fixed"]:
        assert issues[issue_id]["status"] in {"claimed_fixed", "verified"}


def test_successor_84_changes_no_runtime_or_authorized_action() -> None:
    record = _read(RECORD)
    assert all(value is False for value in record["scope"].values())
