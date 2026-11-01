"""Forward contract for importing the R-0053/R-0054 design review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_94_r0054_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_94_binds_both_results_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 94
    assert record["classification"] == "R0053_SUPERSEDED_R0054_PASS_IMPORTED"
    assert record["reviews"]["R-0053"]["classification"] == "SUPERSEDED"
    assert record["reviews"]["R-0054"]["classification"] == "PASS"
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0054_verifies_v60_without_changing_the_fix_source() -> None:
    record = _read(RECORD)
    assert record["registry_updates"] == {
        "V-60": {
            "status": "verified",
            "fixed_in": "f5dbdbef43d1aa16e9c7638093df5c5c5ae58170",
            "verified_by": "R-0054",
        }
    }
    issue = next(item for item in _read(REGISTRY)["issues"] if item["id"] == "V-60")
    assert issue["status"] == "verified"
    assert issue["fixed_in"] == record["registry_updates"]["V-60"]["fixed_in"]
    assert issue["verified_by"] == "R-0054"


def test_import_records_the_accurate_timeline_without_rewriting_history() -> None:
    correction = _read(RECORD)["timeline_correction"]
    assert correction == {
        "d0018_confirmed_sync_commit": (
            "4cf34051cd5d4f47801eaabf0bdc7cd8490d6367"
        ),
        "d0018_confirmed_before_r0053_commit": True,
        "codex_received_user_confirmation_after_r0053_commit": True,
        "successor_93_rewritten": False,
        "r0053_rewritten": False,
        "r0054_rewritten": False,
    }


def test_import_changes_no_runtime_or_authorization() -> None:
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
