"""Forward contract for importing the R-0051 S-17 final review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_90_r0051_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_90_binds_r0051_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 90
    assert record["classification"] == "R0051_PASS_IMPORTED_S17_COMPLETE_PENDING_PUSH"
    assert record["review"] == {
        "archive": "/home/fansenwei/pssolver_v0_4_claude_s17_r0051_review_20261031_v1",
        "archive_checksums_entries": 75,
        "archive_checksums_sha256": (
            "28974aec2f24be686d85be14fd536a533b35ae4f7904477185fb45620995531b"
        ),
        "classification": "PASS",
        "request": "R-0051",
        "result_sha256": (
            "3daf8ca14a6edda2d1dbb67313123dbb9768334e46ea86b3f971758833681a66"
        ),
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0051_verifies_v59_at_the_reviewed_runtime_fix() -> None:
    record = _read(RECORD)
    assert record["registry_updates"] == {
        "V-59": {
            "fixed_in": "223b86dde0eb22ff1816e5a91ee93195ff5718a7",
            "status": "verified",
            "verified_by": "R-0051",
        }
    }
    issue = next(item for item in _read(REGISTRY)["issues"] if item["id"] == "V-59")
    assert issue["status"] == "verified"
    assert issue["fixed_in"] == record["registry_updates"]["V-59"]["fixed_in"]
    assert issue["verified_by"] == "R-0051"


def test_r0051_import_changes_no_runtime_or_authorization() -> None:
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
