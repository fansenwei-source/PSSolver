"""Forward contract for importing the R-0047 S-16 final review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_82_r0047_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_82_binds_the_r0047_result_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 82
    assert record["classification"] == "R0047_PASS_IMPORTED_S16_COMPLETE_PENDING_PUSH"
    assert record["review"] == {
        "archive": "/home/fansenwei/pssolver_v0_4_claude_s16_r0047_review_20261030_v1",
        "archive_checksums_entries": 62,
        "archive_checksums_sha256": (
            "ea3ee798bb710c0cbd577bb347999531e777d27178b2856556784e96ca12f551"
        ),
        "classification": "PASS",
        "request": "R-0047",
        "result_sha256": (
            "43918f38012fdc8d402a3f9d8e125e42c8ffc6bb83632b10b30971ad24ccf2e3"
        ),
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0047_verifies_v55_at_the_reviewed_implementation() -> None:
    record = _read(RECORD)
    assert record["registry_updates"] == {
        "V-55": {
            "status": "verified",
            "fixed_in": "625c3ca6b2c64bb674d43430d39ec3409f78c736",
            "verified_by": "R-0047",
        }
    }
    issue = next(item for item in _read(REGISTRY)["issues"] if item["id"] == "V-55")
    assert issue["status"] == "verified"
    assert issue["fixed_in"] == record["registry_updates"]["V-55"]["fixed_in"]
    assert issue["verified_by"] == "R-0047"


def test_r0047_import_changes_no_runtime_or_gate() -> None:
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
