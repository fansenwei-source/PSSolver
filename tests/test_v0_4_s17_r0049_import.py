"""Forward contract for importing the R-0049 S-17 rework review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_86_r0049_import.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_86_binds_r0049_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 86
    assert record["classification"] == "R0049_PASS_IMPORTED_S17_REWORK_VERIFIED"
    assert record["review"] == {
        "archive": "/home/fansenwei/pssolver_control_s17_claude_r0049_review_20261030_v1",
        "archive_checksums_entries": 51,
        "archive_checksums_sha256": (
            "2875b14e20060c6c6244d030b4fcd4ae346003a6ac02c86eca74fbad6f183871"
        ),
        "classification": "PASS",
        "request": "R-0049",
        "result_sha256": (
            "26c0dbf16d3d2b1bad377fe77664f5eacc74b980e8a29e1ca05aac60a87c8d11"
        ),
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0049_record_verifies_the_three_reviewed_fixes() -> None:
    record = _read(RECORD)
    assert record["registry_updates"] == {
        "V-56": {
            "fixed_in": "7e719bbbcd961f54532756ca8929fac3d441b995",
            "status": "verified",
            "verified_by": "R-0049",
        },
        "V-57": {
            "fixed_in": "7e719bbbcd961f54532756ca8929fac3d441b995",
            "status": "verified",
            "verified_by": "R-0049",
        },
        "V-58": {
            "fixed_in": "d98472aa975776fcd78f95b133d2abb747bce83b",
            "status": "verified",
            "verified_by": "R-0049",
        },
    }
    assert record["next_actions"]["preservation_choice"] == "D-0017 option A"
    assert record["next_actions"]["r3_r6_implementation_started"] is False


def test_r0049_import_changes_no_runtime_or_authorization() -> None:
    record = _read(RECORD)
    assert all(value is False for value in record["scope"].values())
    assert record["authorization"] == {
        "local_registration": True,
        "control_or_pssolver_push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
