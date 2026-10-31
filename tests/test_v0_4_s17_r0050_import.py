"""Forward contract for importing the R-0050 S-17 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_88_r0050_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_88_binds_r0050_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 88
    assert record["classification"] == "R0050_REWORK_REQUIRED_V59_REGISTERED"
    assert record["review"] == {
        "archive": (
            "/home/fansenwei/pssolver_v0_4_claude_s17_r0050_review_20261031_v1"
        ),
        "archive_checksums_entries": 79,
        "archive_checksums_sha256": (
            "875a66ee9b47a2683715f5f68fd69353e6b2839fefd16b7ef350552e61124ed6"
        ),
        "classification": "REWORK_REQUIRED",
        "request": "R-0050",
        "result_sha256": (
            "d28ef1b45e259742a4f22f7f58c03c08d648ea346d634f8e5783ff7a519d07dc"
        ),
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0050_registers_v59_without_pinning_its_forward_status() -> None:
    record = _read(RECORD)
    assert record["registry_updates"] == {
        "V-59": {
            "status": "registered",
            "target_release": "0.4 S-17",
        }
    }
    issue = next(item for item in _read(REGISTRY)["issues"] if item["id"] == "V-59")
    assert issue["status"] in {"registered", "claimed_fixed", "verified"}
    assert issue["source_report"] == "PSSolver_sync/results/R-0050.result.json"
    assert issue["affects_control_paths"] is False
    assert issue["exercised_by_current_control_research"] is False


def test_r0050_import_changes_no_runtime_or_authorization() -> None:
    record = _read(RECORD)
    assert all(value is False for value in record["scope"].values())
    assert record["authorization"] == {
        "local_registration": True,
        "runtime_fix": False,
        "push": False,
        "tag_or_release": False,
        "pypi": False,
        "hpcc": False,
        "slurm_or_h100": False,
    }
