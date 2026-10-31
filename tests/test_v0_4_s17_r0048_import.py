"""Forward-only import of the R-0048 S-17 review findings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_83_r0048_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_83_binds_r0048_and_every_forward_source() -> None:
    record = _read(RECORD)
    assert record["successor"] == 83
    assert record["review"] == {
        "archive": (
            "/home/fansenwei/pssolver_control_s17_claude_r0048_review_20261030_v1"
        ),
        "archive_checksums_entries": 50,
        "archive_checksums_sha256": (
            "bba03b15ba180cf5ec37148370c579671dd5697dd6451cc294ed2ceb4173bff5"
        ),
        "classification": "REWORK_REQUIRED",
        "request": "R-0048",
        "result_sha256": (
            "d576e47398965f3bfafc622fa0b6a0f49f6e659ca14f821ba4a9f7f1ff075b8a"
        ),
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0048_findings_are_registered_without_pinning_future_status() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert set(record["new_findings"]) == {"V-56", "V-57", "V-58"}
    for issue_id, snapshot in record["new_findings"].items():
        assert snapshot["status"] == "registered"
        assert issues[issue_id]["title"] == snapshot["title"]
        assert issues[issue_id]["severity"] == snapshot["severity"]
        assert issues[issue_id]["source_report"] == snapshot["source_report"]
        assert issues[issue_id]["status"] in {
            "registered",
            "claimed_fixed",
            "verified",
        }


def test_r0048_import_changes_no_runtime_or_gate() -> None:
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
