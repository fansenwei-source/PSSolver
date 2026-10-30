"""Forward contract for importing the R-0046 S-16 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_80_r0046_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_successor_80_binds_the_r0046_result_and_forward_sources() -> None:
    record = _read(RECORD)
    assert record["successor"] == 80
    assert record["classification"] == "R0046_REWORK_REQUIRED_IMPORTED"
    assert record["review"] == {
        "archive": "/home/fansenwei/pssolver_v0_4_claude_s16_r0046_review_20261030_v1",
        "archive_checksums_entries": 62,
        "archive_checksums_sha256": (
            "c0a69dffeb0f60bd64e0d22005192b8efb13848077430d9b65965d11a06fbd1a"
        ),
        "classification": "REWORK_REQUIRED",
        "request": "R-0046",
        "result_sha256": (
            "9391ffedde0913f769a9462c2bc1434ae6975b6a31104310c5507b4f2ad011ed"
        ),
    }
    for relative, identity in record["historical_successor_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, identity["historical_sha256"]
        )
    for relative, expected in record["introduced_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_r0046_verifies_v54_and_registers_v55_without_pinning_v55() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert issues["V-54"]["status"] == "verified"
    assert issues["V-54"]["fixed_in"] == (
        "9cf4b34c37d5fa266ad1900eae09f143c5c98b63"
    )
    assert issues["V-54"]["verified_by"] == "R-0046"
    assert issues["V-55"]["status"] in {"registered", "claimed_fixed", "verified"}
    assert issues["V-55"]["source_report"] == "PSSolver_sync/results/R-0046.result.json"
    assert issues["V-55"]["target_release"] == "0.4 S-16"


def test_r0046_import_changes_no_runtime_or_gate() -> None:
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
