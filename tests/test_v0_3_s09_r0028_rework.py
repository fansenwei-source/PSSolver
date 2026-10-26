"""Bindings for the R-0028 rework import and exact revert-proxy support."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_43_r0028_rework.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0028.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0028_rework_result_and_archive_are_bound() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "R0028_REWORK_IMPORTED_AND_EXACT_REVERT_PROXY_PREPARED"
    )
    assert _sha256(RESULT) == record["sync_result"]["sha256"]
    assert record["sync_result"]["classification"] == "REWORK_REQUIRED"
    archive = record["independent_archive"]
    root = Path(archive["path"])
    assert (root / "COMPLETE").is_file()
    assert sum(1 for _ in (root / "checksums.sha256").open(encoding="utf-8")) == 1257
    assert _sha256(root / "checksums.sha256") == archive["checksums_sha256"]


def test_r0028_rework_sources_and_forward_successors_are_exact() -> None:
    record = _read(RECORD)
    for relative, expected in record["added_sources"].items():
        assert _sha256(ROOT / relative) == expected
    for relative, successor in record["historical_successor_sources"].items():
        live = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == live
        assert resolve_s03_successor(relative, successor["current_sha256"]) == live


def test_r0028_rework_changes_no_runtime_or_external_state() -> None:
    record = _read(RECORD)
    assert record["record_scope"]["runtime_changed"] is False
    assert record["record_scope"]["wheel_changed"] is False
    assert record["rework"]["optional_fourth_role_added"] is False
    assert record["registry_import"]["status"] == "registered"
    assert record["authorization"] == {
        "local_rework": True,
        "push": False,
        "hpcc_write": False,
        "slurm_or_h100": False,
        "tag_or_release": False,
    }
