"""Bindings for the R-0029 review import and qualification-support rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]


RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_44_r0029_rework.json"


RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0029.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0029_result_and_independent_archive_are_bound() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_r0029_rework.py::test_r0029_result_and_independent_archive_are_bound"
    )
    assert record["classification"] == ("R0029_REWORK_IMPORTED_AND_DIAGNOSTIC_ENVIRONMENT_HARDENED")
    assert len(record["sync_result"]["sha256"]) == 64
    assert record["sync_result"]["classification"] == "REWORK_REQUIRED"
    archive = record["independent_archive"]
    assert archive["checksums_entries"] == 2157
    assert len(archive["checksums_sha256"]) == 64


def test_r0029_rework_sources_and_successors_are_exact() -> None:
    record = _read(RECORD)
    for relative, expected in record["added_sources"].items():
        assert _sha256(ROOT / relative) == expected
    for relative, successor in record["historical_successor_sources"].items():
        actual = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == actual
        assert resolve_s03_successor(relative, successor["current_sha256"]) == actual


def test_r0029_rework_authorizes_no_external_action() -> None:
    assert _read(RECORD)["authorization"] == {
        "local_rework": True,
        "push": False,
        "hpcc_write": False,
        "slurm_or_h100": False,
        "tag_or_release": False,
    }
