"""Evidence-only binding for the R-0032 S-10 review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_55_r0032_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0032_import_binds_the_exact_independent_review() -> None:
    record = _read(RECORD)
    assert record["plan_step"] == "S-10"
    assert record["sync_result"] == {
        "id": "R-0032",
        "path": "PSSolver_sync/results/R-0032.result.json",
        "sha256": "ce87ee51b3f423f6fb3cb1ddb56f7b09dca96dd3d608bf797e50bd58f9e5aa4e",
        "classification": "REWORK_REQUIRED",
    }
    assert set(record["verdict"]["passed"]) == {
        "S10-R7-PARTIAL",
        "S10-R9-PARTIAL",
        "S10-D0011-BOUNDARY",
        "S10-S09-SEPARATION",
    }
    assert record["verdict"]["new_findings"] == ["V-43"]


def test_r0032_import_registers_v43_without_closing_v33() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["registry_import"] == {
        "V-33": "claimed_fixed",
        "V-43": "registered",
        "next_review": "R-0033",
    }
    assert issues["V-33"]["status"] in {"claimed_fixed", "verified"}
    assert issues["V-43"]["status"] in {
        "registered", "claimed_fixed", "verified"
    }
    assert issues["V-43"]["source_report"] == (
        "PSSolver_sync/results/R-0032.result.json"
    )
    assert issues["V-43"]["target_release"] == "0.3.0 (S-10 rework)"


def test_r0032_import_is_forward_only_and_evidence_only() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)
    assert record["scope"]["evidence_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "evidence_only"
    )
    assert record["next_request"] == "R-0033"
