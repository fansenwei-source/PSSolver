"""Bindings for the S-10 historical-test prequalification rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_47_s10_prequalification_rework.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s10_prequalification_rework_is_forward_only_and_complete() -> None:
    record = _read(RECORD)
    assert record["plan_step"] == "S-10"
    assert record["initial_gate"] == {
        "environment": "D-0008 conda baseline hard-link clone",
        "passed": 4772,
        "skipped": 1,
        "deselected": 7,
        "subtests_passed": 8,
        "failed": 11,
        "failure_domain": "historical evidence-test forward compatibility",
        "log_preserved": True,
        "junit_preserved": True,
    }
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_s10_prequalification_rework_does_not_relax_qualification() -> None:
    record = _read(RECORD)
    contract = record["rework_contract"]
    assert contract["historical_records_unchanged"] is True
    assert contract["historical_snapshots_remain_exact"] is True
    assert contract["live_registry_may_advance_through_successors"] is True
    assert contract["payload_hashes_remain_exact"] is True
    assert contract["runtime_changed"] is False
    assert contract["numerical_threshold_changed"] is False
    assert contract["skip_or_xfail_added"] is False
    assert record["authorization"]["local_rework"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_rework"
    )
    assert record["next_request"] == "R-0032"
