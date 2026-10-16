"""Evidence contract for the rc5 G-1 Consumer recovery-v5 fixes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_consumer_recovery_v5.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_recovery_record_binds_the_two_independent_v4_failures():
    record = _record()
    assert record["classification"] == (
        "READY_RC5_G1_CONSUMER_GUARD_ORDER_AND_HELPER_IMPORT_RECOVERY_V5"
    )
    predecessor = record["failed_predecessor"]
    assert predecessor["job_id"] == "10885604"
    assert predecessor["manifest_entries_passed"] == 46
    assert predecessor["manifest_entries_failed"] == 0
    assert predecessor["historical_failure_classification_preserved"] is True
    consumer = record["consumer"]
    assert consumer["stale_pin_rejected_before_state_access"] is True
    assert consumer["control_algorithm_changed"] is False
    support = record["provider_qualification_support"]
    assert support["runtime_package_changed"] is False
    assert support["local_cpu"] == {
        "passed": 3724,
        "skipped": 0,
        "deselected": 7,
        "failed": 0,
        "xfailed": 0,
        "subtests_passed": 8,
    }


def test_recovery_record_registers_exact_historical_successors():
    record = _record()
    successors = record["historical_successor_sources"]
    assert set(successors) == {
        "benchmarks/run_rc5_g1_consumer.py",
        "tests/test_run_rc5_g1_foundation_stages.py",
    }
    for relative, value in successors.items():
        assert _sha256(ROOT / relative) == value["sha256"]
        assert value["introduced_by_commit"] == (
            "9a6d56d49c2dc7662e51ed7b74736783724d342b"
        )
        assert value["historical_sha256"] != value["sha256"]
        assert "production default" in value["reason"]


def test_recovery_record_does_not_rewrite_historical_freezes():
    for value in _record()["historical_records"].values():
        assert value["rewritten"] is False
        assert _sha256(ROOT / value["path"]) == value["sha256"]


def test_recovery_record_preserves_authorization_boundary():
    record = _record()
    assert record["qualification_complete"] is False
    assert not any(record["authorization"].values())
    assert record["next_gate"].startswith("build fresh Provider and Consumer")
