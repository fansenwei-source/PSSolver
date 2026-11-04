"""Repository-only snapshots for the R-0062 failed-job review import."""

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_122_r0062_import.json"


def _read():
    return json.loads(RECORD.read_text())


def test_r0062_binds_the_review_and_readonly_archive():
    review = _read()["review"]
    assert review["request"] == "R-0062"
    assert review["classification"] == "REWORK_REQUIRED"
    assert review["sync_commit"] == "866546277d36930cbe22e551f6df98820f855403"
    assert review["result_sha256"] == "20bd9621615db012b1ed8b252a0340f6759a1ff9399c2aba28e2011384e59041"
    assert review["archive"]["checksums_entries"] == 9
    assert review["archive"]["checksums_sha256"] == "4e2c5924073e21c092ad496546e0ec140f575dd40615e0a34eeaafff48923370"


def test_r0062_snapshot_registers_v64_without_closing_v63():
    record = _read()
    assert record["registry_updates"] == []
    assert len(record["new_findings"]) == 1
    finding = record["new_findings"][0]
    entry = record["registry_snapshot"]["V-64"]
    assert entry["id"] == finding["proposed_id"] == "V-64"
    assert entry["title"] == finding["title"]
    assert entry["severity"] == finding["severity"] == "high"
    assert entry["status"] == finding["suggested_status"] == "open"
    assert entry["target_release"] == finding["suggested_target_release"] == "0.4.0 (S-19 retry)"
    assert entry["affects_control_paths"] is finding["affects_control_paths"] is False
    assert entry["exercised_by_current_control_research"] is False
    assert "fixed_in" not in entry and "verified_by" not in entry
    for key in ("detail", "repro", "fix"):
        assert finding[key] in entry["notes"]
    before = record["preserved_registry_entries"]
    assert set(before) == {"V-62", "V-63"}
    for issue_id, original in before.items():
        assert record["registry_snapshot"][issue_id] == original
    assert before["V-62"]["status"] == "verified"
    assert before["V-62"]["verified_by"] == "R-0060"
    assert before["V-62"]["fixed_in"] == "949970e5f32126ff50f17ef812cfe288f8eb9729"
    assert before["V-63"]["status"] == "open"


def test_r0062_import_does_not_qualify_runtime_or_start_a_retry():
    record = _read()
    assert not any(record["scope"].values())
    assert record["git_identity"]["implementation_parent"] == "cd9162fe220b8a97023623a2999748fcdac21db6"
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["job_snapshot"] == {
        "id": 10944635, "scheduler_state": "FAILED", "exit_code": "1:0",
        "elapsed_seconds": 4, "qualification_evidence_produced": False,
    }


def test_successor_122_bindings_resolve_forward_without_external_reads():
    record = _read()
    bindings = {p: v["current_sha256"] for p, v in record["historical_successor_sources"].items()}
    assert set(bindings) == {
        "notes/pssolver_v0_2_x_open_issues.json",
        "notes/pssolver_v0_2_x_open_issues_zh.md",
        "tests/test_v0_2_x_open_issues.py",
    }
    bindings.update(record["introduced_sources"])
    for relative, expected in bindings.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == resolve_s03_successor(relative, expected)
