"""Repository record checks for the rc5 G-1 cumulative closure plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.analyze_rc5_g1_cumulative_closure import (
    EXPECTED_ARCHIVES,
    FOCUSED_CELLS,
    PASS_CLASSIFICATION,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_cumulative_closure_plan_v17.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cumulative_closure_plan_binds_exact_sources_and_archives():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == "READY_FOR_RC5_G1_CPU_ONLY_CUMULATIVE_CLOSURE"
    assert record["parent_commit"] == "044fd95cd8351a857bdc87b5b7a626d91fee16d2"
    assert record["success_contract"]["classification"] == PASS_CLASSIFICATION
    assert record["success_contract"]["new_h100_submission_count"] == 0
    assert record["scope"]["new_h100_submission_authorized"] is False
    assert record["scope"]["source_runtime_modified"] is False
    assert record["scope"]["thresholds_modified"] is False
    assert record["scope"]["production_default_changed"] is False
    assert {
        tuple(cell) for cell in record["evidence_contract"]["focused_cell_set"]
    } == FOCUSED_CELLS
    assert record["evidence_contract"]["reused_v6_passed_cells"] == 12
    assert record["evidence_contract"]["requalified_v16_cells"] == 4
    assert record["archive_identities"] == {
        name: {
            "entries": expected["entries"],
            "checksums_sha256": expected["manifest_sha256"],
        }
        for name, expected in EXPECTED_ARCHIVES.items()
    }
    for relative, digest in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == digest


def test_plan_does_not_claim_execution_or_release_authority():
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["execution_status"] == {
        "cumulative_analyzer_executed": False,
        "closure_archive_created": False,
        "g1_complete": False,
    }
    assert record["success_contract"]["eligible_for_automatic_merge"] is False
    assert record["success_contract"]["eligible_for_default_promotion"] is False
    assert record["success_contract"]["post_release_verification_required"] is True
