"""Repository record checks for the rc5 G-1 cumulative closure plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_cumulative_closure_plan_v17.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
