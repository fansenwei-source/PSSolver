"""Frozen record for the rc5 G-1 focused performance recovery plan."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_focused_performance_recovery_v7.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_focused_recovery_record_preserves_failure_and_authorization_boundaries():
    record = _record()
    assert record["source_failure"]["historical_result_reclassified"] is False
    assert record["source_failure"]["thresholds_changed"] is False
    assert record["adjudication_boundary"]["focused_pass_is_g1_complete"] is False
    assert (
        record["adjudication_boundary"]["focused_pass_only_authorizes_cumulative_reassembly"]
        is True
    )
    assert record["adjudication_boundary"]["production_default_changed"] is False
    assert record["authorization"] == {
        "push_authorized": True,
        "h100_submission_authorized": True,
        "automatic_retry_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
    }
