"""Static contract tests for the P9.5 H100 gradient diagnostic."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
PLAN = NOTES / "phase_9_p95_gradient_diagnostic_plan.json"


def test_plan_is_analysis_only_and_preserves_failed_qualification():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["phase"] == "P9.5-gradient-diagnostic"
    assert plan["trigger"]["p9_5_qualification_complete"] is False
    assert plan["trigger"]["eligible_for_p9_6"] is False
    assert plan["execution"]["analysis_only"] is True
    assert plan["execution"]["trials_per_grid"] == 3
    assert plan["execution"]["diagnostic_process_count"] == 6
    assert plan["execution"]["fresh_process_per_trial"] is True
    assert plan["execution"]["submission_count_max"] == 1
    assert plan["execution"]["automatic_retry"] is False
    assert plan["stopping_rules"]["does_not_qualify_p9_5"] is True
    assert plan["stopping_rules"]["does_not_authorize_epsilon_change"] is True


def test_future_archive_lists_diagnostic_after_p95_plan():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(encoding="utf-8")
    p95 = '"phase_9_p95_h100_qualification_plan.json"'
    diagnostic_md = '"phase_9_p95_gradient_diagnostic_plan.md"'
    diagnostic_json = '"phase_9_p95_gradient_diagnostic_plan.json"'

    assert source.count(diagnostic_md) == 1
    assert source.count(diagnostic_json) == 1
    assert source.index(p95) < source.index(diagnostic_md) < source.index(
        diagnostic_json
    )
