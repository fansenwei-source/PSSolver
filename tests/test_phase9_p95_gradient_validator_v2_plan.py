"""Static contracts for the P9.5 validator-v2 diagnostic plan."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def test_plan_remains_analysis_only_and_does_not_authorize_p96():
    plan = json.loads(
        (NOTES / "phase_9_p95_gradient_validator_v2_diagnostic_plan.json").read_text(
            encoding="utf-8"
        )
    )

    assert plan["phase"] == "P9.5-gradient-validator-v2-diagnostic"
    assert plan["execution"]["diagnostic_process_count"] == 2
    assert plan["execution"]["analysis_only"] is True
    assert plan["stopping_rules"]["does_not_modify_frozen_validator"] is True
    assert plan["stopping_rules"]["does_not_qualify_p9_5"] is True
    assert plan["stopping_rules"]["does_not_authorize_p9_6"] is True
    assert plan["nonclaims"]["p9_5_passed"] is False


def test_future_archive_lists_validator_v2_after_first_diagnostic():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(encoding="utf-8")
    previous = '"phase_9_p95_gradient_diagnostic_plan.json"'
    validator_md = '"phase_9_p95_gradient_validator_v2_diagnostic_plan.md"'
    validator_json = '"phase_9_p95_gradient_validator_v2_diagnostic_plan.json"'

    assert source.count(validator_md) == 1
    assert source.count(validator_json) == 1
    assert source.index(previous) < source.index(validator_md) < source.index(
        validator_json
    )
