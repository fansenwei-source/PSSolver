"""Static closure record for P9.7.1 Channel functional declarations."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD = NOTES / "phase_9_p971_channel_functional_declarations.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_p971_record_closes_declarations_without_claiming_execution():
    record = _record()
    assert record["phase"] == "P9.7.1"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_P9_7_1_CHANNEL_FUNCTIONAL_DECLARATIONS"
    )
    assert record["scope"]["functional_kind"] == "channel_activity_batch_one"
    assert record["scope"]["executable"] is False
    assert record["scope"]["stable_package_root_changed"] is False
    assert record["ownership"] == {
        "repository_changed": "PSSolver_only",
        "control_repository_changed": False,
        "legacy_pssolver_control_changed": False,
        "external_control_dependency_added": False,
    }


def test_p971_record_freezes_state_control_observation_and_pressure_nonclaims():
    record = _record()
    assert record["state"]["components"] == ["q_physical", "q_spectral"]
    assert record["state"]["pressure_guess_in_state"] is False
    assert record["control"]["equation_term"] == "div(beta * alpha * Q)"
    assert record["observations"]["names"] == ["Q", "velocity", "pressure"]
    assert record["observations"]["terminal_without_control"] == ["Q"]
    assert record["pressure_policy"] == {
        "solver": "channel_no_slip_modal_stokes_pcg",
        "production_warm_start": True,
        "functional_warm_start": False,
        "functional_initial_guess": "zero",
        "transpose_action": "not_implemented",
        "gradient": "not_qualified",
    }
    assert record["capabilities"]["pure_step"] is False
    assert record["capabilities"]["differentiability"] == "not_qualified"
    assert record["capabilities"]["differentiable_inputs"] == []


def test_p971_record_authorizes_only_the_next_planning_boundary():
    authorization = _record()["authorization"]
    assert authorization["p9_7_1_complete"] is True
    assert authorization["p9_7_2_planning_eligible"] is True
    assert authorization["p9_7_2_implementation_authorized"] is False
    assert authorization["p9_7_3_through_p9_7_6_authorized"] is False
    assert authorization["h100_authorized"] is False
    assert authorization["phase_9_complete"] is False
    assert authorization["production_default_changed"] is False


def test_future_archive_lists_p971_without_regenerating_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    p970 = '"phase_9_p970_channel_functional_plan.json"'
    p971_md = '"phase_9_p971_channel_functional_declarations.md"'
    p971_json = '"phase_9_p971_channel_functional_declarations.json"'
    assert source.count(p971_md) == 1
    assert source.count(p971_json) == 1
    assert source.index(p970) < source.index(p971_md) < source.index(p971_json)
    assert _record()["authorization"]["verbatim_pdf_regenerated"] is False
