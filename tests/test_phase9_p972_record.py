"""Static closure record for P9.7.2 Channel pressure transpose."""

from __future__ import annotations

import json
from pathlib import Path

import pssolver
import pssolver.functional as functional


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD = NOTES / "phase_9_p972_channel_pressure_transpose.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_p972_record_closes_only_the_operator_level_transpose_slice():
    record = _record()

    assert record["phase"] == "P9.7.2"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_P9_7_2_CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL"
    )
    assert record["scope"]["functional_runtime_executable"] is False
    assert record["nonclaims"]["custom_implicit_autograd_rule"] is False
    assert record["nonclaims"]["pressure_gradient_qualified"] is False
    assert record["nonclaims"]["h100_qualified"] is False


def test_p972_preserves_the_frozen_production_solver_and_warm_start():
    record = _record()
    compatibility = record["compatibility"]
    assert (ROOT / compatibility["canonical_production_solver"]).is_file()
    # This is a historical P9.7.2 digest, not a permanent lock on the live
    # solver, which later acquired the rc2 even-grid Nyquist repair.
    assert compatibility["canonical_production_solver_sha256"] == (
        "4503f62df5204f7f48da7ca58674ebda45a3f775b7742dcc7bc7ab535c511b16"
    )
    assert compatibility["canonical_production_solver_modified"] is False
    assert compatibility["production_pressure_pcg_modified"] is False
    assert compatibility["production_pressure_warm_start_modified"] is False
    assert compatibility["channel_runtime_modified"] is False


def test_p972_api_is_provisional_and_control_ownership_does_not_move():
    record = _record()

    assert functional.CHANNEL_PRESSURE_TRANSPOSE_PROTOCOL_VERSION == "1"
    assert hasattr(functional, "ChannelPressureTransposeOperator")
    assert hasattr(functional, "ChannelPressureTransposeProtocol")
    assert not hasattr(pssolver, "ChannelPressureTransposeOperator")
    assert record["ownership"] == {
        "repository_changed": "PSSolver_only",
        "control_repository_changed": False,
        "legacy_pssolver_control_changed": False,
        "external_control_dependency_added": False,
    }


def test_p972_authorizes_only_p973_planning_and_updates_future_archive():
    record = _record()
    authorization = record["authorization"]

    assert authorization["p9_7_2_complete"] is True
    assert authorization["p9_7_3_planning_eligible"] is True
    assert authorization["p9_7_3_implementation_authorized"] is False
    assert authorization["p9_7_4_through_p9_7_6_authorized"] is False
    assert authorization["h100_authorized"] is False
    assert authorization["phase_9_complete"] is False

    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    p971 = '"phase_9_p971_channel_functional_declarations.json"'
    p972_md = '"phase_9_p972_channel_pressure_transpose.md"'
    p972_json = '"phase_9_p972_channel_pressure_transpose.json"'
    assert source.count(p972_md) == 1
    assert source.count(p972_json) == 1
    assert source.index(p971) < source.index(p972_md) < source.index(p972_json)
    assert authorization["verbatim_pdf_regenerated"] is False
