"""Static closure record for P9.7.3 implicit Channel pressure adjoint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver
import pssolver.functional as functional


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD = NOTES / "phase_9_p973_channel_pressure_implicit_adjoint.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_p973_record_closes_the_implicit_pressure_vjp_only():
    record = _record()

    assert record["phase"] == "P9.7.3"
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_P9_7_3_CHANNEL_IMPLICIT_PRESSURE_ADJOINT"
    )
    assert record["scope"]["differentiable_input"] == "pressure_rhs_hat"
    assert record["scope"]["functional_runtime_executable"] is False
    assert record["nonclaims"]["activity_gradient_qualified"] is False
    assert record["nonclaims"]["h100_qualified"] is False


def test_p973_record_freezes_graph_memory_and_oracle_scope():
    record = _record()
    implicit = record["implicit_adjoint"]
    oracle = record["oracle"]

    assert implicit["forward_iteration_tensors_saved"] == 0
    assert implicit["backward_iteration_tensors_saved"] == 0
    assert implicit["graph_size_independent_of_iteration_count"] is True
    assert implicit["higher_order_derivatives"] is False
    assert oracle["role"] == "small_grid_cpu_qualification_only"
    assert oracle["maximum_modes"] == 512
    assert oracle["production_fallback"] is False
    assert oracle["implicit_vjp_matches_unrolled"] is True


def test_p973_preserves_frozen_solver_and_provisional_api_boundary():
    record = _record()
    compatibility = record["compatibility"]
    path = ROOT / compatibility["canonical_production_solver"]

    assert hashlib.sha256(path.read_bytes()).hexdigest() == compatibility[
        "canonical_production_solver_sha256"
    ]
    assert compatibility["canonical_production_solver_modified"] is False
    assert compatibility["p9_7_2_protocol_modified"] is False
    assert functional.CHANNEL_PRESSURE_IMPLICIT_ADJOINT_VERSION == "1"
    assert hasattr(functional, "ChannelImplicitPressureAdjoint")
    assert hasattr(functional, "unrolled_channel_pressure_solve_oracle")
    assert not hasattr(pssolver, "ChannelImplicitPressureAdjoint")


def test_p973_authorizes_only_p974_planning_and_updates_future_archive():
    record = _record()
    authorization = record["authorization"]

    assert authorization["p9_7_3_complete"] is True
    assert authorization["p9_7_4_planning_eligible"] is True
    assert authorization["p9_7_4_implementation_authorized"] is False
    assert authorization["p9_7_5_through_p9_7_6_authorized"] is False
    assert authorization["h100_authorized"] is False
    assert authorization["phase_9_complete"] is False

    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    p972 = '"phase_9_p972_channel_pressure_transpose.json"'
    p973_md = '"phase_9_p973_channel_pressure_implicit_adjoint.md"'
    p973_json = '"phase_9_p973_channel_pressure_implicit_adjoint.json"'
    assert source.count(p973_md) == 1
    assert source.count(p973_json) == 1
    assert source.index(p972) < source.index(p973_md) < source.index(p973_json)
    assert authorization["verbatim_pdf_regenerated"] is False
