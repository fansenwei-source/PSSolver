"""Static evidence and planning gates for P9.7.6--P9.8.0."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _record(name: str) -> dict[str, object]:
    return json.loads((NOTES / name).read_text(encoding="utf-8"))


def test_p976_closure_authorizes_only_p98_planning():
    record = _record("phase_9_p976_h100_closure.json")
    assert record["classification"] == (
        "PASS_P9_7_6_INDEPENDENT_CHANNEL_CONSUMER_INSTALLED_WHEEL_H100"
    )
    assert record["provider_commit"] == (
        "3b6fb9c71d3cd68a72caf3b184bdf0a61826e39c"
    )
    assert record["external_consumer_commit"] == (
        "73c022b00fff71b964b91a4d497419fff21e34ef"
    )
    assert record["archive"]["checksum_passed"] == 54
    authorization = record["authorization"]
    assert authorization["p9_7_complete"] is True
    assert authorization["p9_8_planning_eligible"] is True
    assert authorization["p9_8_implementation_authorized"] is False
    assert authorization["phase_9_complete"] is False


def test_p976_binds_gradient_memory_and_negative_evidence():
    record = _record("phase_9_p976_h100_closure.json")
    h100 = record["h100_functional"]
    assert h100["state"] == "COMPLETED"
    assert h100["formal_submission_count"] == 1
    assert h100["production_functional_byte_identity"] is True
    assert h100["checkpointed_full_history_gradient_relative_l2"] == 0.0
    assert h100["central_fd_max_relative_error"] < h100["central_fd_limit"]
    assert h100["negative_gates"] == "8/8 pass"
    pressure = record["pressure_adjoint"]
    assert pressure["forward_iteration_tensors_saved"] == 0
    assert pressure["backward_iteration_tensors_saved"] == 0
    assert len(set(pressure["graph_nodes_by_fixed_iteration_count"].values())) == 1


def test_p980_preserves_numerics_and_defers_the_version_literal():
    record = _record("phase_9_p980_public_api_stabilization_plan.json")
    assert record["prerequisites"]["independent_consumers_qualified"] == 2
    assert record["invariants"]["spectral_discretization_changed"] is False
    assert record["invariants"]["pressure_operator_changed"] is False
    assert record["invariants"]["production_default_changed"] is False
    assert record["versioning"]["current_functional_api_version"] == (
        "0.1-provisional"
    )
    assert record["versioning"]["p9_8_0_changes_version"] is False
    assert record["versioning"]["stable_version_literal_preselected"] is False


def test_p980_makes_pressure_termination_semantics_explicit():
    record = _record("phase_9_p980_public_api_stabilization_plan.json")
    pressure = record["pressure_convergence_contract"]
    assert pressure["current_fixed_iteration_semantics"] == (
        "deterministic_work_cap_with_early_convergence"
    )
    assert pressure["relative_tolerance_is_always_a_postcondition"] is False
    required = set(pressure["required_metadata"])
    assert {
        "achieved_iteration_count",
        "achieved_relative_residual",
        "tolerance_role",
        "termination_reason",
        "accepted_under_selected_mode",
    } <= required


def test_p980_slices_are_ordered_and_only_p981_planning_is_eligible():
    record = _record("phase_9_p980_public_api_stabilization_plan.json")
    assert [item["id"] for item in record["slices"]] == [
        "P9.8.1",
        "P9.8.2",
        "P9.8.3",
        "P9.8.4",
        "P9.8.5",
        "P9.8.6",
    ]
    authorization = record["authorization"]
    assert authorization["p9_8_0_complete"] is True
    assert authorization["p9_8_1_planning_eligible"] is True
    assert authorization["p9_8_1_implementation_authorized"] is False
    assert authorization["p9_8_2_through_p9_8_6_authorized"] is False
    assert authorization["h100_authorized"] is False
    assert authorization["phase_9_complete"] is False


def test_archive_tracks_new_records_without_regenerating_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "phase_9_p976_h100_closure.md",
        "phase_9_p976_h100_closure.json",
        "phase_9_p980_public_api_stabilization_plan.md",
        "phase_9_p980_public_api_stabilization_plan.json",
    ):
        assert f'"{name}"' in source
    assert _record("phase_9_p976_h100_closure.json")["authorization"][
        "verbatim_pdf_regenerated"
    ] is False
    assert _record("phase_9_p980_public_api_stabilization_plan.json")[
        "authorization"
    ]["verbatim_pdf_regenerated"] is False
