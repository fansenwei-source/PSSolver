"""Static contracts for the planning-only P8.0 capability expansion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pssolver.configuration.public_simulation_runner import (
    public_compiler_capabilities,
)
from pssolver.core.boundary import BoundaryKind


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
PLAN_PATH = NOTES / "phase_8_p80_capability_expansion_plan.json"


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p80_is_planning_only_and_binds_the_phase7_closure() -> None:
    plan = _plan()
    baseline = plan["baseline"]
    authorization = plan["authorization"]
    closure = ROOT / baseline["phase_7_record"]

    assert plan["schema_version"] == 1
    assert plan["phase"] == "P8.0"
    assert plan["status"] == (
        "P8_0_PLANNING_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED"
    )
    assert plan["classification"] == (
        "PASS_P8_0_CAPABILITY_EXPANSION_PLAN"
    )
    assert baseline["commit"] == "f8789461ee71391885d5e04c6906a13bc83a26b0"
    assert _sha256(closure) == baseline["phase_7_record_sha256"]
    assert baseline["phase_7_complete"] is True
    assert baseline["eligible_for_phase_8_planning"] is True
    assert baseline["phase_8_execution_previously_authorized"] is False
    assert authorization == {
        "p8_0_planning_authorized": True,
        "p8_0_complete": True,
        "p8_1_implementation_authorized": False,
        "phase_8_execution_authorized": False,
        "h100_authorized": False,
        "phase_9_authorized": False,
        "production_default_changed": False,
        "existing_phase_6_or_phase_7_evidence_may_be_modified": False,
    }


def test_p80_preserves_the_pre_p81_public_declaration_surface() -> None:
    surface = _plan()["current_public_surface"]

    assert [entry["equation_variant"] for entry in surface["models"]] == [
        "complete_stress_beris_edwards",
        "legacy_active_force_active_nematics",
    ]
    assert [entry["public_constructor_status"] for entry in surface["models"]] == [
        "stable",
        "canonical_internal_request_without_convenience_constructor",
    ]
    assert [entry["name"] for entry in surface["geometries"]] == [
        "periodic_box",
        "plane_slab",
        "rectangular_channel",
    ]
    assert set(surface["low_level_boundary_kinds"]) == {
        f"homogeneous_{BoundaryKind.DIRICHLET.value}",
        f"homogeneous_{BoundaryKind.NEUMANN.value}",
        BoundaryKind.PERIODIC.value,
    }


def test_p80_qualified_matrix_matches_the_live_compiler_registry() -> None:
    plan = _plan()
    expected = {
        (entry["equation_variant"], entry["geometry"]): (
            entry["application"],
            tuple(entry["runtime_paths"]),
        )
        for entry in plan["qualified_applications"]
    }
    observed = {
        (entry["equation_variant"], entry["geometry_name"]): entry[
            "application"
        ]
        for entry in public_compiler_capabilities()
    }

    assert set(expected) <= set(observed)
    assert observed[("complete_stress_beris_edwards", "periodic_box")] == (
        "periodic_complete_stress_beris_edwards"
    )
    for key, (application, _runtime_paths) in expected.items():
        assert observed[key] == application

    matrix = {
        entry["equation_variant"]: entry
        for entry in plan["model_geometry_matrix"]
    }
    assert matrix["complete_stress_beris_edwards"] == {
        "equation_variant": "complete_stress_beris_edwards",
        "periodic_box": "planned_p8_2",
        "plane_slab": "qualified",
        "rectangular_channel": "planned_p8_3",
    }
    assert matrix["legacy_active_force_active_nematics"] == {
        "equation_variant": "legacy_active_force_active_nematics",
        "periodic_box": "deferred_not_qualified",
        "plane_slab": "deferred_not_qualified",
        "rectangular_channel": "qualified",
    }


def test_p80_separates_public_gaps_from_new_boundary_physics() -> None:
    plan = _plan()
    gaps = {entry["id"]: entry for entry in plan["capability_gaps"]}
    policies = {
        entry["name"]: entry
        for entry in plan["current_public_surface"]["boundary_policies"]
    }

    assert len(gaps) == 8
    assert policies["no_slip_velocity"]["status"] == (
        "qualified_inside_channel_application_but_not_public_policy"
    )
    assert policies["periodic"]["status"] == (
        "implicit_from_axis_topology"
    )
    assert gaps["P8-GAP-003"]["first_owner"] == "P8.1"
    assert gaps["P8-GAP-007"]["first_owner"] == "P8.4"
    assert gaps["P8-GAP-008"]["first_owner"] == "P8.5"


def test_p80_freezes_slice_order_and_non_regression_contract() -> None:
    plan = _plan()
    slices = plan["slices"]
    invariants = plan["architectural_invariants"]
    non_regression = plan["non_regression_contract"]

    assert [entry["id"] for entry in slices] == [
        "P8.0",
        "P8.1",
        "P8.2",
        "P8.3",
        "P8.4",
        "P8.5",
        "P8.6",
    ]
    assert slices[0]["authorized"] is True
    assert all(entry["authorized"] is False for entry in slices[1:])
    assert slices[1]["requires_h100"] is False
    assert all(entry["requires_h100"] is True for entry in slices[2:])
    assert invariants["registry_lookup_in_timestep"] is False
    assert invariants["runtime_fallback_allowed"] is False
    assert invariants["pressure_gauge_is_distinct_from_wall_boundary_law"]
    assert non_regression["plane_default"] == "legacy_production"
    assert non_regression["channel_default"] == "legacy_channel"
    assert non_regression["existing_equations_may_change"] is False
    assert non_regression["existing_boundary_laws_may_change"] is False
    assert non_regression["existing_checkpoint_schema_may_change"] is False
    assert non_regression["hot_loop_registry_dispatch_allowed"] is False
    assert (
        non_regression["silent_runtime_or_capability_fallback_allowed"]
        is False
    )


def test_future_verbatim_archive_lists_p80_without_regenerating_the_pdf() -> None:
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    names = (
        "phase_7_final_closure.json",
        "phase_8_p80_capability_expansion_plan.md",
        "phase_8_p80_capability_expansion_plan.json",
    )
    positions = []
    for name in names:
        token = f'"{name}"'
        assert source.count(token) == 1
        positions.append(source.index(token))
    assert positions == sorted(positions)
