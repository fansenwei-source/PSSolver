"""Static contracts for Phase 6 closure and Phase 7 Channel planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _json(name: str) -> dict:
    return json.loads((NOTES / name).read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_phase6_closure_records_the_long_run_without_promoting_a_default():
    value = _json("phase_6_final_closure.json")

    assert value["classification"] == "PASS_PHASE6_LONG_RUN_BYTE_IDENTICAL"
    assert value["equivalence"]["paired_arrays_byte_identical"] == "243/243"
    assert value["equivalence"]["first_mismatch"] is None
    assert value["eligibility"] == {
        "qualification_complete": True,
        "phase_6_complete": True,
        "eligible_for_phase_7_planning": True,
        "phase_7_execution_authorized": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
        "compiled_v2_promoted": False,
    }
    assert value["claims"]["physical_stationarity_established"] is False
    assert value["claims"]["channel_geometry_qualified"] is False


def test_phase7_inventory_binds_the_unchanged_channel_oracle_sources():
    value = _json("phase_7_channel_inventory.json")
    oracle = value["legacy_oracle"]
    assets = value["existing_architecture_assets"]

    assert _sha256(oracle["entry_point"]) == oracle["entry_point_sha256"]
    assert _sha256(oracle["reusable_module"]) == oracle["reusable_module_sha256"]
    assert _sha256(assets["geometry_declaration"].split("::", 1)[0]) == assets[
        "geometry_sha256"
    ]
    assert _sha256(assets["experimental_stokes_adapter"]) == assets[
        "experimental_stokes_adapter_sha256"
    ]
    assert _sha256(assets["experimental_beris_edwards"]) == assets[
        "experimental_beris_edwards_sha256"
    ]
    assert _sha256(assets["stage_g_tests"]) == assets["stage_g_tests_sha256"]
    assert _sha256(assets["legacy_channel_tests"]) == assets[
        "legacy_channel_tests_sha256"
    ]


def test_phase7_boundary_and_pressure_contract_is_geometry_specific():
    value = _json("phase_7_channel_inventory.json")
    oracle = value["legacy_oracle"]

    assert oracle["q_boundaries"] == ["periodic", "neumann", "neumann"]
    assert oracle["velocity_boundaries"] == [
        "periodic",
        "dirichlet",
        "dirichlet",
    ]
    assert oracle["pressure_modal_boundaries"] == [
        "periodic",
        "neumann",
        "neumann",
    ]
    assert oracle["pressure_gauge"] == "zero_mean"
    assert oracle["tangential_zero_mode_policy"] == "not_applicable"
    assert value["known_separation"]["plane_qualification_covers_channel"] is False


def test_phase7_plan_is_planning_only_and_excludes_boundary_and_control_work():
    value = _json("phase_7_channel_migration_plan.json")

    assert value["status"] == "P7_0_PLANNING_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED"
    assert value["phase_6_prerequisite"]["phase_6_complete"] is True
    assert value["slices"][0] == "P7.0_inventory_and_oracle_freeze"
    assert value["slices"][-1] == (
        "P7.6_h100_performance_memory_long_run_closure"
    )
    assert "strong_anchoring" in value["out_of_scope"]
    assert "optimal_control" in value["out_of_scope"]
    assert value["authorization"] == {
        "phase_7_planning_authorized": True,
        "p7_1_implementation_authorized": False,
        "phase_7_execution_authorized": False,
        "h100_authorized": False,
        "production_default_changed": False,
        "plane_evidence_may_be_modified": False,
    }
