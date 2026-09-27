"""Freeze the P8.6 cumulative Phase 8 closure plan."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pssolver


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
PLAN_PATH = NOTES / "phase_8_p860_closure_plan.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_p860_binds_every_authoritative_phase8_slice_record():
    plan = _plan()

    assert plan["phase"] == "P8.6.0"
    assert plan["classification"] == (
        "PASS_P8_6_0_PLANNING_CUMULATIVE_CLOSURE_NOT_EXECUTED"
    )
    assert len(plan["prerequisites"]) == 5
    for record in plan["prerequisites"]:
        path = ROOT / record["path"]
        value = json.loads(path.read_text(encoding="utf-8"))
        assert _sha256(path) == record["sha256"]
        assert value["classification"] == record["classification"]


def test_p860_catalog_snapshot_matches_live_public_registry():
    plan = _plan()
    frozen = plan["public_catalog"]
    catalog = pssolver.capability_catalog()

    assert len(catalog.models) == frozen["model_count"] == 2
    assert len(catalog.geometries) == frozen["geometry_count"] == 3
    assert len(catalog.boundary_policies) == (
        frozen["boundary_policy_count"]
    ) == 8
    assert len(catalog.qualified_combinations) == (
        frozen["qualified_combination_count"]
    ) == 4
    assert [item.key for item in catalog.models] == frozen["models"]
    assert [item.key for item in catalog.geometries] == frozen["geometries"]
    assert [
        item.key for item in catalog.boundary_policies
    ] == frozen["boundary_policies"]
    robin = next(
        item for item in catalog.boundary_policies if item.key == "robin"
    )
    assert robin.executable is False
    assert robin.qualified_applications == ()


def test_p860_freezes_exact_qualified_combinations_and_runtime_paths():
    expected = _plan()["qualified_combinations"]
    actual = [
        {
            "equation_variant": item.equation_variant,
            "geometry_name": item.geometry_name,
            "application": item.application,
            "runtime_paths": list(item.runtime_paths),
        }
        for item in pssolver.available_combinations()
    ]

    assert actual == expected
    assert sum(len(item["runtime_paths"]) for item in actual) == 6


def test_p860_adds_no_runtime_physics_or_default_promotion():
    plan = _plan()

    assert plan["p8_6_1_allowed_changes"] == [
        "repository_owned_fail_closed_analyzer",
        "cumulative_audit_tests",
        "installed_wheel_cpu_audit",
        "h100_command_plan",
        "architecture_records",
    ]
    assert "runtime_or_timestep" in plan["p8_6_1_forbidden_changes"]
    assert "numerical_operator" in plan["p8_6_1_forbidden_changes"]
    assert plan["compatibility"] == {
        "plane_default": "legacy_production",
        "channel_default": "legacy_channel",
        "compiled_plane_promoted": False,
        "compiled_channel_promoted": False,
        "production_default_changed": False,
        "runtime_fallback_allowed": False,
    }
    authorization = plan["authorization"]
    assert authorization["p8_6_1_local_implementation_authorized"] is True
    assert authorization["p8_6_2_h100_authorized"] is False
    assert authorization["phase_9_planning_eligible"] is False
    assert authorization["phase_9_implementation_authorized"] is False


def test_p860_archive_sources_are_registered_once_and_in_order():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    names = (
        "phase_8_p856_h100_qualification.md",
        "phase_8_p856_h100_qualification.json",
        "phase_8_p860_closure_plan.md",
        "phase_8_p860_closure_plan.json",
    )
    positions = []
    for name in names:
        token = f'"{name}"'
        assert source.count(token) == 1
        positions.append(source.index(token))
    assert positions == sorted(positions)
