from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    CHANNEL_BOUNDARIES,
    ChannelRunComponents,
)
from pssolver.configuration.plane_beris_edwards import (
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneBerisEdwardsRunSpec,
)
from pssolver.configuration.plane_beris_edwards_component_graph import (
    PlaneBerisEdwardsRunComponents,
)


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_7_p770_model_geometry_coupling_inventory.json"
)


def _inventory() -> dict[str, object]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def _flatten(groups: dict[str, list[str]]) -> list[str]:
    return [field for values in groups.values() for field in values]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p770_inventory_is_a_non_runtime_authority() -> None:
    inventory = _inventory()

    assert inventory["schema_version"] == 1
    assert inventory["phase"] == "P7.7.0"
    assert inventory["status"] == "P7_7_0_COMPLETE_P7_7_1_NOT_AUTHORIZED"
    assert (
        inventory["classification"]
        == "PASS_P7_7_0_MODEL_GEOMETRY_COUPLING_INVENTORY"
    )
    assert inventory["architectural_diagnosis"] == {
        "current_form": "qualified_vertical_application_slices",
        "orthogonal_model_geometry_composition_available": False,
        "generic_simulation_spec_available": False,
        "generic_equation_system_spec_available": False,
        "generic_capability_lowering_available": False,
        "summary": (
            "Plane Beris-Edwards and Channel active nematics are isolated "
            "and qualified application stacks, but their model, geometry, "
            "boundary, solver, runtime, and workflow choices are still "
            "composed by application-specific types and builders."
        ),
    }
    assert all(value is False for value in inventory["scope"].values())
    assert inventory["next_slice"]["id"] == "P7.7.1"
    assert inventory["next_slice"]["authorized"] is False


def test_p770_source_identities_bind_the_audited_baseline() -> None:
    identities = _inventory()["source_identities"]
    subsequently_connected = {
        "pssolver/applications/plane_beris_edwards.py",
        "pssolver/applications/channel_active_nematics.py",
        "pssolver/models/active_nematics/stokes.py",
    }

    assert len(identities) == 15
    for relative_path, expected in identities.items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        if relative_path in subsequently_connected:
            assert _sha256(path) != expected, relative_path
        else:
            assert _sha256(path) == expected, relative_path


def test_p770_plane_facade_ownership_is_complete_and_disjoint() -> None:
    plane = _inventory()["applications"]["plane_beris_edwards"]
    groups = plane["facade_field_ownership"]
    recorded = _flatten(groups)
    actual = [field.name for field in fields(PlaneBerisEdwardsRunSpec)]

    assert len(actual) == 54
    assert len(recorded) == len(set(recorded))
    assert set(recorded) == set(actual)
    assert plane["component_fields"] == [
        field.name for field in fields(PlaneBerisEdwardsRunComponents)
    ]


def test_p770_channel_facade_ownership_is_complete_and_disjoint() -> None:
    channel = _inventory()["applications"]["channel_active_nematics"]
    groups = channel["facade_field_ownership"]
    recorded = _flatten(groups)
    actual = [field.name for field in fields(ChannelActiveNematicRunSpec)]

    assert len(actual) == 40
    assert len(recorded) == len(set(recorded))
    assert set(recorded) == set(actual)
    assert channel["component_fields"] == [
        field.name for field in fields(ChannelRunComponents)
    ]


def test_p770_records_the_exact_qualified_boundary_contracts() -> None:
    applications = _inventory()["applications"]

    assert applications["plane_beris_edwards"]["boundary_contract"] == {
        key: list(value)
        for key, value in PLANE_FREE_SLIP_BOUNDARIES.to_legacy().items()
    }
    assert (
        applications["channel_active_nematics"]["boundary_contract"]
        == CHANNEL_BOUNDARIES.to_metadata()
    )


def test_p770_does_not_conflate_the_two_qualified_physical_models() -> None:
    matrix = {
        row["model"]: row
        for row in _inventory()["model_geometry_matrix"]
    }

    complete = matrix["complete_stress_beris_edwards"]
    active_force = matrix["legacy_active_force_active_nematics"]
    assert complete["plane_slab"] == "qualified"
    assert complete["rectangular_channel"] == "not_declared_or_lowered"
    assert active_force["plane_slab"] == "not_declared_or_lowered"
    assert active_force["rectangular_channel"] == "qualified"


def test_p770_coupling_ids_and_target_owners_are_unique() -> None:
    inventory = _inventory()
    applications = inventory["applications"]
    ids = [
        entry["id"]
        for application in applications.values()
        for entry in application["hard_couplings"]
    ]
    cross_ids = [entry["id"] for entry in inventory["cross_application_gaps"]]

    assert len(ids) == 20
    assert len(ids) == len(set(ids))
    assert len(cross_ids) == 7
    assert len(cross_ids) == len(set(cross_ids))
    assert set(inventory["target_ownership"]) == {
        "EquationSystemSpec",
        "GeometrySpec",
        "BoundaryAssignment",
        "NumericsSpec",
        "TimeIntegrationSpec",
        "ExecutionSpec",
        "WorkflowSpec",
        "SimulationSpec",
        "CapabilityLowering",
        "ApplicationPresetAdapter",
    }
