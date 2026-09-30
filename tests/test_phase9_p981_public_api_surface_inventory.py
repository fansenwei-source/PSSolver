"""Static inventory gates for Phase 9 P9.8.1."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path

import pssolver
import pssolver.functional as functional
from pssolver.functional import (
    FUNCTIONAL_API_VERSION,
    CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION,
    PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION,
    FunctionalCapabilitySet,
    FunctionalCheckpointState,
    FunctionalControlFieldSpec,
    FunctionalObservationSpec,
    FunctionalRuntimeIdentity,
    FunctionalStateSpec,
    FunctionalTensorSpec,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD = NOTES / "phase_9_p981_public_api_surface_inventory.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _metadata_examples() -> dict[str, set[str]]:
    tensor = FunctionalTensorSpec(
        name="Q",
        shape=(1, 5, 4, 3, 2),
        dtype="float64",
        device="cpu",
        batch_axis=0,
        layout="batch_component_xyz",
        meaning="inventory probe",
        component_names=("xx", "xy", "xz", "yy", "yz"),
    )
    state = FunctionalStateSpec(components=(tensor,))
    control_tensor = FunctionalTensorSpec(
        name="activity",
        shape=(1, 4, 3, 2),
        dtype="float64",
        device="cpu",
        batch_axis=0,
        layout="batch_xyz",
        meaning="inventory probe",
    )
    control = FunctionalControlFieldSpec(
        name="activity",
        tensor=control_tensor,
        equation_term="div(beta * activity * Q)",
        injection_order="multiply_before_differentiation",
        dealiasing_identity="inventory_probe",
        grid_location="nodes",
        broadcast_rules=("batch_one",),
        admissible_min=0.0,
    )
    observation = FunctionalObservationSpec(
        name="Q",
        tensor=tensor,
        convention="physical_Q",
        control_dependent=False,
        terminal_available_without_control=True,
    )
    capabilities = FunctionalCapabilitySet()
    identity = FunctionalRuntimeIdentity(
        scientific={"probe": 1},
        discretization={"probe": 1},
        execution={"probe": 1},
        state_layout=state.to_metadata(),
    )
    return {
        "FunctionalTensorSpec": set(tensor.to_metadata()),
        "FunctionalStateSpec": set(state.to_metadata()),
        "FunctionalControlFieldSpec": set(control.to_metadata()),
        "FunctionalObservationSpec": set(observation.to_metadata()),
        "FunctionalCapabilitySet": set(capabilities.to_metadata()),
        "FunctionalRuntimeIdentity": set(identity.to_metadata()),
    }


def test_inventory_matches_the_exact_functional_export_surface():
    record = _record()
    exported = record["module"]["public_exports"]
    assert record["module"]["export_count"] == 57
    assert set(exported) <= set(functional.__all__)
    assert len(exported) == len(set(exported))
    assert record["module"]["api_version"] == "0.1-provisional"
    assert FUNCTIONAL_API_VERSION == "1.0"
    assert record["module"]["package_root_reexported"] is False
    assert not set(exported) & set(pssolver.__all__)


def test_review_groups_partition_every_export_once():
    record = _record()
    groups = record["export_review_groups"]
    flattened = [name for values in groups.values() for name in values]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == set(record["module"]["public_exports"])


def test_two_consumer_import_inventory_is_a_twelve_name_public_subset():
    record = _record()
    usage = record["qualified_consumer_usage"]
    union = set(usage["periodic_direct_imports"]) | set(
        usage["channel_direct_imports"]
    )
    assert usage["direct_import_union_count"] == 12
    assert union == set(usage["direct_import_union"])
    assert union <= set(record["module"]["public_exports"])
    assert usage["consumer_repository_identity_recorded_here"] is False
    assert usage["consumer_objective_or_experiment_recorded_here"] is False


def test_metadata_key_inventory_matches_constructed_contract_values():
    schemas = _record()["metadata_schemas"]
    for name, keys in _metadata_examples().items():
        assert set(schemas[name]) == keys
    assert set(schemas["FunctionalCheckpointState_dataclass_fields"]) <= {
        field.name for field in fields(FunctionalCheckpointState)
    }


def test_protocol_and_checkpoint_versions_are_distinctly_recorded():
    record = _record()
    protocols = record["protocols"]
    assert set(protocols["runtime_properties"]) == {
        "api_version",
        "state_spec",
        "control_specs",
        "control_field_schema",
        "observation_specs",
        "capabilities",
        "checkpoint_bridge",
    }
    assert set(protocols["runtime_methods"]) == {
        "identity",
        "initial_state",
        "step",
        "observe",
        "step_and_observe",
    }
    checkpoints = record["checkpoint_formats"]
    assert checkpoints["periodic"]["bridge_format_version"] == (
        PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION
    )
    assert checkpoints["channel"]["bridge_format_version"] == (
        CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION
    )
    assert checkpoints["same_integer_version_implies_same_format"] is False
    assert checkpoints["channel"]["pressure_persisted"] is False
    assert checkpoints["channel"]["pressure_warm_start_persisted"] is False


def test_source_bindings_remain_well_formed_historical_baseline_records():
    for binding in _record()["source_bindings"]:
        path = ROOT / binding["path"]
        assert path.is_file()
        assert len(binding["sha256"]) == 64
        assert set(binding["sha256"]) <= set("0123456789abcdef")


def test_p981_only_authorizes_p982_planning_and_tracks_archive_sources():
    record = _record()
    authorization = record["authorization"]
    assert authorization["p9_8_1_complete"] is True
    assert authorization["p9_8_2_planning_eligible"] is True
    assert authorization["p9_8_2_implementation_authorized"] is False
    assert authorization["stable_api_published"] is False
    assert authorization["functional_api_version_changed"] is False
    assert authorization["runtime_or_checkpoint_changed"] is False
    assert authorization["phase_9_complete"] is False

    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "phase_9_p981_public_api_surface_inventory.md",
        "phase_9_p981_public_api_surface_inventory.json",
    ):
        assert f'"{name}"' in source
    assert authorization["verbatim_pdf_regenerated"] is False
