"""P7.7.7 public tensor-free Simulation contract."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import pssolver
from pssolver import Simulation
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation import ExecutionSpec, WorkflowSpec
from pssolver.core.domain import DomainSpec
from pssolver.geometries import PlaneSlab


ROOT = Path(__file__).resolve().parents[1]
API_SOURCE = ROOT / "pssolver/api/simulation.py"
RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_7_p777_public_simulation_api.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reference_spec():
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=ROOT / "unused_p777_plane_output",
        nx=16,
        ny=16,
        nz=8,
        steps=2,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        defect_min_separation=2.0,
        defect_core_radius=0.5,
        twist_modes=(1, 2, 3),
    )
    return compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )


def _simulation() -> Simulation:
    source = _reference_spec()
    return Simulation(
        model=source.equation_system,
        geometry=source.geometry,
        boundaries=source.boundaries,
        numerics=source.numerics,
        time=source.time_integration,
        initial_condition=source.initial_condition,
        execution=source.execution,
        output=source.workflow,
        discretization=source.discretization_parameters,
        invocation=source.invocation,
    )


def _identity_hashes(value: Simulation) -> dict[str, str]:
    return {
        name: record["sha256"]
        for name, record in value.identity_metadata().items()
    }


def test_public_root_exports_only_the_declaration_in_this_slice():
    from pssolver.api import Simulation as ApiSimulation

    assert Simulation is ApiSimulation
    assert pssolver.Simulation is Simulation
    assert "Simulation" in pssolver.__all__
    assert "run_simulation" not in pssolver.__all__
    assert not hasattr(pssolver, "run_simulation")


def test_constructor_signature_freezes_field_ownership_and_order():
    assert tuple(inspect.signature(Simulation).parameters) == (
        "model",
        "geometry",
        "boundaries",
        "numerics",
        "time",
        "initial_condition",
        "execution",
        "output",
        "discretization",
        "invocation",
    )


def test_public_simulation_is_frozen_tensor_free_and_canonical():
    value = _simulation()
    source = _reference_spec()

    assert value.specification.equation_system is value.model
    assert value.specification.geometry is value.geometry
    assert value.specification.boundaries is value.boundaries
    assert value.specification.numerics is value.numerics
    assert value.specification.time_integration is value.time
    assert value.specification.initial_condition is value.initial_condition
    assert value.specification.execution is value.execution
    assert value.specification.workflow is value.output
    assert value.identity_metadata() == source.identity_metadata()
    assert value.canonical_sha256() == value.specification.canonical_sha256()
    json.dumps(value.to_metadata(), allow_nan=False, sort_keys=True)

    with pytest.raises(FrozenInstanceError):
        value.output = WorkflowSpec(steps=3)
    with pytest.raises(TypeError):
        value.discretization["new_option"] = True


def test_existing_composition_errors_are_preserved_at_the_public_boundary():
    value = _simulation()
    incompatible_geometry = PlaneSlab(
        DomainSpec(shape=(16, 16, 8), lengths=(100.0, 100.0, 20.0)),
        wall_normal_axis=1,
    )
    with pytest.raises(ValueError, match="axis"):
        replace(value, geometry=incompatible_geometry)
    with pytest.raises(TypeError, match="equation_system"):
        replace(value, model=object())


def test_public_fields_preserve_the_four_identity_classes():
    value = _simulation()
    original = _identity_hashes(value)

    longer = replace(
        value,
        output=replace(value.output, steps=value.output.steps + 1),
    )
    run_changed = _identity_hashes(longer)
    assert run_changed["run"] != original["run"]
    assert {
        name for name in original if original[name] != run_changed[name]
    } == {"run"}

    other_device = replace(
        value,
        execution=ExecutionSpec(
            backend=value.execution.backend,
            runtime_path=value.execution.runtime_path,
            options={**value.execution.options, "device": "cpu"},
        ),
    )
    execution_changed = _identity_hashes(other_device)
    assert {
        name for name in original if original[name] != execution_changed[name]
    } == {"execution"}

    refined = replace(
        value,
        geometry=PlaneSlab(
            DomainSpec(
                shape=(32, 32, 16),
                lengths=value.geometry.domain.lengths,
            )
        ),
    )
    discretization_changed = _identity_hashes(refined)
    assert {
        name
        for name in original
        if original[name] != discretization_changed[name]
    } == {"discretization"}


def test_api_declaration_has_no_runtime_or_third_party_dependency():
    tree = ast.parse(API_SOURCE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden = {
        "numpy",
        "torch",
        "pssolver.applications",
        "pssolver.backends",
        "pssolver.execution",
        "pssolver.linear_solvers",
        "pssolver.operators",
        "pssolver.runtime",
        "pssolver.transforms",
        "pssolver.workflows",
    }
    assert not {
        name
        for name in imports
        if any(name == item or name.startswith(f"{item}.") for item in forbidden)
    }


def test_machine_record_freezes_scope_and_source_identity():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_P7_7_7_PUBLIC_SIMULATION_API_CONTRACT"
    )
    assert record["public_root_exports"] == ["Simulation"]
    assert record["run_simulation_connected"] is False
    assert record["runtime_construction_connected"] is False
    assert record["phase_8_authorized"] is False
    assert record["production_default_changed"] is False
    assert record["source_sha256"]["pssolver/api/simulation.py"] == _sha256(
        API_SOURCE
    )
