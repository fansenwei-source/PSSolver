from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import pytest

from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation import InitialConditionSource
from pssolver.core.boundary import (
    BoundaryAssignment,
    BoundaryKind,
    BoundarySemantic,
    BoundarySide,
    ComponentBoundaryAssignment,
    FaceBoundaryCondition,
    HomogeneousDirichletBC,
)
from pssolver.core.domain import DomainSpec
from pssolver.geometries import PlaneSlab


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DECLARATIONS = ROOT / "pssolver" / "core" / "boundary.py"
SIMULATION_DECLARATIONS = (
    ROOT / "pssolver" / "configuration" / "simulation.py"
)
ADAPTERS = (
    ROOT
    / "pssolver"
    / "configuration"
    / "active_nematics_simulation_adapters.py"
)
RESULT_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_7_p772_simulation_composition.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plane_components(**overrides: object):
    values = {
        "activity_number": 18.0,
        "output_dir": ROOT / "unused_p772_plane_output",
        "nx": 16,
        "ny": 16,
        "nz": 8,
        "steps": 2,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "defect_min_separation": 2.0,
        "defect_core_radius": 0.5,
        "twist_modes": (1, 2, 3),
    }
    values.update(overrides)
    return decompose_plane_beris_edwards_run_spec(
        create_plane_beris_edwards_run_spec(**values)
    )


def _identity_hashes(value) -> dict[str, str]:
    return {
        name: record["sha256"]
        for name, record in value.identity_metadata().items()
    }


def _replace_component(value, replacement):
    components = tuple(
        replacement if item.component == replacement.component else item
        for item in value.boundaries.components
    )
    return replace(
        value,
        boundaries=replace(value.boundaries, components=components),
    )


def test_p772_declarations_are_tensor_and_runtime_free():
    forbidden = {
        "numpy",
        "torch",
        "pssolver.backends",
        "pssolver.execution",
        "pssolver.linear_solvers",
        "pssolver.operators",
        "pssolver.runtime",
        "pssolver.transforms",
        "pssolver.workflows",
    }
    for path in (BOUNDARY_DECLARATIONS, SIMULATION_DECLARATIONS, ADAPTERS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert not {
            name
            for name in imports
            if any(
                name == blocked or name.startswith(f"{blocked}.")
                for blocked in forbidden
            )
        }, path


def test_plane_adapter_composes_physical_faces_without_modal_leakage():
    components = _plane_components()
    value = compose_plane_beris_edwards_simulation(components)

    assert value.equation_system.variant == "complete_stress_beris_edwards"
    assert value.geometry.periodic_axes == (0, 1)
    assert value.geometry.bounded_axes == (2,)
    assert value.boundaries.component_names == (
        "Qxx",
        "Qxy",
        "Qxz",
        "Qyy",
        "Qyz",
        "p",
        "ux",
        "uy",
        "uz",
    )
    qxx = value.boundaries.for_component("Qxx")
    ux = value.boundaries.for_component("ux")
    uz = value.boundaries.for_component("uz")
    pressure = value.boundaries.for_component("p")
    assert qxx.semantic is BoundarySemantic.PHYSICAL
    assert ux.semantic is BoundarySemantic.PHYSICAL
    assert uz.semantic is BoundarySemantic.PHYSICAL
    assert pressure.semantic is BoundarySemantic.ALGEBRAIC_COMPATIBILITY
    assert {
        face.condition.kind for face in qxx.faces if face.axis == 2
    } == {BoundaryKind.NEUMANN}
    assert {
        face.condition.kind for face in uz.faces if face.axis == 2
    } == {BoundaryKind.DIRICHLET}
    assert {
        face.condition.kind for face in pressure.faces if face.axis == 2
    } == {BoundaryKind.NEUMANN}

    declared = set(value.equation_system.component_names)
    assert set(value.boundaries.component_names) <= declared
    assert not any(
        name.startswith(("dQ", "du", "H", "sigma", "force"))
        for name in value.boundaries.component_names
    )
    assert "distortion_odd_z" not in json.dumps(
        value.boundaries.to_metadata(),
        sort_keys=True,
    )
    assert value.discretization_parameters[
        "legacy_derived_boundary_spaces"
    ]["distortion_odd_z"] == (
        components.effective_boundaries.distortion_odd_z.to_metadata()
    )


def test_channel_adapter_preserves_no_slip_and_pressure_solver_contract():
    components = ChannelActiveNematicRunSpec().components
    value = compose_channel_active_nematics_simulation(components)

    assert value.equation_system.variant == (
        "legacy_active_force_active_nematics"
    )
    assert value.geometry.periodic_axes == (0,)
    assert value.geometry.bounded_axes == (1, 2)
    for component in ("ux", "uy", "uz"):
        assignment = value.boundaries.for_component(component)
        bounded_kinds = {
            face.condition.kind
            for face in assignment.faces
            if face.axis in (1, 2)
        }
        assert bounded_kinds == {BoundaryKind.DIRICHLET}
    assert value.boundaries.for_component("p").semantic is (
        BoundarySemantic.ALGEBRAIC_COMPATIBILITY
    )
    assert value.discretization_parameters["pressure_solver"] == (
        components.pressure_solver.to_metadata()
    )
    assert value.initial_condition.family == "aligned_x_smooth_noise"
    assert value.initial_condition.source is InitialConditionSource.GENERATED


def test_compatibility_adapters_preserve_complete_source_component_metadata():
    plane = _plane_components()
    channel = ChannelActiveNematicRunSpec().components
    plane_value = compose_plane_beris_edwards_simulation(plane)
    channel_value = compose_channel_active_nematics_simulation(channel)

    assert plane_value.compatibility_metadata["source_components"] == (
        plane.to_metadata()
    )
    assert channel_value.compatibility_metadata["source_components"] == (
        channel.to_metadata()
    )
    json.dumps(plane_value.to_metadata(), allow_nan=False, sort_keys=True)
    json.dumps(channel_value.to_metadata(), allow_nan=False, sort_keys=True)


def test_channel_snapshot_source_is_not_relabelled_as_generated_family():
    components = ChannelActiveNematicRunSpec().components
    snapshot_components = replace(
        components,
        initial_condition=replace(
            components.initial_condition,
            mode="snapshot",
        ),
    )
    value = compose_channel_active_nematics_simulation(snapshot_components)

    assert value.initial_condition.family == "external_snapshot"
    assert value.initial_condition.source is InitialConditionSource.SNAPSHOT


def test_face_declaration_can_represent_asymmetric_bounded_wall_laws():
    value = compose_plane_beris_edwards_simulation(_plane_components())
    qxx = value.boundaries.for_component("Qxx")
    faces = tuple(
        replace(face, condition=HomogeneousDirichletBC())
        if face.axis == 2 and face.side is BoundarySide.LOWER
        else face
        for face in qxx.faces
    )
    asymmetric = replace(qxx, faces=faces)
    updated = _replace_component(value, asymmetric)

    wall = {
        face.side: face.condition.kind
        for face in updated.boundaries.for_component("Qxx").faces
        if face.axis == 2
    }
    assert wall == {
        BoundarySide.LOWER: BoundaryKind.DIRICHLET,
        BoundarySide.UPPER: BoundaryKind.NEUMANN,
    }


def test_simulation_rejects_topology_mismatch_missing_and_transient_components():
    value = compose_plane_beris_edwards_simulation(_plane_components())
    qxx = value.boundaries.for_component("Qxx")
    bad_faces = tuple(
        replace(face, condition=HomogeneousDirichletBC())
        if face.axis == 0 and face.side is BoundarySide.LOWER
        else face
        for face in qxx.faces
    )
    with pytest.raises(ValueError, match="periodic axis 0"):
        _replace_component(value, replace(qxx, faces=bad_faces))

    missing = BoundaryAssignment(
        name=value.boundaries.name,
        ndim=value.boundaries.ndim,
        components=value.boundaries.components[:-1],
    )
    with pytest.raises(ValueError, match="cover exactly"):
        replace(value, boundaries=missing)

    transient = replace(qxx, component="dQxx_dx")
    extra = BoundaryAssignment(
        name=value.boundaries.name,
        ndim=value.boundaries.ndim,
        components=(*value.boundaries.components, transient),
    )
    with pytest.raises(ValueError, match="cover exactly"):
        replace(value, boundaries=extra)


def test_evolved_components_cannot_be_labeled_algebraic_compatibility():
    value = compose_plane_beris_edwards_simulation(_plane_components())
    qxx = value.boundaries.for_component("Qxx")
    with pytest.raises(ValueError, match="requires a physical"):
        _replace_component(
            value,
            replace(
                qxx,
                semantic=BoundarySemantic.ALGEBRAIC_COMPATIBILITY,
            ),
        )


def test_four_identity_classes_change_independently():
    value = compose_plane_beris_edwards_simulation(_plane_components())
    baseline = _identity_hashes(value)

    workflow = replace(value.workflow, steps=value.workflow.steps + 1)
    run_changed = replace(value, workflow=workflow)
    observed = _identity_hashes(run_changed)
    assert observed["run"] != baseline["run"]
    assert observed["scientific"] == baseline["scientific"]
    assert observed["discretization"] == baseline["discretization"]
    assert observed["execution"] == baseline["execution"]

    options = dict(value.execution.options)
    options["device"] = "cpu"
    execution_changed = replace(
        value,
        execution=replace(value.execution, options=options),
    )
    observed = _identity_hashes(execution_changed)
    assert observed["execution"] != baseline["execution"]
    assert observed["scientific"] == baseline["scientific"]
    assert observed["discretization"] == baseline["discretization"]
    assert observed["run"] == baseline["run"]

    domain = value.geometry.domain
    refined = PlaneSlab(
        DomainSpec(
            shape=(
                domain.shape[0] * 2,
                domain.shape[1],
                domain.shape[2],
            ),
            lengths=domain.lengths,
            axis_names=domain.axis_names,
            grid_placement=domain.grid_placement,
        )
    )
    discretization_changed = replace(value, geometry=refined)
    observed = _identity_hashes(discretization_changed)
    assert observed["discretization"] != baseline["discretization"]
    assert observed["scientific"] == baseline["scientific"]
    assert observed["execution"] == baseline["execution"]
    assert observed["run"] == baseline["run"]

    compatibility = dict(value.compatibility_metadata)
    compatibility["adapter_note"] = "non_authoritative"
    provenance_changed = replace(
        value,
        compatibility_metadata=compatibility,
    )
    assert _identity_hashes(provenance_changed) == baseline
    assert provenance_changed.canonical_sha256() != value.canonical_sha256()

    renamed_boundaries = replace(
        value,
        boundaries=replace(value.boundaries, name="equivalent_label"),
    )
    assert _identity_hashes(renamed_boundaries) == baseline
    assert renamed_boundaries.canonical_sha256() != value.canonical_sha256()


def test_simulation_spec_is_frozen_and_deterministic():
    first = compose_plane_beris_edwards_simulation(_plane_components())
    second = compose_plane_beris_edwards_simulation(_plane_components())

    assert first == second
    assert first.canonical_sha256() == second.canonical_sha256()
    assert len(first.canonical_sha256()) == 64
    with pytest.raises(FrozenInstanceError):
        first.geometry = second.geometry


def test_p772_remains_disconnected_from_applications_and_package_roots():
    plane_application = (
        ROOT / "pssolver" / "applications" / "plane_beris_edwards.py"
    ).read_text(encoding="utf-8")
    channel_application = (
        ROOT / "pssolver" / "applications" / "channel_active_nematics.py"
    ).read_text(encoding="utf-8")
    configuration_root = (
        ROOT / "pssolver" / "configuration" / "__init__.py"
    ).read_text(encoding="utf-8")
    core_root = (ROOT / "pssolver" / "core" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert "active_nematics_simulation_adapters" not in plane_application
    assert "active_nematics_simulation_adapters" not in channel_application
    assert "from .simulation" not in configuration_root
    assert "BoundaryAssignment" not in core_root


def test_p772_machine_readable_record_matches_sources_and_scope():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["classification"] == (
        "PASS_P7_7_2_BOUNDARY_ASSIGNMENT_SIMULATION_COMPOSITION"
    )
    assert record["runtime_connection"] is False
    assert record["basis_or_solver_lowering"] is False
    assert record["production_default_changed"] is False
    assert record["phase_8_authorized"] is False
    assert record["source_sha256"] == {
        "pssolver/configuration/active_nematics_simulation_adapters.py": (
            _sha256(ADAPTERS)
        ),
        "pssolver/configuration/simulation.py": (
            _sha256(SIMULATION_DECLARATIONS)
        ),
        "pssolver/core/boundary.py": _sha256(BOUNDARY_DECLARATIONS),
    }
