"""P8.4.3 active-nematic Q conveniences and Plane lifting lowering."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation_binding import bind_simulation_runtime
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.core.boundary import BoundarySide
from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    Q_convention_metadata,
    prescribed_q,
    strong_homeotropic_q,
    strong_planar_q,
)
from pssolver.planning import TransformKind


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / (
    "notes/architecture_v0_2/phase_8_p843_q_lifting_lowering.json"
)
IMPLEMENTATION_SOURCES = (
    "pssolver/api/capabilities.py",
    "pssolver/models/active_nematics/__init__.py",
    "pssolver/models/active_nematics/boundaries.py",
    "pssolver/planning/simulation.py",
    "pssolver/configuration/simulation_lowering.py",
    "pssolver/configuration/simulation_binding.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plane_simulation(tmp_path: Path):
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "unused_p843_plane_output",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
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


def _with_q_policy(simulation, policy):
    boundaries = assign_boundaries(
        model=simulation.equation_system,
        geometry=simulation.geometry,
        policies={
            "Q": policy,
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
        name="qualified_plane_strong_q",
    )
    return replace(simulation, boundaries=boundaries)


def _z_face_normals():
    return {
        (2, "lower"): (0.0, 0.0, -1.0),
        (2, "upper"): (0.0, 0.0, 1.0),
    }


def _face_component_values(policy):
    return {
        (value.component, value.axis, value.side): value.value.value
        for value in policy.face_values
    }


def test_q_conveniences_use_the_canonical_convention_and_component_order():
    convention = Q_convention_metadata()
    assert convention["id"] == "de_gennes_S_lambda_max_v1"
    assert convention["definition"] == "Q=(3S/2)(nn-I/3)"

    homeotropic = strong_homeotropic_q(
        scalar_order=0.6,
        face_normals=_z_face_normals(),
    )
    values = _face_component_values(homeotropic)
    assert homeotropic.field_name == "Q"
    assert len(homeotropic.face_values) == 2 * len(Q_COMPONENTS)
    for side in BoundarySide:
        assert values[("Qxx", 2, side)] == pytest.approx(-0.3)
        assert values[("Qyy", 2, side)] == pytest.approx(-0.3)
        for component in ("Qxy", "Qxz", "Qyz"):
            assert values[(component, 2, side)] == 0.0

    planar = strong_planar_q(
        scalar_order=0.6,
        face_directors={
            (2, "lower"): (1.0, 0.0, 0.0),
            (2, "upper"): (1.0, 0.0, 0.0),
        },
        face_normals=_z_face_normals(),
    )
    planar_values = _face_component_values(planar)
    for side in BoundarySide:
        assert planar_values[("Qxx", 2, side)] == pytest.approx(0.6)
        assert planar_values[("Qyy", 2, side)] == pytest.approx(-0.3)


def test_explicit_q_preserves_independent_symmetric_traceless_wall_data():
    lower = np.diag((0.6, -0.3, -0.3))
    upper = np.diag((-0.3, 0.6, -0.3))
    first = prescribed_q({(2, "lower"): lower, (2, "upper"): upper})
    second = prescribed_q({(2, "upper"): upper, (2, "lower"): lower})
    values = _face_component_values(first)

    assert first == second
    assert first.canonical_sha256() == second.canonical_sha256()
    assert values[("Qxx", 2, BoundarySide.LOWER)] == pytest.approx(0.6)
    assert values[("Qyy", 2, BoundarySide.LOWER)] == pytest.approx(-0.3)
    assert values[("Qxx", 2, BoundarySide.UPPER)] == pytest.approx(-0.3)
    assert values[("Qyy", 2, BoundarySide.UPPER)] == pytest.approx(0.6)


def test_q_conveniences_reject_invalid_order_structure_and_orientation():
    with pytest.raises(ValueError, match="non-negative"):
        strong_homeotropic_q(
            scalar_order=-0.1,
            face_normals=_z_face_normals(),
        )
    with pytest.raises(ValueError, match="unit length"):
        strong_homeotropic_q(
            scalar_order=0.6,
            face_normals={(2, "lower"): (0.0, 0.0, 2.0)},
        )
    with pytest.raises(ValueError, match="tangent"):
        strong_planar_q(
            scalar_order=0.6,
            face_directors={(2, "lower"): (0.0, 0.0, 1.0)},
            face_normals={(2, "lower"): (0.0, 0.0, -1.0)},
        )
    with pytest.raises(ValueError, match="same oriented faces"):
        strong_planar_q(
            scalar_order=0.6,
            face_directors={(2, "lower"): (1.0, 0.0, 0.0)},
            face_normals={(2, "upper"): (0.0, 0.0, 1.0)},
        )
    nonsymmetric = np.array(
        [[0.5, 0.2, 0.0], [0.0, -0.25, 0.0], [0.0, 0.0, -0.25]]
    )
    with pytest.raises(ValueError, match="symmetric"):
        prescribed_q({(2, "lower"): nonsymmetric})
    spatial = np.stack((np.diag((0.6, -0.3, -0.3)),) * 2)
    with pytest.raises(ValueError, match="spatially constant"):
        prescribed_q({(2, "lower"): spatial})


@pytest.mark.parametrize(
    "policy",
    (
        prescribed_q(
            {
                (2, "lower"): np.diag((0.6, -0.3, -0.3)),
                (2, "upper"): np.diag((-0.3, 0.6, -0.3)),
            }
        ),
        strong_homeotropic_q(
            scalar_order=0.6,
            face_normals=_z_face_normals(),
        ),
        strong_planar_q(
            scalar_order=0.6,
            face_directors={
                (2, "lower"): (1.0, 0.0, 0.0),
                (2, "upper"): (0.0, 1.0, 0.0),
            },
            face_normals=_z_face_normals(),
        ),
    ),
)
def test_plane_q_application_lowering_binds_lift_and_dst_remainder(
    tmp_path,
    policy,
):
    simulation = _with_q_policy(_plane_simulation(tmp_path), policy)
    plan = lower_simulation_spec(simulation)
    lift = plan.lifting_plan

    assert lift is not None
    assert lift.component_order == Q_COMPONENTS
    assert lift.source_equation_sha256 == (
        simulation.equation_system.canonical_sha256()
    )
    assert plan.source_simulation_sha256 == simulation.canonical_sha256()
    for component in Q_COMPONENTS:
        assert plan.basis_for(component).transform_kinds == (
            TransformKind.FFT,
            TransformKind.FFT,
            TransformKind.DST,
        )
        assert lift.for_component(component).field_name == "Q"
    metadata = plan.to_metadata()
    assert metadata["lifting_plan_sha256"] == lift.canonical_sha256()
    assert metadata["lifting_plan"]["evolved_representation"] == (
        "homogeneous_remainder"
    )
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_homogeneous_plane_lowering_metadata_remains_without_lifting_key(
    tmp_path,
):
    simulation = _plane_simulation(tmp_path)
    plan = lower_simulation_spec(simulation)

    assert plan.lifting_plan is None
    assert "lifting_plan" not in plan.to_metadata()
    assert "lifting_plan_sha256" not in plan.to_metadata()
    for component in Q_COMPONENTS:
        assert plan.basis_for(component).transform_kinds == (
            TransformKind.FFT,
            TransformKind.FFT,
            TransformKind.DCT,
        )


def test_historical_p843_runtime_rejection_is_superseded_by_p844(tmp_path):
    simulation = _with_q_policy(
        _plane_simulation(tmp_path),
        strong_homeotropic_q(
            scalar_order=0.6,
            face_normals=_z_face_normals(),
        ),
    )
    plan = lower_simulation_spec(simulation)

    binding = bind_simulation_runtime(simulation, plan)
    assert binding.runtime_path == "legacy_production"
    assert binding.lowering_plan_sha256 == plan.canonical_sha256()


def test_q_convenience_layer_does_not_own_geometry_planning_or_runtime():
    path = ROOT / "pssolver/models/active_nematics/boundaries.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imports.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    for forbidden in (
        "pssolver.geometries",
        "pssolver.operators",
        "pssolver.planning",
        "pssolver.runtime",
        "pssolver.transforms",
        "pssolver.workflows",
    ):
        assert not any(
            name == forbidden or name.startswith(f"{forbidden}.")
            for name in imports
        )


def test_p843_record_freezes_lowering_only_scope():
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))

    assert record["phase"] == "P8.4.3"
    assert record["classification"] == (
        "PASS_P8_4_3_Q_CONVENIENCES_AND_PLANE_APPLICATION_LOWERING"
    )
    assert record["q_convention_id"] == "de_gennes_S_lambda_max_v1"
    assert record["production_runtime_connected"] is False
    assert record["workflow_or_restart_changed"] is False
    assert record["h100_used"] is False
    assert record["authorization"]["p8_4_4_eligible_for_planning"] is True
    assert record["authorization"]["p8_4_4_implementation_authorized"] is False
    assert set(record["source_sha256"]) == set(IMPLEMENTATION_SOURCES)
    superseded_by_p844 = {
        "pssolver/api/capabilities.py",
        "pssolver/configuration/simulation_binding.py",
    }
    for relative, expected in record["source_sha256"].items():
        if relative in superseded_by_p844:
            continue
        assert _sha256(ROOT / relative) == expected


def test_future_verbatim_archive_lists_p843_without_regenerating_pdf():
    source = (
        ROOT / "notes/architecture_v0_2/build_verbatim_archive_pdf.py"
    ).read_text(encoding="utf-8")
    for name in (
        "phase_8_p843_q_lifting_lowering.md",
        "phase_8_p843_q_lifting_lowering.json",
    ):
        assert source.count(f'"{name}"') == 1
