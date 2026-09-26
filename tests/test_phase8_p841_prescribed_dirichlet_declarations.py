"""P8.4.1 generic static prescribed-Dirichlet declaration contracts."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import pssolver.boundaries as public_boundaries
from pssolver.boundaries import (
    HomogeneousBoundaryPolicy,
    PrescribedFaceValue,
    StaticPrescribedDirichletPolicy,
    assign_boundaries,
    prescribed_dirichlet,
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
from pssolver.configuration.simulation_lowering import (
    LoweringRejectionCode,
    SimulationLoweringError,
    lower_simulation_spec,
)
from pssolver.core.boundary import (
    BoundaryCondition,
    BoundaryKind,
    BoundarySemantic,
    BoundarySide,
    PeriodicBC,
    PrescribedDirichletBC,
    StaticConstantBoundaryValue,
)
from pssolver.core.domain import DomainSpec
from pssolver.core.fields import FieldRole
from pssolver.geometries import PlaneSlab
from pssolver.systems.algebraic import AlgebraicSystemSpec
from pssolver.systems.equations import (
    EquationFieldSpec,
    EquationSystemSpec,
    EquationTermSpec,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / (
    "notes/architecture_v0_2/"
    "phase_8_p841_prescribed_dirichlet_declarations.json"
)
GENERIC_SOURCES = (
    "pssolver/core/boundary.py",
    "pssolver/boundaries/prescribed.py",
    "pssolver/boundaries/homogeneous.py",
    "pssolver/configuration/simulation_lowering.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generic_system() -> EquationSystemSpec:
    return EquationSystemSpec(
        name="generic_transport",
        variant="generic_scalar_transport",
        fields=(
            EquationFieldSpec(
                "concentration",
                FieldRole.EVOLVED,
                ("c",),
            ),
            EquationFieldSpec("flux", FieldRole.TRANSIENT, ("j",)),
            EquationFieldSpec(
                "multiplier",
                FieldRole.ALGEBRAIC,
                ("lambda_value",),
            ),
        ),
        evolution_laws=(
            EquationTermSpec(
                "evolve_concentration",
                "generic_transport_rhs",
                ("c",),
                ("c", "j"),
            ),
        ),
        constitutive_laws=(
            EquationTermSpec(
                "construct_flux",
                "generic_flux",
                ("j",),
                ("c",),
            ),
        ),
        algebraic_systems=(
            AlgebraicSystemSpec(
                "solve_multiplier",
                "generic_constraint",
                ("lambda_value",),
                ("c", "j"),
            ),
        ),
        parameters={"diffusivity": 1.0},
    )


def _geometry() -> PlaneSlab:
    return PlaneSlab(DomainSpec((8, 6, 4), (4.0, 3.0, 2.0)))


def _multiplier_policy() -> HomogeneousBoundaryPolicy:
    return HomogeneousBoundaryPolicy(
        field_name="multiplier",
        semantic=BoundarySemantic.ALGEBRAIC_COMPATIBILITY,
        kind="neumann",
    )


def _generic_assignment():
    model = _generic_system()
    geometry = _geometry()
    return assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "concentration": prescribed_dirichlet(
                "concentration",
                {
                    ("c", 2, "lower"): 1.25,
                    ("c", 2, BoundarySide.UPPER): -0.5,
                },
            ),
            "multiplier": _multiplier_policy(),
        },
    )


def _plane_simulation():
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=ROOT / "unused_p841_plane_output",
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


def test_static_constant_value_is_finite_canonical_and_content_addressed():
    value = StaticConstantBoundaryValue(-0.0)
    same = StaticConstantBoundaryValue(0)

    assert value.value == 0.0
    assert value.to_metadata() == {
        "representation": "constant_scalar",
        "time_dependence": "static",
        "value": 0.0,
    }
    assert value.canonical_sha256() == same.canonical_sha256()
    assert len(value.canonical_sha256()) == 64
    json.dumps(value.to_metadata(), allow_nan=False, sort_keys=True)

    with pytest.raises(TypeError, match="real scalar"):
        StaticConstantBoundaryValue(True)
    with pytest.raises(TypeError, match="real scalar"):
        StaticConstantBoundaryValue("1.0")
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            StaticConstantBoundaryValue(invalid)


def test_prescribed_condition_cannot_omit_its_value_identity():
    condition = PrescribedDirichletBC(2.5)

    assert condition.kind is BoundaryKind.DIRICHLET
    assert condition.is_homogeneous is False
    assert condition.value == StaticConstantBoundaryValue(2.5)
    assert condition.to_metadata()["value_sha256"] == (
        condition.value.canonical_sha256()
    )
    with pytest.raises(ValueError, match="homogeneous boundary contracts only"):
        BoundaryCondition(BoundaryKind.DIRICHLET, is_homogeneous=False)


def test_public_policy_has_deterministic_per_component_face_identity():
    first = prescribed_dirichlet(
        "concentration",
        {
            ("c", 2, "upper"): -0.5,
            ("c", 2, "lower"): 1.25,
        },
    )
    second = StaticPrescribedDirichletPolicy(
        "concentration",
        (
            PrescribedFaceValue("c", 2, BoundarySide.LOWER, 1.25),
            PrescribedFaceValue("c", 2, BoundarySide.UPPER, -0.5),
        ),
    )

    assert first == second
    assert first.canonical_sha256() == second.canonical_sha256()
    assert [value.side for value in first.face_values] == [
        BoundarySide.LOWER,
        BoundarySide.UPPER,
    ]
    with pytest.raises(ValueError, match="must be unique"):
        StaticPrescribedDirichletPolicy(
            "concentration",
            (second.face_values[0], second.face_values[0]),
        )


def test_generic_evolved_field_composition_preserves_independent_wall_values():
    assignment = _generic_assignment()
    concentration = assignment.for_component("c")

    for axis in (0, 1):
        assert {
            type(face.condition)
            for face in concentration.faces
            if face.axis == axis
        } == {PeriodicBC}
    wall = {
        face.side: face.condition
        for face in concentration.faces
        if face.axis == 2
    }
    assert wall[BoundarySide.LOWER] == PrescribedDirichletBC(1.25)
    assert wall[BoundarySide.UPPER] == PrescribedDirichletBC(-0.5)
    assert concentration.semantic is BoundarySemantic.PHYSICAL


def test_composition_rejects_missing_extra_periodic_and_wrong_role_values():
    model = _generic_system()
    geometry = _geometry()
    with pytest.raises(ValueError, match="missing=.*upper"):
        assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "concentration": prescribed_dirichlet(
                    "concentration",
                    {("c", 2, "lower"): 1.0},
                ),
                "multiplier": _multiplier_policy(),
            },
        )
    with pytest.raises(ValueError, match="extra=.*'c'.*0.*lower"):
        assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "concentration": prescribed_dirichlet(
                    "concentration",
                    {
                        ("c", 0, "lower"): 0.0,
                        ("c", 2, "lower"): 1.0,
                        ("c", 2, "upper"): 1.0,
                    },
                ),
                "multiplier": _multiplier_policy(),
            },
        )
    with pytest.raises(ValueError, match="supported only for evolved"):
        assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "concentration": HomogeneousBoundaryPolicy(
                    "concentration",
                    BoundarySemantic.PHYSICAL,
                    "neumann",
                ),
                "multiplier": prescribed_dirichlet(
                    "multiplier",
                    {
                        ("lambda_value", 2, "lower"): 0.0,
                        ("lambda_value", 2, "upper"): 0.0,
                    },
                ),
            },
        )


def test_current_lowering_rejects_prescribed_data_before_selecting_dst():
    simulation = _plane_simulation()
    qxx = simulation.boundaries.for_component("Qxx")
    prescribed = replace(
        qxx,
        faces=tuple(
            replace(
                face,
                condition=PrescribedDirichletBC(
                    1.0 if face.side is BoundarySide.LOWER else 2.0
                ),
            )
            if face.axis == 2
            else face
            for face in qxx.faces
        ),
    )
    boundaries = replace(
        simulation.boundaries,
        components=tuple(
            prescribed if value.component == "Qxx" else value
            for value in simulation.boundaries.components
        ),
    )
    candidate = replace(simulation, boundaries=boundaries)

    with pytest.raises(SimulationLoweringError) as caught:
        lower_simulation_spec(candidate)
    assert caught.value.rejection.code is (
        LoweringRejectionCode.UNSUPPORTED_PRESCRIBED_BOUNDARY
    )
    assert caught.value.rejection.to_metadata()["context"] == {
        "component": "Qxx",
        "faces": [
            {"axis": 2, "side": "lower"},
            {"axis": 2, "side": "upper"},
        ],
    }


def test_generic_declaration_layer_has_no_model_tensor_or_runtime_dependency():
    forbidden_imports = {
        "numpy",
        "torch",
        "pssolver.applications",
        "pssolver.backends",
        "pssolver.execution",
        "pssolver.models",
        "pssolver.operators",
        "pssolver.runtime",
        "pssolver.transforms",
        "pssolver.workflows",
    }
    for relative in (
        "pssolver/core/boundary.py",
        "pssolver/boundaries/prescribed.py",
    ):
        path = ROOT / relative
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
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
                for blocked in forbidden_imports
            )
        }
        lowered = source.lower()
        assert "director" not in lowered
        assert "scalar_order" not in lowered
        assert "nematic" not in lowered


def test_public_exports_and_record_freeze_p841_scope():
    assert {
        "PrescribedFaceValue",
        "StaticPrescribedDirichletPolicy",
        "prescribed_dirichlet",
    } <= set(public_boundaries.__all__)

    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    assert record["phase"] == "P8.4.1"
    assert record["classification"] == (
        "PASS_P8_4_1_GENERIC_PRESCRIBED_DIRICHLET_DECLARATIONS"
    )
    assert record["generic_field_support"] is True
    assert record["numerical_lifting_implemented"] is False
    assert record["new_executable_combinations"] == []
    assert record["h100_used"] is False
    assert record["authorization"]["p8_4_2_eligible_for_planning"] is True
    assert record["authorization"]["p8_4_2_implementation_authorized"] is False
    assert set(record["source_sha256"]) == set(GENERIC_SOURCES)
    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected


def test_future_verbatim_archive_lists_p841_without_regenerating_pdf():
    source = (
        ROOT
        / "notes/architecture_v0_2/build_verbatim_archive_pdf.py"
    ).read_text(encoding="utf-8")
    for name in (
        "phase_8_p841_prescribed_dirichlet_declarations.md",
        "phase_8_p841_prescribed_dirichlet_declarations.json",
    ):
        assert source.count(f'"{name}"') == 1
