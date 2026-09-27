"""P8.5.1 field-neutral static Robin declaration contracts."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import pssolver
import pssolver.boundaries as public_boundaries
from pssolver.boundaries import (
    HomogeneousBoundaryPolicy,
    RobinFaceLaw,
    StaticRobinBoundaryPolicy,
    assign_boundaries,
    robin,
)
from pssolver.configuration.simulation_lowering import (
    LoweringRejectionCode,
    SimulationLoweringError,
    lower_simulation_spec,
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
from pssolver.core.boundary import (
    BoundaryCondition,
    BoundaryKind,
    BoundarySemantic,
    BoundarySide,
    PeriodicBC,
    StaticConstantBoundaryValue,
    StaticRobinBC,
    StaticRobinCoefficients,
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
NOTES = ROOT / "notes" / "architecture_v0_2"
RECORD_PATH = NOTES / "phase_8_p851_robin_declarations.json"
P851_IMPLEMENTATION_COMMIT = "34c4c8f1884aefe8f700ad387a2be324a6a3e078"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_at_commit(relative: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{P851_IMPLEMENTATION_COMMIT}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


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


def _policy() -> StaticRobinBoundaryPolicy:
    return robin(
        "concentration",
        {
            ("c", 2, "lower"): (2.0, 3.0, -4.0),
            ("c", 2, BoundarySide.UPPER): (5.0, 7.0, 11.0),
        },
    )


def _assignment():
    return assign_boundaries(
        model=_generic_system(),
        geometry=_geometry(),
        policies={
            "concentration": _policy(),
            "multiplier": _multiplier_policy(),
        },
    )


def _plane_simulation():
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=ROOT / "unused_p851_plane_output",
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


def test_static_robin_coefficients_are_finite_raw_and_content_addressed():
    coefficients = StaticRobinCoefficients(2, -0.0, 3.5)
    same = StaticRobinCoefficients(2.0, 0.0, 3.5)

    assert coefficients.alpha == StaticConstantBoundaryValue(2.0)
    assert coefficients.beta == StaticConstantBoundaryValue(0.0)
    assert coefficients.gamma == StaticConstantBoundaryValue(3.5)
    assert coefficients.is_homogeneous is False
    assert coefficients.canonical_sha256() == same.canonical_sha256()
    assert coefficients.to_metadata() == {
        "alpha": StaticConstantBoundaryValue(2.0).to_metadata(),
        "beta": StaticConstantBoundaryValue(0.0).to_metadata(),
        "gamma": StaticConstantBoundaryValue(3.5).to_metadata(),
        "canonical_form": "alpha*phi+beta*(n_dot_grad_phi)=gamma",
        "coefficient_roles": {
            "alpha": "field_multiplier",
            "beta": "outward_normal_gradient_multiplier",
            "gamma": "right_hand_side",
        },
        "normal_derivative_convention": "outward_unit_normal",
        "normalization": "raw_coefficients",
    }
    json.dumps(coefficients.to_metadata(), allow_nan=False, sort_keys=True)

    with pytest.raises(ValueError, match="cannot both be zero"):
        StaticRobinCoefficients(0.0, -0.0, 1.0)
    with pytest.raises(ValueError, match="alpha, beta, and gamma"):
        robin(
            "concentration",
            {("c", 2, "lower"): (1.0, 2.0)},
        )
    with pytest.raises(TypeError, match="real scalar"):
        StaticRobinCoefficients(True, 1.0, 0.0)
    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            StaticRobinCoefficients(1.0, invalid, 0.0)


def test_static_robin_condition_preserves_homogeneous_semantics_and_identity():
    homogeneous = StaticRobinBC(1.0, 2.0, 0.0)
    prescribed = StaticRobinBC(1.0, 2.0, 3.0)

    assert homogeneous.kind is BoundaryKind.ROBIN
    assert homogeneous.is_homogeneous is True
    assert prescribed.kind is BoundaryKind.ROBIN
    assert prescribed.is_homogeneous is False
    assert prescribed.to_metadata()["coefficients_sha256"] == (
        prescribed.coefficients.canonical_sha256()
    )
    assert prescribed.to_metadata()["coefficients"][
        "normal_derivative_convention"
    ] == "outward_unit_normal"
    with pytest.raises(ValueError, match="require StaticRobinBC"):
        BoundaryCondition(BoundaryKind.ROBIN)


def test_public_policy_is_deterministic_and_keeps_raw_scaling_identity():
    first = _policy()
    second = StaticRobinBoundaryPolicy(
        "concentration",
        (
            RobinFaceLaw("c", 2, "upper", (5.0, 7.0, 11.0)),
            RobinFaceLaw("c", 2, "lower", (2.0, 3.0, -4.0)),
        ),
    )
    physically_scaled = robin(
        "concentration",
        {
            ("c", 2, "lower"): (4.0, 6.0, -8.0),
            ("c", 2, "upper"): (10.0, 14.0, 22.0),
        },
    )

    assert first == second
    assert first.canonical_sha256() == second.canonical_sha256()
    assert first.canonical_sha256() != physically_scaled.canonical_sha256()
    assert [value.side for value in first.face_laws] == [
        BoundarySide.LOWER,
        BoundarySide.UPPER,
    ]
    with pytest.raises(ValueError, match="must be unique"):
        StaticRobinBoundaryPolicy(
            "concentration",
            (second.face_laws[0], second.face_laws[0]),
        )


def test_generic_evolved_field_composition_preserves_independent_face_laws():
    assignment = _assignment()
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
    assert wall[BoundarySide.LOWER] == StaticRobinBC(2.0, 3.0, -4.0)
    assert wall[BoundarySide.UPPER] == StaticRobinBC(5.0, 7.0, 11.0)
    assert concentration.semantic is BoundarySemantic.PHYSICAL


def test_composition_rejects_missing_extra_periodic_and_wrong_role_laws():
    model = _generic_system()
    geometry = _geometry()
    with pytest.raises(ValueError, match="missing=.*upper"):
        assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "concentration": robin(
                    "concentration",
                    {("c", 2, "lower"): (1.0, 1.0, 0.0)},
                ),
                "multiplier": _multiplier_policy(),
            },
        )
    with pytest.raises(ValueError, match="extra=.*'c'.*0.*lower"):
        assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "concentration": robin(
                    "concentration",
                    {
                        ("c", 0, "lower"): (1.0, 1.0, 0.0),
                        ("c", 2, "lower"): (1.0, 1.0, 0.0),
                        ("c", 2, "upper"): (1.0, 1.0, 0.0),
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
                "multiplier": robin(
                    "multiplier",
                    {
                        ("lambda_value", 2, "lower"): (1.0, 1.0, 0.0),
                        ("lambda_value", 2, "upper"): (1.0, 1.0, 0.0),
                    },
                ),
            },
        )


def test_current_lowering_rejects_robin_before_transform_selection():
    simulation = _plane_simulation()
    qxx = simulation.boundaries.for_component("Qxx")
    robin_qxx = replace(
        qxx,
        faces=tuple(
            replace(face, condition=StaticRobinBC(2.0, 3.0, 4.0))
            if face.axis == 2
            else face
            for face in qxx.faces
        ),
    )
    candidate = replace(
        simulation,
        boundaries=replace(
            simulation.boundaries,
            components=tuple(
                robin_qxx if value.component == "Qxx" else value
                for value in simulation.boundaries.components
            ),
        ),
    )

    with pytest.raises(SimulationLoweringError) as caught:
        lower_simulation_spec(candidate)
    assert caught.value.rejection.code is (
        LoweringRejectionCode.UNSUPPORTED_ROBIN_BOUNDARY
    )
    assert caught.value.rejection.to_metadata()["context"] == {
        "component": "Qxx",
        "faces": [
            {"axis": 2, "side": "lower"},
            {"axis": 2, "side": "upper"},
        ],
    }


def test_generic_robin_modules_have_no_model_tensor_operator_or_runtime_dependency():
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
        "pssolver/boundaries/robin.py",
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


def test_public_exports_and_capability_catalog_do_not_claim_execution():
    assert {
        "RobinFaceLaw",
        "StaticRobinBoundaryPolicy",
        "robin",
    } <= set(public_boundaries.__all__)
    capability = {
        value.key: value for value in pssolver.available_boundary_policies()
    }["robin"]
    assert capability.constructor == "pssolver.boundaries.robin"
    assert capability.executable is False
    assert capability.qualified_applications == ()


def test_p851_record_binds_plan_and_content_addressed_implementation():
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))

    assert record["phase"] == "P8.5.1"
    assert record["classification"] == (
        "PASS_P8_5_1_FIELD_NEUTRAL_STATIC_ROBIN_DECLARATIONS"
    )
    assert record["implementation_commit"] == P851_IMPLEMENTATION_COMMIT
    assert _sha256(ROOT / record["baseline"]["planning_record"]) == (
        record["baseline"]["planning_record_sha256"]
    )
    for relative, expected in record["source_sha256"].items():
        assert _sha256_at_commit(relative) == expected


def test_p851_record_freezes_declaration_only_scope_and_authorization():
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))

    assert record["canonical_law"] == {
        "form": "alpha*phi+beta*(n_dot_grad_phi)=gamma",
        "normal_derivative_convention": "outward_unit_normal",
        "coefficient_representation": "static_finite_real_constant",
        "raw_coefficient_identity": "canonical_json_sha256",
        "operator_normalization_performed": False,
        "alpha_and_beta_both_zero_allowed": False,
        "homogeneous_when": "gamma==0",
    }
    assert record["numerical_robin_operator_implemented"] is False
    assert record["finite_q_anchoring_implemented"] is False
    assert record["new_executable_combinations"] == []
    assert record["h100_used"] is False
    assert record["execution_boundary"]["dct_or_dst_selected"] is False
    assert record["execution_boundary"][
        "dirichlet_or_neumann_fallback"
    ] is False
    assert record["local_verification"] == {
        "targeted": "89 passed",
        "full": "2337 passed, 8 subtests passed",
        "git_diff_check": "pass",
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 0,
    }
    authorization = record["authorization"]
    assert authorization["p8_5_1_complete"] is True
    assert authorization["eligible_for_p8_5_2_method_adr_planning"] is True
    assert authorization["p8_5_2_implementation_authorized"] is False
    assert authorization["h100_authorized"] is False
    assert authorization["nonhomogeneous_neumann_authorized"] is False
    assert authorization["phase_9_authorized"] is False
    assert authorization["production_default_changed"] is False


def test_future_verbatim_archive_lists_p851_without_regenerating_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "phase_8_p851_robin_declarations.md",
        "phase_8_p851_robin_declarations.json",
    ):
        assert source.count(f'"{name}"') == 1
