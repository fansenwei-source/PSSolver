"""P5.1 disconnected Plane compiled-v2 declaration tests."""

from __future__ import annotations

import dataclasses
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pssolver.planning.plane_compiled_v2 import (
    PLANE_COMPILED_V2_IDENTITY,
    PlaneCompiledStage,
    PlaneFieldLayoutDeclaration,
    PlaneFieldRole,
    PlaneTransformFamily,
    PlaneTransformGroupDeclaration,
    PlaneWorkspaceLifetime,
    PlaneWorkspaceRequirement,
    plane_compiled_v2_declaration,
)


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = (
    ROOT / "notes/architecture_v0_2/phase_5_p51_compiled_declarations.json"
)


def test_canonical_declaration_freezes_plane_layout_and_basis_groups():
    declaration = plane_compiled_v2_declaration()
    assert declaration.identity == PLANE_COMPILED_V2_IDENTITY
    assert declaration.connected_runtime is None
    assert declaration.layout.component_order == (
        "Qxx",
        "Qxy",
        "Qxz",
        "Qyy",
        "Qyz",
        "ux",
        "uy",
        "uz",
        "p",
    )
    assert declaration.layout.to_metadata()["transform_groups"] == [
        {
            "name": "q_neumann",
            "components": ["Qxx", "Qxy", "Qxz", "Qyy", "Qyz"],
            "role": "evolved",
            "transform": "fft_fft_dct",
        },
        {
            "name": "tangential_velocity_neumann",
            "components": ["ux", "uy"],
            "role": "algebraic",
            "transform": "fft_fft_dct",
        },
        {
            "name": "normal_velocity_dirichlet",
            "components": ["uz"],
            "role": "algebraic",
            "transform": "fft_fft_dst",
        },
        {
            "name": "pressure_modal_neumann",
            "components": ["p"],
            "role": "algebraic",
            "transform": "fft_fft_dct",
        },
    ]


def test_canonical_declaration_matches_qualified_euler_stage_order():
    declaration = plane_compiled_v2_declaration()
    assert tuple(stage.value for stage in declaration.operation_order) == (
        "prepare_algebraic",
        "pre_update_callback",
        "explicit_rhs",
        "spectral_add_dt_rhs",
        "spectral_divide_by_denominator",
        "project_dynamic_spectra",
        "inverse_dynamic_spectra",
        "scheduled_spectral_refresh",
        "commit_progress",
    )
    assert declaration.persistent_state.integrator == (
        "projected_semi_implicit_euler"
    )
    assert declaration.persistent_state.history_depth == 0
    assert "representation_ledger" in declaration.persistent_state.items


def test_all_workspace_categories_are_bounded_and_construction_bound():
    declaration = plane_compiled_v2_declaration()
    assert len(declaration.workspace_requirements) == 7
    assert {
        requirement.name for requirement in declaration.workspace_requirements
    } == {
        "explicit_q_rhs",
        "q_gradient_cache",
        "constitutive_scratch",
        "nematic_force",
        "stokes_solve_scratch",
        "projected_transform_scratch",
        "inverse_transform_scratch",
    }
    assert all(
        requirement.bounded and requirement.construction_bound
        for requirement in declaration.workspace_requirements
    )


def test_declarations_are_deeply_immutable_value_objects():
    declaration = plane_compiled_v2_declaration()
    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.identity = "legacy_production"
    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.layout.evolved_components = ("Qxx",)
    with pytest.raises(dataclasses.FrozenInstanceError):
        declaration.workspace_requirements[0].bounded = False


def test_layout_rejects_duplicate_missing_and_role_incompatible_groups():
    q_group = PlaneTransformGroupDeclaration(
        name="q_neumann",
        components=("Qxx",),
        role=PlaneFieldRole.EVOLVED,
        transform=PlaneTransformFamily.FFT_FFT_DCT,
    )
    duplicate = dataclasses.replace(
        q_group,
        name="q_duplicate",
    )
    with pytest.raises(ValueError, match="exactly one"):
        PlaneFieldLayoutDeclaration(
            evolved_components=("Qxx",),
            algebraic_components=("ux",),
            transform_groups=(q_group, duplicate),
        )
    with pytest.raises(ValueError, match="exactly cover"):
        PlaneFieldLayoutDeclaration(
            evolved_components=("Qxx",),
            algebraic_components=("ux",),
            transform_groups=(q_group,),
        )
    wrong_role = PlaneTransformGroupDeclaration(
        name="velocity_wrong_role",
        components=("ux",),
        role=PlaneFieldRole.EVOLVED,
        transform=PlaneTransformFamily.FFT_FFT_DCT,
    )
    with pytest.raises(ValueError, match="algebraic role"):
        PlaneFieldLayoutDeclaration(
            evolved_components=("Qxx",),
            algebraic_components=("ux",),
            transform_groups=(q_group, wrong_role),
        )


def test_compiled_declaration_rejects_stage_reordering_and_unbounded_workspace():
    declaration = plane_compiled_v2_declaration()
    reordered = declaration.operation_order[:-2] + (
        PlaneCompiledStage.COMMIT_PROGRESS,
        PlaneCompiledStage.SCHEDULED_SPECTRAL_REFRESH,
    )
    with pytest.raises(ValueError, match="production order"):
        dataclasses.replace(declaration, operation_order=reordered)
    with pytest.raises(ValueError, match="bounded and construction-bound"):
        PlaneWorkspaceRequirement(
            name="bad_scratch",
            owner="integrator",
            lifetime=PlaneWorkspaceLifetime.GENERATION,
            bounded=False,
        )


def test_direct_import_is_tensor_free_and_does_not_connect_runtime():
    source_path = ROOT / "pssolver/planning/plane_compiled_v2.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module != "__future__"
    )
    assert imported_roots == {"dataclasses", "enum"}
    code = """
import sys
import pssolver.planning.plane_compiled_v2 as module
declaration = module.plane_compiled_v2_declaration()
assert declaration.connected_runtime is None
assert 'pssolver.runtime.plane_beris_edwards' not in sys.modules
assert 'pssolver.runtime.plane_legacy' not in sys.modules
assert 'pssolver.integrators.step_program' not in sys.modules
assert 'torch' not in module.__dict__
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_compiled_declaration_remains_private_and_selector_free():
    import pssolver
    import pssolver.planning as planning
    from pssolver.configuration import PlaneRuntimePath

    assert not hasattr(pssolver, "plane_compiled_v2_declaration")
    assert not hasattr(planning, "plane_compiled_v2_declaration")
    assert {member.value for member in PlaneRuntimePath} == {
        "legacy_production",
        "separated_canary",
    }


def test_p51_qualification_record_matches_disconnected_scope():
    record = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_P5_1_DISCONNECTED_COMPILED_DECLARATIONS"
    )
    assert record["authorization"] == {
        "p5_1_implementation_authorized": True,
        "p5_2_implementation_authorized": False,
        "phase_6_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
    assert record["connection"] == {
        "runtime_selector_added": False,
        "production_import_added": False,
        "tensor_binding_added": False,
        "runtime_execution_added": False,
        "fallback_allowed": False,
    }
