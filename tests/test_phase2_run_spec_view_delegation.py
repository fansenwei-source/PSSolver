"""P2.3 gates for dependency-safe Plane RunSpec view delegation."""

from __future__ import annotations

import ast
from pathlib import Path

from pssolver.core import NumericsConfig
from pssolver.functional.legacy_declarations import plane_beris_edwards as facade
from pssolver.functional.legacy_declarations import (
    plane_beris_edwards_component_graph as component_graph,
)
from pssolver.functional.legacy_declarations.active_nematics_specifications import (
    BerisEdwardsMaterialRequest,
    ExtrudedDefectGasInitialConditionSpec,
)
from pssolver.functional.legacy_declarations.plane_beris_edwards import (
    PlaneBerisEdwardsRunSpec,
    PlaneFreeSlipBoundaryConditions,
    PlaneRuntimePath,
    SpectralRefreshSpec,
    create_plane_beris_edwards_run_spec,
)
from pssolver.functional.legacy_declarations.systems_stokes import IncompressibleStokesSystemSpec
from pssolver.geometries import PlaneSlab
from pssolver.presets.shendruk import (
    ShendrukPlaneParameterRequest,
    ShendrukPlanePreset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION_ROOT = PROJECT_ROOT / "pssolver" / "configuration"
FACADE_PATH = CONFIGURATION_ROOT / "plane_beris_edwards.py"
DECLARATIONS_PATH = (
    CONFIGURATION_ROOT / "plane_beris_edwards_declarations.py"
)
BUILDERS_PATH = CONFIGURATION_ROOT / "plane_beris_edwards_builders.py"
COMPONENTS_PATH = CONFIGURATION_ROOT / "plane_beris_edwards_components.py"
COMPONENT_GRAPH_PATH = (
    CONFIGURATION_ROOT / "plane_beris_edwards_component_graph.py"
)

VIEW_BUILDERS = {
    "domain": "build_plane_beris_edwards_domain",
    "geometry": "build_plane_beris_edwards_geometry",
    "numerics": "build_plane_beris_edwards_numerics",
    "shendruk_preset": "build_plane_beris_edwards_shendruk_preset",
}

COMPONENT_TYPES = (
    "PlaneBerisEdwardsPhysicsSpec",
    "PlaneTimeSteppingSpec",
    "PlaneBerisEdwardsExecutionSpec",
    "PlaneWorkflowSpec",
    "PlaneInvocationSpec",
    "PlaneBerisEdwardsRunComponents",
)

EXPECTED_COMPONENT_HINTS = {
    "PlaneBerisEdwardsPhysicsSpec": {
        "material": BerisEdwardsMaterialRequest,
        "shendruk_request": ShendrukPlaneParameterRequest,
        "stokes": IncompressibleStokesSystemSpec,
        "requested_friction_mode_fric": float,
    },
    "PlaneTimeSteppingSpec": {
        "dt": float,
        "spectral_refresh": SpectralRefreshSpec,
    },
    "PlaneBerisEdwardsExecutionSpec": {
        "device": str,
        "tf32": str,
        "molecular_field_linear_space": str,
        "stress_divergence_sum_space": str,
        "pointwise_execution": str,
        "disable_q_gradient_reuse": bool,
        "runtime_path": PlaneRuntimePath,
    },
    "PlaneWorkflowSpec": {
        "output_dir": Path,
        "steps": int,
        "save_start_step": int,
        "save_interval": int,
        "diagnostic_interval": int,
        "diagnostics": bool,
        "save_hydrodynamics": bool,
        "checkpoint_interval": int | None,
        "restart_from": Path | None,
    },
    "PlaneInvocationSpec": {
        "validation_config_sha256": str | None,
        "dry_run": bool,
    },
    "PlaneBerisEdwardsRunComponents": {
        "geometry": PlaneSlab,
        "boundaries": PlaneFreeSlipBoundaryConditions,
        "effective_boundaries": PlaneFreeSlipBoundaryConditions,
        "numerics": NumericsConfig,
        "physics": component_graph.PlaneBerisEdwardsPhysicsSpec,
        "preset": ShendrukPlanePreset,
        "time_stepping": component_graph.PlaneTimeSteppingSpec,
        "initial_condition": ExtrudedDefectGasInitialConditionSpec,
        "execution": component_graph.PlaneBerisEdwardsExecutionSpec,
        "workflow": component_graph.PlaneWorkflowSpec,
        "invocation": component_graph.PlaneInvocationSpec,
    },
}


def _spec(**overrides: object) -> PlaneBerisEdwardsRunSpec:
    values = {
        "activity_number": 18.0,
        "output_dir": "oracle/p2.3-view-delegation",
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_node(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _property_node(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _top_level_imports(tree: ast.Module) -> tuple[ast.Import | ast.ImportFrom, ...]:
    return tuple(
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )


def test_each_facade_view_calls_exactly_the_builders_needed_for_that_view(
    monkeypatch,
):
    spec = _spec()
    calls: dict[str, list[tuple[tuple[object, ...], dict[str, object]]]] = {
        name: [] for name in VIEW_BUILDERS.values()
    }

    for builder_name in VIEW_BUILDERS.values():
        original = getattr(facade, builder_name)

        def spy(*args, __name=builder_name, __target=original, **kwargs):
            calls[__name].append((args, kwargs))
            return __target(*args, **kwargs)

        monkeypatch.setattr(facade, builder_name, spy)

    expected_calls = {
        "domain": {"build_plane_beris_edwards_domain"},
        "geometry": {
            "build_plane_beris_edwards_domain",
            "build_plane_beris_edwards_geometry",
        },
        "numerics": {"build_plane_beris_edwards_numerics"},
        "shendruk_preset": {
            "build_plane_beris_edwards_shendruk_preset"
        },
    }

    for property_name in VIEW_BUILDERS:
        before = {name: len(items) for name, items in calls.items()}
        getattr(spec, property_name)
        for builder_name in VIEW_BUILDERS.values():
            increment = int(builder_name in expected_calls[property_name])
            assert len(calls[builder_name]) == before[builder_name] + increment
