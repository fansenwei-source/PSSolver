"""P2.3 gates for dependency-safe Plane RunSpec view delegation."""

from __future__ import annotations

import ast
from dataclasses import fields
import os
from pathlib import Path
import subprocess
import sys
from typing import get_type_hints

import pytest

import pssolver
import pssolver.configuration as configuration
from pssolver.configuration import plane_beris_edwards as facade
from pssolver.configuration import plane_beris_edwards_builders as builders
from pssolver.configuration import (
    plane_beris_edwards_component_graph as component_graph,
)
from pssolver.configuration import (
    plane_beris_edwards_components as components,
)
from pssolver.configuration import (
    plane_beris_edwards_declarations as declarations,
)
from pssolver.configuration import (
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneBerisEdwardsRunSpec,
    PlaneFreeSlipBoundaryConditions,
    PlaneRuntimePath,
    SpectralRefreshSpec,
    create_plane_beris_edwards_run_spec,
)
from pssolver.core import NumericsConfig
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics.specifications import (
    BerisEdwardsMaterialRequest,
    ExtrudedDefectGasInitialConditionSpec,
)
from pssolver.plane import PLANE_HERMITIAN_AXIS as RUNTIME_HERMITIAN_AXIS
from pssolver.presets.shendruk import (
    ShendrukPlaneParameterRequest,
    ShendrukPlanePreset,
)
from pssolver.systems.stokes import IncompressibleStokesSystemSpec


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


def test_neutral_legacy_and_stable_paths_share_exact_declaration_objects():
    names = (
        "PlaneRuntimePath",
        "PlaneFreeSlipBoundaryConditions",
        "SpectralRefreshSpec",
        "PLANE_FREE_SLIP_BOUNDARIES",
    )
    for name in names:
        canonical = getattr(declarations, name)
        assert getattr(facade, name) is canonical
        assert getattr(configuration, name) is canonical
        assert getattr(components, name) is canonical
        assert getattr(component_graph, name) is canonical
        assert not hasattr(pssolver, name)

    run_fields = {
        item.name: item for item in fields(PlaneBerisEdwardsRunSpec)
    }
    assert run_fields["runtime_path"].default is (
        PlaneRuntimePath.LEGACY_PRODUCTION
    )
    assert run_fields["boundaries"].default is PLANE_FREE_SLIP_BOUNDARIES
    assert facade.PLANE_HERMITIAN_AXIS == RUNTIME_HERMITIAN_AXIS


def test_lower_graph_and_upper_adapter_share_exact_component_types():
    for name in COMPONENT_TYPES:
        canonical = getattr(component_graph, name)
        assert getattr(components, name) is canonical
        assert name in component_graph.__all__
        assert name in components.__all__
        assert canonical.__module__ == components.__name__
        assert canonical.__qualname__ == name
        assert all(
            isinstance(annotation, str)
            for annotation in canonical.__annotations__.values()
        )
        assert get_type_hints(canonical) == EXPECTED_COMPONENT_HINTS[name]
        assert not hasattr(configuration, name)
        assert not hasattr(pssolver, name)

    builder_name = "build_plane_beris_edwards_run_components"
    assert set(component_graph.__all__) == set(COMPONENT_TYPES) | {
        builder_name
    }
    assert set(components.__all__) == set(COMPONENT_TYPES) | {
        "decompose_plane_beris_edwards_run_spec"
    }
    target = getattr(component_graph, builder_name)
    assert target.__module__ == component_graph.__name__
    for module in (components, facade, configuration, pssolver):
        assert builder_name not in getattr(module, "__all__", ())
        assert not hasattr(module, builder_name)


def test_builder_names_are_direct_module_only_and_facade_imports_exact_objects():
    for name in VIEW_BUILDERS.values():
        target = getattr(builders, name)
        assert getattr(facade, name) is target
        assert name in builders.__all__
        assert name not in configuration.__all__
        assert not hasattr(configuration, name)
        assert not hasattr(pssolver, name)


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


def test_facade_property_bodies_delegate_without_legacy_construction_or_imports():
    tree = _module_tree(FACADE_PATH)
    run_spec = _class_node(tree, "PlaneBerisEdwardsRunSpec")
    forbidden_calls = {
        "DomainSpec",
        "NumericsConfig",
        "PlaneSlab",
        "decompose_plane_beris_edwards_run_spec",
        "resolve_shendruk_plane_preset",
    }

    for property_name, builder_name in VIEW_BUILDERS.items():
        function = _property_node(run_spec, property_name)
        calls = {
            node.func.id
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert builder_name in calls
        assert calls.isdisjoint(forbidden_calls)
        assert not any(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(function)
        )


def test_p2_3_module_graph_is_one_way_and_has_no_hidden_import_workaround():
    paths = (
        DECLARATIONS_PATH,
        BUILDERS_PATH,
        COMPONENT_GRAPH_PATH,
        FACADE_PATH,
        COMPONENTS_PATH,
    )
    trees = {path: _module_tree(path) for path in paths}
    for path, tree in trees.items():
        assert len(_top_level_imports(tree)) == sum(
            isinstance(node, (ast.Import, ast.ImportFrom))
            for node in ast.walk(tree)
        ), path
        source = path.read_text(encoding="utf-8")
        assert "__import__" not in source
        assert "import_module" not in source

    module_names = {
        DECLARATIONS_PATH: "plane_beris_edwards_declarations",
        BUILDERS_PATH: "plane_beris_edwards_builders",
        COMPONENT_GRAPH_PATH: "plane_beris_edwards_component_graph",
        FACADE_PATH: "plane_beris_edwards",
        COMPONENTS_PATH: "plane_beris_edwards_components",
    }
    internal_names = set(module_names.values())
    observed_internal_edges = {
        (module_names[path], node.module)
        for path, tree in trees.items()
        for node in _top_level_imports(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module in internal_names
    }
    assert observed_internal_edges == {
        ("plane_beris_edwards", "plane_beris_edwards_builders"),
        ("plane_beris_edwards", "plane_beris_edwards_declarations"),
        (
            "plane_beris_edwards_component_graph",
            "plane_beris_edwards_builders",
        ),
        (
            "plane_beris_edwards_component_graph",
            "plane_beris_edwards_declarations",
        ),
        ("plane_beris_edwards_components", "plane_beris_edwards"),
        (
            "plane_beris_edwards_components",
            "plane_beris_edwards_component_graph",
        ),
        (
            "plane_beris_edwards_components",
            "plane_beris_edwards_declarations",
        ),
    }

    declaration_source = DECLARATIONS_PATH.read_text(encoding="utf-8")
    builder_source = BUILDERS_PATH.read_text(encoding="utf-8")
    graph_source = COMPONENT_GRAPH_PATH.read_text(encoding="utf-8")
    facade_source = FACADE_PATH.read_text(encoding="utf-8")
    component_source = COMPONENTS_PATH.read_text(encoding="utf-8")

    for forbidden in (
        "plane_beris_edwards_components",
        "plane_beris_edwards_component_graph",
        "plane_beris_edwards_builders",
        "from .plane_beris_edwards import",
        "pssolver.applications",
        "pssolver.runtime",
        "pssolver.workflows",
    ):
        assert forbidden not in declaration_source
    for forbidden in (
        "plane_beris_edwards_components",
        "plane_beris_edwards_component_graph",
        "from .plane_beris_edwards import",
        "pssolver.applications",
        "pssolver.runtime",
        "pssolver.workflows",
    ):
        assert forbidden not in builder_source
    for forbidden in (
        "from .plane_beris_edwards import",
        "pssolver.applications",
        "pssolver.runtime",
        "pssolver.workflows",
    ):
        assert forbidden not in graph_source
    graph_tree = trees[COMPONENT_GRAPH_PATH]
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "plane_beris_edwards_builders"
        for node in _top_level_imports(graph_tree)
    )
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.level == 1
        and node.module == "plane_beris_edwards_declarations"
        for node in _top_level_imports(graph_tree)
    )
    assert "plane_beris_edwards_components" not in facade_source
    assert "plane_beris_edwards_component_graph" not in facade_source
    assert "plane_beris_edwards_declarations" in facade_source
    assert "plane_beris_edwards_builders" in facade_source
    assert "plane_beris_edwards_declarations" in component_source
    assert "plane_beris_edwards_component_graph" in component_source
    assert "plane_beris_edwards_builders" not in component_source

    facade_edges = [
        node
        for node in _top_level_imports(trees[COMPONENTS_PATH])
        if isinstance(node, ast.ImportFrom)
        and node.module == "plane_beris_edwards"
        and node.level == 1
    ]
    assert len(facade_edges) == 1
    assert [alias.name for alias in facade_edges[0].names] == [
        "PLANE_HERMITIAN_AXIS",
        "PlaneBerisEdwardsRunSpec",
    ]


@pytest.mark.parametrize(
    "imports",
    (
        """
from pssolver.configuration import plane_beris_edwards_declarations as declarations
from pssolver.configuration import plane_beris_edwards_builders as builders
from pssolver.configuration import plane_beris_edwards_component_graph as component_graph
from pssolver.configuration import plane_beris_edwards as facade
from pssolver.configuration import plane_beris_edwards_components as components
""",
        """
from pssolver.configuration import plane_beris_edwards_builders as builders
from pssolver.configuration import plane_beris_edwards_component_graph as component_graph
from pssolver.configuration import plane_beris_edwards_components as components
from pssolver.configuration import plane_beris_edwards_declarations as declarations
from pssolver.configuration import plane_beris_edwards as facade
""",
        """
from pssolver.configuration import plane_beris_edwards_components as components
from pssolver.configuration import plane_beris_edwards as facade
from pssolver.configuration import plane_beris_edwards_component_graph as component_graph
from pssolver.configuration import plane_beris_edwards_builders as builders
from pssolver.configuration import plane_beris_edwards_declarations as declarations
""",
        """
from pssolver.configuration import plane_beris_edwards as facade
from pssolver.configuration import plane_beris_edwards_declarations as declarations
from pssolver.configuration import plane_beris_edwards_components as components
from pssolver.configuration import plane_beris_edwards_component_graph as component_graph
from pssolver.configuration import plane_beris_edwards_builders as builders
""",
    ),
)
def test_p2_3_modules_are_cycle_free_under_fresh_process_import_orders(imports):
    script = imports + """
import pssolver
import pssolver.configuration as configuration

for name in (
    'PlaneRuntimePath',
    'PlaneFreeSlipBoundaryConditions',
    'SpectralRefreshSpec',
    'PLANE_FREE_SLIP_BOUNDARIES',
):
    target = getattr(declarations, name)
    assert getattr(facade, name) is target
    assert getattr(configuration, name) is target
    assert getattr(components, name) is target
    assert getattr(component_graph, name) is target
    assert not hasattr(pssolver, name)

for name in (
    'PlaneBerisEdwardsPhysicsSpec',
    'PlaneTimeSteppingSpec',
    'PlaneBerisEdwardsExecutionSpec',
    'PlaneWorkflowSpec',
    'PlaneInvocationSpec',
    'PlaneBerisEdwardsRunComponents',
):
    assert getattr(components, name) is getattr(component_graph, name)

for name in (
    'build_plane_beris_edwards_domain',
    'build_plane_beris_edwards_geometry',
    'build_plane_beris_edwards_numerics',
    'build_plane_beris_edwards_shendruk_preset',
):
    assert getattr(facade, name) is getattr(builders, name)
    assert not hasattr(configuration, name)

spec = facade.create_plane_beris_edwards_run_spec(
    activity_number=18.0,
    output_dir='oracle/p2.3-import-order',
)
parts = components.decompose_plane_beris_edwards_run_spec(spec)
assert spec.domain == parts.geometry.domain
assert spec.geometry == parts.geometry
assert spec.numerics == parts.numerics
assert spec.shendruk_preset == parts.preset
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_graph_first_import_preserves_annotations_pickle_and_legacy_resolution():
    script = """
import hashlib
import pickle
import sys
from typing import get_type_hints
from pssolver.configuration import plane_beris_edwards_component_graph as graph

adapter_name = 'pssolver.configuration.plane_beris_edwards_components'
assert adapter_name not in sys.modules
oracles = {
    'PlaneBerisEdwardsPhysicsSpec': (101, '9b249577e6c264d6fa5e93f46914e380ed8005aa49c7f1ed5e05418471df94c4'),
    'PlaneTimeSteppingSpec': (94, '16aadde85f367f1ccf355d4d5b752edc4174cbdb54b23413d6bd1b5d17f70b43'),
    'PlaneBerisEdwardsExecutionSpec': (103, 'e76958558d254ab8ba50331ac23b5117e06e630e635cf49de05515eaccc23d26'),
    'PlaneWorkflowSpec': (90, 'c4c86b5a106976a5641234070ef89be82943af6b341ef2875f69a94d3f497bf7'),
    'PlaneInvocationSpec': (92, '21460de5b3bff745342169653ef7d855dc9995095e62d5cf80369a1ff4843719'),
    'PlaneBerisEdwardsRunComponents': (103, 'f92fab39c5c5f07bd59966b5fa8176cd53813b2516ed1809b5a0bb1a4ceb8dad'),
}
for name, (expected_length, expected_sha256) in oracles.items():
    target = getattr(graph, name)
    assert target.__module__ == adapter_name
    assert all(isinstance(value, str) for value in target.__annotations__.values())
    payload = pickle.dumps(target, protocol=4)
    assert len(payload) == expected_length
    assert hashlib.sha256(payload).hexdigest() == expected_sha256
    assert pickle.loads(payload) is target

assert adapter_name in sys.modules
from pssolver.configuration import plane_beris_edwards_components as adapter
assert not hasattr(adapter, 'build_plane_beris_edwards_run_components')
for name in (
    'PlaneBerisEdwardsPhysicsSpec',
    'PlaneTimeSteppingSpec',
    'PlaneBerisEdwardsExecutionSpec',
    'PlaneWorkflowSpec',
    'PlaneInvocationSpec',
    'PlaneBerisEdwardsRunComponents',
):
    assert getattr(adapter, name) is getattr(graph, name)
    assert get_type_hints(getattr(adapter, name))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
