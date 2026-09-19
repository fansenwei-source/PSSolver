"""Executable import boundaries for the v0.2 architecture migration.

The test is intentionally a ratchet, not a claim that the v0.1.2 tree already
has the target architecture.  New dependencies must follow the accepted
direction.  Seven remaining legacy edges are named exactly so that a broad
exemption cannot hide new coupling and removing a debt requires removing its
allowance.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "pssolver"


@dataclass(frozen=True, order=True)
class ImportEdge:
    source: str
    destination: str


# Allowed internal dependency layers for packages governed in Phase 0.
# Applications are composition roots and are checked separately below.
ALLOWED_INTERNAL_LAYERS = {
    "core": frozenset({"core"}),
    "geometries": frozenset({"core", "geometries"}),
    "planning": frozenset({"core", "planning"}),
    "execution": frozenset(
        {
            "backends",
            "core",
            "execution",
            "linear_solvers",
            "operators",
            "planning",
        }
    ),
    "backends": frozenset({"backends", "core", "planning"}),
    "operators": frozenset({"backends", "core", "operators", "planning"}),
    "linear_solvers": frozenset(
        {"backends", "core", "linear_solvers", "operators", "planning"}
    ),
    "integrators": frozenset({"core", "execution", "integrators", "planning"}),
    "models": frozenset({"core", "execution", "models"}),
    "adapters": frozenset({"adapters", "core", "planning"}),
    "presets": frozenset({"models", "presets"}),
    "configuration": frozenset(
        {"adapters", "configuration", "core", "geometries", "models", "presets"}
    ),
    "runtime": frozenset(
        {
            "backends",
            "configuration",
            "core",
            "execution",
            "integrators",
            "linear_solvers",
            "models",
            "operators",
            "planning",
            "runtime",
        }
    ),
    "workflows": frozenset(
        {"configuration", "io", "models", "run_metadata", "runtime", "workflows"}
    ),
    "io": frozenset({"core", "io", "runtime"}),
    "control": frozenset({"control", "models", "snapshots"}),
    # The historical facade is allowed to re-export the canonical Phase 1
    # implementations, but canonical modules never depend on the facade.
    "transforms": frozenset({"backends", "linear_solvers", "operators"}),
}


# These are migration debts present in the frozen v0.1.2 source.  Each entry
# is narrow, observed, documented by the Phase 0 ADRs, and scheduled for
# removal rather than being treated as an acceptable layer-wide dependency.
EXPECTED_LEGACY_EXCEPTIONS = {
    ImportEdge("pssolver.models.active_nematics.stokes", "pssolver.transforms"),
    ImportEdge("pssolver.configuration.plane_beris_edwards", "pssolver.plane"),
    ImportEdge(
        "pssolver.configuration.plane_beris_edwards", "pssolver.transforms"
    ),
    ImportEdge("pssolver.runtime.plane_legacy", "pssolver.integrator"),
    ImportEdge("pssolver.runtime.plane_legacy", "pssolver.plane"),
    ImportEdge("pssolver.runtime.plane_legacy", "pssolver.solver"),
    ImportEdge(
        "pssolver.runtime.plane_beris_edwards",
        "pssolver.experimental.plane_shadow_driver",
    ),
}


TENSOR_FREE_LAYERS = frozenset({"core", "geometries", "planning"})


def _module_name(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_name(path: Path) -> str:
    module = _module_name(path)
    if path.name == "__init__.py":
        return module
    return module.rpartition(".")[0]


def _resolve_from_import(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = _package_name(path).split(".")
    keep = len(package_parts) - (node.level - 1)
    if keep < 1:
        raise AssertionError(f"invalid relative import in {path}:{node.lineno}")
    target_parts = package_parts[:keep]
    if node.module:
        target_parts.extend(node.module.split("."))
    return ".".join(target_parts)


def _known_internal_modules() -> frozenset[str]:
    return frozenset(_module_name(path) for path in PACKAGE_ROOT.rglob("*.py"))


KNOWN_INTERNAL_MODULES = _known_internal_modules()


def _from_import_targets(path: Path, node: ast.ImportFrom) -> set[str]:
    base = _resolve_from_import(path, node)
    targets = {base}
    # ``from pssolver import experimental`` names a real submodule even though
    # the AST stores only ``pssolver`` in node.module.  Add alias-qualified
    # targets only when they resolve to an actual package module; ordinary
    # symbol imports such as ``from pssolver.transforms import Foo`` retain the
    # module edge without pretending that Foo is a module.
    for alias in node.names:
        if alias.name == "*":
            continue
        candidate = f"{base}.{alias.name}" if base else alias.name
        if candidate in KNOWN_INTERNAL_MODULES:
            targets.add(candidate)
    return targets


def _literal_dynamic_import(node: ast.Call) -> str | None:
    function = node.func
    recognized = (
        isinstance(function, ast.Name)
        and function.id in {"__import__", "import_module"}
    ) or (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "importlib"
        and function.attr == "import_module"
    )
    if not recognized or not node.args:
        return None
    argument = node.args[0]
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    return None


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.update(_from_import_targets(path, node))
        elif isinstance(node, ast.Call):
            dynamic_import = _literal_dynamic_import(node)
            if dynamic_import is not None:
                imports.add(dynamic_import)
    return imports


def _layer(module: str) -> str:
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "pssolver":
        return ""
    return parts[1]


def _all_edges() -> set[ImportEdge]:
    edges: set[ImportEdge] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = _module_name(path)
        for destination in _imports(path):
            if destination == "pssolver" or destination.startswith("pssolver."):
                edges.add(ImportEdge(source, destination))
    return edges


def test_tensor_free_declaration_and_planning_layers_use_no_third_party_runtime():
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = _module_name(path)
        if _layer(source) not in TENSOR_FREE_LAYERS:
            continue
        for destination in sorted(_imports(path)):
            top_level = destination.split(".", 1)[0]
            if top_level == "pssolver" or top_level in sys.stdlib_module_names:
                continue
            violations.append(f"{source} -> {destination}")
    assert violations == [], "tensor-free layer imported runtime dependency:\n" + "\n".join(
        violations
    )


def test_internal_imports_follow_the_phase_zero_dependency_ratchet():
    observed_exceptions: set[ImportEdge] = set()
    violations: list[ImportEdge] = []

    for edge in sorted(_all_edges()):
        source_layer = _layer(edge.source)
        if source_layer not in ALLOWED_INTERNAL_LAYERS:
            continue
        destination_layer = _layer(edge.destination)
        if destination_layer in ALLOWED_INTERNAL_LAYERS[source_layer]:
            continue
        if edge in EXPECTED_LEGACY_EXCEPTIONS:
            observed_exceptions.add(edge)
            continue
        violations.append(edge)

    assert violations == [], "new architecture dependency violation:\n" + "\n".join(
        f"{edge.source} -> {edge.destination}" for edge in violations
    )
    assert observed_exceptions == EXPECTED_LEGACY_EXCEPTIONS, (
        "the exact legacy-debt inventory changed; remove a retired allowance or "
        "document and review any new edge:\n"
        f"expected={sorted(EXPECTED_LEGACY_EXCEPTIONS)!r}\n"
        f"observed={sorted(observed_exceptions)!r}"
    )


def test_applications_do_not_reach_into_experimental_implementations():
    violations = sorted(
        edge
        for edge in _all_edges()
        if _layer(edge.source) == "applications"
        and _layer(edge.destination) == "experimental"
    )
    assert violations == []


def test_only_the_frozen_runtime_bridge_reaches_into_experimental_code():
    expected = {
        ImportEdge(
            "pssolver.runtime.plane_beris_edwards",
            "pssolver.experimental.plane_shadow_driver",
        )
    }
    observed = {
        edge
        for edge in _all_edges()
        if _layer(edge.source) != "experimental"
        and _layer(edge.destination) == "experimental"
    }
    assert observed == expected


def test_import_scanner_covers_package_aliases_and_literal_dynamic_imports():
    representative_path = PACKAGE_ROOT / "runtime" / "__init__.py"

    absolute = ast.parse("from pssolver import experimental").body[0]
    relative = ast.parse("from .. import experimental").body[0]
    symbol = ast.parse(
        "from pssolver.transforms import TensorProductTransformBackend"
    ).body[0]
    assert isinstance(absolute, ast.ImportFrom)
    assert isinstance(relative, ast.ImportFrom)
    assert isinstance(symbol, ast.ImportFrom)

    assert "pssolver.experimental" in _from_import_targets(
        representative_path, absolute
    )
    assert "pssolver.experimental" in _from_import_targets(
        representative_path, relative
    )
    assert _from_import_targets(representative_path, symbol) == {
        "pssolver.transforms"
    }

    dynamic = ast.parse(
        'importlib.import_module("pssolver.experimental.hidden")'
    ).body[0]
    assert isinstance(dynamic, ast.Expr)
    assert isinstance(dynamic.value, ast.Call)
    assert _literal_dynamic_import(dynamic.value) == "pssolver.experimental.hidden"
