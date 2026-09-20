"""Close and continuously audit production provenance package facades."""

from __future__ import annotations

import ast
from pathlib import Path

from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _untracked_imported_package_facades() -> dict[str, tuple[str, ...]]:
    inventory = set(PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES)
    consumers_by_facade: dict[str, set[str]] = {}

    for relative_path in PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES:
        if not relative_path.endswith(".py"):
            continue
        source_path = PROJECT_ROOT / relative_path
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module is None or not node.module.startswith("pssolver"):
                continue
            facade = f"{node.module.replace('.', '/')}/__init__.py"
            if not (PROJECT_ROOT / facade).is_file() or facade in inventory:
                continue
            consumers_by_facade.setdefault(facade, set()).add(relative_path)

    return {
        facade: tuple(sorted(consumers))
        for facade, consumers in sorted(consumers_by_facade.items())
    }


def test_phase2_provenance_tracks_all_imported_package_facades():
    assert _untracked_imported_package_facades() == {}


def test_phase2_provenance_contains_the_two_closure_facades():
    inventory = set(PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES)
    assert {
        "pssolver/core/__init__.py",
        "pssolver/geometries/__init__.py",
    } <= inventory
