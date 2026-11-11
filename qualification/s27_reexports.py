"""Resolve explicit package re-exports from S-27's immutable parent Git tree."""

import ast
import subprocess
from functools import cache
from pathlib import Path

from qualification.s25_import_scan import imports


BASE = "327a192ba879da0991110eca99f4b69b9a45c84e"
REPOSITORY = Path(__file__).resolve().parents[1]


@cache
def package_exports(package):
    """Read declarations, never import the historical runtime or inspect the draft."""
    if not package.startswith("pssolver"):
        return {}
    relative = package.replace(".", "/") + "/__init__.py"
    present = subprocess.run(
        ["git", "ls-tree", BASE, "--", relative],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    if not present.stdout:
        return {}
    source = subprocess.run(
        ["git", "show", BASE + ":" + relative],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom):
            resolved = list(imports(ast.Module(body=[node], type_ignores=[]), package, True))
            for (_, origin, name), alias in zip(resolved, node.names, strict=True):
                if name != "*":
                    result[alias.asname or name] = (origin, name)
    return result


def resolve_reexport(package, name):
    """Follow explicit imports in baseline package roots, failing on cycles."""
    seen = set()
    while name and name in package_exports(package):
        key = (package, name)
        if key in seen:
            raise ValueError(f"cyclic baseline re-export: {key}")
        seen.add(key)
        package, name = package_exports(package)[name]
    return package
