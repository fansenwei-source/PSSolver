"""Read source and direct AST imports without importing the numerical package."""

from __future__ import annotations

import ast
from pathlib import Path


def read_toml(path: Path) -> dict:
    """Use the standard parser; Python 3.10 tooling requires its tomli backport."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    return tomllib.loads(path.read_text(encoding="utf-8"))


def module_name(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports(tree: ast.AST, module: str, is_package: bool) -> list[tuple[int, str]]:
    """Return absolute targets, including 'from package import submodule'."""
    result = []
    package = module.split(".") if is_package else module.split(".")[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = package[: len(package) - node.level + 1] if node.level else []
            if node.module:
                prefix += node.module.split(".")
            target = ".".join(prefix)
            if target:
                result.append((node.lineno, target))
            result.extend(
                (node.lineno, ".".join([*prefix, alias.name]))
                for alias in node.names
                if alias.name != "*"
            )
    return result
