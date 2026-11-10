"""Source-only audit of retained S-24 units; no runtime or external archive access."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


BASE = "79558be0d1d65fb75a9b5fd41940314572310bf6"


def source_units(source: str) -> dict[str, tuple[str, str]]:
    """Return top-level definitions and assignments with decorators and exact bytes."""
    lines = source.splitlines(keepends=True)
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign):
            name = "assign:" + ",".join(ast.dump(item) for item in node.targets)
        elif isinstance(node, ast.AnnAssign):
            name = "assign:" + ast.dump(node.target)
        else:
            continue
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        result[name] = (ast.dump(node), "".join(lines[first - 1 : node.end_lineno]))
    return result


def retained_changes(
    root: Path,
    directories: tuple[str, ...] = ("pssolver", "tests"),
    *,
    base: str = BASE,
    endpoint: str | None = None,
) -> dict:
    """Compare real Git baseline objects, never tolerate missing historical sources."""
    changed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=M",
            base,
            *([endpoint] if endpoint else []),
            "--",
            *directories,
        ],
        cwd=root,
        text=True,
    ).splitlines()
    formatting = []
    semantics = []
    unchanged = []
    for relative in changed:
        if not relative.endswith(".py"):
            continue
        before = subprocess.check_output(["git", "show", base + ":" + relative], cwd=root)
        old = source_units(before.decode())
        after = (
            subprocess.check_output(["git", "show", endpoint + ":" + relative], cwd=root).decode()
            if endpoint
            else (root / relative).read_text()
        )
        new = source_units(after)
        for name in sorted(old.keys() & new.keys()):
            item = {"path": relative, "unit": name}
            if old[name][0] != new[name][0]:
                semantics.append(item)
            elif old[name][1] != new[name][1]:
                formatting.append(item)
            else:
                unchanged.append(item)
    return {"format_only": formatting, "semantic": semantics, "unchanged": unchanged}
