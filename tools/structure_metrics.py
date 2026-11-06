"""Reproducible source-structure trends, not numerical or quality thresholds."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path


if __package__:
    from ._source import imports
else:
    from _source import imports


ORCHESTRATION = {
    "configuration",
    "workflows",
    "runtime",
    "planning",
    "execution",
    "applications",
    "api",
    "systems",
    "presets",
    "adapters",
}
NUMERICS = {
    "operators",
    "linear_solvers",
    "integrators",
    "models",
    "boundaries",
    "backends",
    "geometries",
}


def sources(root: Path, revision: str | None) -> dict[str, str]:
    if revision:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision, "--", "pssolver"],
            cwd=root,
            text=True,
        ).splitlines()
        return {
            name: subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=root).decode()
            for name in names
            if name.endswith(".py")
        }
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted((root / "pssolver").rglob("*.py"))
    }


def cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Report strongly connected components (including self-import cycles)."""
    index = 0
    indices, low, stack, active, found = {}, {}, [], set(), []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = low[node] = index
        index += 1
        stack.append(node)
        active.add(node)
        for target in sorted(graph[node]):
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            component = []
            while True:
                item = stack.pop()
                active.remove(item)
                component.append(item)
                if item == node:
                    break
            if len(component) > 1 or node in graph[node]:
                found.append(sorted(component))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return sorted(found)


def measure(source: dict[str, str]) -> dict:
    """Count nonblank, non-comment lines, retaining docstrings as in the baseline."""
    line_counts, normalized, modules, parsed, long_functions = {}, {}, {}, {}, []
    for name, text in sorted(source.items()):
        lines = [
            (i, line.strip())
            for i, line in enumerate(text.splitlines(), 1)
            if line.strip() and not line.lstrip().startswith("#")
        ]
        line_counts[name] = len(lines)
        normalized[name] = lines
        module = name[:-3].replace("/", ".").removesuffix(".__init__")
        modules[module] = name
        tree = ast.parse(text, filename=name)
        parsed[module] = tree
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                if length > 150:
                    long_functions.append(
                        {"file": name, "function": node.name, "line": node.lineno, "lines": length}
                    )
    graph = {module: set() for module in modules}
    for module, tree in parsed.items():
        for _, target in imports(tree, module, modules[module].endswith("/__init__.py")):
            while target and target not in modules:
                target = target.rpartition(".")[0]
            if target and target != module:
                graph[module].add(target)
    windows = defaultdict(list)
    for name, lines in normalized.items():
        for offset in range(len(lines) - 7):
            window = tuple(text for _, text in lines[offset : offset + 8])
            windows[window].append((name, offset))
    duplicate_lines = set()
    for locations in windows.values():
        if len(locations) > 1:
            for name, offset in locations:
                duplicate_lines.update((name, i) for i in range(offset, offset + 8))
    total = sum(line_counts.values())

    def layer_lines(layers: set[str]) -> int:
        return sum(n for name, n in line_counts.items() if name.split("/")[1] in layers)

    experimental = layer_lines({"experimental"})
    orchestration, numerics = layer_lines(ORCHESTRATION), layer_lines(NUMERICS)
    imported = set().union(*graph.values())
    return {
        "schema": "pssolver.structure_metrics.v1",
        "method": {
            "lines": "nonblank and not comment-only; docstrings count",
            "duplication": "union of source positions in repeated 8-stripped-line windows",
            "functions": "AST def through end_lineno, including blank/comment lines",
            "imports": "direct static AST imports, all scopes; dynamic imports not resolved",
            "unimported": "no incoming static package edge; not a dead-code conclusion",
            "thresholds": None,
        },
        "source_sha256": {
            name: hashlib.sha256(text.encode()).hexdigest() for name, text in sorted(source.items())
        },
        "package_modules": len(source),
        "package_lines": total,
        "experimental_lines": experimental,
        "experimental_fraction": experimental / total,
        "legacy_declarations_lines": sum(
            n
            for name, n in line_counts.items()
            if name.startswith("pssolver/functional/legacy_declarations/")
        ),
        "duplicate_window_lines": len(duplicate_lines),
        "duplicate_fraction": len(duplicate_lines) / total,
        "functions_over_150": long_functions,
        "functions_over_150_count": len(long_functions),
        "orchestration_lines": orchestration,
        "numerical_lines": numerics,
        "orchestration_to_numerics_ratio": orchestration / numerics if numerics else None,
        "import_cycles": cycles(graph),
        "unimported_modules": sorted(set(modules) - imported),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", help="Read immutable Git objects instead of working files")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(sources(args.root, args.revision))
    if args.revision:
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", f"{args.revision}^{{commit}}"],
            cwd=args.root,
            text=True,
        ).strip()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
