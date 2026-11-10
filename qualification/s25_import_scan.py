"""Static import closure and isolated import probes for the S-25 trial."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


DELETED = frozenset(
    {
        "pssolver.configuration.periodic_beris_edwards",
        "pssolver.configuration.public_periodic_simulation_compiler",
        "pssolver.applications.periodic_beris_edwards",
        "pssolver.runtime.periodic_beris_edwards",
        "pssolver.workflows.periodic_beris_edwards",
        "pssolver.workflows.periodic_checkpoint",
        "pssolver.execution.state",
        "pssolver.execution.workspace",
        "pssolver.workflows.diagnostic_io",
        "pssolver.workflows.state_validation",
    }
)
SCOPES = ("benchmarks", "qualification", "scripts_plane", "scripts_channel")


def module_name(relative):
    return relative.removesuffix(".py").replace("/", ".").removesuffix(".__init__")


def sources(root):
    result = {
        module_name(p.relative_to(root).as_posix()): p
        for p in root.rglob("*.py")
        if not any(
            part.startswith(".") or part in {"build", "__pycache__"}
            for part in p.relative_to(root).parts
        )
    }
    loader = root / "qualification/_legacy_imports.py"
    if loader.is_file():
        tree = ast.parse(loader.read_text())
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "ALLOWED" for target in node.targets
            )
        )
        value = assignment.value
        allowed = ast.literal_eval(value.args[0] if isinstance(value, ast.Call) else value)
        for name in allowed:
            path = root / "qualification/stages" / (name.rsplit(".", 1)[-1] + ".py")
            result["pssolver.experimental." + name] = path
    return result


def imports(tree, module, package=False, *, nested=False):
    nodes = ast.walk(tree) if nested else tree.body
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name, None
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = module.split(".") if package else module.split(".")[:-1]
                base = ".".join(
                    parts[: len(parts) - node.level + 1] + ([node.module] if node.module else [])
                )
            else:
                base = node.module or ""
            for alias in node.names:
                yield node.lineno, base, alias.name


def definitions(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(
                n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
    return names


def inventory(root, *, nested=False):
    paths = sources(root)
    trees = {name: ast.parse(path.read_text(), filename=str(path)) for name, path in paths.items()}
    graph, errors = {}, {}
    for module, tree in trees.items():
        edges, missing = [], []
        for line, base, name in imports(
            tree, module, paths[module].name == "__init__.py", nested=nested
        ):
            target = base + "." + name if name and base + "." + name in paths else base
            if target in paths:
                edges.append(target)
                if (
                    name
                    and target == base
                    and name != "*"
                    and name not in definitions(trees[target])
                ):
                    # Module __getattr__ implements explicit compatibility exports.
                    if "__getattr__" not in definitions(trees[target]):
                        missing.append(dict(line=line, target=base, name=name, kind="missing_name"))
            elif base.split(".")[0] in {"pssolver", "tests", *SCOPES}:
                missing.append(dict(line=line, target=base, name=name, kind="missing_module"))
        graph[module] = sorted(set(edges))
        errors[module] = missing
    return paths, trees, graph, errors


def closure_errors(start, graph, errors):
    pending = [(start, [start])]
    seen, result = set(), []
    while pending:
        module, chain = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        result.extend(
            dict(error, chain=chain + [error["target"]]) for error in errors.get(module, [])
        )
        pending.extend(
            (target, chain + [target]) for target in graph.get(module, []) if target not in seen
        )
    return result


def scan(root, output, *, actual):
    paths, _, graph, errors = inventory(root)
    selected = {
        name: path
        for name, path in paths.items()
        if path.parent == root or path.relative_to(root).parts[0] in SCOPES
    }
    report = {"root": str(root), "modules": {}, "actual": {}}
    for name, path in sorted(selected.items()):
        report["modules"][name] = {
            "path": path.relative_to(root).as_posix(),
            "errors": closure_errors(name, graph, errors),
        }

    def probe(item):
        name, path = item
        code = (
            "import sys,importlib; sys.path.insert(0," + repr(str(root)) + "); "
            "from qualification._legacy_imports import install; install(); "
            "importlib.import_module(" + repr(name) + ")"
        )
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code],
                cwd=root,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return name, dict(exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr)
        except subprocess.TimeoutExpired as exc:
            return name, dict(
                exit_code=124, stdout=str(exc.stdout or ""), stderr=str(exc.stderr or "")
            )

    if actual:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for name, value in pool.map(
                probe,
                [
                    (n, p)
                    for n, p in sorted(selected.items())
                    if p.relative_to(root).parts[0] in {"benchmarks", "qualification"}
                ],
            ):
                report["actual"][name] = value
                print(name, value["exit_code"], flush=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--actual", action="store_true")
    args = parser.parse_args()
    scan(args.root, args.output, actual=args.actual)
