"""Four fail-closed S-26 import, state-write and classification checks."""

import argparse
import ast
import json
import subprocess
from pathlib import Path

from qualification.s25_boundary_check import check as boundary_check
from qualification.s25_import_scan import imports
from qualification.s26_guards import planning_closure
from qualification.s26_records import retirement, retirement_references
from qualification.s26_test_classify import check as classification_check


BASE = "63f19383ec2d05aedc8b0354f058302f74f4a751"
DELETED = {row["module"] for row in retirement()["modules"]}
STATE_WRITES = {
    ("pssolver/engine/engine.py", "state", "solver"),
    ("pssolver/engine/program.py", "out", "constrained_current"),
    ("pssolver/engine/program.py", "out", "completed_steps"),
    ("pssolver/engine/program.py", "out", "refresh_phase"),
}


def check(root):
    deleted_imports, layers, writes, counts = [], [], [], {}
    for path in sorted((root / "pssolver").rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        module = relative[:-3].replace("/", ".").removesuffix(".__init__")
        tree = ast.parse(path.read_text())
        for line, base, name in imports(tree, module, path.name == "__init__.py", nested=True):
            full = base + "." + name if name else base
            if base in DELETED or full in DELETED:
                deleted_imports.append(dict(path=relative, line=line, base=base, name=name))
            if module.startswith(("pssolver.planning.", "pssolver.core.")):
                numerical = {"torch", "numpy", "scipy"}
                components = {
                    "engine",
                    "models",
                    "presets",
                    "backends",
                    "operators",
                    "linear_solvers",
                }
                late_factory = (
                    module == "pssolver.planning.assemble"
                    and base == "pssolver.engine.engine"
                    and name == "Engine"
                )
                if base.split(".")[0] in numerical or (
                    base.startswith("pssolver.")
                    and base.split(".")[1] in components
                    and not late_factory
                ):
                    layers.append(dict(path=relative, line=line, target=base))
        if not module.startswith(("pssolver.planning.", "pssolver.engine.")):
            continue
        old = subprocess.run(["git", "show", BASE + ":" + relative], cwd=root, capture_output=True)
        previous = set()
        if old.returncode == 0:
            previous = {ast.dump(node) for node in ast.walk(ast.parse(old.stdout))}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                target = (
                    node.value.id if isinstance(node.value, ast.Name) else ast.unparse(node.value)
                )
                key = (relative, target, node.attr)
                if key in STATE_WRITES:
                    counts[str(key)] = counts.get(str(key), 0) + 1
                elif target != "self" and ast.dump(node) not in previous:
                    writes.append(dict(path=relative, line=node.lineno, target=ast.unparse(node)))
            elif isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name)
                and node.func.id == "setattr"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "__setattr__"
            ):
                if ast.dump(node) not in previous and (
                    not node.args
                    or not isinstance(node.args[0], ast.Name)
                    or node.args[0].id != "self"
                ):
                    writes.append(dict(path=relative, line=node.lineno, call=ast.unparse(node)))
    boundary = boundary_check(root)
    closure = planning_closure(root)
    references = retirement_references(root)
    classification = classification_check(
        root, json.loads((root / "qualification/s26_test_disposition.json").read_text())
    )
    deleted_present = [
        name for name in DELETED if (root / (name.replace(".", "/") + ".py")).exists()
    ]
    return dict(
        deleted_imports=deleted_imports,
        deleted_present=deleted_present,
        layers=layers,
        new_external_attribute_writes=writes,
        state_writes=counts,
        boundary=boundary,
        planning_closure=closure,
        retirement_references=references,
        classification=classification,
        passed=not (
            deleted_imports
            or deleted_present
            or layers
            or writes
            or boundary["violations"]
            or closure["violations"]
            or not references["passed"]
            or classification["unclassified"]
        )
        and sum(counts.values()) == 5,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                passed=result["passed"], unclassified=len(result["classification"]["unclassified"])
            )
        )
    )
    raise SystemExit(not result["passed"])
