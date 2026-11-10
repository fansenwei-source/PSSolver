"""Rebuild the R-0080 unit/import table on actual S-25 candidate paths."""

import argparse
import ast
import json
from pathlib import Path

from qualification.s25_import_scan import imports


FORBID = {
    "io": {
        "configuration",
        "functional",
        "api",
        "applications",
        "workflows",
        "runtime",
        "models",
        "planning",
    },
    "engine": {"api", "applications", "workflows", "configuration", "functional", "runtime", "io"},
    "planning": {
        "engine",
        "api",
        "applications",
        "workflows",
        "configuration",
        "functional",
        "runtime",
        "io",
        "models",
    },
    "models": {"configuration", "functional", "runtime", "io", "api", "applications", "workflows"},
    "configuration": {"functional", "api", "applications", "workflows", "runtime"},
    "runtime": {"functional", "api", "applications", "workflows"},
    "functional": {"api", "applications", "workflows", "runtime"},
}
EXACT = [
    "configuration/legacy_periodic.py",
    "models/active_nematics/engine_terms.py",
    "functional/periodic_activity.py",
    "functional/periodic_checkpoint.py",
    "functional/validation.py",
    "runtime/legacy_entry.py",
    "runtime/runner.py",
    "runtime/observations.py",
    "io/diagnostic_io.py",
    "io/engine_state.py",
    "planning/engine_plan.py",
    "planning/periodic.py",
]


def check(root):
    paths = [root / "pssolver" / p for p in EXACT]
    paths += list((root / "pssolver/engine").glob("*.py"))
    paths += list((root / "pssolver/io/legacy").glob("*.py"))
    edges, violations, units = [], [], []
    for path in sorted(set(paths)):
        module = path.relative_to(root).as_posix()[:-3].replace("/", ".")
        tree = ast.parse(path.read_text())
        caller_layer = module.split(".")[1]
        for node in tree.body:
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign),
            ):
                units.append(
                    {
                        "module": module,
                        "line": node.lineno,
                        "unit": getattr(node, "name", "<assignment>"),
                    }
                )
        for line, target, name in imports(tree, module, path.name == "__init__.py", nested=True):
            if not target.startswith("pssolver."):
                continue
            row = dict(module=module, line=line, target=target, name=name)
            edges.append(row)
            if target.split(".")[1] in FORBID.get(caller_layer, set()):
                violations.append(row)
    return dict(
        units=units,
        edges=edges,
        violations=violations,
        summary=dict(
            modules=len(set(paths)), units=len(units), edges=len(edges), violations=len(violations)
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = check(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))
    raise SystemExit(bool(result["violations"]))
