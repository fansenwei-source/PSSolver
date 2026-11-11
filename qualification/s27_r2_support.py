"""Import-safe access to the exact r2 gate's static owner scanner."""

import ast
import json
from pathlib import Path


def tables():
    """Read the installed, independently reviewed authority, without executing it."""
    return json.loads(Path(__file__).with_name("s27_r2_tables.json").read_text())


def edges(root):
    """Execute only the gate's two pure AST-scanning function definitions."""
    path = Path(__file__).with_name("s27_guard_crosscheck.py")
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"modmap", "edges"}
    ]
    if {node.name for node in definitions} != {"modmap", "edges"}:
        raise ValueError("r2 owner scanner definitions missing")
    namespace = {"ast": ast}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["edges"](root)


def owner_violations(observed, record):
    """Require the exact owner set for every reviewed and rehomed edge."""
    return [
        dict(
            source=row["source"],
            destination=row["destination"],
            expected=sorted(row["owners"]),
            observed=sorted(observed.get((row["source"], row["destination"]), set())),
        )
        for row in record["edges"]
        if observed.get((row["source"], row["destination"]), set()) != set(row["owners"])
    ]
