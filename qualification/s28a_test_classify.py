"""Inventory S-28a test references in both the parent and candidate trees."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from qualification.s25_import_scan import imports, inventory, module_name
from qualification.s27_records import reader_edges, record_edges


def units(tree):
    """Use pytest's function and class::method spelling."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name.startswith("test"):
                    yield f"{node.name}::{child.name}", child


def inventory_tests(root, table):
    """Follow exact imports through test helpers and frozen script callers."""
    paths, trees, _, _ = inventory(root, nested=True)
    changed = set(table["retired_modules"]) | {
        module_name(path) for path in table["changed_paths"] if path.startswith("pssolver/")
    }
    records = [
        row
        for row in record_edges(root, changed)
        if row["src"] != "notes/PSSolver_v0_3_0_s03_successor_147_s28a_channel.json"
    ]
    direct, helpers = {}, {}
    for module, tree in trees.items():
        direct[module], helpers[module] = set(), set()
        for line, base, name in imports(
            tree, module, paths[module].name == "__init__.py", nested=True
        ):
            named = f"{base}.{name}" if name else base
            target = named if named in paths or named in changed else base
            if target in changed:
                direct[module].add(f"{target}:{line}")
            elif target.startswith(("tests.", "benchmarks.", "qualification.")):
                helpers[module].add(target)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                direct[module].update(
                    f"embedded:{target}:{node.lineno}"
                    for target in changed
                    if target in node.value or target.replace(".", "/") + ".py" in node.value
                )
    rows = {}
    for module, path in sorted(paths.items()):
        if not module.startswith("tests.") or not path.name.startswith("test_"):
            continue
        visited, pending, evidence = set(), [(module,)], []
        while pending:
            chain = pending.pop()
            target = chain[-1]
            if target in visited:
                continue
            visited.add(target)
            evidence.extend((*chain, reference) for reference in sorted(direct.get(target, ())))
            pending.extend((*chain, helper) for helper in sorted(helpers.get(target, ())))
        readers = reader_edges(trees[module], records)
        for unit, _node in units(trees[module]):
            rows[(path.relative_to(root).as_posix(), unit)] = sorted(
                set(evidence)
                | {
                    (module + "::" + unit, row["src"], row["pointer"], row["target"])
                    for row in readers.get(unit, ())
                }
            )
    return rows


def collect(baseline, candidate):
    """Report a changed test even if all its retired imports have disappeared."""
    table = json.loads((candidate / "qualification/s28a_tables.json").read_bytes())
    before, after = (inventory_tests(tree, table) for tree in (baseline, candidate))
    changed_files = {
        path
        for path, _ in before.keys() | after.keys()
        if not (baseline / path).is_file()
        or not (candidate / path).is_file()
        or (baseline / path).read_bytes() != (candidate / path).read_bytes()
    }
    return [
        {
            "path": path,
            "unit": unit,
            "before": before.get((path, unit)),
            "after": after.get((path, unit)),
        }
        for path, unit in sorted(before.keys() | after.keys())
        if path in changed_files or before.get((path, unit)) or after.get((path, unit))
    ]


def chains_ref(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def verify(rows, record):
    required = {(row["path"], row["unit"]): chains_ref(row) for row in rows}
    present = {(row["path"], row["unit"]): row for row in record["affected_tests"]}
    unclassified = [
        path + "::" + unit
        for (path, unit), digest in required.items()
        if (path, unit) not in present or present[(path, unit)]["chains_ref"] != digest
    ]
    invalid = [
        f"{path}::{unit}"
        for (path, unit), row in present.items()
        if (path, unit) not in required
        or row["rule"]
        not in {
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
            "R6",
            "R7",
            "R8",
            "R9",
        }
    ]
    return {
        "unclassified": unclassified,
        "invalid": invalid,
        "tests": len(required),
        "passed": not (unclassified or invalid) and len(present) == len(record["affected_tests"]),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    disposition = json.loads(
        (args.candidate / "qualification/s28a_test_disposition.json").read_bytes()
    )
    rows = collect(args.baseline, args.candidate)
    result = verify(rows, disposition)
    args.output.write_text(json.dumps(dict(result, evidence=rows), indent=2) + "\n")
    print(result)
    raise SystemExit(not result["passed"])
