"""Account for changed tests, frozen original tests, and affected unchanged callers."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from qualification.s25_import_scan import imports, module_name, sources
from qualification.s28a_test_classify import units


REGISTRY = "qualification/s28b_test_disposition.json"
HISTORY = "qualification/s28b_historical_tests.json"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _details(root, changed_modules):
    module_paths = sources(root)
    impacts = {}
    for module, path in module_paths.items():
        if not module.startswith(("tests.", "qualification.", "benchmarks.")):
            continue
        tree = ast.parse(path.read_bytes())
        related, callers = set(), set()
        for lineno, base, name in imports(tree, module, path.name == "__init__.py", nested=True):
            member = f"{base}.{name}" if name else base
            if any(target in changed_modules for target in (base, member)):
                related.add(f"{member}:{lineno}")
            target = member if member in module_paths else base
            if target.startswith(("tests.", "qualification.", "benchmarks.")):
                callers.add(target)
        impacts[module] = (related, callers)
    result = {}
    for module, path in module_paths.items():
        if not module.startswith("tests.") or not path.name.startswith("test_"):
            continue
        pending = [module]
        seen, reasons = set(), set()
        while pending:
            entry = pending.pop()
            if entry in seen:
                continue
            seen.add(entry)
            direct, calls = impacts.get(entry, (set(), set()))
            reasons.update(f"{entry} -> {reason}" for reason in direct)
            pending.extend(sorted(calls))
        relative = path.relative_to(root).as_posix()
        result[relative] = {
            "sha256": _sha(path),
            "test_units": [name for name, _ in units(ast.parse(path.read_bytes()))],
            "changed_module_references": sorted(reasons),
        }
    return result


def collect(basis, tree):
    table = json.loads((tree / "qualification/s28b_tables.json").read_bytes())
    changed = {module_name(path) for path in table["changed_paths"] if path.startswith("pssolver/")}
    before, after = (_details(root, changed) for root in (basis, tree))
    return {
        path: {"before": before.get(path), "after": after.get(path)}
        for path in sorted(before.keys() | after.keys())
        if before.get(path) != after.get(path) or before[path]["changed_module_references"]
    }


def verify(rows, tree, record):
    recorded = record["files"]
    history = json.loads((tree / HISTORY).read_bytes())
    historical = {row["path"] for row in history["files"]}
    mismatches = [
        path
        for path in sorted(rows.keys() | recorded.keys())
        if path not in recorded or path not in rows or rows[path] != recorded[path]["evidence"]
    ]
    invalid = []
    for path in rows.keys() & recorded.keys():
        row = recorded[path]
        rule = row["rule"]
        if rule not in {"T1", "T2", "T3", "T4", "T5", "T6", "unchanged"} or not row["reason"]:
            invalid.append(path)
        elif (rule == "T3") != (path in historical):
            invalid.append(path)
        elif rows[path]["after"] is None and rule != "T3":
            invalid.append(path)
        elif rule == "unchanged" and (
            rows[path]["before"]["sha256"] != rows[path]["after"]["sha256"]
        ):
            invalid.append(path)
    return {
        "passed": not (mismatches or invalid),
        "unclassified": mismatches,
        "invalid": sorted(invalid),
        "files": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basis", type=Path)
    parser.add_argument("tree", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record = json.loads((args.tree / REGISTRY).read_bytes())
    report = verify(collect(args.basis, args.tree), args.tree, record)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
