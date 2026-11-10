"""S-26 static test inventory, including transitive helpers and embedded names."""

import argparse
import ast
import json
from pathlib import Path

from qualification.s25_import_scan import imports, inventory
from qualification.s26_records import reader_edges, record_edges, retirement


CHANGED = {
    "pssolver.engine.engine",
    "pssolver.engine.program",
    "pssolver.engine.memberwise_transforms",
    "pssolver.planning.engine_plan",
    "pssolver.configuration.legacy_periodic",
    "pssolver.core.terms",
    "pssolver.planning.assemble",
    "pssolver.planning.term_analysis",
    "pssolver.planning.capabilities",
    "pssolver.engine.terms",
    "pssolver.presets.beris_edwards",
}
CHANGED.update(row["module"] for row in retirement()["modules"])
CHANGED.update(path[:-3].replace("/", ".") for path in retirement()["retired_tests"])


def affected(root):
    paths, trees, _, _ = inventory(root, nested=True)
    records = record_edges(root, CHANGED)
    direct, graph = {}, {}
    for module, tree in trees.items():
        direct[module], graph[module] = [], []
        for line, base, name in imports(
            tree, module, paths[module].name == "__init__.py", nested=True
        ):
            target = (
                base + "." + name
                if name and (base + "." + name in paths or base + "." + name in CHANGED)
                else base
            )
            if target in CHANGED:
                direct[module].append(f"{target}:{line}")
            elif target.startswith(("tests.", "benchmarks.", "qualification.")):
                graph[module].append(target)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            for old in sorted(CHANGED):
                parent, _, short = old.rpartition(".")
                if (
                    old in node.value
                    or old.replace(".", "/") in node.value
                    or (parent in node.value and short in node.value)
                ):
                    direct[module].append("embedded:" + old + ":" + str(node.lineno))

    def chains(module):
        seen, pending, found = set(), [(module, [module])], []
        while pending:
            name, chain = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            found.extend(chain + [target] for target in direct.get(name, []))
            pending.extend(
                (target, chain + [target]) for target in graph.get(name, []) if target not in seen
            )
        return found

    result = []
    for module, path in sorted(paths.items()):
        if not path.name.startswith("test_") or not module.startswith("tests."):
            continue
        evidence = chains(module)
        readers = reader_edges(trees[module], records)
        for node in trees[module].body:
            units = []
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                units = [node.name]
            elif isinstance(node, ast.ClassDef):
                units = [
                    node.name + "::" + f.name
                    for f in node.body
                    if isinstance(f, ast.FunctionDef) and f.name.startswith("test")
                ]
            for unit in units:
                unit_evidence = evidence + [
                    [
                        module + "::" + unit,
                        row["src"],
                        "record_string:" + row["pointer"] + ":" + row["target"],
                    ]
                    for row in readers.get(unit, [])
                ]
                if not unit_evidence:
                    continue
                result.append(
                    dict(path=path.relative_to(root).as_posix(), unit=unit, chains=unit_evidence)
                )
    return result


def check(root, disposition):
    rows = affected(root)
    known = {(r["path"], r["unit"]): r for r in disposition["affected_tests"]}
    unclassified = [r for r in rows if (r["path"], r["unit"]) not in known]
    return dict(
        root=str(root),
        tests=rows,
        unclassified=unclassified,
        summary=dict(
            files=len({r["path"] for r in rows}), tests=len(rows), unclassified=len(unclassified)
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("disposition", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = check(args.root, json.loads(args.disposition.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))
    raise SystemExit(bool(report["unclassified"]))
