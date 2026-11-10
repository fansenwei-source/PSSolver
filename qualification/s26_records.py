"""Record-string edges and test-reader attribution, without executing repository code."""

import ast
import fnmatch
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def retirement():
    """Read the single reviewed retirement registry next to these tools."""
    return json.loads(Path(__file__).with_name("s26_retired_modules.json").read_text())


def retirement_references(root):
    """Reject references outside the one registry and exact guard locations."""
    registry = retirement()
    targets = {row["module"] for row in registry["modules"]}
    allowed = {(r["file"], r["line"], r["kind"], r["module"]) for r in registry["guard_references"]}
    observed = set()
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if any(
            p.startswith(".") or p in {"build", "__pycache__"} for p in path.relative_to(root).parts
        ):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for target in targets:
                    if target in node.value:
                        observed.add((rel, node.lineno, "dotted_string", target))
                    if target.replace(".", "/") + ".py" in node.value:
                        observed.add((rel, node.lineno, "file_string", target))
    records = [
        edge
        for edge in record_edges(root, targets)
        if edge["src"] != "qualification/s26_retired_modules.json"
    ]
    expected, invalid = [], []
    for row in registry.get("provenance_references", []):
        relative = row.get("file", "")
        path = root / relative
        if (
            row.get("kind") != "record_string"
            or not relative.endswith(".json")
            or any(char in relative for char in "*?[]")
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.resolve().is_relative_to(root.resolve())
            or not re.fullmatch(r"[0-9a-f]{64}", row.get("file_sha256", ""))
        ):
            invalid.append(row)
            continue
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            digest = None
        if digest != row["file_sha256"]:
            invalid.append(row)
        expected.append((relative, row["pointer"], row["value"], row["kind"]))
    actual = [(r["src"], r["pointer"], r["value"], r["kind"]) for r in records]
    unexpected_provenance = sorted((Counter(actual) - Counter(expected)).elements())
    missing_provenance = sorted((Counter(expected) - Counter(actual)).elements())
    return dict(
        unexpected=sorted(observed - allowed),
        missing_guards=sorted(allowed - observed),
        record_references=records,
        unexpected_provenance=unexpected_provenance,
        missing_provenance=missing_provenance,
        invalid_provenance=invalid,
        passed=(
            observed == allowed
            and not unexpected_provenance
            and not missing_provenance
            and not invalid
        ),
    )


def record_edges(root, targets):
    """Preserve each JSON string/key occurrence and its JSON pointer as evidence."""
    result = []
    for path in sorted(root.rglob("*.json")):
        if any(
            p.startswith(".") or p in {"build", "__pycache__"} for p in path.relative_to(root).parts
        ):
            continue
        try:
            value = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        pending = [("", value)]
        while pending:
            pointer, item = pending.pop()
            if isinstance(item, dict):
                for key, content in item.items():
                    pending.extend([(pointer + "/@" + key, key), (pointer + "/" + key, content)])
            elif isinstance(item, list):
                pending.extend((pointer + "/" + str(i), v) for i, v in enumerate(item))
            elif isinstance(item, str):
                for target in targets:
                    if re.fullmatch(re.escape(target) + r"(?::[A-Za-z_][\w.]*)?", item) or (
                        item == target.replace(".", "/") + ".py"
                        or item.startswith(target.replace(".", "/") + ".py::")
                    ):
                        result.append(
                            dict(
                                src=path.relative_to(root).as_posix(),
                                target=target,
                                kind="record_string",
                                pointer=pointer,
                                value=item,
                            )
                        )
    return result


def reader_edges(tree, records):
    """Attribute literal record paths through globals and same-file helpers to tests.

    Computed paths without any literal record name remain an explicit static limit.
    No record contents are imported or treated as executable instructions.
    """
    by_path = {}
    by_name = {}
    for edge in records:
        by_path.setdefault(edge["src"], []).append(edge)
        by_name.setdefault(Path(edge["src"]).name, set()).add(edge["src"])
    definitions = {}
    tests = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                tests[node.name] = node
            elif isinstance(node, ast.ClassDef):
                tests.update(
                    (node.name + "::" + f.name, f)
                    for f in node.body
                    if isinstance(f, ast.FunctionDef) and f.name.startswith("test")
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = node.targets if isinstance(node, ast.Assign) else [node.target]
            for name in names:
                if isinstance(name, ast.Name):
                    definitions[name.id] = node
    result = {}
    for unit, test in tests.items():
        pending, seen, strings = [test], set(), set()
        while pending:
            node = pending.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    strings.add(child.value)
                elif isinstance(child, ast.Name) and child.id in definitions:
                    pending.append(definitions[child.id])
        selected = set()
        for value in strings:
            if not value.endswith(".json"):
                continue
            if value in by_path:
                selected.add(value)
            selected.update(by_name.get(value, ()))
            if any(char in value for char in "*?["):
                selected.update(
                    rel
                    for rel in by_path
                    if fnmatch.fnmatch(rel, value) or fnmatch.fnmatch(Path(rel).name, value)
                )
        matches = [edge for rel in sorted(selected) for edge in by_path[rel]]
        if matches:
            result[unit] = matches
    return result
