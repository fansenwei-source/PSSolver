"""Record-string edges and test-reader attribution, without executing repository code."""

import ast
import fnmatch
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


BINDING_RECORD = "notes/PSSolver_v0_3_0_s03_successor_146_s27_plane.json"


def authority():
    """The installed r2 table is the sole retirement-category authority."""
    return json.loads(Path(__file__).with_name("s27_r2_tables.json").read_text())


REGISTRY_FILES = frozenset(authority()["registry_files"]) - {BINDING_RECORD}


def baseline_bytes(root, relative):
    """Read only the reachable S-27 parent; a missing file is not frozen."""
    result = subprocess.run(
        ["git", "show", "327a192ba879da0991110eca99f4b69b9a45c84e:" + relative],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def retirement():
    """Read the single reviewed retirement registry next to these tools."""
    return json.loads(Path(__file__).with_name("s27_retired_modules.json").read_text())


def retirement_references(root):
    """Apply r2 section 6.3 categories, not the superseded per-line registry."""
    table = authority()
    registry = retirement()
    targets = set(table["retired_modules"])
    needles = targets | {name.replace(".", "/") + ".py" for name in targets}
    unexpected, counts = [], Counter()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.suffix not in {".py", ".json"}:
            continue
        if any(p.startswith(".") or p in {"build", "__pycache__"} for p in relative.parts):
            continue
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore")
        if not any(needle in text for needle in needles):
            continue
        rel = relative.as_posix()
        before = baseline_bytes(root, rel)
        unchanged = before == data
        lines = (
            set(before.decode("utf-8", errors="ignore").splitlines())
            if before is not None
            else set()
        )
        for number, line in enumerate(text.splitlines(), 1):
            if not any(needle in line for needle in needles):
                continue
            category = None
            if path.suffix == ".py":
                if rel.startswith("pssolver/"):
                    category = "VIOLATION_package"
                elif unchanged and line in lines:
                    category = "frozen_script"
                elif line in lines:
                    category = "historical_line"
                elif rel in table["guard_files"]:
                    category = "guard"
                elif any(fnmatch.fnmatch(rel, pattern) for pattern in table["own_file_patterns"]):
                    category = "s27_own_file"
            elif unchanged:
                category = "provenance"
            elif rel in table["registry_files"]:
                category = "registry"
            if category is None or category.startswith("VIOLATION"):
                unexpected.append(dict(file=rel, line=number, category=category))
            else:
                counts[category] += 1
    invalid_registry = sorted(set(registry.get("registry_files", [])) ^ REGISTRY_FILES)
    invalid_modules = sorted({row["module"] for row in registry["modules"]} ^ targets)
    return dict(
        unexpected=unexpected,
        counts=dict(counts),
        invalid_registry_files=invalid_registry,
        invalid_modules=invalid_modules,
        passed=not (unexpected or invalid_registry or invalid_modules),
    )


def record_edges(root, targets):
    """Preserve each JSON string/key occurrence and its JSON pointer as evidence."""
    result = []
    patterns = {
        target: re.compile(re.escape(target) + r"(?::[A-Za-z_][\w.]*)?") for target in targets
    }
    file_paths = {target: target.replace(".", "/") + ".py" for target in targets}
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
                    if patterns[target].fullmatch(item) or (
                        item == file_paths[target] or item.startswith(file_paths[target] + "::")
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
