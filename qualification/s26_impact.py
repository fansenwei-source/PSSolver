"""S-26 impact inventory (T-0019). Static AST analysis of one PSSolver tree; read-only.

Usage: python3 -I -B s26_impact.py <tree_root> <output_dir>
<tree_root> is an extracted tree (git archive), never a worktree. Writes only into <output_dir>.

Sections (one JSON each, plus summary.txt):
  modules.json      every parsed file -> module name, kind (package/test/qualification/other)
  edges.json        every reference edge: importer -> target module (+name), kind in
                    {import, from_import, nested_import, relative, import_module_string,
                     dotted_string, file_string, ref_string}
  targets.json      for each S-26 target module: importers grouped by area and kind
  deletable.json    maximal subset of the S-26 deletion candidates whose package importers
                    all lie inside the subset (iterated closure); blockers for the rest
  registries.json   module-level literal containers holding import-path strings or
                    application keys ("where is 'what can run' registered")
  layers.json       planning/ and core/ modules importing torch, numpy or component /
                    numeric layers (target rule S-22 3.1)
  tests.json        test functions touching S-26 targets (name-level attribution,
                    same-file helper propagation) -> target set
  side_effects.json attribute writes / setattr / MRO for classes in the S-26 touch set
Limits: static; dynamic getattr on computed names and eval are not modelled; attribution of
tests through helpers is by name within one file (plus imported helper modules listed as edges).
"""

import ast
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def main():
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qualification.s26_records import reader_edges, record_edges, retirement

    ROOT = Path(sys.argv[1]).resolve()
    OUT = Path(sys.argv[2]).resolve()
    OUT.mkdir(parents=True, exist_ok=False)

    # ---- S-26 target sets (reviewable data; S-22 section 4 E8 part 2, S-26 scope) -------------
    DELETE_CANDIDATES = [row["module"] for row in retirement()["modules"]]
    # Package __init__ modules that S-26 edits anyway: their re-export edges do not block deletion,
    # they are reported as required edits instead.
    EDITABLE_REEXPORTERS = {"pssolver.planning", "pssolver.models"}
    CHANGE_TARGETS = [
        "pssolver.configuration.legacy_periodic",
        "pssolver.api.capabilities",
        "pssolver.configuration.application_registry",
        "pssolver.engine.engine",
        "pssolver.engine.program",
        "pssolver.engine.constraints",
        "pssolver.engine.linear",
        "pssolver.planning.engine_plan",
        "pssolver.planning",
        "pssolver.presets",
        "pssolver.models.active_nematics.engine_terms",
        "pssolver.io.engine_state",
        "pssolver.runtime.runner",
        "pssolver.runtime.legacy_entry",
        "pssolver.functional.periodic_activity",
        "pssolver.functional.validation",
        "pssolver.core.terms",
        "pssolver.planning.assemble",
        "pssolver.planning.capabilities",
        "pssolver.planning.term_analysis",
        "pssolver.engine.terms",
        "pssolver.engine.memberwise_transforms",
        "pssolver.presets.beris_edwards",
    ]
    TARGETS = DELETE_CANDIDATES + CHANGE_TARGETS
    COMPONENT_LAYERS = {
        "models",
        "presets",
        "geometries",
        "boundaries",
        "integrators",
        "engine",
        "backends",
        "operators",
        "linear_solvers",
        "runtime",
        "io",
        "functional",
        "api",
        "applications",
        "workflows",
        "adapters",
        "configuration",
        "execution",
        "systems",
        "experimental",
        "control",
    }
    DOTTED = re.compile(r"^pssolver(?:\.[A-Za-z_]\w*)+(?::[A-Za-z_][\w.]*)?$")
    FILEPATH = re.compile(r"pssolver/[\w/]+\.py")

    def area(rel):
        top = rel.split("/")[0]
        if top == "pssolver":
            return "package"
        if top in {"tests", "qualification", "benchmarks", "tools", "examples", "jobs", "docs"}:
            return top
        return "other"

    def module_name(rel):
        name = rel[:-3].replace("/", ".")
        return name[: -len(".__init__")] if name.endswith(".__init__") else name

    files = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if any(
            part in {"build", ".git", "__pycache__"} or part.endswith(".egg-info")
            for part in rel.split("/")
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError) as exc:
            files[rel] = dict(module=module_name(rel), area=area(rel), parse_error=str(exc))
            continue
        files[rel] = dict(
            module=module_name(rel), area=area(rel), tree=tree, is_pkg=path.name == "__init__.py"
        )
    package_modules = {info["module"] for info in files.values() if info["area"] == "package"}

    def resolve_relative(info, level, module):
        parts = info["module"].split(".")
        base = parts if info["is_pkg"] else parts[:-1]
        base = base[: len(base) - (level - 1)] if level > 1 else base
        return ".".join(base + ([module] if module else []))

    def target_of(name):
        """Longest package module prefix of a dotted name."""
        parts = name.split(".")
        for end in range(len(parts), 0, -1):
            candidate = ".".join(parts[:end])
            if candidate in package_modules or candidate in DELETE_CANDIDATES:
                return candidate, ".".join(parts[end:])
        return None, None

    edges = []
    for rel, info in files.items():
        tree = info.get("tree")
        if tree is None:
            continue
        top_level = {id(node) for node in tree.body}
        for node in ast.walk(tree):
            kind = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod, rest = target_of(alias.name)
                    if mod:
                        edges.append(
                            dict(
                                src=rel,
                                line=node.lineno,
                                target=mod,
                                name=rest,
                                kind="import" if id(node) in top_level else "nested_import",
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                rel_kind = None
                if node.level:
                    base = resolve_relative(info, node.level, node.module)
                    rel_kind = "relative"
                for alias in node.names:
                    full = f"{base}.{alias.name}"
                    mod, rest = target_of(full)
                    if mod is None:
                        mod, rest = target_of(base)
                        rest = alias.name if rest == "" else f"{rest}.{alias.name}"
                    if mod:
                        kind = rel_kind or (
                            "from_import" if id(node) in top_level else "nested_import"
                        )
                        edges.append(
                            dict(src=rel, line=node.lineno, target=mod, name=rest, kind=kind)
                        )
            elif isinstance(node, ast.Call):
                func = node.func
                fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if (
                    fname == "import_module"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    mod, rest = target_of(node.args[0].value)
                    if mod:
                        edges.append(
                            dict(
                                src=rel,
                                line=node.lineno,
                                target=mod,
                                name=rest,
                                kind="import_module_string",
                            )
                        )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                if DOTTED.match(value):
                    left = value.split(":")[0]
                    mod, rest = target_of(left)
                    if mod:
                        edges.append(
                            dict(
                                src=rel,
                                line=node.lineno,
                                target=mod,
                                name=rest,
                                kind="ref_string" if ":" in value else "dotted_string",
                            )
                        )
                for match in FILEPATH.findall(value):
                    mod = module_name(match)
                    if mod in package_modules or mod in DELETE_CANDIDATES:
                        edges.append(
                            dict(src=rel, line=node.lineno, target=mod, name="", kind="file_string")
                        )

    records = record_edges(ROOT, TARGETS)
    for edge in records:
        files.setdefault(edge["src"], dict(module=edge["src"], area=area(edge["src"])))
    edges.extend(dict(edge, line=0, name=edge["pointer"]) for edge in records)
    readers = {
        rel: reader_edges(info["tree"], records)
        for rel, info in files.items()
        if "tree" in info and info["area"] == "tests"
    }
    (OUT / "record_readers.json").write_text(json.dumps(readers, indent=1) + "\n")

    (OUT / "modules.json").write_text(
        json.dumps(
            {rel: {k: v for k, v in info.items() if k != "tree"} for rel, info in files.items()},
            indent=1,
        )
        + "\n"
    )
    (OUT / "edges.json").write_text(json.dumps(edges, indent=1) + "\n")

    def in_target(target, mod):
        return mod == target or mod.startswith(target + ".")

    # ---- targets ---------------------------------------------------------------------------
    targets = {}
    for target in TARGETS:
        rows = [e for e in edges if e["target"] == target and module_name(e["src"]) != target]
        grouped = defaultdict(lambda: defaultdict(list))
        for e in rows:
            grouped[files[e["src"]]["area"]][e["kind"]].append(
                f"{e['src']}:{e['line']}:{e['name']}"
            )
        targets[target] = {
            a: {k: sorted(set(v)) for k, v in kinds.items()} for a, kinds in grouped.items()
        }
    (OUT / "targets.json").write_text(json.dumps(targets, indent=1) + "\n")

    # ---- deletable closure -----------------------------------------------------------------
    candidate = {m for m in DELETE_CANDIDATES if m in package_modules}
    blockers = {}
    changed = True
    while changed:
        changed = False
        for mod in sorted(candidate):
            outside = sorted(
                {
                    f"{e['src']}:{e['line']}({e['kind']})"
                    for e in edges
                    if e["target"] == mod
                    and files[e["src"]]["area"] == "package"
                    and module_name(e["src"]) not in candidate
                    and module_name(e["src"]) not in EDITABLE_REEXPORTERS
                }
            )
            if outside:
                candidate.discard(mod)
                blockers[mod] = outside
                changed = True
    deletable = sorted(candidate)
    non_package_refs = {
        m: sorted(
            {
                f"{e['src']}:{e['line']}({e['kind']})"
                for e in edges
                if e["target"] == m and files[e["src"]]["area"] != "package"
            }
        )
        for m in deletable
    }
    required_edits = {
        m: sorted(
            {
                f"{e['src']}:{e['line']}({e['kind']}):{e['name']}"
                for e in edges
                if e["target"] == m and module_name(e["src"]) in EDITABLE_REEXPORTERS
            }
        )
        for m in deletable
    }
    (OUT / "deletable.json").write_text(
        json.dumps(
            dict(
                deletable_in_s26=deletable,
                already_deleted=sorted(set(DELETE_CANDIDATES) - package_modules),
                blocked=blockers,
                required_reexport_edits=required_edits,
                deletable_non_package_references=non_package_refs,
            ),
            indent=1,
        )
        + "\n"
    )

    # ---- registries ------------------------------------------------------------------------
    APP_KEY = re.compile(
        r"^(periodic|plane|channel)_[a-z_]*beris_edwards$|^(periodic_spectral|"
        r"legacy_production|compiled_v2|separated_canary)$"
    )
    registries = []
    for _rel, info in files.items():
        tree = info.get("tree")
        if tree is None or info["area"] != "package":
            continue
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
                continue
            names = [
                t.id
                for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
                if isinstance(t, ast.Name)
            ]
            strings = [
                c.value
                for c in ast.walk(node.value)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            ]
            dotted = [s for s in strings if DOTTED.match(s)]
            keys = [s for s in strings if APP_KEY.match(s)]
            calls = [c for c in ast.walk(node.value) if isinstance(c, ast.Call)]
            if len(dotted) >= 2 or len(keys) >= 2:
                registries.append(
                    dict(
                        module=info["module"],
                        line=node.lineno,
                        names=names,
                        dotted_refs=len(dotted),
                        application_keys=len(keys),
                        constructor_calls=len(calls),
                        sample=sorted(set(dotted + keys))[:6],
                    )
                )
    (OUT / "registries.json").write_text(json.dumps(registries, indent=1) + "\n")

    # ---- layers ----------------------------------------------------------------------------
    layers = {}
    for rel, info in files.items():
        tree = info.get("tree")
        if tree is None or not rel.startswith(("pssolver/planning/", "pssolver/core/")):
            continue
        found = []
        late_factory = set()
        if info["module"] == "pssolver.planning.assemble":
            for cls in tree.body:
                if isinstance(cls, ast.ClassDef) and cls.name == "AssembledProblem":
                    for method in cls.body:
                        if isinstance(method, ast.FunctionDef) and method.name == "engine":
                            late_factory.update(
                                id(n)
                                for n in ast.walk(method)
                                if isinstance(n, ast.ImportFrom)
                                and n.module == "pssolver.engine.engine"
                                and [(a.name, a.asname) for a in n.names] == [("Engine", None)]
                            )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if id(node) in late_factory:
                    continue
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [
                        resolve_relative(info, node.level, node.module)
                        if node.level
                        else (node.module or "")
                    ]
                )
                for n in names:
                    top = n.split(".")[0]
                    if top in {"torch", "numpy"}:
                        found.append(f"{node.lineno}:{n}")
                    elif n.startswith("pssolver."):
                        layer = n.split(".")[1]
                        if layer in COMPONENT_LAYERS:
                            found.append(f"{node.lineno}:{n}")
        layers[info["module"]] = sorted(set(found))
    (OUT / "layers.json").write_text(json.dumps(layers, indent=1) + "\n")

    # ---- tests -----------------------------------------------------------------------------
    def touched(target_mods, mod):
        return any(in_target(t, mod) for t in target_mods)

    tests = {}
    for rel, info in files.items():
        tree = info.get("tree")
        if tree is None or info["area"] != "tests":
            continue
        imported = {}  # local name -> target module
        file_hits = set()
        for e in edges:
            if e["src"] != rel:
                continue
            if any(e["target"] == t for t in TARGETS):
                file_hits.add(e["target"])
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                base = (
                    resolve_relative(info, node.level, node.module) if node.level else node.module
                )
                for alias in node.names:
                    full = f"{base}.{alias.name}"
                    mod, _ = target_of(full)
                    if mod is None:
                        mod, _ = target_of(base)
                    if mod in TARGETS:
                        imported[alias.asname or alias.name] = mod
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    mod, _ = target_of(alias.name)
                    if mod in TARGETS and alias.asname:
                        imported[alias.asname] = mod
        if not file_hits and not imported:
            continue
        funcs = {
            n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        direct = {}
        for name, fn in funcs.items():
            hit = set()
            for sub in ast.walk(fn):
                if isinstance(sub, ast.Name) and sub.id in imported:
                    hit.add(imported[sub.id])
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    value = sub.value
                    if DOTTED.match(value):
                        mod, _ = target_of(value.split(":")[0])
                        if mod in TARGETS:
                            hit.add(mod)
                    for match in FILEPATH.findall(value):
                        if module_name(match) in TARGETS:
                            hit.add(module_name(match))
            direct[name] = hit
        # same-file helper propagation (fixed point)
        calls = {
            name: {
                s.func.id
                for s in ast.walk(fn)
                if isinstance(s, ast.Call) and isinstance(s.func, ast.Name) and s.func.id in funcs
            }
            for name, fn in funcs.items()
        }
        total = {k: set(v) for k, v in direct.items()}
        grew = True
        while grew:
            grew = False
            for name in funcs:
                for callee in calls[name]:
                    if not total[callee] <= total[name]:
                        total[name] |= total[callee]
                        grew = True
        rows = {name: sorted(hit) for name, hit in total.items() if name.startswith("test") and hit}
        if rows or file_hits:
            tests[rel] = dict(file_level=sorted(file_hits), functions=rows)
    (OUT / "tests.json").write_text(json.dumps(tests, indent=1) + "\n")

    # ---- side effects / MRO ----------------------------------------------------------------
    SIDE_MODULES = [
        "pssolver.engine.engine",
        "pssolver.engine.program",
        "pssolver.engine.constraints",
        "pssolver.models.active_nematics.engine_terms",
        "pssolver.models.active_nematics.stokes",
        "pssolver.configuration.legacy_periodic",
        "pssolver.api.capabilities",
        "pssolver.functional.periodic_activity",
    ]
    side = {}
    for _rel, info in files.items():
        if info.get("module") not in SIDE_MODULES or info.get("tree") is None:
            continue
        writes, classes = [], {}
        for node in ast.walk(info["tree"]):
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets_ = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets_:
                    if isinstance(t, ast.Attribute):
                        owner = ast.unparse(t.value)
                        writes.append(f"{node.lineno}:{owner}.{t.attr}")
            elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setattr":
                writes.append(
                    f"{node.lineno}:setattr({ast.unparse(node.args[0]) if node.args else ''},...)"
                )
            elif isinstance(node, ast.ClassDef):
                classes[node.name] = [ast.unparse(b) for b in node.bases]
        side[info["module"]] = dict(
            attribute_writes_non_self_init=sorted(
                {
                    w
                    for w in writes
                    if ".last_" in w or "setattr" in w or not w.split(":")[1].startswith("self.")
                }
            ),
            attribute_writes_total=len(writes),
            class_bases=classes,
        )
    (OUT / "side_effects.json").write_text(json.dumps(side, indent=1) + "\n")

    # ---- summary ---------------------------------------------------------------------------
    lines = [
        f"tree: {ROOT}",
        f"parsed files: {len(files)}  package modules: {len(package_modules)}",
        f"edges: {len(edges)}",
        "",
    ]
    lines.append("deletable in S-26 (closure): " + ", ".join(deletable))
    for mod, rows in required_edits.items():
        if rows:
            lines.append(f"  requires re-export edit for {mod}: {rows}")
    for mod, why in blockers.items():
        lines.append(f"BLOCKED {mod}: {len(why)} package refs, e.g. {why[:3]}")
    lines.append("")
    for target in TARGETS:
        counts = {
            a: sum(len(v) for v in kinds.values()) for a, kinds in targets.get(target, {}).items()
        }
        lines.append(f"target {target}: {counts}")
    lines.append("")
    lines.append(
        f"registries (package, >=2 import-path strings or application keys): {len(registries)}"
    )
    for r in registries:
        lines.append(
            f"  {r['module']}:{r['line']} {r['names']} dotted={r['dotted_refs']} keys={r['application_keys']}"
        )
    lines.append("")
    bad_layers = {m: v for m, v in layers.items() if v}
    lines.append(
        f"planning/core modules with torch/numpy/component imports: {len(bad_layers)} of {len(layers)}"
    )
    for m, v in sorted(bad_layers.items()):
        lines.append(f"  {m}: {v[:6]}{' ...' if len(v) > 6 else ''}")
    lines.append("")
    nfun = sum(len(v["functions"]) for v in tests.values())
    lines.append(f"test files touching targets: {len(tests)}; test functions attributed: {nfun}")
    per_target = defaultdict(int)
    for v in tests.values():
        for hits in v["functions"].values():
            for h in hits:
                per_target[h] += 1
    for t, n in sorted(per_target.items()):
        lines.append(f"  {t}: {n} test functions")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
