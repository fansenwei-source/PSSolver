"""S-28a release gate (T-0022): check a tree against the single S-28a authority tables.

Usage:
  python3 -I -B s28a_guard.py design <baseline_tree 4b72093> <s28a_tables.json> <out.json>
      [--simulate-install]
  python3 -I -B s28a_guard.py draft <baseline_tree 4b72093> <draft_tree> <s28a_tables.json> <out.json>
      [--simulate-install]

design: before implementation, the tables are checked against the baseline facts (D1-D6): retired modules
exist and are scheduled for deletion; every exact or mechanical move names an existing source unit with the
recorded AST hash; moved sets are closed under module-level references; planned paths are consistent; every
allowed edge obeys the S-22 layer rule; the criteria keep exactly the two section-9 exclusions.
draft: after implementation, the draft tree is checked against the baseline (G1-G5 and OWNERS, as the S-27
r2.3 gate): new import edges, layers and the torch-free closure, retired names, changed paths and frozen
files, and units (exact moves AST-identical, mechanical moves AST-identical after their declared
substitutions, required units present, every new or rewritten unit registered).

Installation view (R-0098, R-0099): --simulate-install overlays this gate and the tables at their installed
paths; without it an installed copy must be byte-identical to the running copy, and in both views the running
gate must have the sha256 bound by the tables ('gate_sha256').  Version-control metadata (.git, .hg, .svn) is
not part of a checked tree (R-0100).  Importing this module only defines functions.
Exit status 1 if any count is non-zero, 2 on a usage error.
"""

import ast
import fnmatch
import hashlib
import json
import re
import sys
from pathlib import Path


VCS_METADATA = frozenset({".git", ".hg", ".svn"})
USAGE = (
    "usage: s28a_guard.py design <baseline> <tables> <out.json> [--simulate-install]\n"
    "       s28a_guard.py draft <baseline> <draft> <tables> <out.json> [--simulate-install]"
)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def installation_view(tree, tables, tables_path, gate_self, simulate_install):
    """Return (overlay, issues) for the checked tree."""
    installed = tables["installed_paths"]
    overlay = {
        installed["gate"]: gate_self.read_bytes(),
        installed["tables"]: tables_path.read_bytes(),
    }
    issues = []
    bound = tables.get("gate_sha256")
    if bound is not None and sha_bytes(overlay[installed["gate"]]) != bound:
        issues.append(dict(kind="running_gate_not_bound_by_tables", path=installed["gate"]))
    if not simulate_install:
        for rel, data in overlay.items():
            p = tree / rel
            if p.exists() and p.read_bytes() != data:
                issues.append(dict(kind="installed_copy_differs_from_running_copy", path=rel))
        for rel in tables["installation_removes"]:
            if (tree / rel).exists():
                issues.append(dict(kind="installation_remove_still_present", path=rel))
    return overlay, issues


def tree_files(root, overlay):
    """Relative path -> bytes of the checked tree (VCS metadata, build and egg-info excluded)."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if (
            not p.is_file()
            or rel.startswith("build/")
            or ".egg-info" in rel
            or not VCS_METADATA.isdisjoint(rel.split("/"))
        ):
            continue
        out[rel] = p.read_bytes()
    if overlay:
        out.update(overlay)
    return out


def modmap(root):
    out = {}
    for p in sorted((root / "pssolver").rglob("*.py")):
        rel = p.relative_to(root).as_posix()
        name = rel[:-3].replace("/", ".")
        out[name[:-9] if name.endswith(".__init__") else name] = p
    return out


def edges(root):
    mods = modmap(root)
    res = {}
    for mod, path in mods.items():
        tree = ast.parse(path.read_text())
        is_pkg = path.name == "__init__.py"
        users = {}
        for top in tree.body:
            unit = getattr(top, "name", None)
            if unit is None:
                continue
            for sub in ast.walk(top):
                if isinstance(sub, ast.Name):
                    users.setdefault(sub.id, set()).add(unit)
        for top in tree.body:
            unit = getattr(top, "name", "<module>")
            for node in ast.walk(top):
                if isinstance(node, ast.Import):
                    pairs = [(a.name, (a.asname or a.name).split(".")[0]) for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        parts = mod.split(".") if is_pkg else mod.split(".")[:-1]
                        parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                        base = ".".join(parts + ([node.module] if node.module else []))
                    pairs = [
                        (
                            f"{base}.{a.name}" if f"{base}.{a.name}" in mods else base,
                            a.asname or a.name,
                        )
                        for a in node.names
                    ]
                else:
                    continue
                for t, bound in pairs:
                    if not t.startswith("pssolver"):
                        continue
                    while t not in mods and "." in t:
                        t = t.rsplit(".", 1)[0]
                    if t not in mods or t == mod:
                        continue
                    owners = {unit} if unit != "<module>" else (users.get(bound) or {"<re-export>"})
                    res.setdefault((mod, t), set()).update(owners)
    return res


def layer(mod):
    p = mod.split(".")
    return p[1] if len(p) > 1 else p[0]


def ratchet(root):
    src = (root / "tests/test_architecture_import_boundaries.py").read_text()
    ns = {"frozenset": frozenset}

    class ImportEdge(tuple):
        def __new__(cls, a, b):
            return tuple.__new__(cls, (a, b))

    ns["ImportEdge"] = ImportEdge
    for node in ast.parse(src).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id.isupper()
        ):
            try:
                ns[node.targets[0].id] = eval(compile(ast.Expression(node.value), "r", "eval"), ns)
            except Exception:
                pass
    reviewed = set()
    for v in ns.values():
        vals = v.values() if isinstance(v, dict) else [v]
        for vv in vals:
            if (
                isinstance(vv, (set, frozenset))
                and vv
                and all(isinstance(e, tuple) and len(e) == 2 for e in vv)
            ):
                reviewed |= {tuple(e) for e in vv}
    return {k: frozenset(v) for k, v in ns.get("ALLOWED_INTERNAL_LAYERS", {}).items()}, reviewed


def units_of_source(text):
    out = {}
    for n in ast.parse(text).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = n
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                if isinstance(t, ast.Name):
                    out[t.id] = n
    return out


def find_unit(root, ref):
    """'path:name' or 'path:Class.method' -> AST node, or None."""
    path, name = ref.split(":")
    p = root / path
    if not p.is_file():
        return None
    found = units_of_source(p.read_text())
    if "." not in name:
        return found.get(name)
    cls, meth = name.split(".")
    node = found.get(cls)
    if not isinstance(node, ast.ClassDef):
        return None
    for sub in node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == meth:
            return sub
    return None


def ast_sha(node):
    return sha_bytes(ast.dump(node).encode())


def substituted(node, substitutions):
    """Apply the declared text substitutions to the unparsed unit; return the new AST node or None."""
    text = ast.unparse(node)
    for old, new in substitutions:
        if old not in text:
            return None
        text = text.replace(old, new)
    try:
        return ast.parse(text).body[0]
    except SyntaxError:
        return None


def target_ref(move):
    name = move["unit"]
    return f"{move['target']}:{name}"


def all_moves(tables):
    return [("exact", m) for m in tables["exact_moves"]] + [
        ("mechanical", m) for m in tables["mechanical_moves"]
    ]


def class_members(cls_node, dynamic=()):
    """Methods, self attributes stored, literal register_buffer names and declared dynamic buffers."""
    names = set(dynamic)
    for node in ast.walk(cls_node):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register_buffer"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            names.add(node.args[0].value)
    for sub in cls_node.body:
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(sub.name)
        for node in ast.walk(sub):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Store)
            ):
                names.add(node.attr)
    return names


# ---- design mode ---------------------------------------------------------------------------------
def design_retired(tables, base, changed):
    out = []
    frozen = set(tables["frozen_files"])
    for m in tables["retired_modules"]:
        rel = m.replace(".", "/") + ".py"
        if not (base / rel).is_file():
            out.append(dict(kind="retired_module_missing_at_baseline", module=m))
        if rel in frozen:
            out.append(dict(kind="retired_module_is_frozen", module=m))
        if not changed.get(rel, "").startswith("delete"):
            out.append(dict(kind="retired_module_not_scheduled_for_deletion", module=m))
    return out


def design_moves(tables, base, changed):
    out = []
    seen = {}
    for kind, m in all_moves(tables) + [("test", m) for m in tables["test_moves"]]:
        node = find_unit(base, m["source"])
        if node is None:
            out.append(dict(kind="move_source_missing", source=m["source"]))
            continue
        if ast_sha(node) != m["source_ast_sha256"]:
            out.append(dict(kind="move_source_hash_differs", source=m["source"]))
        if kind == "mechanical" and substituted(node, m["substitutions"]) is None:
            out.append(dict(kind="substitution_not_applicable", source=m["source"]))
        target = m["target"]
        if kind == "test":
            if not target.startswith("tests/"):
                out.append(dict(kind="test_move_outside_tests", target=target))
        elif not changed.get(target, "").startswith(("add", "modify")):
            out.append(dict(kind="move_target_not_planned", target=target, unit=m["unit"]))
        key = (target, m["unit"])
        if key in seen and seen[key] != m["source_ast_sha256"]:
            out.append(dict(kind="conflicting_moves_for_one_unit", target=target, unit=m["unit"]))
        seen[key] = m["source_ast_sha256"]
    return out


def design_closure(tables, base):
    """Moved units must carry every module-level helper they use (or the target already has it)."""
    out = []
    by_pair = {}
    for _, m in all_moves(tables):
        path, name = m["source"].split(":")
        by_pair.setdefault((path, m["target"]), set()).add(name.split(".")[-1])
    for _, m in all_moves(tables):
        path, name = m["source"].split(":")
        src_units = units_of_source((base / path).read_text())
        node = find_unit(base, m["source"])
        if node is None:
            continue
        if "." in name:
            target_cls = m["unit"].split(".")[0]
            tnode = units_of_source((base / m["target"]).read_text()).get(target_cls)
            if not isinstance(tnode, ast.ClassDef):
                out.append(dict(kind="mechanical_target_class_missing", unit=m["unit"]))
                continue
            moved = {
                x["unit"].split(".")[1] for x in tables["mechanical_moves"] if "." in x["unit"]
            }
            dynamic = tables.get("mechanical_target_dynamic_members", {}).get(target_cls, ())
            available = class_members(tnode, dynamic) | moved
            sub = substituted(node, m["substitutions"])
            if sub is None:
                continue
            for attr in sorted(
                {
                    a.attr
                    for a in ast.walk(sub)
                    if isinstance(a, ast.Attribute)
                    and isinstance(a.value, ast.Name)
                    and a.value.id == "self"
                }
                - available
            ):
                out.append(dict(kind="method_reference_unresolved", unit=m["unit"], attribute=attr))
            continue
        refs = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in src_units} - {
            name
        }
        for ref in sorted(refs - by_pair.get((path, m["target"]), set())):
            out.append(dict(kind="moved_unit_reference_not_moved", unit=m["source"], reference=ref))
    return out


def design_paths(tables, base, changed):
    out = []
    frozen = set(tables["frozen_files"])
    for rel, reason in changed.items():
        exists = (base / rel).is_file()
        if reason.startswith("add") and exists:
            out.append(dict(kind="added_path_exists_at_baseline", path=rel))
        if reason.startswith(("modify", "delete")) and not exists:
            out.append(dict(kind="planned_path_missing_at_baseline", path=rel))
        if rel in frozen:
            out.append(dict(kind="frozen_file_planned_to_change", path=rel))
    for mod, row in tables["new_modules"].items():
        rel = mod.replace(".", "/") + ".py"
        if not changed.get(rel, "").startswith("add"):
            out.append(dict(kind="new_module_not_planned_as_add", module=mod))
        if not row["required_units"]:
            out.append(dict(kind="new_module_without_required_units", module=mod))
    for f in tables["frozen_files"]:
        p = base / f
        if not p.is_file() or sha_bytes(p.read_bytes()) != tables["frozen_sha256"].get(f):
            out.append(dict(kind="frozen_file_hash_differs", path=f))
    return out


def design_edges(tables, base):
    out = []
    mods = set(modmap(base)) | set(tables["new_modules"])
    forbid = {k: set(v) for k, v in tables["forbid"].items()}
    for row in tables["allowed_edges"]:
        s, d = row["source"], row["destination"]
        for end in (s, d):
            if end not in mods:
                out.append(dict(kind="edge_endpoint_unknown", module=end))
        if layer(d) in forbid.get(layer(s), set()):
            out.append(dict(kind="edge_violates_layer_rule", edge=f"{s} -> {d}"))
        if d.replace(".", "/") + ".py" in {
            m.replace(".", "/") + ".py" for m in tables["retired_modules"]
        }:
            out.append(dict(kind="edge_to_retired_module", edge=f"{s} -> {d}"))
    return out


def design_criteria(tables):
    out = []
    c = tables["criteria"]
    names = [e["name"] for e in c["section9_exclusions"]]
    if names != ["progress.constraints_current", "identity.runtime.adapter"]:
        out.append(dict(kind="section9_exclusions_not_exactly_two", names=names))
    for e in c["section9_exclusions"]:
        rx = re.compile(e["pointer_regex"])
        for pattern in e["dry_run_patterns"]:
            if not rx.search(pattern):
                out.append(dict(kind="dry_run_pattern_outside_exclusion", pattern=pattern))
    keys = c["channel_identity_runtime_keys_equal"]
    if not keys or "adapter" in keys:
        out.append(dict(kind="identity_keys_invalid", keys=keys))
    dry = c["dry_run"]
    if dry["channel_v04_self_differences"] or dry["functional_v04_self_differences"]:
        out.append(dict(kind="v04_not_self_consistent"))
    return out


def run_design(base, tables, tables_path, gate_self, simulate_install):
    overlay, install_issues = installation_view(
        base, tables, tables_path, gate_self, simulate_install
    )
    changed = tables["changed_paths"]
    report = {
        "D1_retired": design_retired(tables, base, changed),
        "D2_moves": design_moves(tables, base, changed),
        "D3_closure": design_closure(tables, base),
        "D4_paths": design_paths(tables, base, changed),
        "D5_edges_layers": design_edges(tables, base),
        "D6_criteria_install": design_criteria(tables) + install_issues,
    }
    files = tree_files(base, overlay if simulate_install else None)
    report["view"] = dict(
        installation_view="simulated" if simulate_install else "as found", files=len(files)
    )
    report["summary"] = {k: len(v) for k, v in report.items() if isinstance(v, list)}
    return report


# ---- draft mode ----------------------------------------------------------------------------------
def draft_units(tables, base, draft, base_mods, draft_mods, touched):
    out = []
    for kind, m in all_moves(tables):
        src = find_unit(base, m["source"])
        dst = find_unit(draft, target_ref(m))
        if dst is None:
            out.append(dict(kind="moved_unit_missing", unit=target_ref(m)))
            continue
        want = src if kind == "exact" else substituted(src, m["substitutions"])
        if want is None or ast.dump(want) != ast.dump(dst):
            out.append(dict(kind=f"{kind}_move_not_identical", unit=target_ref(m)))
    for m in tables["test_moves"]:
        if find_unit(draft, target_ref(m)) is None:
            out.append(dict(kind="test_move_missing", unit=target_ref(m)))
    registry_path = draft / tables["unit_registry"]
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    listed = {(m["target"], m["unit"].split(".")[0]) for _, m in all_moves(tables)}
    for mod, row in tables["new_modules"].items():
        rel = mod.replace(".", "/") + ".py"
        if mod not in draft_mods:
            out.append(dict(kind="new_module_missing", module=mod))
            continue
        actual = units_of_source(draft_mods[mod].read_text())
        for u in row["required_units"]:
            if u not in actual:
                out.append(dict(kind="required_unit_missing", module=mod, unit=u))
            listed.add((rel, u))
    for mod in touched:
        rel = mod.replace(".", "/") + ".py"
        actual = units_of_source(draft_mods[mod].read_text())
        before = units_of_source(base_mods[mod].read_text()) if mod in base_mods else {}
        for u, node in actual.items():
            if u.startswith("__") or (rel, u) in listed:
                continue
            changed = u not in before or ast.dump(before[u]) != ast.dump(node)
            if changed and u not in registry.get(mod, {}):
                out.append(dict(kind="unregistered_new_or_rewritten_unit", module=mod, unit=u))
    for mod, rows in registry.items():
        for u, row in rows.items():
            if mod not in draft_mods or u not in units_of_source(draft_mods[mod].read_text()):
                out.append(dict(kind="registry_row_without_unit", module=mod, unit=u))
            if not row.get("authority"):
                out.append(dict(kind="registry_row_without_authority", module=mod, unit=u))
    return out


def run_draft(base, draft, tables, tables_path, gate_self, simulate_install):
    overlay, install_issues = installation_view(
        draft, tables, tables_path, gate_self, simulate_install
    )
    report = {}
    base_layers, base_reviewed = ratchet(base)
    draft_layers, _ = ratchet(draft)
    allowed_layers = dict(base_layers)
    for k, v in tables["layers_extra"].items():
        allowed_layers[k] = frozenset(v)
    base_e, draft_e = edges(base), edges(draft)
    rows = {(e["source"], e["destination"]) for e in tables["allowed_edges"]}
    g1, owner_rows = [], []
    if draft_layers != allowed_layers:
        g1.append(dict(kind="layer_table_differs_from_baseline_plus_extra"))
    for (s, d), owners in sorted(draft_e.items()):
        if (s, d) in base_e:
            continue
        inside = layer(s) == layer(d) or layer(d) in allowed_layers.get(layer(s), frozenset())
        if not (inside or (s, d) in base_reviewed or (s, d) in rows):
            g1.append(dict(kind="unlisted_edge", edge=f"{s} -> {d}", owners=sorted(owners)))
        if (s, d) in rows:
            owner_rows.append(dict(edge=f"{s} -> {d}", owners=sorted(owners)))
    report["G1_import_edges"] = dict(violations=g1, table_rows_used=owner_rows)
    base_mods, draft_mods = modmap(base), modmap(draft)
    touched = [
        m
        for m, p in draft_mods.items()
        if m not in base_mods or sha_bytes(p.read_bytes()) != sha_bytes(base_mods[m].read_bytes())
    ]
    registry_path = draft / tables["unit_registry"]
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    known = {
        (m["target"].replace("/", ".")[:-3], m["unit"].split(".")[0]) for _, m in all_moves(tables)
    }
    for mod, row in tables["new_modules"].items():
        known |= {(mod, u) for u in row["required_units"]}
    for mod, rows_ in registry.items():
        known |= {(mod, u) for u in rows_}
    report["OWNERS_unauthorized"] = [
        dict(edge=r["edge"], owner=o)
        for r in owner_rows
        for o in r["owners"]
        if o != "<re-export>" and (r["edge"].split(" -> ")[0], o) not in known
    ]
    forbid = {k: set(v) for k, v in tables["forbid"].items()}
    g2 = [
        dict(kind="layer_rule", edge=f"{s} -> {d}")
        for (s, d) in draft_e
        if s in touched and (s, d) not in base_e and layer(d) in forbid.get(layer(s), set())
    ]
    deferred = {tuple(x) for x in tables["torch_free_deferred_imports"]}
    component = set(tables["component_layers"])
    closure, stack = set(), list(tables["torch_free_modules"])
    while stack:
        m = stack.pop()
        if m in closure or m not in draft_mods:
            continue
        closure.add(m)
        for node in ast.walk(ast.parse(draft_mods[m].read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            for n in names:
                if (m, n) in deferred:
                    continue
                if n.split(".")[0] in ("torch", "numpy") or (
                    n.startswith("pssolver.") and layer(n) in component
                ):
                    g2.append(dict(kind="torch_free_closure", module=m, imports=n))
                elif n.startswith("pssolver.") and n in draft_mods:
                    stack.append(n)
    report["G2_layers"] = dict(violations=g2)
    draft_files = tree_files(draft, overlay if simulate_install else None)
    base_files = tree_files(base, None)
    retired = tables["retired_modules"]
    needles = set(retired) | {r.replace(".", "/") + ".py" for r in retired}
    g3 = []
    for rel, data in sorted(draft_files.items()):
        if not rel.endswith((".py", ".json")):
            continue
        text = data.decode("utf-8", errors="ignore")
        if not any(n in text for n in needles):
            continue
        base_lines = set(base_files.get(rel, b"").decode("utf-8", "ignore").splitlines())
        for i, line in enumerate(text.splitlines(), 1):
            if not any(n in line for n in needles):
                continue
            own = any(fnmatch.fnmatch(rel, p) for p in tables["own_file_patterns"])
            if rel.startswith("pssolver/") or not (line in base_lines or own):
                g3.append(dict(file=rel, line=i))
    report["G3_retired_names"] = dict(violations=g3)
    changed = []
    for rel, data in draft_files.items():
        if rel not in base_files:
            changed.append((rel, "add"))
        elif base_files[rel] != data:
            changed.append((rel, "modify"))
    changed += [(rel, "delete") for rel in base_files if rel not in draft_files]
    plan = tables["changed_paths"]
    g4 = []
    for rel, change in sorted(changed):
        listed = plan.get(rel, "").split(":")[0]
        by_class = any(fnmatch.fnmatch(rel, c["pattern"]) for c in tables["changed_path_classes"])
        if listed != change and not by_class:
            g4.append(dict(kind="unplanned_change", path=rel, change=change))
        frozen = rel in tables["frozen_files"] or any(
            fnmatch.fnmatch(rel, p) for p in tables["frozen_patterns"]
        )
        if frozen and change != "add":
            g4.append(dict(kind="frozen_file_changed", path=rel, change=change))
    for rel, reason in plan.items():
        if reason.startswith("delete") and rel in draft_files:
            g4.append(dict(kind="planned_deletion_still_present", path=rel))
        if reason.startswith("add") and rel not in draft_files:
            g4.append(dict(kind="planned_addition_missing", path=rel))
    g4.extend(install_issues)
    report["G4_changes"] = dict(
        violations=g4,
        installation_view="simulated" if simulate_install else "as found",
        changed=len(changed),
    )
    report["G5_units"] = dict(
        violations=draft_units(tables, base, draft, base_mods, draft_mods, touched)
    )
    report["summary"] = {
        k: len(v["violations"]) if isinstance(v, dict) else len(v) for k, v in report.items()
    }
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    simulate = "--simulate-install" in argv
    args = [a for a in argv if not a.startswith("--")]
    if len(args) == 4 and args[0] == "design":
        base, tables_path, out = Path(args[1]), Path(args[2]), Path(args[3])
        tables = json.loads(tables_path.read_text())
        report = run_design(base, tables, tables_path, Path(__file__).resolve(), simulate)
    elif len(args) == 5 and args[0] == "draft":
        base, draft, tables_path, out = (Path(a) for a in args[1:])
        tables = json.loads(tables_path.read_text())
        report = run_draft(base, draft, tables, tables_path, Path(__file__).resolve(), simulate)
    else:
        print(USAGE, file=sys.stderr)
        return 2
    summary = report["summary"]
    out.write_text(json.dumps(report, indent=1, default=sorted) + "\n")
    print(json.dumps(summary))
    for k, v in report.items():
        rows = v.get("violations") if isinstance(v, dict) else v if isinstance(v, list) else None
        if rows:
            print("==", k)
            for row in rows[:40]:
                print("  ", json.dumps(row)[:300])
    return 1 if any(summary.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
