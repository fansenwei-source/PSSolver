"""S-27 r2 release gate (R-0097): check a draft tree against the single r2 tables.

Usage:
  python3 -I -B s27_guard_crosscheck.py <baseline_tree 327a192> <draft_tree>
      <s27_r2_tables.json> <out.json> [--simulate-install]

--simulate-install: check the tree as it will look after installing this gate and the tables at
r2 'installed_paths' (and removing r2 'installation_removes').  Without it, any installed copy
must be byte-identical to the running gate and tables (R-0098: an installed authority must
register itself).  In both views the running gate must have the sha256 that the tables bind in
r2 'gate_sha256' (R-0099).

Import-safe (r2.2, R-0099): importing this module only defines functions; the command line is
read in main() under the __main__ guard.  modmap() and edges() stay pure AST functions that need
only `ast`.

The r2 tables (design S-27 r2, appendix A) are the only authority.  Five guards plus owner
authorization:
  G1 import edges   every new pssolver->pssolver edge is (i) inside the baseline
                    ALLOWED_INTERNAL_LAYERS plus the r2 'layers_extra', or in a baseline reviewed
                    edge set, or (ii) an r2 'edges' row whose owner set equals the observed owner
                    set; rehome rows must have only exact-move owners.  The draft ratchet layer
                    table must equal baseline + layers_extra (no widening).
  G2 layers         S-22 layer rule (FORBID) on every new edge of a new or changed package
                    module; planning/core/presets.beris_edwards static import closure free of
                    torch/numpy/component layers except the listed deferred import.
  G3 retired names  every occurrence of a retired module name/path is classified by rule:
                    package .py -> violation; non-package .py byte-identical to baseline with the
                    line already present -> frozen_script; other non-package .py line already
                    present verbatim at baseline -> historical_line; r2 'guard_files' -> guard;
                    S-27 own files (r2 'own_file_patterns') -> s27_own_file; JSON unchanged since
                    baseline -> provenance; r2 'registry_files' -> registry; anything else ->
                    violation.
  G4 changes        every changed, added or deleted path is listed in r2 'changed_paths'
                    (package, root, tools, notes, packaging) or matches an r2
                    'changed_path_classes' rule (tests/**, qualification/**: governed by the S-27
                    test disposition and registries); no frozen file (r2 'frozen_files',
                    'frozen_patterns') is modified or deleted (qualification/s24_frozen_sources.json
                    is a forward-bound record, not a freeze); plus the installation-view issues.
  G5 units          every top-level unit of the r2 destination modules is listed; 'exact' units
                    are AST-identical to their baseline source unit; units added to other changed
                    package modules are listed.
  OWNERS            owners of every non-layer edge are listed r2 units (or exact moves).
Exit status 1 if any guard count is non-zero, 2 on a usage error.
"""

import ast
import fnmatch
import hashlib
import json
import sys
from pathlib import Path


USAGE = (
    "usage: s27_guard_crosscheck.py <baseline_tree> <draft_tree> <s27_r2_tables.json> <out.json>"
    " [--simulate-install]"
)


# ---- installation view (R-0098, R-0099) -------------------------------------------------------
# The gate and its tables are themselves installed into the checked tree (r2 'installed_paths').
# With --simulate-install the check runs on the tree as it will look after installation: the two
# files are overlaid at their installed paths with the bytes being used now, and r2
# 'installation_removes' are treated as absent.  Without the flag, an installed copy must be
# byte-identical to the files in use, so the gate never evaluates a tree against tables other than
# the ones it carries.  In both views the running gate must have the sha256 the tables bind
# (r2 'gate_sha256'), so gate and tables travel as one pair of original bytes.
def installation_view(draft, tables, tables_path, gate_self, simulate_install):
    """Return (overlay, removed, install_issues) for the checked draft tree."""
    installed = tables["installed_paths"]
    overlay = {
        installed["gate"]: gate_self.read_bytes(),
        installed["tables"]: tables_path.read_bytes(),
    }
    removed = set(tables["installation_removes"]) if simulate_install else set()
    issues = []
    bound = tables.get("gate_sha256")
    if bound is not None and hashlib.sha256(overlay[installed["gate"]]).hexdigest() != bound:
        issues.append(dict(kind="running_gate_not_bound_by_tables", path=installed["gate"]))
    if not simulate_install:
        for rel, data in overlay.items():
            p = draft / rel
            if p.exists() and p.read_bytes() != data:
                issues.append(dict(kind="installed_copy_differs_from_running_copy", path=rel))
        for rel in tables["installation_removes"]:
            if (draft / rel).exists():
                issues.append(dict(kind="installation_remove_still_present", path=rel))
    return overlay, removed, issues


def tree_files(root, removed, overlay):
    """Relative path -> bytes of the checked tree; the overlay gives the installation view."""
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if not p.is_file() or rel.startswith("build/") or ".egg-info" in rel or rel in removed:
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


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def units_of(path):
    tree = ast.parse(path.read_text())
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[n.name] = n
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                if isinstance(t, ast.Name):
                    out[t.id] = n
    return out


def check_import_edges(tables, base, draft, base_e, draft_e, exact_set, listed):
    """G1 and OWNERS."""
    base_layers, base_reviewed = ratchet(base)
    draft_layers, _ = ratchet(draft)
    allowed_layers = dict(base_layers)
    for k, v in tables["layers_extra"].items():
        allowed_layers[k] = frozenset(v)
    g1 = []
    widening = sorted(
        k
        for k in set(draft_layers) | set(allowed_layers)
        if draft_layers.get(k) != allowed_layers.get(k)
    )
    if widening:
        g1.append(dict(kind="layer_table_differs_from_baseline_plus_extra", layers=widening))
    table_edges = {(e["source"], e["destination"]): e for e in tables["edges"]}
    owner_rows, unauthorized = [], []
    seen_table = set()
    for (s, d), owners in sorted(draft_e.items()):
        if (s, d) in base_e:
            continue
        inside = layer(s) == layer(d) or layer(d) in allowed_layers.get(layer(s), frozenset())
        if inside or (s, d) in base_reviewed:
            owner_rows.append(
                dict(
                    edge=f"{s} -> {d}",
                    rule="layer_table" if inside else "baseline_reviewed",
                    owners=sorted(owners),
                )
            )
            continue
        row = table_edges.get((s, d))
        if row is None:
            g1.append(dict(kind="unlisted_edge", edge=f"{s} -> {d}", owners=sorted(owners)))
            continue
        seen_table.add((s, d))
        if set(row["owners"]) != owners:
            g1.append(
                dict(
                    kind="owner_mismatch",
                    edge=f"{s} -> {d}",
                    listed=sorted(row["owners"]),
                    observed=sorted(owners),
                )
            )
        if row["kind"] == "rehome" and not all((s, o) in exact_set for o in owners):
            g1.append(
                dict(kind="rehome_with_non_exact_owner", edge=f"{s} -> {d}", owners=sorted(owners))
            )
        for o in owners:
            if (s, o) not in listed and o not in ("<re-export>",):
                unauthorized.append(dict(edge=f"{s} -> {d}", owner=o))
        owner_rows.append(
            dict(
                edge=f"{s} -> {d}",
                rule=row["kind"],
                owners=sorted(owners),
                authority=row["authority"],
            )
        )
    for key in table_edges:
        if key not in seen_table:
            g1.append(dict(kind="table_edge_not_present", edge=f"{key[0]} -> {key[1]}"))
    return dict(violations=g1, new_edges=owner_rows), unauthorized


def check_layers(tables, base_e, draft_e, draft_mods, touched):
    """G2."""
    forbid = {k: set(v) for k, v in tables["forbid"].items()}
    g2 = []
    for m in touched:
        lay = layer(m)
        if lay not in forbid:
            continue
        for (s, d), owners in draft_e.items():
            if s == m and (s, d) not in base_e and layer(d) in forbid[lay]:
                g2.append(dict(kind="layer_rule", edge=f"{s} -> {d}", owners=sorted(owners)))
    deferred_ok = {tuple(x) for x in tables["torch_free_deferred_imports"]}
    component = set(tables["component_layers"])
    closure, stack = set(), list(tables["torch_free_modules"])
    while stack:
        m = stack.pop()
        if m in closure or m not in draft_mods:
            continue
        closure.add(m)
        tree = ast.parse(draft_mods[m].read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            for n in names:
                if (m, n) in deferred_ok:
                    continue
                if n.split(".")[0] in ("torch", "numpy") or (
                    n.startswith("pssolver.") and layer(n) in component
                ):
                    g2.append(
                        dict(kind="torch_free_closure", module=m, imports=n, line=node.lineno)
                    )
                elif n.startswith("pssolver.") and n in draft_mods:
                    stack.append(n)
    return dict(violations=g2, torch_free_closure=sorted(closure))


def check_retired_names(tables, base, draft_files):
    """G3."""
    retired = tables["retired_modules"]
    needles = set(retired) | {r.replace(".", "/") + ".py" for r in retired}
    guard_files, registry_files = set(tables["guard_files"]), set(tables["registry_files"])
    g3, counts = [], {}
    for rel, data in sorted(draft_files.items()):
        if not rel.endswith((".py", ".json")):
            continue
        text = data.decode("utf-8", errors="ignore")
        if not any(n in text for n in needles):
            continue
        base_p = base / rel
        unchanged = base_p.exists() and base_p.read_bytes() == data
        base_lines = (
            set(base_p.read_text(errors="ignore").splitlines()) if base_p.exists() else set()
        )
        for i, line in enumerate(text.splitlines(), 1):
            if not any(n in line for n in needles):
                continue
            if rel.endswith(".py"):
                if rel.startswith("pssolver/"):
                    cat = "VIOLATION_package"
                elif unchanged and line in base_lines:
                    cat = "frozen_script"
                elif line in base_lines:
                    cat = "historical_line"
                elif rel in guard_files:
                    cat = "guard"
                elif any(fnmatch.fnmatch(rel, pat) for pat in tables["own_file_patterns"]):
                    cat = "s27_own_file"
                else:
                    cat = "VIOLATION_unlisted_py"
            else:
                if unchanged:
                    cat = "provenance"
                elif rel in registry_files:
                    cat = "registry"
                else:
                    cat = "VIOLATION_changed_json"
            counts[cat] = counts.get(cat, 0) + 1
            if cat.startswith("VIOLATION"):
                g3.append(dict(file=rel, line=i, category=cat))
    return dict(violations=g3, counts=counts)


def check_changes(tables, draft_files, base_files, install_issues, simulate_install):
    """G4."""
    changed = []
    for rel, data in draft_files.items():
        if rel not in base_files:
            changed.append((rel, "added"))
        elif base_files[rel] != data:
            changed.append((rel, "modified"))
    for rel in base_files:
        if rel not in draft_files:
            changed.append((rel, "deleted"))
    changed.sort()
    paths_table = tables["changed_paths"]
    class_rules = tables["changed_path_classes"]

    def by_class(rel):
        return next((c["reason"] for c in class_rules if fnmatch.fnmatch(rel, c["pattern"])), None)

    g4 = [
        dict(path=r, change=c) for r, c in changed if r not in paths_table and by_class(r) is None
    ]
    class_counts = {}
    for r, _ in changed:
        if r not in paths_table and by_class(r):
            class_counts[by_class(r)] = class_counts.get(by_class(r), 0) + 1
    stale = sorted(set(paths_table) - {r for r, _ in changed})
    # frozen files: never modified or deleted (additions under frozen directories are not
    # modifications)
    frozen_exact = set(tables["frozen_files"])
    for r, c in changed:
        if c != "added" and (
            r in frozen_exact or any(fnmatch.fnmatch(r, pat) for pat in tables["frozen_patterns"])
        ):
            g4.append(dict(path=r, change=c, kind="frozen_file_changed"))
    g4.extend(install_issues)
    return dict(
        violations=g4,
        installation_view="simulated" if simulate_install else "as found",
        changed=len(changed),
        listed=len(paths_table),
        class_permitted=class_counts,
        listed_but_unchanged=stale,
    )


def check_units(units_table, added_table, base, base_mods, draft_mods, touched, listed):
    """G5."""
    g5 = []
    for m, table in list(units_table.items()) + [(m, t) for m, t in added_table.items()]:
        full = m in units_table
        if m not in draft_mods:
            g5.append(dict(kind="module_missing", module=m))
            continue
        actual = units_of(draft_mods[m])
        for u in actual if full else []:
            if u not in table and not u.startswith("__"):
                g5.append(dict(kind="unlisted_unit", module=m, unit=u))
        for u, row in table.items():
            if u not in actual:
                g5.append(dict(kind="listed_unit_missing", module=m, unit=u))
                continue
            if row["kind"] == "exact":
                src_path, src_unit = row["source"].split(":")
                src = units_of(base / src_path).get(src_unit)
                if src is None or ast.dump(src) != ast.dump(actual[u]):
                    g5.append(
                        dict(kind="exact_not_identical", module=m, unit=u, source=row["source"])
                    )
    for m in touched:
        if m in units_table or m not in base_mods:
            continue
        added = set(units_of(draft_mods[m])) - set(units_of(base_mods[m]))
        for u in sorted(added):
            if (m, u) not in listed and not u.startswith("__"):
                g5.append(dict(kind="unlisted_added_unit", module=m, unit=u))
    return dict(violations=g5)


def crosscheck(base, draft, tables_path, *, simulate_install=False, gate_self=None):
    """Run the six guards and return the report (with 'summary'); performs no writes."""
    tables = json.loads(tables_path.read_text())
    gate_self = Path(__file__).resolve() if gate_self is None else gate_self
    overlay, removed, install_issues = installation_view(
        draft, tables, tables_path, gate_self, simulate_install
    )
    units_table = tables["units"]  # full listing: {module: {unit: {kind, source?, authority}}}
    added_table = tables["added_units"]  # added units of otherwise modified modules
    exact_set = {
        (m, u)
        for tab in (units_table, added_table)
        for m, us in tab.items()
        for u, row in us.items()
        if row["kind"] == "exact"
    }
    listed = {(m, u) for tab in (units_table, added_table) for m, us in tab.items() for u in us}
    base_e, draft_e = edges(base), edges(draft)
    base_mods, draft_mods = modmap(base), modmap(draft)
    touched = [
        m for m, p in draft_mods.items() if m not in base_mods or sha(p) != sha(base_mods[m])
    ]
    draft_files = tree_files(draft, removed, overlay if simulate_install else None)
    base_files = tree_files(base, removed, None)

    report = {}
    report["G1_import_edges"], report["OWNERS_unauthorized"] = check_import_edges(
        tables, base, draft, base_e, draft_e, exact_set, listed
    )
    report["G2_layers"] = check_layers(tables, base_e, draft_e, draft_mods, touched)
    report["G3_retired_names"] = check_retired_names(tables, base, draft_files)
    report["G4_changes"] = check_changes(
        tables, draft_files, base_files, install_issues, simulate_install
    )
    report["G5_units"] = check_units(
        units_table, added_table, base, base_mods, draft_mods, touched, listed
    )
    report["summary"] = {
        k: len(v["violations"]) if isinstance(v, dict) else len(v) for k, v in report.items()
    }
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 4:
        print(USAGE, file=sys.stderr)
        return 2
    base, draft, tables_path, out = (Path(a) for a in args)
    report = crosscheck(base, draft, tables_path, simulate_install="--simulate-install" in argv)
    summary = report["summary"]
    out.write_text(json.dumps(report, indent=1, default=sorted) + "\n")
    print(json.dumps(summary))
    for k, v in report.items():
        if isinstance(v, dict) and v.get("violations"):
            print("==", k)
            for row in v["violations"][:40]:
                print("  ", json.dumps(row)[:300])
        elif isinstance(v, list) and v:
            print("==", k)
            for row in v[:40]:
                print("  ", json.dumps(row)[:300])
    return 1 if any(summary.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
