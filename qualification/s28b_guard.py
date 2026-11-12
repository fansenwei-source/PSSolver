"""S-28b release gate (T-0023): check a tree against the single S-28b authority tables.

Usage:
  python3 -I -B s28b_guard.py design <baseline_tree 4b72093> <s28a_tables.json> <out.json>
      [--simulate-install] [--current=<tree>]
  python3 -I -B s28b_guard.py draft <baseline_tree 4b72093> <draft_tree> <s28a_tables.json> <out.json>
      [--simulate-install]

S-28b r1.4 (R-0118) adds to G7 the validation-digest contract (Output.validation_config_sha256 enters the declaration
hash, not the Plan; an uppercase digest is rejected with the v0.4 "64 lowercase hexadecimal" text).
S-28b r1.2 (R-0116) adds to G2 the check that the draft introduces no new direct import cycle (import-time edges,
parent/child package edges excluded) and to G7 the compiled-artifact contract (source, validate_integrity, output
directory outside the Plan identity).
S-28b r1.1 (R-0115) adds D12/G7: the public compile paths (Periodic, Channel, both Plane writings, the declared
rejections) actually run in a child process that blocks torch and numpy imports; D12 replays the planned chains on the
basis (planned exact copies applied, lazy root stub) against the 49d1c32 record, G7 compiles through the draft's
public API.  Exact copies (rows with "copy": true) leave their source in place.
S-28b (r1.0) extends the S-28a r1.2 gate: layer rows may be removed (layers_removed); D10/G6 check the public API
table (old root names each have one disposition; the draft root exports exactly the new names through a static
_EXPORTS map); D11/G2b check that the planned torch-free import modules reach neither torch nor numpy at module
level (the process-level probe runs in the test suite); D6 checks the S-28b criteria table.
design: before implementation, the tables are checked against the baseline facts (D1-D6): retired modules
exist and are scheduled for deletion; every exact or mechanical move names an existing source unit with the
recorded AST hash; moved sets are closed under module-level references; planned paths are consistent; every
allowed edge obeys the S-22 layer rule; the criteria keep exactly the two section-9 exclusions; and (D7,
R-0106) every direct import or literal reference to a retired module anywhere in the tree (package, tests,
benchmarks, qualification, notes, scripts, tools, examples, root files) is covered by an authorized change; (D8,
R-0107) the import closure of every exact or mechanical move, resolved to the defining modules, is covered in the
target module by the layer table, a baseline reviewed edge or a registered edge with matching owners, and no moved
unit carries a retired module name; (D9) every module-level edge the design plans for rewritten and new code is
authorized and obeys the layer rule.
draft: after implementation, the draft tree is checked against the baseline (G1-G5 and OWNERS, as the S-27
r2.3 gate): new import edges, layers and the torch-free closure, retired names, changed paths and frozen
files, and units (exact moves AST-identical, mechanical moves byte-identical to the basis source text after
their declared source substitutions, required units present, every new or rewritten unit registered).

S-28b r2.2 (R-0122) removes every ast.unparse text route for mechanical moves: the authority is the basis source
text with the table's source substitutions (target_text_sha256) and the AST of that text (target_ast_sha256), so
D2, D3, D8 and G5 give the same result on every Python 3.12 patch release (ast.unparse changed its f-string quotes
between 3.12.3 and 3.12.16).

Installation view (R-0098, R-0099): --simulate-install overlays this gate and the tables at their installed
paths; without it an installed copy must be byte-identical to the running copy, and in both views the running
gate must have the sha256 bound by the tables ('gate_sha256').  Version-control metadata (.git, .hg, .svn) is
not part of a checked tree (R-0100).  Importing this module only defines functions.
Exit status 1 if any count is non-zero, 2 on a usage error.
"""

import ast
import fnmatch
import hashlib
import io
import json
import re
import sys
import tokenize
from pathlib import Path


VCS_METADATA = frozenset({".git", ".hg", ".svn"})
USAGE = (
    "usage: s28b_guard.py design <baseline> <tables> <out.json> [--simulate-install]\n"
    "       s28b_guard.py draft <baseline> <draft> <tables> <out.json> [--simulate-install]"
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


def allowed_layer_table(tables, base_layers):
    """Baseline ratchet minus removed rows plus the table's replacement rows."""
    allowed = {
        k: v for k, v in base_layers.items() if k not in set(tables.get("layers_removed", []))
    }
    for k, v in tables["layers_extra"].items():
        allowed[k] = frozenset(v)
    return allowed


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
def module_rel(root, mod):
    """Module name -> relative path (module file or package __init__)."""
    rel = mod.replace(".", "/") + ".py"
    pkg = mod.replace(".", "/") + "/__init__.py"
    return pkg if not (root / rel).is_file() and (root / pkg).is_file() else rel


def unit_locations(tables):
    """(source module, unit) -> target module, for every exact and mechanical move."""
    out = {}
    for _, m in all_moves(tables):
        if m.get("copy"):
            continue
        path, name = m["source"].split(":")
        src = path[:-3].replace("/", ".").removesuffix(".__init__")
        out[(src, name.split(".")[0])] = (
            m["target"][:-3].replace("/", ".").removesuffix(".__init__")
        )
    return out


def design_retired(tables, base, changed):
    out = []
    frozen = set(tables["frozen_files"])
    for m in tables["retired_modules"]:
        rel = module_rel(base, m)
        if not (base / rel).is_file():
            out.append(dict(kind="retired_module_missing_at_baseline", module=m))
        if rel in frozen:
            out.append(dict(kind="retired_module_is_frozen", module=m))
        if not changed.get(rel, "").startswith("delete"):
            out.append(dict(kind="retired_module_not_scheduled_for_deletion", module=m))
    return out


def unit_source(text, name):
    """Source text of a top-level unit, decorators included (the span tools/check.py uses)."""
    for node in ast.parse(text).body:
        names = [getattr(node, "name", None)]
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [t.id for t in targets if isinstance(t, ast.Name)]
        if name in names:
            start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
            return "".join(text.splitlines(keepends=True)[start - 1 : node.end_lineno])
    return None


def pep701_quote_reuse(text):
    """True when an f-string reuses its own quote inside a replacement field (3.12-only syntax)."""
    fstring_start = getattr(tokenize, "FSTRING_START", None)
    if fstring_start is None:
        return False
    stack = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == fstring_start:
            quote = token.string.lstrip("rRbBfFuU")
            if stack and quote[0] == stack[-1][0]:
                return True
            stack.append(quote)
        elif token.type == tokenize.FSTRING_END:
            stack.pop()
        elif token.type == tokenize.STRING and stack:
            if token.string.lstrip("rRbBuU")[0] == stack[-1][0]:
                return True
    return False


def mechanical_text(base, move):
    """The basis unit text with the table's source substitutions applied (R-0121), or None."""
    path, name = move["source"].split(":")
    text = unit_source((base / path).read_text(), name)
    if text is None:
        return None
    for old, new in move.get("source_substitutions", []):
        if text.count(old) != 1:
            return None
        text = text.replace(old, new)
    return text


def mechanical_node(base, move):
    """r2.2: the AST of mechanical_text (no ast.unparse route, which differs between Python patch releases)."""
    text = mechanical_text(base, move)
    if text is None:
        return None
    try:
        return ast.parse(text).body[0]
    except SyntaxError:
        return None


def historical_exact_units(base):
    """(target, unit) of every accepted exact move before S-28b (S-25..S-27 manifest, S-28a table)."""
    rows = []
    manifest = base / "qualification/s27_exact_moves.json"
    channel = base / "qualification/s28a_tables.json"
    if manifest.is_file():
        rows += [(e["target"], e["unit"]) for e in json.loads(manifest.read_text())["exact_moves"]]
    if channel.is_file():
        rows += [
            (m["target"], m["source"].split(":", 1)[1])
            for m in json.loads(channel.read_text())["exact_moves"]
        ]
    return rows


def historical_routes(tables, changed):
    moved = {
        tuple(m["source"].split(":", 1)): kind for kind, m in all_moves(tables) if not m.get("copy")
    }
    replaced = {(r["path"], r["unit"]) for r in tables.get("replaced_exact_units", [])}
    deleted = {p for p, why in changed.items() if why.startswith("delete")}
    return moved, replaced, deleted


def design_moves(tables, base, changed):
    out = []
    seen = {}
    moved, replaced, deleted = historical_routes(tables, changed)
    history = historical_exact_units(base)
    for target, unit in history:  # R-0121: at most one S-28b route per historical exact unit
        routes = [
            name
            for name, hit in (
                ("moved", (target, unit) in moved),
                ("replaced", (target, unit) in replaced),
                ("retired_with_file", target in deleted and (target, unit) not in moved),
            )
            if hit
        ]
        if len(routes) > 1:
            out.append(dict(kind="historical_unit_routes_conflict", target=target, unit=unit))
    for path, unit in sorted(replaced):
        if not changed.get(path, "").startswith("modify") or (path, unit) not in set(history):
            out.append(
                dict(kind="replaced_unit_not_historical_in_modified_file", path=path, unit=unit)
            )
    for kind, m in all_moves(tables) + [("test", m) for m in tables["test_moves"]]:
        node = find_unit(base, m["source"])
        if node is None:
            out.append(dict(kind="move_source_missing", source=m["source"]))
            continue
        if ast_sha(node) != m["source_ast_sha256"]:
            out.append(dict(kind="move_source_hash_differs", source=m["source"]))
        # R-0121/r2.2: byte-exact source form, recorded AST, valid for Python 3.10
        if kind == "mechanical":
            text = mechanical_text(base, m)
            moved = mechanical_node(base, m)
            if (
                text is None
                or moved is None
                or sha_bytes(text.encode()) != m.get("target_text_sha256")
                or ast_sha(moved) != m.get("target_ast_sha256")
                or pep701_quote_reuse(text)
            ):
                out.append(dict(kind="mechanical_source_text_invalid", source=m["source"]))
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
    for kind, m in all_moves(tables):
        path, name = m["source"].split(":")
        src_units = units_of_source((base / path).read_text())
        node = find_unit(base, m["source"])
        if node is None:
            continue
        if kind == "mechanical" and "." not in name:
            node = mechanical_node(base, m)
            if node is None:
                out.append(dict(kind="mechanical_source_text_invalid", source=m["source"]))
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
            sub = mechanical_node(base, m)
            if sub is None:
                out.append(dict(kind="mechanical_source_text_invalid", source=m["source"]))
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
        if not tables["changed_paths"].get(path, "").startswith("delete"):
            moved_any = {
                x["source"].split(":")[1].split(".")[0]
                for _, x in all_moves(tables)
                if x["source"].startswith(path + ":")
            }
            refs -= set(src_units) - moved_any
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
        if rel not in changed:
            rel = mod.replace(".", "/") + "/__init__.py"
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
    allowed = allowed_layer_table(tables, ratchet(base)[0])
    for src_layer, dst_layers in sorted(
        allowed.items()
    ):  # R-0119: a layer pair is allowed or forbidden, not both
        both = sorted(set(dst_layers) & forbid.get(src_layer, set()))
        if both:
            out.append(
                dict(kind="layer_pair_allowed_and_forbidden", layer=src_layer, destinations=both)
            )
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


CRITERION_KEYS = ("what", "format", "writer", "comparison_object", "dry_run")
REFERENCE_FAMILIES = {
    "periodic": 13,
    "plane": 65,
    "channel": 20,
    "functional_periodic": 2,
    "functional_channel": 4,
}
API_DISPOSITIONS = ("kept", "replaced", "moved", "removed")


def design_criteria(tables):
    """D6: two section-9 exclusions only; every criterion names format, writer and comparison object."""
    out = []
    c = tables["criteria"]
    names = [e["name"] for e in c["section9_exclusions"]]
    if names != ["progress.constraints_current", "identity.runtime.adapter"]:
        out.append(dict(kind="section9_exclusions_not_exactly_two", names=names))
    for key, row in sorted(c["criteria"].items()):
        missing = [k for k in CRITERION_KEYS if not row.get(k)]
        if missing:
            out.append(dict(kind="criterion_incomplete", criterion=key, missing=missing))
    ref = tables["plan_identity_reference"]
    counts = {}
    for sid, row in ref.items():
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("plan_identity_sha256", "")):
            out.append(dict(kind="reference_identity_invalid", scene=sid))
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    if counts != REFERENCE_FAMILIES:
        out.append(dict(kind="reference_family_counts_differ", counts=counts))
    for row in c["rejections"]:
        if row["scene"] in ref:
            out.append(dict(kind="rejected_scene_has_reference", scene=row["scene"]))
    trace = tables.get(
        "metadata_trace"
    )  # R-0118: every v0.4 public field has a source or an explicit retirement
    if not trace or trace.get("unmatched") != 0 or not trace.get("fields"):
        out.append(dict(kind="metadata_trace_incomplete"))
    else:
        used = {row[2] for row in trace["rules"]} | {v[0] for v in trace["cli"].values()}
        used |= {v[0] for v in trace["api_inputs"].values()}
        for category in sorted(used - set(trace["value_rules"])):
            out.append(dict(kind="metadata_category_without_value_rule", category=category))
        rules = [(k, re.compile(rx), c) for k, rx, c, _ in trace["rules"]]
        whole = trace["whole_object_prefixes"]  # category -> prefix of the object it defines
        prefixes = tuple(whole.values())
        for kind, path, category in trace[
            "paths"
        ]:  # R-0119: one rule per path, no presence inside a whole object
            matches = [c for k, rx, c in rules if k == kind and rx.search(path)]
            if matches != [category]:
                out.append(dict(kind="trace_rule_not_exclusive", path=path, matches=matches))
            presence = trace["presence"][category]
            if path.startswith(prefixes) and presence == "present":
                out.append(dict(kind="presence_required_inside_whole_object", path=path))
            if presence == "whole_object" and not path.startswith(whole.get(category, "\0")):
                out.append(dict(kind="whole_object_rule_outside_its_prefix", path=path))
        for category, presence in trace["presence"].items():
            if presence == "whole_object" and category not in whole:
                out.append(dict(kind="whole_object_category_without_prefix", category=category))
        if sorted(trace.get("schema_version_v05", {})) != ["channel", "periodic", "plane"]:
            out.append(dict(kind="schema_version_v05_incomplete"))
        if not trace.get("key_closure"):
            out.append(dict(kind="metadata_key_closure_missing"))
    return out


EQUAL_VALUED = frozenset({"DECL", "DERIVED", "CONST", "INIT", "RUN"})


def design_accepted_classes(tables, base):
    """D6 (r2, R-0120): a metadata key that S-27 accepted as 'implementation' may not be traced as equal to v0.4."""
    out = []
    path = base / "qualification/s27_metadata_classes.json"
    trace = tables.get("metadata_trace") or {}
    if not path.is_file() or not trace.get("paths"):
        return [dict(kind="accepted_metadata_classes_unavailable")]
    for rule in json.loads(path.read_text())["rules"]:
        if rule["class"] != "implementation":
            continue
        pattern = re.escape(rule["pattern"]).replace(r"/\*\*", "(/.*)?").replace(r"\*", "[^/]*")
        rx = re.compile(f"^{pattern}$")
        for kind, traced, category in trace["paths"]:
            if kind == "meta" and rx.match(traced) and category in EQUAL_VALUED:
                out.append(
                    dict(
                        kind="accepted_implementation_key_compared_equal",
                        path=traced,
                        category=category,
                    )
                )
    return out


def root_all(root):
    for n in ast.parse((root / "pssolver/__init__.py").read_text()).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "__all__":
            return list(ast.literal_eval(n.value))
    return None


def design_public_api(tables, base, changed):
    """D10: each basis root name has exactly one disposition; new names come from planned or existing modules."""
    out = []
    api = tables["public_api"]
    basis = root_all(base)
    if sorted(api["old_root"]) != sorted(basis or []):
        out.append(
            dict(
                kind="old_root_table_differs_from_basis",
                missing=sorted(set(basis or []) - set(api["old_root"])),
                extra=sorted(set(api["old_root"]) - set(basis or [])),
            )
        )
    new = api["new_root"]
    for name, row in sorted(api["old_root"].items()):
        if row.get("disposition") not in API_DISPOSITIONS or not row.get("now"):
            out.append(dict(kind="disposition_invalid", name=name))
            continue
        if row["disposition"] in ("kept", "replaced") and name not in new:
            out.append(dict(kind="kept_name_not_exported", name=name))
        if row["disposition"] in ("moved", "removed") and name in new:
            out.append(dict(kind="removed_name_still_exported", name=name))
    mods = set(modmap(base))
    for name, module in sorted(new.items()):
        rel = module.replace(".", "/")
        planned = changed.get(rel + ".py", "") or changed.get(rel + "/__init__.py", "")
        if module not in mods and not planned.startswith("add"):
            out.append(dict(kind="export_module_unknown", name=name, module=module))
        if planned.startswith("delete"):
            out.append(dict(kind="export_module_retired", name=name, module=module))
    return out


def module_level_imports(path, mod):
    """Absolute names imported by statements that execute at import time (not inside def/class bodies)."""
    tree = ast.parse(path.read_text())
    is_pkg = path.name == "__init__.py"
    names = []

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
                continue
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = mod.split(".") if is_pkg else mod.split(".")[:-1]
                    parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                    base = ".".join(parts + ([node.module] if node.module else []))
                names.append(base)
                names.extend(f"{base}.{a.name}" for a in node.names)
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(node, field, None)
                if isinstance(sub, list):
                    visit(sub)
            for handler in getattr(node, "handlers", []) or []:
                visit(handler.body)

    visit(tree.body)
    return names


def direct_import_cycles(root):
    """Strongly connected sets of import-time edges between modules (parent/child package edges excluded)."""
    mods = modmap(root)
    graph = {}
    for mod, path in mods.items():
        edges = set()
        for name in module_level_imports(path, mod):
            if not name.startswith("pssolver"):
                continue
            target = name
            while target not in mods and "." in target:
                target = target.rsplit(".", 1)[0]
            if (
                target in mods
                and target != mod
                and not mod.startswith(target + ".")
                and not target.startswith(mod + ".")
            ):
                edges.add(target)
        graph[mod] = edges
    index, low, on, stack, found = {}, {}, set(), [], []
    counter = [0]

    def visit(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in graph[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = set()
            while True:
                w = stack.pop()
                on.discard(w)
                comp.add(w)
                if w == v:
                    break
            if len(comp) > 1:
                found.append(frozenset(comp))

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10000))
    try:
        for v in graph:
            if v not in index:
                visit(v)
    finally:
        sys.setrecursionlimit(limit)
    return found


def torch_free_violations(root, modules, stop=()):
    """Module-level import closure (parent packages included) must not reach torch or numpy."""
    mods = modmap(root)
    out, seen, stack = [], set(), [(m, m) for m in modules]
    while stack:
        mod, origin = stack.pop()
        if mod in seen or mod not in mods or mod in stop:
            continue
        seen.add(mod)
        parts = mod.split(".")
        stack.extend((".".join(parts[:i]), origin) for i in range(1, len(parts)))
        for name in module_level_imports(mods[mod], mod):
            top = name.split(".")[0]
            if top in ("torch", "numpy"):
                out.append(
                    dict(kind="torch_free_import_closure", origin=origin, module=mod, imports=name)
                )
            elif top == "pssolver":
                target = name
                while target not in mods and "." in target:
                    target = target.rsplit(".", 1)[0]
                stack.append((target, origin))
    return out


def design_torch_free(tables, base, changed):
    """D11: existing torch-free import modules that are not planned to change are already torch-free."""
    planned = {
        rel[:-3].replace("/", ".").removesuffix(".__init__")
        for rel, why in changed.items()
        if rel.startswith("pssolver/") and rel.endswith(".py") and not why.startswith("add")
    }
    mods = set(modmap(base))
    existing = [m for m in tables["torch_free_import_modules"] if m in mods and m not in planned]
    return torch_free_violations(base, existing, stop=planned)


def mention_pattern(needles):
    """Whole-token match: 'pssolver.integrator' does not match 'pssolver.integrators'."""
    alternatives = "|".join(re.escape(n) for n in sorted(needles, key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9_])(?:{alternatives})(?![A-Za-z0-9_])")


CLOSURE_SUFFIXES = (".py", ".json", ".toml", ".in", ".cfg", ".txt")


def references_to_retired(root, retired):
    """Relative path -> sorted 'import:line'/'literal:line' hits on the retired set (relative imports resolved)."""
    needles = set(retired) | {r.replace(".", "/") + ".py" for r in retired}
    pattern = mention_pattern(needles)
    rows = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if (
            not p.is_file()
            or not rel.endswith(CLOSURE_SUFFIXES)
            or rel.startswith("build/")
            or ".egg-info" in rel
            or not VCS_METADATA.isdisjoint(rel.split("/"))
        ):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        hits = {
            f"literal:{i}" for i, line in enumerate(text.splitlines(), 1) if pattern.search(line)
        }
        if rel.endswith(".py"):
            me = rel[:-3].replace("/", ".")
            me = me[:-9] if me.endswith(".__init__") else me
            try:
                nodes = list(ast.walk(ast.parse(text)))
            except SyntaxError:
                nodes = []
            for node in nodes:
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = node.module or ""
                    if node.level:
                        parts = me.split(".") if rel.endswith("__init__.py") else me.split(".")[:-1]
                        parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                        base = ".".join(parts + ([node.module] if node.module else []))
                    names = [base] + [f"{base}.{a.name}" for a in node.names]
                if any(n in retired for n in names):
                    hits.add(f"import:{node.lineno}")
        if hits:
            rows[rel] = sorted(hits)
    return rows


def authorized_data_lines(tables):
    """Target file -> stripped source lines of the retired-name data units moved into it (R-0119 B8)."""
    by_source = {m["source"]: m["target"] for _, m in all_moves(tables) if not m.get("copy")}
    out = {}
    for row in tables.get("retired_name_data_units", []):
        target = by_source.get(row["unit"])
        if target is not None:
            out.setdefault(target, set()).update(
                x.strip() for x in row["source_lines"] if x.strip()
            )
    return out


def design_deletion_closure(tables, tree, changed):
    """D7: every reference to the retired set is covered by an authorized change or a frozen/historical rule."""
    retired = tables["retired_modules"]
    retired_files = {module_rel(tree, m) for m in retired} | {
        m.replace(".", "/") + ".py" for m in retired
    }
    frozen = set(tables["frozen_files"])
    out = []
    authorized = authorized_data_lines(tables)
    for rel, hits in references_to_retired(tree, retired).items():
        planned = changed.get(rel, "")
        if rel in retired_files and planned.startswith("delete"):
            continue
        if rel.startswith("pssolver/"):
            if planned.startswith("add") and all(h.startswith("literal") for h in hits):
                lines = (tree / rel).read_text(encoding="utf-8", errors="ignore").splitlines()
                allowed = authorized.get(rel, set())
                hits = [h for h in hits if lines[int(h.split(":")[1]) - 1].strip() not in allowed]
                if not hits:
                    continue  # R-0119 B8: only frozen data lines of units authorized to move into this file
            if not planned.startswith(("modify", "delete")):
                out.append(dict(kind="package_reference_not_authorized", path=rel, hits=hits))
            continue
        if planned.startswith(("modify", "delete")):
            continue
        if any(fnmatch.fnmatch(rel, c["pattern"]) for c in tables["changed_path_classes"]):
            continue
        if rel in frozen or any(fnmatch.fnmatch(rel, pat) for pat in tables["frozen_patterns"]):
            continue
        if (
            rel.startswith("notes/")
            and rel.endswith(".json")
            and all(h.startswith("literal") for h in hits)
        ):
            continue
        out.append(dict(kind="reference_not_authorized", path=rel, hits=hits))
    return out


def module_bindings(root, rel):
    """Module-level name -> ('def', module) | ('from', module, name) | ('module', module) for one source file."""
    mod = rel[:-3].replace("/", ".")
    mod = mod[:-9] if mod.endswith(".__init__") else mod
    tree = ast.parse((root / rel).read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                out[(a.asname or a.name).split(".")[0]] = (
                    "module",
                    a.name if a.asname else a.name.split(".")[0],
                )
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = mod.split(".") if rel.endswith("__init__.py") else mod.split(".")[:-1]
                parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join(parts + ([node.module] if node.module else []))
            for a in node.names:
                out[a.asname or a.name] = ("from", base, a.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ("def", mod)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                if isinstance(t, ast.Name):
                    out[t.id] = ("def", mod)
    return out


def defining_module(root, mods, module, name, depth=0):
    """Follow re-exports to the module that defines `name` (or the submodule it names)."""
    if f"{module}.{name}" in mods:
        return f"{module}.{name}"
    if module not in mods or depth > 8:
        return module if module.startswith("pssolver") else None
    rel = mods[module].relative_to(root).as_posix()
    binding = module_bindings(root, rel).get(name)
    if binding is None or binding[0] == "def":
        return module
    if binding[0] == "module":
        return binding[1] if binding[1].startswith("pssolver") else None
    return defining_module(root, mods, binding[1], binding[2], depth + 1)


def planned_modules(tables, base):
    """Modules that exist after the change: basis modules not deleted, plus added paths."""
    out = set(modmap(base)) | set(tables["new_modules"])
    for rel, why in tables["changed_paths"].items():
        if rel.startswith("pssolver/") and rel.endswith(".py"):
            mod = rel[:-3].replace("/", ".").removesuffix(".__init__")
            if why.startswith("add"):
                out.add(mod)
            elif why.startswith("delete"):
                out.discard(mod)
    return out


def edge_authorized(tables, base, src, dst, owners, allowed_layers, reviewed):
    """Return None when the edge is authorized, else a violation kind."""
    forbid = {k: set(v) for k, v in tables["forbid"].items()}
    if layer(dst) in forbid.get(layer(src), set()):
        return "edge_forbidden_by_layer_rule"
    if layer(src) == layer(dst) or layer(dst) in allowed_layers.get(layer(src), frozenset()):
        return None
    if (src, dst) in reviewed:
        return None
    for row in tables["allowed_edges"]:
        if (row["source"], row["destination"]) == (src, dst):
            pinned = row.get("owners")
            if pinned is not None and not set(owners) <= set(pinned):
                return "edge_owner_not_registered"
            return None
    return "edge_not_authorized"


def design_move_imports(tables, base):
    """D8: the import closure of moved units, in their target modules, is authorized."""
    out = []
    mods = modmap(base)
    retired = set(tables["retired_modules"])
    needles = retired | {r.replace(".", "/") + ".py" for r in retired}
    base_layers, reviewed = ratchet(base)
    allowed_layers = allowed_layer_table(tables, base_layers)
    required = {}
    located = unit_locations(tables)
    renames = tables.get("renames", {})
    for _, m in all_moves(tables):
        path, name = m["source"].split(":")
        node = find_unit(base, m["source"])
        if node is None:
            continue
        data_units = {row["unit"] for row in tables.get("retired_name_data_units", [])}
        if m["source"] not in data_units and mention_pattern(needles).search(ast.unparse(node)):
            out.append(dict(kind="moved_unit_carries_retired_name", unit=m["source"]))
        target = m["target"][:-3].replace("/", ".")
        target = target[:-9] if target.endswith(".__init__") else target
        if "source_substitutions" in m:
            node = mechanical_node(base, m)
            if node is None:
                out.append(dict(kind="mechanical_source_text_invalid", source=m["source"]))
                continue
        bindings = module_bindings(base, path)
        target_rel = m["target"]
        target_defs = (
            set(units_of_source((base / target_rel).read_text()))
            if (base / target_rel).is_file()
            else set()
        )
        moved_here = {
            x["unit"].split(".")[0] for _, x in all_moves(tables) if x["target"] == target_rel
        }
        for ref in sorted(
            {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & set(bindings)
        ):
            kind = bindings[ref]
            if ref in moved_here or ref in target_defs:
                continue
            if kind[0] == "def":
                source_mod = path[:-3].replace("/", ".").removesuffix(".__init__")
                dst = located.get((source_mod, ref), source_mod)
                if dst == target:
                    continue
                if dst == source_mod and tables["changed_paths"].get(path, "").startswith("delete"):
                    out.append(
                        dict(kind="moved_unit_needs_unmoved_helper", unit=m["source"], name=ref)
                    )
                    continue
                required.setdefault((target, dst), set()).add(m["unit"].split(".")[0])
                continue
            dst = (
                defining_module(base, mods, kind[1], kind[2])
                if kind[0] == "from"
                else (kind[1] if kind[1].startswith("pssolver") else None)
            )
            if dst is not None and kind[0] == "from":
                dst = located.get((dst, kind[2]), renames.get(dst, dst))
            elif dst is not None:
                dst = renames.get(dst, dst)
            if dst is None or dst == target:
                continue
            if dst in retired:
                out.append(
                    dict(
                        kind="moved_unit_needs_retired_module",
                        unit=m["source"],
                        module=dst,
                        name=ref,
                    )
                )
                continue
            required.setdefault((target, dst), set()).add(m["unit"].split(".")[0])
    for (src, dst), owners in sorted(required.items()):
        verdict = edge_authorized(tables, base, src, dst, owners, allowed_layers, reviewed)
        if verdict:
            out.append(dict(kind=verdict, edge=f"{src} -> {dst}", owners=sorted(owners)))
    return out, {f"{s} -> {d}": sorted(o) for (s, d), o in sorted(required.items())}


def design_planned_edges(tables, base):
    """D9: module-level edges planned for rewritten and new code are authorized."""
    out = []
    base_layers, reviewed = ratchet(base)
    allowed_layers = allowed_layer_table(tables, base_layers)
    mods = planned_modules(tables, base)
    retired = set(tables["retired_modules"])
    for row in tables.get("planned_edges", []):
        src, dst = row["source"], row["destination"]
        if src not in mods or dst not in mods or dst in retired:
            out.append(dict(kind="planned_edge_endpoint_invalid", edge=f"{src} -> {dst}"))
            continue
        verdict = edge_authorized(tables, base, src, dst, [], allowed_layers, reviewed)
        if verdict:
            out.append(dict(kind=verdict, edge=f"{src} -> {dst}"))
    return out


def run_design(base, tables, tables_path, gate_self, simulate_install, current=None):
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
        "D6_criteria_install": design_criteria(tables)
        + design_accepted_classes(tables, base)
        + install_issues,
        "D7_deletion_closure": design_deletion_closure(tables, base, changed)
        + (design_deletion_closure(tables, current, changed) if current is not None else []),
        "D10_public_api": design_public_api(tables, base, changed),
        "D11_torch_free": design_torch_free(tables, base, changed),
        "D12_compile_probe": design_compile_probe(tables, base),
    }
    report["D8_move_imports"], required_edges = design_move_imports(tables, base)
    report["D9_planned_edges"] = design_planned_edges(tables, base)
    files = tree_files(base, overlay if simulate_install else None)
    report["view"] = dict(
        installation_view="simulated" if simulate_install else "as found",
        files=len(files),
        current=str(current) if current is not None else None,
        move_import_edges=required_edges,
    )
    report["summary"] = {k: len(v) for k, v in report.items() if isinstance(v, list)}
    return report


# ---- R-0115: process-level compile probes (D12 design mode, G7 draft mode) -----------------------------
_PROBE_COMMON = r"""
import dataclasses, importlib.abc, json, math, sys
ATTEMPTS = []


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("torch", "numpy"):
            ATTEMPTS.append(name)
            raise ImportError("blocked by the S-28b compile probe: " + name)
        return None


sys.meta_path.insert(0, _Block())


def enc(v):
    if isinstance(v, float):
        return {"float": v.hex()}
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        return {k: enc(x) for k, x in v.items()}
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: enc(getattr(v, f.name)) for f in dataclasses.fields(v)}
    return v


def npy_bytes(shape, descr="<f8", itemsize=8, truncate=None):
    header = "{'descr': '%s', 'fortran_order': False, 'shape': %r, }" % (descr, tuple(shape))
    pad = 64 - (10 + len(header) + 1) % 64
    header = header + " " * pad + "\n"
    blob = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode("latin1")
    count = 1
    for n in shape:
        count *= n
    blob += bytes(count * itemsize)
    return blob if truncate is None else blob[:truncate]
"""

_D12_PROBE = (
    _PROBE_COMMON
    + r"""
root, cases_path, ref_path, out_path = sys.argv[1:5]
sys.path.insert(0, root)
cases = json.load(open(cases_path))
reference = json.load(open(ref_path))
violations = []


def dec(v):
    if isinstance(v, dict) and set(v) == {"float"}:
        return float.fromhex(v["float"])
    if isinstance(v, dict) and set(v) == {"Parameter"}:
        return Parameter(dec(v["Parameter"]))
    if isinstance(v, list):
        return tuple(dec(x) for x in v)
    if isinstance(v, dict):
        return {k: dec(x) for k, x in v.items()}
    return v


try:
    from pssolver.core.terms import Parameter
    from pssolver.planning.assemble import (AssemblyProblem, EulerInput, ExecutionInput, GeometryInput,
                                            NumericsInput, OutputInput, assemble)
    from pssolver.presets.beris_edwards import CompleteStressBerisEdwards
    from pssolver.presets import shendruk
except Exception as exc:
    json.dump(dict(violations=[dict(kind="probe_import_failed", error=f"{type(exc).__name__}: {exc}"[:300])]
                   + [dict(kind="blocked_import_attempt", module=m) for m in ATTEMPTS], replayed=0),
              open(out_path, "w"), indent=1)
    raise SystemExit(0)


def check(name, what, recorded, value):
    if recorded != enc(value):
        violations.append(dict(kind="replay_differs", case=name, what=what))


def identity(case, model_kwargs, refresh_interval=None):
    a = case["assemble"]
    integrator = dec(a["integrator"])
    if refresh_interval is not None:
        integrator["refresh_interval"] = refresh_interval
    problem = AssemblyProblem(CompleteStressBerisEdwards(**model_kwargs), GeometryInput(**dec(a["geometry"])),
                              dec(a["boundaries"]), EulerInput(**integrator), NumericsInput(**dec(a["numerics"])),
                              ExecutionInput(**dec(a["execution"])), OutputInput(**dec(a["outputs"])))
    return assemble(problem).plan.identity_sha256


def plane_chain(name, case, resolve_kwargs):
    recorded = case["resolve_shendruk_plane_preset"][0]
    preset = shendruk.resolve_shendruk_plane_preset(**resolve_kwargs)
    check(name, "shendruk preset", recorded["result"], preset)
    refresh = case["resolve_spectral_refresh"][0]
    spec = shendruk._resolve_spectral_refresh(**dec(refresh["kwargs"]))
    check(name, "spectral refresh", refresh["result"], spec)
    model = dec(case["model"])
    material = dict(model["material"])
    material["gamma"] = preset.rotational_viscosity
    rebuilt = dict(model, material=material, ldg_l1=preset.ldg_l1, alpha=Parameter(preset.zeta))
    for key in ("ldg_l1",):
        check(name, key, case["model"][key], rebuilt[key])
    check(name, "gamma", case["model"]["material"]["gamma"], preset.rotational_viscosity)
    check(name, "alpha", case["model"]["alpha"]["Parameter"], preset.zeta)
    return identity(case, rebuilt, spec.effective_interval_steps)


replayed = 0
for name, case in sorted(cases.items()):
    try:
        if name.startswith(("periodic_default", "channel_default")):
            got = identity(case, dec(case["model"]))
            want = case["assemble"]["plan_identity_sha256"]
            if got != want or got != reference[name]["plan_identity_sha256"]:
                violations.append(dict(kind="plan_identity_differs", case=name))
        elif name.startswith("plane_shendruk:"):
            sid = name.split(":", 1)[1]
            got = plane_chain(name, case, dec(case["resolve_shendruk_plane_preset"][0]["kwargs"]))
            if got != case["assemble"]["plan_identity_sha256"] or got != reference[sid]["plan_identity_sha256"]:
                violations.append(dict(kind="plan_identity_differs", case=name))
        elif name.startswith("plane_csbe:"):
            d = case["declaration"]
            s_eq = shendruk.positive_equilibrium_S(d["ldg_a"], d["ldg_b"], d["ldg_c"])
            check(name, "S_eq", case["positive_equilibrium_S"][0]["result"], s_eq)
            equilibrium_q = 1.5 * s_eq
            frank_k = 2.0 * equilibrium_q**2 * d["ldg_l1"]
            activity_number = d["height"] * math.sqrt(d["activity"] / frank_k)
            run_spec_kwargs = case["create_plane_run_spec_from_public"][0]["kwargs"]
            check(name, "frank_k", run_spec_kwargs["frank_k"], frank_k)
            check(name, "activity_number", run_spec_kwargs["activity_number"], activity_number)
            kwargs = dec(case["resolve_shendruk_plane_preset"][0]["kwargs"])
            kwargs.update(activity_number=activity_number, frank_k=frank_k)
            got = plane_chain(name, case, kwargs)
            if got != case["assemble"]["plan_identity_sha256"]:
                violations.append(dict(kind="plan_identity_differs", case=name))
        elif name == "plane_csbe_error:no_real_equilibrium":
            d = case["declaration"]
            try:
                shendruk.positive_equilibrium_S(d["ldg_a"], d["ldg_b"], d["ldg_c"])
                violations.append(dict(kind="expected_error_missing", case=name))
            except ValueError as exc:
                reason = ("the qualified Plane translation requires bulk coefficients "
                          f"with a positive real uniaxial equilibrium: {exc}")
                if case["message"] != f"{case['code']}: {reason}":
                    violations.append(dict(kind="error_text_differs", case=name))
        elif name == "plane_csbe_error:refresh_not_multiple":
            try:
                shendruk._resolve_spectral_refresh(**case["refresh_kwargs"])
                violations.append(dict(kind="expected_error_missing", case=name))
            except ValueError as exc:
                if case["message"] != f"{case['code']}: {exc}":
                    violations.append(dict(kind="error_text_differs", case=name))
        else:
            continue
        replayed += 1
    except Exception as exc:  # report, do not hide
        violations.append(dict(kind="replay_raised", case=name, error=f"{type(exc).__name__}: {exc}"[:300]))
for name in ("torch", "numpy"):
    if name in sys.modules:
        violations.append(dict(kind="module_loaded", module=name))
violations.extend(dict(kind="blocked_import_attempt", module=m) for m in ATTEMPTS)
json.dump(dict(violations=violations, replayed=replayed), open(out_path, "w"), indent=1)
"""
)


def _run_probe(script, args):
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="s28b_probe_") as tmp:
        path = Path(tmp) / "probe.py"
        path.write_text(script)
        out = Path(tmp) / "out.json"
        proc = subprocess.run(
            [sys.executable, "-I", "-B", str(path), *[str(a) for a in args(Path(tmp))], str(out)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0 or not out.is_file():
            return [
                dict(kind="probe_failed", returncode=proc.returncode, stderr=proc.stderr[-800:])
            ], 0
        report = json.loads(out.read_text())
        return report["violations"], report["replayed"]


def _apply_copies(tree_root, base, tables):
    """Planned exact copies (R-0115 B2) applied to a probe copy of the basis."""
    for row in [m for m in tables["exact_moves"] if m.get("copy")]:
        node = find_unit(base, row["source"])
        lines = (base / row["source"].split(":")[0]).read_text().splitlines()
        start = (
            node.decorator_list[0].lineno if getattr(node, "decorator_list", None) else node.lineno
        )
        segment = "\n".join(lines[start - 1 : node.end_lineno])
        target = tree_root / row["target"]
        text = target.read_text()
        if row.get("replaces_import"):
            text = (
                "\n".join(x for x in text.splitlines() if x.strip() != row["replaces_import"])
                + "\n"
            )
        target.write_text(text + "\n\n" + segment + "\n")


def design_compile_probe(tables, base):
    """D12: the planned public compile chains run without importing torch or numpy and reproduce the record."""
    import shutil

    def args(tmp):
        probe_root = tmp / "tree"
        shutil.copytree(
            base / "pssolver", probe_root / "pssolver", ignore=shutil.ignore_patterns("__pycache__")
        )
        (probe_root / "pssolver/__init__.py").write_text("from ._version import __version__\n")
        _apply_copies(probe_root, base, tables)
        (tmp / "cases.json").write_text(json.dumps(tables["compile_probe_cases"]))
        (tmp / "ref.json").write_text(json.dumps(tables["plan_identity_reference"]))
        return [probe_root, tmp / "cases.json", tmp / "ref.json"]

    violations, replayed = _run_probe(_D12_PROBE, args)
    expected = sum(1 for k in tables["compile_probe_cases"] if not k.startswith("periodic_header"))
    if replayed != expected:
        violations.append(dict(kind="cases_not_all_replayed", replayed=replayed, expected=expected))
    return violations


_G7_PROBE = (
    _PROBE_COMMON
    + r"""
from pathlib import Path

draft_root, cases_path, work, out_path = sys.argv[1:5]
sys.path.insert(0, draft_root)
cases = json.load(open(cases_path))
violations, replayed = [], 0
import pssolver as P  # noqa: E402


def build(spec, workdir):
    snapshot = spec.get("snapshot")
    initial_dir = workdir / "initial"
    initial_dir.mkdir(parents=True, exist_ok=True)
    if snapshot is not None:
        content = bytes.fromhex(snapshot["raw_hex"]) if "raw_hex" in snapshot else npy_bytes(
            snapshot["shape"], truncate=snapshot.get("truncate"))
        (initial_dir / "Q_0.npy").write_bytes(content)
    model_name, model_kwargs = spec["model"]
    geometry_name, geometry_kwargs = spec["geometry"]
    ic = spec["initial_condition"]
    if ic[0] == "SnapshotInitialCondition":
        initial = P.SnapshotInitialCondition(initial_dir, **ic[1])
    else:
        initial = getattr(P, ic[0])(*ic[1], **ic[2])
    return P.Simulation(
        model=getattr(P, model_name)(**model_kwargs),
        geometry=getattr(P, geometry_name)(**{k: tuple(v) for k, v in geometry_kwargs.items()}),
        boundaries={field: getattr(P, policy)() for field, policy in spec["boundaries"].items()},
        numerics=P.SpectralNumerics(**spec["numerics"]),
        time=P.SemiImplicitEuler(**spec["time"]),
        initial_condition=initial,
        execution=P.TorchSpectralExecution(**spec["execution"]),
        output=P.Output(directory=workdir / "output", **spec["output"]),
    )


for name, case in sorted(cases.items()):
    spec = case.get("v05")
    if spec is None:
        continue
    try:
        simulation = build(spec, Path(work) / name.replace(":", "_"))
        expect = spec["expect"]
        try:
            compiled = P.compile_simulation(simulation)
            got = dict(plan_identity_sha256=compiled.plan_identity_sha256)
        except P.PublicSimulationCompilationError as exc:
            got = dict(rejection=dict(code=getattr(exc.code, "value", exc.code), kind=exc.kind, reason=exc.reason))
        if got != expect:
            violations.append(dict(kind="public_compile_differs", case=name, expected=expect, got=got))
        replayed += 1
    except Exception as exc:
        violations.append(dict(kind="public_compile_raised", case=name, error=f"{type(exc).__name__}: {exc}"[:300]))
try:  # R-0116 B3: source, integrity, and the output directory outside the Plan
    spec = cases["periodic_default_float64_default"]["v05"]
    pair = [build(spec, Path(work) / name) for name in ("pair_a", "pair_b")]
    compiled = [P.compile_simulation(sim) for sim in pair]
    for sim, c in zip(pair, compiled):
        if c.source is not sim:
            violations.append(dict(kind="compiled_source_is_not_the_declaration"))
        if c.validate_integrity() is not None or c.declaration_sha256 != sim.canonical_sha256():
            violations.append(dict(kind="compiled_integrity_contract"))
    if compiled[0].plan_identity_sha256 != compiled[1].plan_identity_sha256:
        violations.append(dict(kind="output_directory_changed_plan_identity"))
    if compiled[0].declaration_sha256 == compiled[1].declaration_sha256 or compiled[0] == compiled[1]:
        violations.append(dict(kind="declaration_hash_ignores_output_directory"))
    # R-0118 B6: the validation digest is a declared Output field: in the declaration hash, not in the Plan
    digests = []
    for value in ("a" * 64, "b" * 64):
        variant = dict(spec, output=dict(spec["output"], validation_config_sha256=value))
        digests.append(P.compile_simulation(build(variant, Path(work) / ("digest_" + value[0]))))
    if digests[0].plan_identity_sha256 != digests[1].plan_identity_sha256:
        violations.append(dict(kind="validation_digest_changed_plan_identity"))
    if digests[0].declaration_sha256 == digests[1].declaration_sha256:
        violations.append(dict(kind="validation_digest_not_in_declaration_hash"))
    try:
        build(dict(spec, output=dict(spec["output"], validation_config_sha256="A" * 64)), Path(work) / "digest_A")
        violations.append(dict(kind="uppercase_validation_digest_accepted"))
    except ValueError as exc:
        if "64 lowercase hexadecimal" not in str(exc):
            violations.append(dict(kind="validation_digest_error_text", error=str(exc)[:200]))
except Exception as exc:
    violations.append(dict(kind="compiled_contract_raised", error=f"{type(exc).__name__}: {exc}"[:300]))
try:
    P.capability_catalog().to_metadata()
except Exception as exc:
    violations.append(dict(kind="catalog_raised", error=f"{type(exc).__name__}: {exc}"[:300]))
for module in ("torch", "numpy"):
    if module in sys.modules:
        violations.append(dict(kind="module_loaded", module=module))
violations.extend(dict(kind="blocked_import_attempt", module=m) for m in ATTEMPTS)
json.dump(dict(violations=violations, replayed=replayed), open(out_path, "w"), indent=1)
"""
)


def draft_compile_probe(tables, draft):
    """G7: the draft's public compile paths (both Plane writings included) import neither torch nor numpy."""

    def args(tmp):
        (tmp / "cases.json").write_text(json.dumps(tables["compile_probe_cases"]))
        (tmp / "work").mkdir()
        return [draft, tmp / "cases.json", tmp / "work"]

    violations, replayed = _run_probe(_G7_PROBE, args)
    expected = sum(1 for c in tables["compile_probe_cases"].values() if "v05" in c)
    if replayed != expected:
        violations.append(dict(kind="cases_not_all_compiled", replayed=replayed, expected=expected))
    return violations


# ---- draft mode ----------------------------------------------------------------------------------
def draft_units(tables, base, draft, base_mods, draft_mods, touched):
    out = []
    for kind, m in all_moves(tables):
        src = find_unit(base, m["source"])
        dst = find_unit(draft, target_ref(m))
        if dst is None:
            out.append(dict(kind="moved_unit_missing", unit=target_ref(m)))
            continue
        want = src if kind == "exact" else mechanical_node(base, m)
        if want is None or ast.dump(want) != ast.dump(dst):
            out.append(dict(kind=f"{kind}_move_not_identical", unit=target_ref(m)))
        elif kind == "mechanical":  # R-0121: the moved text itself, not only its AST
            text = unit_source((draft / m["target"]).read_text(), m["unit"])
            if text != mechanical_text(base, m):
                out.append(dict(kind="mechanical_move_text_differs", unit=target_ref(m)))
    moved, replaced, deleted = historical_routes(tables, tables["changed_paths"])
    for target, unit in historical_exact_units(
        base
    ):  # R-0121: kept historical exact units stay byte-identical
        if target in deleted or (target, unit) in moved or (target, unit) in replaced:
            continue
        path = draft / target
        kept = unit_source(path.read_text(), unit) if path.is_file() else None
        if kept is None or kept != unit_source((base / target).read_text(), unit):
            out.append(dict(kind="historical_exact_unit_changed", target=target, unit=unit))
    for m in tables["test_moves"]:
        if find_unit(draft, target_ref(m)) is None:
            out.append(dict(kind="test_move_missing", unit=target_ref(m)))
    registry_path = draft / tables["unit_registry"]
    registry = json.loads(registry_path.read_text()) if registry_path.is_file() else {}
    # a class that receives moved methods is not itself registered by those moves (R-0105)
    listed = {(m["target"], m["unit"]) for _, m in all_moves(tables) if "." not in m["unit"]}
    for mod, units in tables.get("required_units_modified", {}).items():
        if mod == "authority":
            continue
        actual = units_of_source(draft_mods[mod].read_text()) if mod in draft_mods else {}
        for u in units:
            if u not in actual:
                out.append(dict(kind="required_unit_missing", module=mod, unit=u))
            listed.add((mod.replace(".", "/") + ".py", u))
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
            if row.get(
                "copy_of"
            ):  # R-0115 B2: further pure scalar copies must be AST-identical to the original
                src = find_unit(base, row["copy_of"])
                dst = (
                    units_of_source(draft_mods[mod].read_text()).get(u)
                    if mod in draft_mods
                    else None
                )
                if src is None or dst is None or ast.dump(src) != ast.dump(dst):
                    out.append(dict(kind="registered_copy_not_identical", module=mod, unit=u))
    return out


def draft_public_api(tables, draft):
    """G6: the draft root exports exactly the table's names through a static _EXPORTS map and imports nothing heavy."""
    out = []
    new = tables["public_api"]["new_root"]
    if sorted(root_all(draft) or []) != sorted(new):
        out.append(
            dict(kind="root_all_differs", expected=len(new), found=len(root_all(draft) or []))
        )
    root = draft / "pssolver/__init__.py"
    exports = None
    for n in ast.parse(root.read_text()).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_EXPORTS":
            exports = ast.literal_eval(n.value)
    lazy = {k: v for k, v in new.items() if k != "__version__"}
    if exports != lazy:
        out.append(dict(kind="root_exports_map_differs"))
    eager = [x for x in module_level_imports(root, "pssolver") if x.startswith("pssolver")]
    if sorted(set(eager) - {"pssolver._version", "pssolver._version.__version__"}):
        out.append(dict(kind="root_eager_import", imports=sorted(set(eager))))
    mods = modmap(draft)
    for name, module in sorted(lazy.items()):
        if module not in mods:
            out.append(dict(kind="export_module_missing", name=name, module=module))
            continue
        rel = mods[module].relative_to(draft).as_posix()
        if name not in module_bindings(draft, rel):
            out.append(dict(kind="export_name_unbound", name=name, module=module))
    return out


def run_draft(base, draft, tables, tables_path, gate_self, simulate_install):
    overlay, install_issues = installation_view(
        draft, tables, tables_path, gate_self, simulate_install
    )
    report = {}
    base_layers, base_reviewed = ratchet(base)
    draft_layers, _ = ratchet(draft)
    allowed_layers = allowed_layer_table(tables, base_layers)
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
    for mod, units in tables.get("required_units_modified", {}).items():
        if mod != "authority":
            known |= {(mod, u) for u in units}
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
    g2.extend(torch_free_violations(draft, tables["torch_free_import_modules"]))
    renames = tables.get("renames", {})

    def renamed(mod):
        for old in sorted(renames, key=len, reverse=True):
            if mod == old or mod.startswith(old + "."):
                return renames[old] + mod[len(old) :]
        return mod

    base_cycles = {frozenset(renamed(m) for m in c) for c in direct_import_cycles(base)}
    g2.extend(
        dict(kind="new_import_cycle", modules=sorted(c))
        for c in direct_import_cycles(draft)
        if c not in base_cycles
    )
    report["G2_layers"] = dict(violations=g2)
    report["G6_public_api"] = dict(violations=draft_public_api(tables, draft))
    report["G7_compile_probe"] = dict(violations=draft_compile_probe(tables, draft))
    draft_files = tree_files(draft, overlay if simulate_install else None)
    base_files = tree_files(base, None)
    retired = tables["retired_modules"]
    needles = set(retired) | {r.replace(".", "/") + ".py" for r in retired}
    pattern = mention_pattern(needles)
    data_lines = set()
    for row in tables.get("retired_name_data_units", []):
        node = find_unit(base, row["unit"])
        if node is not None:
            path = row["unit"].split(":")[0]
            lines = (base / path).read_text().splitlines()[node.lineno - 1 : node.end_lineno]
            data_lines |= {x.strip() for x in lines if x.strip()}
    g3 = []
    for rel, data in sorted(draft_files.items()):
        if not rel.endswith((".py", ".json")):
            continue
        text = data.decode("utf-8", errors="ignore")
        if not pattern.search(text):
            continue
        base_lines = set(base_files.get(rel, b"").decode("utf-8", "ignore").splitlines())
        for i, line in enumerate(text.splitlines(), 1):
            if not pattern.search(line):
                continue
            own = any(fnmatch.fnmatch(rel, p) for p in tables["own_file_patterns"])
            if rel.startswith("pssolver/") and line.strip() in data_lines:
                continue
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
    current = next((a.split("=", 1)[1] for a in argv if a.startswith("--current=")), None)
    if len(args) == 4 and args[0] == "design":
        base, tables_path, out = Path(args[1]), Path(args[2]), Path(args[3])
        tables = json.loads(tables_path.read_text())
        report = run_design(
            base,
            tables,
            tables_path,
            Path(__file__).resolve(),
            simulate,
            Path(current) if current else None,
        )
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
