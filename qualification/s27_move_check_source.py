"""R-0091: mechanical check of every S-27 'verbatim move + allowed rewrite' provision.

Usage: python3 -I -B s27_move_check.py <tree_root(327a192 git archive)> <out.json>
For every top-level unit (function, class, assignment) of the source modules named by the
S-27 design as verbatim-move sources, and for the two Fields methods named in section 4.2:
  - free names and where they resolve (local unit / import origin / builtin);
  - attribute chains rooted at each parameter (e.g. adapter.solver.model.static_model.x);
  - same-module units it calls (transitively, closure);
  - dependencies on the S-27 delete set or on the old runtime adapter surface.
Classification per unit:
  MOVABLE      closure touches nothing deleted and no adapter/solver surface;
  NEEDS_REBIND depends on adapter/solver/Fields surface or deleted names (listed);
  INLINE       the 'unit' named by the design is not a separable top-level unit.
"""

import ast
import builtins
import json
import sys
from pathlib import Path


def main():
    ROOT = Path(sys.argv[1])
    OUT = Path(sys.argv[2])
    DELETED = {
        r["module"]
        for r in json.loads(Path(__file__).with_name("s27_retired_modules.json").read_text())[
            "modules"
        ]
    }
    # Names re-exported by pssolver.runtime / pssolver.workflows that originate in the delete set
    ADAPTER_SURFACE = {
        "PlaneRuntimeAdapterProtocol",
        "LegacyPlaneRuntimeAdapter",
        "plane_physical_component",
        "plane_lifting_restart_metadata",
        "plane_lifting_fresh_construction_provenance",
        "verify_plane_lifting_identity",
        "build_plane_beris_edwards_runtime",
        "PlaneRuntimeBuildRequest",
        "build_legacy_plane_runtime",
    }
    ADAPTER_ATTRS = {
        "solver",
        "fields",
        "synchronize_for_observation",
        "projected_normal_force",
        "flow_diagnostics",
        "advance",
        "completed_steps",
        "restore_progress",
        "backend_restart_metadata",
        "runtime_path",
        "projector",
        "integrator",
        "model",
    }
    SOURCES = json.loads(Path(__file__).with_name("s27_impact_targets.json").read_text())[
        "move_sources"
    ]
    FIELDS_METHODS = (
        "Field.py",
        "Fields",
        (
            "select_spectral_group",
            "store_spatial_group",
            "select_spatial_group",
            "store_spectral_group",
            "_transform_group_indexer",
        ),
    )
    BUILTINS = set(dir(builtins))

    def module_name(rel):
        name = rel[:-3].replace("/", ".")
        return name[:-9] if name.endswith(".__init__") else name

    def imports_of(tree, mod, is_pkg):
        out = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = mod.split(".") if is_pkg else mod.split(".")[:-1]
                    parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                    base = ".".join(parts + ([node.module] if node.module else []))
                for a in node.names:
                    out[a.asname or a.name] = f"{base}.{a.name}"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    out[(a.asname or a.name).split(".")[0]] = a.name
        return out

    def reexport_origin(qualified):
        """Resolve pssolver.<pkg>.<Name> re-exports to the defining module (one hop)."""
        pkg, _, name = qualified.rpartition(".")
        init = ROOT / (pkg.replace(".", "/") + "/__init__.py")
        if not init.exists():
            return qualified
        tree = ast.parse(init.read_text())
        origin = imports_of(tree, pkg, True).get(name)
        return origin or qualified

    def chains(node, params):
        found = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute):
                parts = []
                cur = sub
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id in params:
                    found.add(".".join([cur.id] + list(reversed(parts))))
        # keep maximal chains only
        return sorted(c for c in found if not any(o != c and o.startswith(c + ".") for o in found))

    def unit_info(node, imports, local_units, mod):
        params = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            params = {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
            if a.vararg:
                params.add(a.vararg.arg)
            if a.kwarg:
                params.add(a.kwarg.arg)
        assigned = {
            n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        inner = {
            n.name
            for n in ast.walk(node)
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n is not node
        }
        for n in ast.walk(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not node:
                assigned |= {x.arg for x in n.args.args + n.args.kwonlyargs}
            if isinstance(n, ast.comprehension):
                assigned |= {t.id for t in ast.walk(n.target) if isinstance(t, ast.Name)}
            if isinstance(n, ast.ExceptHandler) and n.name:
                assigned.add(n.name)
        free = sorted(
            {
                n.id
                for n in ast.walk(node)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            - params
            - assigned
            - inner
            - BUILTINS
        )
        resolved, local_calls = {}, []
        for name in free:
            if name in local_units:
                resolved[name] = f"{mod}:{name}"
                local_calls.append(name)
            elif name in imports:
                origin = imports[name]
                if origin.startswith("pssolver.") and origin.count(".") >= 2:
                    origin = reexport_origin(origin)
                resolved[name] = origin
            else:
                resolved[name] = "UNRESOLVED"
        return dict(
            params=sorted(params),
            free=resolved,
            local_calls=local_calls,
            param_chains=chains(node, params),
            lines=[node.lineno, node.end_lineno],
        )

    report = {"units": {}, "fields_methods": {}}
    for rel, why in SOURCES.items():
        path = ROOT / "pssolver" / rel
        mod = module_name("pssolver/" + rel)
        tree = ast.parse(path.read_text())
        imports = imports_of(tree, mod, False)
        local = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                for t in node.targets if isinstance(node, ast.Assign) else [node.target]:
                    if isinstance(t, ast.Name):
                        local[t.id] = node
        infos = {name: unit_info(node, imports, local, mod) for name, node in local.items()}
        # transitive closure over local calls
        for name, info in infos.items():
            seen, stack = set(), list(info["local_calls"])
            while stack:
                n = stack.pop()
                if n in seen or n not in infos:
                    continue
                seen.add(n)
                stack.extend(infos[n]["local_calls"])
            deps_deleted, deps_adapter = set(), set()
            for n in {name} | seen:
                for k, origin in infos[n]["free"].items():
                    if any(origin == d or origin.startswith(d + ".") for d in DELETED):
                        deps_deleted.add(f"{k}<-{origin}")
                    if k in ADAPTER_SURFACE or origin.rsplit(".", 1)[-1] in ADAPTER_SURFACE:
                        deps_adapter.add(f"{k}<-{origin}")
                for c in infos[n]["param_chains"]:
                    parts = c.split(".")
                    if len(parts) > 1 and parts[1] in ADAPTER_ATTRS:
                        deps_adapter.add(f"{n}:{c}")
            info["closure"] = sorted(seen)
            info["deleted_dependencies"] = sorted(deps_deleted)
            info["adapter_dependencies"] = sorted(deps_adapter)
            info["class"] = "MOVABLE" if not deps_deleted and not deps_adapter else "NEEDS_REBIND"
            info["size"] = info["lines"][1] - info["lines"][0] + 1
        report["units"][rel] = {"why": why, "units": infos}

    # Fields methods named by section 4.2
    fpath = ROOT / "pssolver" / FIELDS_METHODS[0]
    ftree = ast.parse(fpath.read_text())
    for node in ftree.body:
        if isinstance(node, ast.ClassDef) and node.name == FIELDS_METHODS[1]:
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and m.name in FIELDS_METHODS[2]:
                    selfattrs = sorted(
                        {
                            f"self.{a.attr}"
                            for a in ast.walk(m)
                            if isinstance(a, ast.Attribute)
                            and isinstance(a.value, ast.Name)
                            and a.value.id == "self"
                        }
                    )
                    report["fields_methods"][m.name] = dict(
                        lines=[m.lineno, m.end_lineno],
                        self_attributes=selfattrs,
                        source=ast.get_source_segment(fpath.read_text(), m),
                    )

    # the inline question: is metadata construction a separable unit in the application module?
    app = ROOT / "pssolver" / next(p for p in SOURCES if p.startswith("applications/"))
    atree = ast.parse(app.read_text())
    owned = next(
        n
        for n in atree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_plane_beris_edwards_owned"
    )
    meta_assigns = [
        n.lineno
        for n in ast.walk(owned)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in ("metadata", "implementation_provenance")
            for t in n.targets
        )
    ]
    report["inline"] = dict(
        function="_run_plane_beris_edwards_owned",
        lines=[owned.lineno, owned.end_lineno],
        metadata_assignment_lines=meta_assigns,
        nested_defs=[
            n.name for n in ast.walk(owned) if isinstance(n, ast.FunctionDef) and n is not owned
        ],
    )
    OUT.write_text(json.dumps(report, indent=1) + "\n")

    for rel, block in report["units"].items():
        print(f"== {rel}  ({block['why']})")
        for name, info in sorted(block["units"].items(), key=lambda kv: kv[1]["lines"][0]):
            flag = info["class"]
            extra = ""
            if flag != "MOVABLE":
                extra = (
                    " | deleted: "
                    + ", ".join(info["deleted_dependencies"][:6])
                    + " | adapter: "
                    + ", ".join(info["adapter_dependencies"][:8])
                )
            print(f"  {info['lines'][0]:>5}-{info['lines'][1]:<5} {flag:12} {name}{extra}")
    print("== Fields methods (S-27 §4.2):")
    for k, v in report["fields_methods"].items():
        print("  ", k, v["lines"], v["self_attributes"])
    print("== inline:", json.dumps(report["inline"]))


if __name__ == "__main__":
    main()
