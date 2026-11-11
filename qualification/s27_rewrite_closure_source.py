"""R-0093: close every residual dependency of the R-0091 'only replace X' function bodies.

Usage: python3 -I -B s27_rewrite_closure.py <tree(327a192 git archive)> <out.json>
For each function body that R-0091 allows to move with enumerated substitutions, list every dependency:
  - self.<attr> uses (methods), parameter attribute chains (adapter.solver.integrator.step_count ...),
  - parameter-level uses that are not attribute chains (isinstance(adapter, ...), f(adapter)),
  - free names (imports, module units, builtins excluded).
Each dependency is tagged with the authorization that covers it (AUTH table below, R-0091 + R-0093),
or KEPT (still available unchanged in the new context), or RESIDUAL. Exit 1 if any RESIDUAL remains.
"""

import ast
import builtins
import json
import sys
from pathlib import Path


def main():
    ROOT = Path(sys.argv[1])
    OUT = Path(sys.argv[2])
    BUILTINS = set(dir(builtins))

    TARGETS = [
        ("workflows/plane_observation.py", None, "capture_plane_observation"),
        ("workflows/plane_observation.py", None, "capture_plane_diagnostic"),
        ("runtime/plane_beris_edwards.py", "LegacyPlaneRuntimeAdapter", "projected_normal_force"),
        ("runtime/plane_beris_edwards.py", "LegacyPlaneRuntimeAdapter", "flow_diagnostics"),
        ("workflows/plane_checkpoint.py", None, "capture_plane_checkpoint"),
        ("workflows/plane_checkpoint.py", None, "restore_plane_checkpoint"),
        ("applications/plane_beris_edwards.py", None, "main"),
        ("applications/plane_beris_edwards.py", None, "_print_workflow_summary"),
        ("Field.py", "Fields", "_normalize_field_indices"),
        ("Field.py", "Fields", "_normalize_field_key"),
        ("Field.py", "Fields", "_get_field_index"),
        ("Field.py", "Fields", "_transform_group_indexer"),
        ("Field.py", "Fields", "select_spatial_group"),
        ("Field.py", "Fields", "select_spectral_group"),
        ("Field.py", "Fields", "_store_group"),
        ("Field.py", "Fields", "store_spatial_group"),
        ("Field.py", "Fields", "store_spectral_group"),
    ]

    # dependency -> authorization id (R-0091 C1-C8, F2a-F2c, F5a-F5b, F6a-F6c; R-0093 adds the rest)
    AUTH = {
        "capture_plane_observation": {
            "param:adapter": "C1/C4 (adapter -> fields: FieldView)",
            "free:PlaneRuntimeAdapterProtocol": "C1",
            "param:synchronize": "C2",
            "adapter.synchronize_for_observation": "C2",
            "adapter.completed_steps": "C3",
            "adapter.fields": "C4",
            "adapter.solver.batchsize": "C5",
            "free:plane_physical_component": "C6",
        },
        "capture_plane_diagnostic": {
            "param:adapter": "C4",
            "adapter.fields": "C4",
            "free:PlaneRuntimeAdapterProtocol": "C4 (parameter annotation -> FieldView)",
            "adapter.projected_normal_force": "C7",
            "adapter.flow_diagnostics": "C8",
        },
        "projected_normal_force": {
            "self._solver.model.static_model.last_projected_normal_force": "C7",
            "param:self": "C7 (method -> function of constraint_diagnostics)",
        },
        "flow_diagnostics": {
            "self._solver.model.static_model": "C8",
            "param:self": "C8 (method -> function of constraint_diagnostics)",
        },
        "capture_plane_checkpoint": {
            "param:adapter": "F2a/R-0093 G1 (adapter -> engine, state)",
            "free:PlaneRuntimeAdapterProtocol": "R-0093 G1",
            "adapter.synchronize_for_observation": "F2a (-> engine.prepare_constraints_(state))",
            "adapter.fields": "F2a (-> engine.field_view(state))",
            "adapter.solver.integrator": "F2a/R-0093 G3 (refresh counters -> state.refresh_phase, plan interval)",
            "adapter.completed_steps": "F2a (-> state.completed_steps)",
            "adapter.backend_restart_metadata": "F2a (-> parameter backend_restart)",
            "adapter.runtime_path": "F2a (-> run_spec.runtime_path)",
            "free:plane_lifting_restart_metadata": "R-0093 G2 (-> None)",
            "free:plane_lifting_fresh_construction_provenance": "R-0093 G2 (-> None)",
        },
        "restore_plane_checkpoint": {
            "param:adapter": "F2b/R-0093 G1 (adapter -> engine, state)",
            "free:PlaneRuntimeAdapterProtocol": "R-0093 G1",
            "adapter.runtime_path": "F2b (-> run_spec.runtime_path)",
            "adapter.fields": "F2b (-> engine.field_view(state); copy_ writes State)",
            "adapter.backend_restart_metadata": "F2b (-> literal by run_spec.runtime_path)",
            "adapter.synchronize_for_observation": "F2b (-> constrained_current=False; prepare_constraints_)",
            "adapter.restore_progress": "F2b/R-0093 G4",
            "adapter.completed_steps": "F2b (-> state.completed_steps)",
            "free:plane_lifting_restart_metadata": "R-0093 G2 (-> None)",
            "free:verify_plane_lifting_identity": "R-0093 G2 (call removed; no-op for None lifting)",
        },
        "main": {
            "free:run_plane_beris_edwards": "F5a (-> new entry)",
            "free:_print_workflow_summary": "F5 (moved with main)",
            "free:parse_args": "F5 (moved verbatim)",
        },
        "_print_workflow_summary": {
            "param:workflow_result": "F5b (new result type)",
            "free:PlaneWorkflowResult": "F5b",
            "workflow_result.elapsed_seconds": "F5b/R-0093 G5 (result must expose)",
            "workflow_result.diagnostics": "F5b/R-0093 G5 (tuple of PlaneDiagnostic)",
        },
        "_normalize_field_indices": {
            "param:self": "F6",
            "self._normalize_field_key": "R-0093 G6",
            "self.boundary_conditions": "R-0093 G6 (-> component_count)",
        },
        "_normalize_field_key": {"param:self": "F6", "self._get_field_index": "R-0093 G6"},
        "_get_field_index": {"param:self": "F6", "self.name_to_idx": "R-0093 G6 (-> name_to_idx)"},
        "_transform_group_indexer": {
            "param:self": "F6",
            "self.transform_group_indexing": "F6 (-> indexing)",
        },
        "select_spatial_group": {
            "param:self": "F6",
            "self._normalize_field_indices": "R-0093 G6",
            "self._transform_group_indexer": "F6",
            "self.spatial": "F6 (-> tensor)",
        },
        "select_spectral_group": {
            "param:self": "F6",
            "self._normalize_field_indices": "R-0093 G6",
            "self._transform_group_indexer": "F6",
            "self.spectral": "F6 (-> tensor)",
        },
        "_store_group": {
            "param:self": "F6",
            "self._normalize_field_indices": "R-0093 G6",
            "self._transform_group_indexer": "F6",
        },
        "store_spatial_group": {
            "param:self": "F6",
            "self._store_group": "F6",
            "self.spatial": "F6 (-> tensor)",
        },
        "store_spectral_group": {
            "param:self": "F6",
            "self._store_group": "F6",
            "self.spectral": "F6 (-> tensor)",
        },
    }
    # names that stay available unchanged in the new context (moved with the unit or kept modules)
    KEPT = {
        "np",
        "torch",
        "math",
        "Q_COMPONENTS",
        "VELOCITY_COMPONENTS",
        "PlaneObservation",
        "PlaneDiagnostic",
        "PlaneBerisEdwardsRunSpec",
        "PlaneWorkflowCheckpoint",
        "PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
        "PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION",
        "build_plane_checkpoint_compatibility_identity",
        "_validate_plane_checkpoint_embedded_identity",
        "_lifting_compatibility_identity",
        "CheckpointCompatibilityError",
        "require_finite_tensors",
        "decompose_plane_beris_edwards_run_spec",
        "trange",
        "Sequence",
        "Mapping",
    }
    # attribute chains on non-adapter parameters that are data (run_spec, checkpoint, fields-as-FieldView)
    DATA_PARAMS = {
        "run_spec",
        "checkpoint",
        "argv",
        "step",
        "viscosity",
        "friction",
        "keys",
        "values",
        "destination",
        "indices",
        "key",
        "name",
    }

    def find(rel, cls, name):
        text = (ROOT / "pssolver" / rel).read_text()
        tree = ast.parse(text)
        body = tree.body
        if cls:
            body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
        return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name), text

    report, residual_total = {}, 0
    for rel, cls, name in TARGETS:
        fn, text = find(rel, cls, name)
        params = [a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs]
        assigned = {
            n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        for n in ast.walk(fn):
            if isinstance(n, ast.comprehension):
                assigned |= {t.id for t in ast.walk(n.target) if isinstance(t, ast.Name)}
            if isinstance(n, ast.ExceptHandler) and n.name:
                assigned.add(n.name)
            if isinstance(n, (ast.FunctionDef, ast.Lambda)) and n is not fn:
                assigned |= {a.arg for a in n.args.args + n.args.kwonlyargs + n.args.posonlyargs}
                if isinstance(n, ast.FunctionDef):
                    assigned.add(n.name)
        deps = set()
        # attribute chains rooted at parameters (maximal prefix up to the first non-attribute)
        chain_nodes = set()
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Attribute):
                parts, cur = [], sub
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id in params:
                    deps.add(".".join([cur.id] + list(reversed(parts))))
                    chain_nodes.add(id(cur))
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id in params and id(sub) not in chain_nodes and sub.id not in DATA_PARAMS:
                    deps.add(f"param:{sub.id}")
                elif sub.id not in params and sub.id not in assigned and sub.id not in BUILTINS:
                    deps.add(f"free:{sub.id}")
        # keep maximal chains only
        deps = {d for d in deps if not any(o != d and o.startswith(d + ".") for o in deps)}
        rows = []
        auth = AUTH.get(name, {})
        for d in sorted(deps):
            tag = None
            for key, value in auth.items():
                if d == key or d.startswith(key + "."):
                    tag = value
                    break
            if tag is None:
                root = d.split(":", 1)[-1].split(".")[0]
                if d.startswith("free:") and root in KEPT:
                    tag = "KEPT"
                elif not d.startswith(("free:", "param:")) and root in DATA_PARAMS | {
                    "fields",
                    "integrator",
                    "final_diagnostic",
                    "components",
                    "result",
                }:
                    tag = "KEPT (data/FieldView or local)"
                elif d.startswith("free:") and root == name:
                    tag = "KEPT (recursion)"
            if tag is None:
                tag = "RESIDUAL"
                residual_total += 1
            rows.append(dict(dependency=d, covered_by=tag))
        report[f"{rel}::{(cls + '.') if cls else ''}{name}"] = dict(
            lines=[fn.lineno, fn.end_lineno], params=params, dependencies=rows
        )
    OUT.write_text(json.dumps(dict(residual=residual_total, functions=report), indent=1) + "\n")
    for key, value in report.items():
        print(f"== {key} {value['lines']}")
        for row in value["dependencies"]:
            print(f"   {row['covered_by'][:60]:60} {row['dependency']}")
    print("RESIDUAL TOTAL:", residual_total)
    sys.exit(1 if residual_total else 0)


if __name__ == "__main__":
    main()
