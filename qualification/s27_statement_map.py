"""Verify the closed R-0091/R-0093 substitutions against reachable source ASTs."""

import argparse
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path


SOURCE_PATHS = json.loads(Path(__file__).with_name("s27_impact_targets.json").read_text())[
    "statement_sources"
]

BASE = "327a192ba879da0991110eca99f4b69b9a45c84e"


def owned_statements(root):
    """Pair unchanged inline statements; expose every reviewed replacement."""
    old_text = subprocess.check_output(
        ["git", "show", BASE + ":pssolver/applications/plane_beris_edwards.py"], cwd=root
    ).decode()
    new_text = (root / "pssolver/runtime/legacy_entry.py").read_text()
    before = function(old_text, "_run_plane_beris_edwards_owned").body
    after = function(new_text, "_run_v04_plane_translation_owned").body
    manifest = json.loads((root / "qualification/s27_statement_map.json").read_text())
    rows, used, violations = [], set(), []
    for index, statement in enumerate(before):
        dump = ast.dump(statement)
        matches = [i for i, n in enumerate(after) if ast.dump(n) == dump and i not in used]
        if matches:
            target = matches[0]
            used.add(target)
            rows.append(dict(source_index=index, target_index=target, disposition="exact"))
        else:
            entry = manifest["source"].get(str(index))
            digest = hashlib.sha256(dump.encode()).hexdigest()
            if not entry or entry["ast_sha256"] != digest:
                violations.append(
                    dict(side="source", index=index, statement=ast.unparse(statement))
                )
            rows.append(dict(source_index=index, disposition=entry))
    for index, statement in enumerate(after):
        if index in used:
            continue
        entry = manifest["candidate"].get(str(index))
        digest = hashlib.sha256(ast.dump(statement).encode()).hexdigest()
        if not entry or entry["ast_sha256"] != digest:
            violations.append(dict(side="candidate", index=index, statement=ast.unparse(statement)))
        rows.append(dict(target_index=index, disposition=entry))
    for side, valid in [
        (
            "source",
            {
                r["source_index"]
                for r in rows
                if "source_index" in r and r["disposition"] != "exact"
            },
        ),
        (
            "candidate",
            {
                r["target_index"]
                for r in rows
                if "target_index" in r and r["disposition"] != "exact"
            },
        ),
    ]:
        if set(map(int, manifest[side])) != valid:
            violations.append(dict(side=side, reason="stale replacement registration"))
    return dict(rows=rows, violations=violations, passed=not violations)


def function(text, name):
    body = ast.parse(text).body
    for part in name.split("."):
        node = next(n for n in body if getattr(n, "name", None) == part)
        body = node.body
    return node


def rewrite(body, expressions=(), statements=()):
    expressions = {
        ast.dump(ast.parse(a, mode="eval").body): ast.parse(b, mode="eval").body
        for a, b in expressions
    }
    statements = {ast.dump(ast.parse(a).body[0]): ast.parse(b).body for a, b in statements}

    class Replace(ast.NodeTransformer):
        def visit(self, node):
            key = ast.dump(node)
            if key in expressions:
                return copy.deepcopy(expressions[key])
            if key in statements:
                return copy.deepcopy(statements[key])
            return super().visit(node)

    tree = ast.Module(body=copy.deepcopy(body), type_ignores=[])
    return Replace().visit(tree).body


def check(root):
    rows = []

    def source(path, name):
        text = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=root).decode()
        return function(text, name).body

    def target(path, name):
        return function((root / path).read_text(), name).body

    def compare(label, expected, actual, authority):
        before, after = [ast.dump(n) for n in expected], [ast.dump(n) for n in actual]
        rows.append(
            dict(
                unit=label,
                authority=authority,
                equal=before == after,
                expected=before,
                actual=after,
            )
        )

    observation = SOURCE_PATHS["source_0"]
    observations = "pssolver/runtime/plane_observations.py"
    old_check = 'if not isinstance(adapter, PlaneRuntimeAdapterProtocol):\n raise TypeError("adapter must implement PlaneRuntimeAdapterProtocol")'
    engine_check = 'if not isinstance(engine, Engine):\n raise TypeError("engine must be an Engine")\nif not isinstance(state, State):\n raise TypeError("state must be a State")'
    compare(
        "capture_plane_observation",
        rewrite(
            source(observation, "capture_plane_observation"),
            [
                ("adapter.completed_steps if step is None else step", "step"),
                ("adapter.solver.batchsize", "fields.spatial.shape[1]"),
                ("plane_physical_component(adapter, name)", "fields[name]"),
            ],
            [
                (
                    old_check,
                    'if not isinstance(fields, FieldView):\n raise TypeError("fields must be a FieldView")',
                ),
                ("if synchronize:\n adapter.synchronize_for_observation()", ""),
                ("fields = adapter.fields", ""),
            ],
        ),
        target(observations, "capture_plane_observation"),
        "C1-C6",
    )
    compare(
        "capture_plane_diagnostic",
        rewrite(
            source(observation, "capture_plane_diagnostic"),
            [
                (
                    "adapter.projected_normal_force()",
                    "_projected_normal_force(constraint_diagnostics)",
                ),
                ("adapter.flow_diagnostics()", "_flow_diagnostics(constraint_diagnostics)"),
            ],
            [("fields = adapter.fields", "")],
        ),
        target(observations, "capture_plane_diagnostic"),
        "C4/C7/C8",
    )
    adapter = SOURCE_PATHS["source_1"]
    compare(
        "projected_normal_force",
        rewrite(
            source(adapter, "LegacyPlaneRuntimeAdapter.projected_normal_force"),
            [
                (
                    "self._solver.model.static_model.last_projected_normal_force",
                    'constraint_diagnostics.get("last_projected_normal_force")',
                ),
            ],
        ),
        target(observations, "_projected_normal_force"),
        "C7",
    )
    compare(
        "flow_diagnostics",
        rewrite(
            source(adapter, "LegacyPlaneRuntimeAdapter.flow_diagnostics"),
            [
                ("model." + name, "constraint_diagnostics[" + repr(name) + "]")
                for name in (
                    "last_pressure_iterations",
                    "last_pressure_residual",
                    "last_pressure_relative_residual",
                )
            ],
            [("model = self._solver.model.static_model", "")],
        ),
        target(observations, "_flow_diagnostics"),
        "C8",
    )
    checkpoint = SOURCE_PATHS["source_2"]
    expr = [
        ("adapter.completed_steps", "state.completed_steps"),
        ("adapter.runtime_path", "run_spec.runtime_path"),
        ("adapter.fields", "engine.field_view(state)"),
        ("adapter.backend_restart_metadata()", "backend_restart"),
        ("plane_lifting_restart_metadata(adapter)", "None"),
        ("plane_lifting_fresh_construction_provenance(adapter)", "None"),
        ("adapter.synchronize_for_observation()", "engine.prepare_constraints_(state)"),
        ("integrator.spectral_refresh_interval", "engine.plan.integrator.refresh_interval"),
        ("integrator.step_count", "state.refresh_phase[0]"),
        ("integrator.refresh_count", "state.refresh_phase[1]"),
    ]
    compare(
        "capture_plane_checkpoint",
        rewrite(
            source(checkpoint, "capture_plane_checkpoint"),
            expr,
            [(old_check, engine_check), ("integrator = adapter.solver.integrator", "")],
        ),
        target("tests/v0_5_s27_plane_v04_checkpoint.py", "plane_v04_checkpoint_from_state"),
        "F2a/G1-G3",
    )

    # Splitting validation from publication is F2b. Every original validation statement is retained.
    restore = source(checkpoint, "restore_plane_checkpoint")
    validation = []
    for statement in restore[2:]:
        if isinstance(statement, ast.AnnAssign):
            break
        if isinstance(statement, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"fields", "target_spatial", "target_spectral"}
            for t in statement.targets
        ):
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "verify_plane_lifting_identity"
        ):
            continue
        validation.append(statement)
    current_validation = target("pssolver/configuration/legacy_plane.py", "validate_plane_restore")
    compare(
        "restore_plane_checkpoint.validation",
        rewrite(
            validation,
            [
                ("adapter.runtime_path", "run_spec.runtime_path"),
                ("plane_lifting_restart_metadata(adapter)", "None"),
                ("adapter.backend_restart_metadata()", "backend_restart"),
            ],
        ),
        current_validation[1:],
        "F2b/G1/G2: docstring describes split; input data parameters replace adapter",
    )
    start = next(i for i, n in enumerate(restore) if isinstance(n, ast.AnnAssign))
    publication = rewrite(
        restore[start:],
        [("adapter.completed_steps", "state.completed_steps")],
        [
            (
                "adapter.synchronize_for_observation()",
                "state.constrained_current = False\nengine.prepare_constraints_(state, diagnostics=diagnostics)",
            ),
            (
                "adapter.restore_progress(completed_steps=checkpoint.completed_steps, spectral_refresh_interval=checkpoint.spectral_refresh_interval, integrator_step_count=checkpoint.integrator_step_count, integrator_refresh_count=checkpoint.integrator_refresh_count)",
                "_restore_plane_progress(state, engine.plan.integrator.refresh_interval, checkpoint.completed_steps, checkpoint.spectral_refresh_interval, checkpoint.integrator_step_count, checkpoint.integrator_refresh_count)",
            ),
        ],
    )
    current_publication = target("pssolver/runtime/legacy_entry.py", "restore_plane_state")
    offset = next(i for i, n in enumerate(current_publication) if isinstance(n, ast.AnnAssign))
    compare(
        "restore_plane_checkpoint.publication",
        publication,
        current_publication[offset:],
        "F2b/G4: same transfer-before-copy order",
    )
    application = SOURCE_PATHS["source_3"]
    compare(
        "main",
        rewrite(
            source(application, "main"), [("run_plane_beris_edwards", "run_v04_plane_translation")]
        ),
        target("pssolver/cli/plane.py", "main"),
        "F5a",
    )
    compare(
        "_print_workflow_summary",
        source(application, "_print_workflow_summary"),
        target("pssolver/cli/plane.py", "_print_workflow_summary"),
        "F5b/G5",
    )
    mappings = [
        ("self.name_to_idx", "name_to_idx"),
        ("self._get_field_index(key)", "_get_field_index(key, name_to_idx)"),
        ("self._normalize_field_key(key)", "_normalize_field_key(key, name_to_idx)"),
        (
            "self._normalize_field_indices(keys)",
            "_normalize_field_indices(keys, name_to_idx, component_count)",
        ),
        ("len(self.boundary_conditions)", "component_count"),
        ("self.transform_group_indexing", "indexing"),
        ("self._transform_group_indexer(indices)", "_transform_group_indexer(indices, indexing)"),
        ("self.spatial", "tensor"),
        ("self.spectral", "tensor"),
    ]
    for old, new in [
        ("_get_field_index", "_get_field_index"),
        ("_normalize_field_key", "_normalize_field_key"),
        ("_normalize_field_indices", "_normalize_field_indices"),
        ("_transform_group_indexer", "_transform_group_indexer"),
        ("select_spatial_group", "select_group"),
        ("select_spectral_group", "select_group"),
        ("_store_group", "store_group"),
    ]:
        compare(
            "Fields." + old,
            rewrite(source("pssolver/Field.py", "Fields." + old), mappings),
            target("pssolver/engine/groups.py", new),
            "F6/G6",
        )
    # Old store wrappers add no operation: their sole delegated body is the verified _store_group.
    for old in ("store_spatial_group", "store_spectral_group"):
        before = source("pssolver/Field.py", "Fields." + old)
        expected = ast.parse(
            "self._store_group(self."
            + ("spatial" if "spatial" in old else "spectral")
            + ", keys, values)"
        ).body
        compare(
            "Fields." + old,
            expected,
            before,
            "F6: one-call wrapper eliminated; store_group body checked above",
        )
    compare(
        "run_plane_beris_edwards",
        rewrite(
            source(application, "run_plane_beris_edwards"),
            [("_run_plane_beris_edwards_owned", "_run_v04_plane_translation_owned")],
        ),
        target("pssolver/runtime/legacy_entry.py", "run_v04_plane_translation"),
        "R-0091 F4: unchanged TF32/dry-run wrapper with renamed entry",
    )
    owned = owned_statements(root)
    return dict(
        rows=rows,
        owned=owned,
        violations=[row["unit"] for row in rows if not row["equal"]] + owned["violations"],
        passed=all(row["equal"] for row in rows) and owned["passed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = check(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], violations=report["violations"])))
    raise SystemExit(not report["passed"])
