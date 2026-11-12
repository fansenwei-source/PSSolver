"""R-0096 exact owner locks and the narrow CLI boundary."""

import json
from pathlib import Path

from qualification.s27_r2_support import edges, owner_violations


ROOT = Path(__file__).resolve().parents[1]


def registry():
    record = json.loads((ROOT / "qualification/s27_r2_tables.json").read_text())
    row, = [
        row for row in record["edges"]
        if (row["source"], row["destination"]) == ("pssolver.runtime.legacy_entry", "pssolver.engine.state")
    ]
    assert row["owners"] == ["restore_plane_state"]
    # R-0107 rehomed this single Channel owner without opening the edge.
    row["owners"].append("restore_channel_state")
    return record


def test_ratchet_edges_are_exactly_the_authoritative_table():
    import ast

    tree = ast.parse((ROOT / "tests/test_architecture_import_boundaries.py").read_text())
    value = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "REVIEWED_S27_EDGES")
    actual = {tuple(ast.literal_eval(arg) for arg in call.args) for call in value.elts}
    assert actual == {(row["source"], row["destination"]) for row in registry()["edges"]}


def test_owner_scanner_sees_another_unit_using_the_same_import(tmp_path):
    package = tmp_path / "pssolver"
    (package / "cli").mkdir(parents=True)
    (package / "runtime").mkdir()
    (package / "runtime/runner.py").write_text("class RunResult: pass\n")
    (package / "cli/plane.py").write_text(
        "from pssolver.runtime.runner import RunResult\n"
        "def _print_workflow_summary(value: RunResult): pass\n"
        "def unauthorized(value: RunResult): pass\n"
    )
    observed = edges(tmp_path)
    assert observed[("pssolver.cli.plane", "pssolver.runtime.runner")] == {
        "_print_workflow_summary", "unauthorized",
    }
    assert any(row["source"] == "pssolver.cli.plane" and row["destination"] == "pssolver.runtime.runner"
               for row in owner_violations(observed, registry()))


def test_library_never_imports_cli():
    assert not [(source, target) for source, target in edges(ROOT)
                if target.startswith("pssolver.cli") and not source.startswith("pssolver.cli")]


def test_plane_observation_exact_moves_keep_original_unit_names():
    moves = json.loads((ROOT / "qualification/s27_exact_moves.json").read_text())
    names = {row["unit"] for row in moves["exact_moves"]
             if row["target"] == "pssolver/runtime/plane_observations.py"}
    assert {"PlaneObservation", "PlaneDiagnostic", "DIAGNOSTIC_DTYPE", "DIAGNOSTIC_HEADER",
            "_atomic_save", "write_plane_observation", "write_plane_diagnostics"} <= names
