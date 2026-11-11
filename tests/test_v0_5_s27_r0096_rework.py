"""R-0096 exact owner locks and the narrow CLI boundary."""

import copy
import json
from pathlib import Path

import pytest

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


def test_reviewed_edge_inventory_is_nine_and_rehomes_are_three():
    record = registry()
    assert len(record["edges"]) == 12
    assert sum(row["kind"] == "rehome" for row in record["edges"]) == 3
    assert all(row["authority"] and row["owners"] for row in record["edges"])
    assert owner_violations(edges(ROOT), record) == []


def test_ratchet_edges_are_exactly_the_authoritative_table():
    import ast

    tree = ast.parse((ROOT / "tests/test_architecture_import_boundaries.py").read_text())
    value = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "REVIEWED_S27_EDGES")
    actual = {tuple(ast.literal_eval(arg) for arg in call.args) for call in value.elts}
    assert actual == {(row["source"], row["destination"]) for row in registry()["edges"]}


@pytest.mark.parametrize("index", range(12))
@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_every_reviewed_owner_set_is_exact(index, mutation):
    record = registry()
    row = record["edges"][index]
    observed = copy.deepcopy(edges(ROOT))
    key = (row["source"], row["destination"])
    if mutation == "extra":
        observed[key].add("unreviewed_owner")
    else:
        observed[key].remove(row["owners"][0])
    failures = owner_violations(observed, record)
    assert len(failures) == 1
    assert (failures[0]["source"], failures[0]["destination"]) == key


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


def test_move_categories_are_exactly_the_r2_units_table():
    table = registry()
    moves = json.loads((ROOT / "qualification/s27_exact_moves.json").read_text())
    expected = {kind: set() for kind in ("exact", "rewritten", "new")}
    for module, entries in [*table["units"].items(), *table["added_units"].items()]:
        path = module.replace(".", "/") + ".py"
        if not (ROOT / path).is_file():
            path = module.replace(".", "/") + "/__init__.py"
        for unit, row in entries.items():
            expected[row["kind"]].add((path, unit))
    actual = {(row["target"], row["unit"]) for row in moves["exact_moves"]}
    inherited = json.loads((ROOT / "qualification/s26_exact_moves.json").read_text())
    inherited = {(row["target"], row["unit"]) for row in inherited["exact_moves"]}
    assert actual == inherited | expected["exact"]
    assert {(r["target"], r["unit"]) for r in moves["rewritten_units"]} == expected["rewritten"]
    assert {(r["target"], r["unit"]) for r in moves["new_units"]} == expected["new"]
    assert "rewritten" not in moves
