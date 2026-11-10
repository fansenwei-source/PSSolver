"""Exact forward provenance, retirement, constructor and record-reader guards."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

from qualification import s26_records
from qualification.s26_guards import planning_closure
from qualification.s26_records import reader_edges, record_edges, retirement, retirement_references
from tools import check


ROOT = Path(__file__).resolve().parents[1]


def test_retired_names_occur_only_at_registered_locations():
    report = retirement_references(ROOT)
    assert report["passed"], report
    assert len(retirement()["provenance_references"]) == 11


@pytest.mark.parametrize("mutation", [
    "none", "unregistered", "missing_pointer", "file_hash", "missing_file", "duplicate",
    "kind", "python_file", "wildcard", "python_reference", "unregistered_python_guard",
])
def test_provenance_references_are_exact_and_fail_closed(tmp_path, monkeypatch, mutation):
    module = retirement()["modules"][0]["module"]
    file = tmp_path / "record.json"
    file.write_text(json.dumps({"retired": module}))
    entry = dict(file="record.json", pointer="/retired", value=module,
                 file_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), kind="record_string")
    registry = dict(modules=[dict(module=module)], guard_references=[], provenance_references=[entry])
    if mutation == "unregistered":
        (tmp_path / "extra.json").write_text(json.dumps({"retired": module}))
    elif mutation == "missing_pointer":
        entry["pointer"] = "/not_present"
    elif mutation == "file_hash":
        file.write_text(file.read_text() + "\n")
    elif mutation == "missing_file":
        file.unlink()
    elif mutation == "duplicate":
        registry["provenance_references"].append(dict(entry))
    elif mutation == "kind":
        entry["kind"] = "dotted_string"
    elif mutation == "python_file":
        entry["file"] = "record.py"
        (tmp_path / "record.py").write_text("value = " + repr(module) + "\n")
    elif mutation == "wildcard":
        entry["file"] = "*.json"
    elif mutation in {"python_reference", "unregistered_python_guard"}:
        (tmp_path / "extra.py").write_text("value = " + repr(module) + "\n")
        if mutation == "unregistered_python_guard":
            registry["provenance_references"].append(dict(
                file="extra.py", pointer="/value", value=module, kind="record_string",
                file_sha256=hashlib.sha256((tmp_path / "extra.py").read_bytes()).hexdigest()))
    monkeypatch.setattr(s26_records, "retirement", lambda: registry)
    report = retirement_references(tmp_path)
    assert report["passed"] is (mutation == "none"), report


def test_physical_source_bindings_cover_all_existing_logical_aliases():
    record = json.loads((ROOT / "notes/PSSolver_v0_3_0_s03_successor_145_s26_assembler.json").read_text())
    sources = record["historical_successor_sources"]
    previous = [json.loads(path.read_text()) for path in (ROOT / "notes").glob("PSSolver_v0_3_0_s03_successor_*.json")
                if path.name != "PSSolver_v0_3_0_s03_successor_145_s26_assembler.json"]
    reverse = {}
    for historical in previous:
        for alias, binding in historical.get("historical_successor_sources", {}).items():
            physical = binding.get("current_path")
            if physical in sources and binding["current_sha256"] == sources[physical]["historical_sha256"]:
                reverse.setdefault(physical, set()).add(alias)
    assert sum(map(len, reverse.values())) == 2
    for physical, aliases in reverse.items():
        for alias in aliases:
            assert sources[alias]["current_path"] == physical
            assert sources[alias]["historical_sha256"] == sources[physical]["historical_sha256"]
            assert sources[alias]["current_sha256"] == sources[physical]["current_sha256"]
    for relative, binding in sources.items():
        path = ROOT / binding.get("current_path", relative)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["current_sha256"]


def test_exact_moves_preserve_every_frozen_source_unit_and_hash():
    old = json.loads((ROOT / "qualification/s25_exact_moves.json").read_text())
    new = json.loads((ROOT / "qualification/s26_exact_moves.json").read_text())
    assert old["source_commit"] == new["source_commit"]
    assert [{k: v for k, v in row.items() if k != "target_lines"} for row in new["exact_moves"]] == [
        {k: v for k, v in row.items() if k != "target_lines"} for row in old["exact_moves"]]
    assert sum(map(len, check.exact_assemblies().values())) == len(old["exact_moves"])


def test_exact_move_guard_rejects_frozen_hash_change(tmp_path, monkeypatch):
    destination = tmp_path / "qualification"
    destination.mkdir()
    for name in ("s25_exact_moves.json", "s26_exact_moves.json"):
        (destination / name).write_bytes((ROOT / "qualification" / name).read_bytes())
    newer = json.loads((destination / "s26_exact_moves.json").read_text())
    newer["exact_moves"][0]["sha256"] = "0" * 64
    (destination / "s26_exact_moves.json").write_text(json.dumps(newer))
    monkeypatch.setattr(check, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="source, unit or frozen hash"):
        check.exact_assemblies()


def test_record_references_are_attributed_to_readers_not_unrelated_tests(tmp_path):
    target = retirement()["modules"][0]["module"]
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "fixture.json").write_text(json.dumps({"file": target.replace(".", "/") + ".py"}))
    tree = ast.parse('REC = "notes/fixture.json"\ndef read():\n return REC\n'
                     'def test_reads():\n read()\ndef test_unrelated():\n pass\n')
    edges = record_edges(tmp_path, {target})
    result = reader_edges(tree, edges)
    assert set(result) == {"test_reads"}
    assert result["test_reads"][0]["kind"] == "record_string"
    assert result["test_reads"][0]["target"] == target


def test_static_closure_rejects_nested_tensor_import(tmp_path):
    report = planning_closure(ROOT)
    for module in report["modules"]:
        relative = module.replace(".", "/") + ".py"
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    path = tmp_path / "pssolver/planning/term_analysis.py"
    path.write_text(path.read_text() + "\ndef forbidden():\n    import torch\n")
    assert any(row.get("target") == "torch" for row in planning_closure(tmp_path)["violations"])
