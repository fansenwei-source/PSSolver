"""R-0095: baseline re-exports, frozen references and final binding order."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest

from qualification import s27_records
from qualification.s27_reexports import resolve_reexport
from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("package,name,module", [
    ("pssolver.runtime", "LegacyPlaneRuntimeAdapter", "pssolver.runtime.plane_beris_edwards"),
    ("pssolver.workflows", "PlaneWorkflowCheckpoint", "pssolver.workflows.plane_checkpoint"),
    ("pssolver.runtime", "build_legacy_plane_runtime", "pssolver.runtime.plane_legacy"),
])
def test_baseline_root_reexports_resolve_even_after_candidate_retirement(package, name, module):
    assert resolve_reexport(package, name) == module
    assert resolve_reexport(package, "unregistered_name") == package


@pytest.mark.parametrize("module", [
    "s27_impact", "s27_metadata_classes", "s27_move_check_source", "s27_rewrite_closure_source",
])
def test_analysis_scripts_import_without_reading_arguments(module, monkeypatch):
    monkeypatch.setattr("sys.argv", ["no_arguments"])
    imported = importlib.import_module("qualification." + module)
    assert callable(imported.main)


@pytest.mark.parametrize("mutation", ["none", "historical_line", "changed_line", "not_frozen", "new_python_reference", "registry_list", "package", "json"])
def test_r2_retirement_categories_fail_closed(tmp_path, monkeypatch, mutation):
    directory = tmp_path / "qualification"
    directory.mkdir()
    target = "pssolver.example_retired"
    content = 'source = "pssolver/example_retired.py"\n'
    file = tmp_path / "frozen.py"
    file.write_text(content)
    monkeypatch.setattr(s27_records, "baseline_bytes", lambda root, relative: content.encode() if relative == "frozen.py" else None)
    table = dict(retired_modules=[target], guard_files=[], own_file_patterns=[], registry_files=[])
    monkeypatch.setattr(s27_records, "authority", lambda: table)
    registry = dict(modules=[dict(module=target)], registry_files=sorted(s27_records.REGISTRY_FILES))
    if mutation == "historical_line":
        file.write_text(content + "\n")
    elif mutation == "changed_line":
        file.write_text("other_" + content)
    elif mutation == "not_frozen":
        monkeypatch.setattr(s27_records, "baseline_bytes", lambda root, relative: None)
    elif mutation == "new_python_reference":
        (tmp_path / "extra.py").write_text(content)
    elif mutation == "registry_list":
        registry["registry_files"].append("arbitrary.json")
    elif mutation == "package":
        (tmp_path / "pssolver").mkdir()
        (tmp_path / "pssolver/extra.py").write_text(content)
    elif mutation == "json":
        (tmp_path / "extra.json").write_text(json.dumps({"module": target}))
    monkeypatch.setattr(s27_records, "retirement", lambda: registry)
    report = s27_records.retirement_references(tmp_path)
    assert report["passed"] is (mutation in {"none", "historical_line"}), report


def test_plane_reference_is_from_v04_and_has_all_sixty_nine_filename_keys():
    record = json.loads((ROOT / "tests/baselines/s27/plane_comparison.json").read_text())
    frozen = json.loads((ROOT / "tests/baselines/v0_4/trajectories.json").read_text())
    expected = {name for name, row in frozen["cases"].items() if row["scene"]["family"] == "plane"}
    assert len(expected) == 69
    assert set(record["cases"]) == expected
    assert record["source_commit"] == "4c12228aba01e8483e80d28a1471ab72384d5fda"
    assert record["fingerprint"] == frozen["fingerprint"]
    for relative, sha in record["source_files"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolve_s03_successor(
            relative, sha
        )


def test_successor_146_is_generated_after_all_registry_files():
    record = json.loads((ROOT / s27_records.BINDING_RECORD).read_text())
    registry = s27_records.retirement()
    assert set(registry["registry_files"]) == s27_records.REGISTRY_FILES
    assert not any(row["file"] in s27_records.REGISTRY_FILES | {s27_records.BINDING_RECORD} for row in registry["provenance_references"])
    for path in s27_records.REGISTRY_FILES:
        assert record["introduced_sources"][path] == hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
