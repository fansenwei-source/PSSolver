"""R-0074 regression checks: narrow retirement and honest retained-unit evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from qualification.s24_source_audit import BASE, retained_changes, source_units
from tests.v0_3_s03_successor import resolve_s03_successor
from tools import check


ROOT = Path(__file__).resolve().parents[1]


def test_rework_is_forward_bound_without_rewriting_successor_141():
    record = json.loads(
        (ROOT / "notes/PSSolver_v0_3_0_s03_successor_142_s24_rework.json").read_text()
    )
    original = ROOT / "notes/PSSolver_v0_3_0_s03_successor_141_s24_retirement.json"
    assert (
        hashlib.sha256(original.read_bytes()).hexdigest() == record["unchanged_record_141_sha256"]
    )
    assert record["historical_records_rewritten"] is False
    for relative, binding in record["historical_successor_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolve_s03_successor(
            relative, binding["historical_sha256"]
        ), relative
    for relative, expected in record["introduced_sources"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolve_s03_successor(
            relative, expected
        ), relative


def test_retained_ast_equal_units_keep_the_exact_baseline_text():
    assert retained_changes(ROOT)["format_only"] == []


def test_every_changed_retained_test_unit_has_a_disposition_and_exact_history():
    actual = {(row["path"], row["unit"]) for row in retained_changes(ROOT, ("tests",))["semantic"]}
    dispositions = json.loads((ROOT / "qualification/s24_test_disposition.json").read_text())
    manifest = json.loads((ROOT / "qualification/historical_stage_tests.json").read_text())
    nodes = set(manifest["nodeids"])
    registered = set()
    for row in dispositions:
        for unit in row["adapted_retained_units"]:
            identity = (row["path"], unit["unit"])
            assert identity not in registered
            registered.add(identity)
            assert unit["change_type"] and unit["reason"]
            assert set(unit["historical_nodeids"]) <= nodes
            assert all(node.startswith(row["path"] + "::") for node in unit["historical_nodeids"])
        changed_names = {unit["unit"] for unit in row["adapted_retained_units"]}
        assert not changed_names.intersection(row["unchanged_current_assertion_functions"])
        for name in row["historical_only_test_functions"]:
            prefix = row["path"] + "::" + name
            assert any(node == prefix or node.startswith(prefix + "[") for node in nodes)
    assert registered == actual


def test_current_lifting_file_and_rc1_rejection_are_restored_byte_exact():
    relative = "tests/test_phase8_p844_lifting_workflow_restart.py"
    original = subprocess.check_output(["git", "show", BASE + ":" + relative], cwd=ROOT)
    assert (ROOT / relative).read_bytes() == original
    relative = "tests/test_v0_3_s08_historical_cpu_coverage.py"
    original = subprocess.check_output(["git", "show", BASE + ":" + relative], cwd=ROOT).decode()
    name = "test_authentic_rc1_lifted_plane_rejects_before_tensor_load"
    assert source_units((ROOT / relative).read_text())[name] == source_units(original)[name]


def test_private_lifting_fixture_is_retained_and_not_rejected_by_s24():
    from pssolver.io._retired_checkpoint import reject_retired_checkpoint

    relative = "tests/fixtures/v0_4_0_s24_retired/private_lifting/checkpoint.json"
    payload = (ROOT / relative).read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest()
        == "1263094dab60e6d5b39bd06eeef7050481ca689d7616b0b5deb0cd25b29f6ce3"
    )
    metadata = json.loads(payload)
    assert metadata["lifting_restart"] is not None
    reject_retired_checkpoint(metadata, operation="read")


def test_existing_lint_is_a_per_rule_ratchet_and_new_files_have_no_allowance(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "ROOT", tmp_path)
    (tmp_path / "old.py").write_bytes(b"candidate")
    (tmp_path / "new.py").write_bytes(b"new")
    monkeypatch.setattr(check.subprocess, "check_output", lambda *args, **kwargs: b"baseline")
    counts = {b"baseline": Counter(F401=2), b"candidate": Counter(F401=2), b"new": Counter()}
    monkeypatch.setattr(check, "lint_counts", lambda executable, path, source: counts[source])
    assert check.lint_ratchet("ruff", ["old.py", "new.py"], ["new.py"], "base")
    counts[b"candidate"] = Counter(F401=1, F841=1)
    assert not check.lint_ratchet("ruff", ["old.py"], [], "base")
    counts[b"new"] = Counter(F401=1)
    assert not check.lint_ratchet("ruff", ["new.py"], ["new.py"], "base")


def test_mypy_line_mapping_requires_identical_text_not_just_old_error_codes():
    old = "def f():\n    unused()\n    return value.missing\n"
    new = "def f():\n    return value.missing\n"
    assert check.unchanged_line_map(old, new)[2] == 3
    changed = "def f():\n    return other.missing\n"
    assert 2 not in check.unchanged_line_map(old, changed)
