"""R-0084: closed reference mapping and exact-move engineering ratchets."""

import hashlib
import json
import subprocess
from dataclasses import asdict

import pytest

from tests import v0_5_s24_registry_reference as reference
from tools import check


@pytest.mark.parametrize("missing", tuple(reference.S25_IDENTITY_MAP))
def test_periodic_mapping_rejects_each_missing_identity(monkeypatch, missing):
    monkeypatch.delitem(reference.S25_IDENTITY_MAP, missing)
    with pytest.raises(AssertionError, match="unmapped Periodic identity"):
        reference.retained_application_records()


def test_closed_mapping_changes_only_the_four_periodic_fields():
    original = reference.release_registry().APPLICATIONS
    fields = {"compiler", "request_type", "runtime_factory", "adapter_protocol"}
    for application in original:
        before = asdict(application)
        after = reference._forward_periodic_value(
            before, periodic=application.geometry_name == "periodic_box"
        )
        changed = {key for key in before if before[key] != after[key]}
        assert changed == (fields if application.geometry_name == "periodic_box" else set())
    with pytest.raises(AssertionError, match="unmapped"):
        reference._forward_periodic_value(
            "pssolver.runtime.periodic_beris_edwards.Unregistered", periodic=True
        )
    with pytest.raises(AssertionError, match="another application"):
        reference._forward_periodic_value(next(iter(reference.S25_IDENTITY_MAP)), periodic=False)


def _repo(tmp_path, monkeypatch):
    monkeypatch.setattr(check, "ROOT", tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "old.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "old.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Ratchet Test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "baseline"], cwd=tmp_path, check=True
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()


def test_exact_rename_uses_reachable_old_path(tmp_path, monkeypatch):
    base = _repo(tmp_path, monkeypatch)
    subprocess.run(["git", "mv", "old.py", "new.py"], cwd=tmp_path, check=True)
    assert check.changed_files(base) == (["new.py"], [], {"new.py": "old.py"})
    calls = []

    def counts(_tool, relative, source):
        calls.append((relative, source))
        return check.Counter({"F401": 1})

    monkeypatch.setattr(check, "lint_counts", counts)
    assert check.lint_ratchet("ruff", ["new.py"], [], base, renames={"new.py": "old.py"})
    assert calls == [("new.py", b"value = 1\n"), ("new.py", b"value = 1\n")]
    with pytest.raises(subprocess.CalledProcessError):
        check.lint_ratchet("ruff", ["new.py"], [], base, renames={"new.py": "missing.py"})


def test_truly_new_file_has_no_historical_baseline(tmp_path, monkeypatch):
    base = _repo(tmp_path, monkeypatch)
    (tmp_path / "new.py").write_text("new_value = 2\n")
    assert check.changed_files(base) == (["new.py"], ["new.py"], {})


def test_modified_existing_file_keeps_its_original_baseline(tmp_path, monkeypatch):
    base = _repo(tmp_path, monkeypatch)
    (tmp_path / "old.py").write_text("value = 2\n")
    assert check.changed_files(base) == (["old.py"], [], {})


def test_changed_rename_is_new_not_an_exact_move(tmp_path, monkeypatch):
    base = _repo(tmp_path, monkeypatch)
    subprocess.run(["git", "mv", "old.py", "new.py"], cwd=tmp_path, check=True)
    (tmp_path / "new.py").write_text("value = 2\n")
    assert check.changed_files(base) == (["new.py"], ["new.py"], {})


def test_assembly_allows_only_old_span_format_edits():
    original = "def old():\n    return  1\n\n\ndef new():\n    return  2\n"
    entries = [{"start": 1, "end": 2}]
    assert not check.assembly_format_changes(original, original.replace("return  1", "return 1"), entries)
    assert check.assembly_format_changes(original, original.replace("return  2", "return 2"), entries)
    assert check.assembly_format_changes(original, "# new\n" + original, entries)
    assert check.assembly_format_changes(original, original.replace("\n\ndef new", "\n# new\n\ndef new"), entries)


@pytest.mark.parametrize("actual", [[], [("F401", 3)], [("F841", 2)], [("F401", 2), ("F401", 4)]])
def test_assembly_lint_requires_exact_rule_and_relative_line(tmp_path, monkeypatch, actual):
    (tmp_path / "new.py").write_text("# source\n")
    monkeypatch.setattr(check, "ROOT", tmp_path)

    def diagnostics(_tool, path, _source):
        rows = [("F401", 11)] if path == "old.py" else actual
        return [{"code": code, "location": {"row": row}} for code, row in rows]

    monkeypatch.setattr(check, "lint_diagnostics", diagnostics)
    entry = dict(source="old.py", original="old", unit="old", old_start=10, old_end=12, start=1, end=3)
    assert not check.assembly_lint("ruff", "new.py", [entry])
    actual[:] = [("F401", 2)]
    assert check.assembly_lint("ruff", "new.py", [entry])


def test_registered_exact_unit_bytes_and_lines_are_enforced(tmp_path, monkeypatch):
    base = _repo(tmp_path, monkeypatch)
    source = "def old():\n    return  1\n"
    (tmp_path / "old.py").write_text(source)
    subprocess.run(["git", "add", "old.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Ratchet Test", "-c", "user.email=test@example.invalid",
         "commit", "-qm", "unit"], cwd=tmp_path, check=True
    )
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    (tmp_path / "new.py").write_text(source)
    (tmp_path / "qualification").mkdir()
    row = dict(source="old.py", target="new.py", unit="old", target_lines=[1, 2],
               sha256=hashlib.sha256(source.encode()).hexdigest())
    (tmp_path / "qualification/s25_exact_moves.json").write_text(
        json.dumps(dict(source_commit=base, exact_moves=[row]))
    )
    (tmp_path / "qualification/s26_exact_moves.json").write_text(
        json.dumps(dict(source_commit=base, exact_moves=[row]))
    )
    assert check.exact_assemblies()["new.py"][0]["unit"] == "old"
    (tmp_path / "new.py").write_text(source.replace("return  1", "return 1"))
    with pytest.raises(ValueError, match="altered exact move"):
        check.exact_assemblies()


def test_exact_moves_require_the_latest_manifest(tmp_path, monkeypatch):
    (tmp_path / "qualification").mkdir()
    (tmp_path / "qualification/s25_exact_moves.json").write_text(
        json.dumps(dict(source_commit="old", exact_moves=[]))
    )
    monkeypatch.setattr(check, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="s26_exact_moves.json"):
        check.exact_assemblies()


def test_tests_are_never_sent_to_the_formatter(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("test formatter invoked")

    monkeypatch.setattr(check, "run", forbidden)
    assert check.format_new_files("ruff", ["tests/new.py"], {})
