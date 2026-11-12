"""Development version, lazy visualization and source-distribution contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import pssolver
from tools.check_distributions import scan


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_matches_the_active_development_record():
    active = json.loads((ROOT / "notes/pssolver_active_version_record.json").read_text())
    record = json.loads((ROOT / active["record"]).read_text())
    assert pssolver.__version__ == record[active["version_field"]]


def test_manifest_is_an_explicit_allowlist_for_only_shipped_sources():
    lines = (ROOT / "MANIFEST.in").read_text().splitlines()
    commands = [line for line in lines if line and not line.startswith("#")]
    assert commands.pop(0) == "global-exclude *"
    assert all(line.startswith("include ") and "*" not in line for line in commands)
    paths = [line.removeprefix("include ") for line in commands]
    assert len(paths) == len(set(paths))
    expected = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "pssolver").rglob("*.py")
        if not path.is_relative_to(ROOT / "pssolver/control")
        and not (
            path.parent == ROOT / "pssolver/experimental"
            and (path.name.startswith("stage_") or path.stem == "h100_shadow_qualification")
        )
    } | {
        "README.md",
        "CHANGELOG.md",
        "pyproject.toml",
        "setup.py",
        "MANIFEST.in",
        "Plane_beris_edwards_stokes.py",
        "docs/migration_0_4_to_0_5.md",
    }
    assert set(paths) == expected
    assert not any(
        p.startswith(("tests/", "notes/", "benchmarks/", "qualification/")) for p in paths
    )


def test_pygame_is_lazy_and_missing_dependency_has_actionable_error():
    source = """
import importlib.abc
import sys
class NoPygame(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pygame" or fullname.startswith("pygame."):
            raise ModuleNotFoundError("blocked pygame")
sys.meta_path.insert(0, NoPygame())
from pssolver.visualizer import Visualizer
assert "pygame" not in sys.modules
v = Visualizer()
assert "pygame" not in sys.modules
for method, args in ((v.run_pygame, ()), (v.animate_pygame, (None,))):
    try:
        method(*args)
    except ImportError as e:
        assert "pssolver[visualization]" in str(e)
    else:
        raise AssertionError("missing pygame was accepted")
"""
    subprocess.run([sys.executable, "-B", "-c", source], cwd=ROOT, check=True)


def test_only_the_two_confirmed_changelog_lines_are_privacy_exceptions():
    policy = json.loads((ROOT / "qualification/package_policy.json").read_text())
    exceptions = policy["private_line_exceptions"]
    assert len(exceptions) == 2
    assert [item["original_line"] for item in exceptions] == [349, 376]
    assert {item["decision"] for item in exceptions} == {"D-0043"}
    content = (ROOT / "CHANGELOG.md").read_bytes()
    assert scan({"CHANGELOG.md": content}, exceptions)["passed"]
    assert not scan({"CHANGELOG.md": content + b"new /scratch entry\n"}, exceptions)["passed"]
    assert not scan({"README.md": b"/scratch1\n"}, exceptions)["passed"]
    line = exceptions[0]["line"].encode()
    assert content.count(line) == 1
    assert not scan({"CHANGELOG.md": content.replace(line, line + b" edit")}, exceptions)["passed"]
    assert not scan({"CHANGELOG.md": content + line + b"\n"}, exceptions)["passed"]


@pytest.mark.parametrize(
    "token", ["/home/", "fansenwei", "/scratch", "caspar", "PSSolver-Control", "pssolver_control"]
)
def test_private_text_in_package_code_is_always_rejected(token):
    assert not scan({"pssolver/example.py": token.encode()}, [])["passed"]
