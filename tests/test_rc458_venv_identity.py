"""Regression tests for RC4.5.8 installed-venv identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.rc458_venv_identity import (
    lexical_python_path,
    lexical_venv_root,
    validate_installed_venv_identity,
)
from benchmarks.run_rc458_profile_matrix_venv_recovery import main as matrix_main


def _layout(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    base = tmp_path / "base" / "bin" / "python3.10"
    base.parent.mkdir(parents=True)
    base.write_bytes(b"base interpreter")
    root = tmp_path / "qualified-venv"
    (root / "bin").mkdir(parents=True)
    (root / "pyvenv.cfg").write_text("home = /base\n", encoding="utf-8")
    python = root / "bin" / "python"
    python.symlink_to(base)
    purelib = root / "lib" / "python3.10" / "site-packages"
    package = purelib / "pssolver" / "__init__.py"
    package.parent.mkdir(parents=True)
    package.write_text("", encoding="utf-8")
    return python, root, purelib, package


def test_python_identity_preserves_symlinked_venv_entry_point(tmp_path):
    python, root, purelib, package = _layout(tmp_path)

    report = validate_installed_venv_identity(
        requested_python=python,
        imported_module=package,
        runtime_prefix=root,
        purelib=purelib,
        reported_executable=python,
    )

    assert lexical_python_path(python) == python.absolute()
    assert lexical_venv_root(python) == root.absolute()
    assert Path(report["requested_python"]) == python.absolute()
    assert report["requested_python_is_symlink"] is True
    assert Path(report["resolved_python_target"]) != python.absolute()
    assert Path(report["resolved_python_target"]) == python.resolve()
    assert Path(report["lexical_venv_root"]) == root.absolute()
    assert report["passed"] is True


def test_resolved_interpreter_parent_is_not_used_as_the_venv_root(tmp_path):
    python, root, purelib, package = _layout(tmp_path)

    buggy_root = python.resolve().parent.parent
    report = validate_installed_venv_identity(
        requested_python=python,
        imported_module=package,
        runtime_prefix=root,
        purelib=purelib,
    )

    assert buggy_root != root.resolve()
    assert Path(report["canonical_venv_root"]) == root.resolve()


def test_import_outside_requested_venv_is_rejected(tmp_path):
    python, root, purelib, _package = _layout(tmp_path)
    outside = tmp_path / "source" / "pssolver" / "__init__.py"
    outside.parent.mkdir(parents=True)
    outside.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the requested venv purelib"):
        validate_installed_venv_identity(
            requested_python=python,
            imported_module=outside,
            runtime_prefix=root,
            purelib=purelib,
        )


def test_runtime_prefix_mismatch_is_rejected_before_import_ownership(tmp_path):
    python, _root, purelib, package = _layout(tmp_path)
    other = tmp_path / "other-venv"
    other.mkdir()

    with pytest.raises(ValueError, match="runtime prefix does not match"):
        validate_installed_venv_identity(
            requested_python=python,
            imported_module=package,
            runtime_prefix=other,
            purelib=purelib,
        )


def test_missing_venv_marker_is_rejected(tmp_path):
    python, root, _purelib, _package = _layout(tmp_path)
    (root / "pyvenv.cfg").unlink()

    with pytest.raises(ValueError, match="venv marker"):
        lexical_venv_root(python)


def test_profile_matrix_cli_preserves_both_requested_python_symlinks(tmp_path):
    base = tmp_path / "base" / "python3.10"
    base.parent.mkdir()
    base.write_bytes(b"base interpreter")
    baseline = tmp_path / "baseline" / "bin" / "python"
    candidate = tmp_path / "candidate" / "bin" / "python"
    baseline.parent.mkdir(parents=True)
    candidate.parent.mkdir(parents=True)
    baseline.symlink_to(base)
    candidate.symlink_to(base)
    command_plan = tmp_path / "command_plan.json"

    assert matrix_main(
        [
            "--baseline-python",
            str(baseline),
            "--candidate-python",
            str(candidate),
            "--baseline-wheel-sha256",
            "a" * 64,
            "--candidate-wheel-sha256",
            "b" * 64,
            "--support-root",
            str(tmp_path / "support"),
            "--output-root",
            str(tmp_path / "output"),
            "--command-plan",
            str(command_plan),
            "--plan-only",
        ]
    ) == 0

    plan = json.loads(command_plan.read_text(encoding="utf-8"))
    assert plan["baseline_python"] == str(baseline.absolute())
    assert plan["candidate_python"] == str(candidate.absolute())
    assert {row["python"] for row in plan["rows"]} == {
        str(baseline.absolute()),
        str(candidate.absolute()),
    }
    assert base.resolve() not in {Path(row["python"]) for row in plan["rows"]}
