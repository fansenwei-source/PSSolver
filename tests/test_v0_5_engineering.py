"""Executable contracts for the source-only engineering tools."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._categories import history_available, load_categories
from tests.v0_5_release_manifest import released_manifest
from tools._source import imports
from tools.check_imports import check
from tools.structure_metrics import measure


ROOT = Path(__file__).resolve().parents[1]


def test_current_import_contracts_pass_without_importing_runtime():
    assert check(ROOT) == []
    source = (ROOT / "tools/check_imports.py").read_text()
    assert not any(
        isinstance(n, ast.Import) and any(a.name == "pssolver" for a in n.names)
        for n in ast.walk(ast.parse(source))
    )


@pytest.mark.parametrize(
    "statement,target",
    [
        ("import torch", "torch"),
        ("from torch import Tensor", "torch.Tensor"),
        ("from ... import runtime", "pssolver.runtime"),
        ("from ...runtime import builder", "pssolver.runtime.builder"),
    ],
)
def test_ast_import_resolution_includes_relative_from_imports(statement, target):
    assert target in {
        name for _, name in imports(ast.parse(statement), "pssolver.operators.inner.sample", False)
    }


@pytest.mark.parametrize(
    "layer,statement",
    [
        ("core", "import torch"),
        ("backends", "from .. import runtime"),
        ("operators", "from pssolver.api import Simulation"),
        ("linear_solvers", "def f():\n    import pssolver.workflows"),
    ],
)
def test_new_forbidden_imports_fail_closed(tmp_path, layer, statement):
    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    package = tmp_path / "pssolver" / layer
    package.mkdir(parents=True)
    (package / "new.py").write_text(statement)
    assert check(tmp_path)


def test_metrics_have_explicit_definitions_and_no_gates():
    report = measure(
        {
            "pssolver/__init__.py": "",
            "pssolver/core/a.py": "# comment\nfrom . import b\nx = 1\n",
            "pssolver/core/b.py": "from . import a\n",
            "pssolver/experimental/unused.py": "value = 1\n",
        }
    )
    assert report["package_lines"] == 4
    assert report["experimental_fraction"] == 0.25
    assert report["import_cycles"] == [["pssolver.core.a", "pssolver.core.b"]]
    assert "pssolver.experimental.unused" in report["unimported_modules"]
    assert report["method"]["thresholds"] is None


def test_structure_baseline_is_bound_to_the_unmodified_parent():
    report = json.loads((ROOT / "qualification/structure_metrics/e32ac21.json").read_text())
    assert report["commit"] == "e32ac213be3ad994a95ec71c53212123530be05c"
    assert report["package_modules"] == 266
    assert report["package_lines"] == 83564
    assert report["experimental_lines"] == 25299
    assert report["method"]["thresholds"] is None


def test_classification_is_total_unique_and_keeps_the_original_gpu_boundary():
    mapping, gpu = load_categories(ROOT)
    assert set(mapping.values()) == {
        "functional",
        "numerical",
        "historical",
        "installed",
        "gpu",
        "baseline_fingerprint",
    }
    assert len(gpu) == 6
    assert all(node.split("::")[0] in mapping for node in gpu)
    for name in (
        "test_release_v0_1.py",
        "test_release_v0_2.py",
        "test_release_v0_2_rc5.py",
        "test_release_v0_2_rc6.py",
    ):
        assert mapping["tests/" + name] == "historical"


def test_missing_history_is_a_distinguishable_failure_not_a_skip(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert not history_available(tmp_path)
    with pytest.raises(RuntimeError, match="PSSOLVER_TEST_PREREQUISITE"):
        released_manifest(tmp_path)
    assert history_available(ROOT)


def test_unclassified_test_is_rejected(tmp_path):
    (tmp_path / "tests").mkdir()
    manifest = """[categories]
functional = []
numerical = []
historical = []
installed = []
gpu = []
baseline_fingerprint = []
[gpu_prerequisites]
nodeids = []
"""
    (tmp_path / "tests/categories.toml").write_text(manifest)
    (tmp_path / "tests/test_new.py").write_text("def test_new(): pass\n")
    with pytest.raises(ValueError, match="Unclassified"):
        load_categories(tmp_path)


@pytest.mark.parametrize("isolated_root", [False, True])
def test_explicit_gpu_selection_fails_before_the_historical_skip_marker(tmp_path, isolated_root):
    node = (
        "tests/test_rc4_plane_nyquist_cuda.py::"
        "test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence"
    )
    isolation = []
    if isolated_root:
        isolation = ["--import-mode=importlib", f"--rootdir={tmp_path}"]
        node = str(ROOT / node)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            *isolation,
            node,
        ],
        cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "PSSOLVER_TEST_PREREQUISITE: CUDA device required" in result.stdout
    assert "skipped" not in result.stdout


def test_lint_baseline_is_record_only_and_classified():
    record = json.loads((ROOT / "qualification/lint_baseline.json").read_text())
    assert record["counts"] == {"F401": 65, "F841": 3, "E741": 1}
    assert len(record["diagnostics"]) == 69
    assert all(
        row["category"] and row["reason"] and row["source_sha256"] for row in record["diagnostics"]
    )
    assert "noqa" not in (ROOT / "tools/check.py").read_text()


def test_mypy_baseline_does_not_accept_new_or_modified_diagnostics(tmp_path, monkeypatch):
    from tools import check as checker

    (tmp_path / "qualification").mkdir()
    (tmp_path / "qualification/type_baseline.json").write_text(
        json.dumps(
            {
                "diagnostics": ["sample.py:1: error: old [attr-defined]"],
                "sources": {"sample.py": "0" * 64},
            }
        )
    )
    (tmp_path / "sample.py").write_text("changed = True\n")
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(
        checker,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 1, "sample.py:1: error: old [attr-defined]\n", ""
        ),
    )
    assert not checker.check_types(sys.executable)
