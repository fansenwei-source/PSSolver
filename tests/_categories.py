"""Explicit test ownership and prerequisite failures, without changing test bodies."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from tools._source import read_toml


ROOT = Path(__file__).resolve().parents[1]
NAMES = ("functional", "numerical", "historical", "installed", "gpu")


def load_categories(root: Path = ROOT) -> tuple[dict[str, str], set[str]]:
    manifest = read_toml(root / "tests/categories.toml")
    categories = manifest["categories"]
    if set(categories) != set(NAMES):
        raise ValueError("Exactly the five documented categories are required")
    mapping = {}
    for category, paths in categories.items():
        for path in paths:
            if path in mapping:
                raise ValueError(f"Duplicate test category: {path}")
            mapping[path] = category
    actual = {p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py")}
    if set(mapping) != actual:
        raise ValueError(f"Unclassified or stale tests: {sorted(set(mapping) ^ actual)}")
    return mapping, set(manifest["gpu_prerequisites"]["nodeids"])


def pytest_configure(config):
    for name in NAMES:
        config.addinivalue_line("markers", f"category_{name}: S-22 explicit test classification")
    config._pssolver_categories = load_categories()
    config._pssolver_history_checked = False


def pytest_collection_modifyitems(config, items):
    mapping, gpu_nodeids = config._pssolver_categories
    for item in items:
        relative = Path(item.path).resolve().relative_to(ROOT).as_posix()
        category = "gpu" if item.nodeid in gpu_nodeids else mapping[relative]
        item.add_marker(getattr(pytest.mark, f"category_{category}"))


def history_available(root: Path) -> bool:
    """A missing object is a prerequisite failure, never a reason to skip."""
    try:
        shallow = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if shallow.returncode or shallow.stdout.strip() != "false":
            return False
        for revision in (
            "4c12228aba01e8483e80d28a1471ab72384d5fda",
            "85ba3aef75548d549d9db453ca4a5ab604b17a9d",
        ):
            result = subprocess.run(
                ["git", "cat-file", "-e", revision + "^{commit}"],
                cwd=root,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                return False
    except OSError:
        return False
    return True


def pytest_runtest_setup(item):
    if item.get_closest_marker("category_historical"):
        if not item.config._pssolver_history_checked:
            if not history_available(ROOT):
                pytest.fail("PSSOLVER_TEST_PREREQUISITE: full Git history required", pytrace=False)
            item.config._pssolver_history_checked = True
    if item.get_closest_marker("category_installed"):
        if not os.environ.get("PSSOLVER_INSTALLED_PYTHON"):
            pytest.fail(
                "PSSOLVER_TEST_PREREQUISITE: PSSOLVER_INSTALLED_PYTHON required", pytrace=False
            )
    if item.nodeid in item.config._pssolver_categories[1]:
        import torch

        if not torch.cuda.is_available():
            pytest.fail("PSSOLVER_TEST_PREREQUISITE: CUDA device required", pytrace=False)
