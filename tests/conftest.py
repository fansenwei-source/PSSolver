"""Shared test-only capability probes."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    """Load test policy by path without adding Provider source to sys.path."""
    loader_spec = importlib.util.spec_from_file_location(
        "_pssolver_legacy_stage_imports",
        Path(__file__).resolve().parents[1] / "qualification/_legacy_imports.py",
    )
    loader = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loader)
    loader.install()
    spec = importlib.util.spec_from_file_location(
        "_pssolver_category_policy", Path(__file__).with_name("_categories.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config.pluginmanager.register(module, name="pssolver-category-policy")


@pytest.fixture(scope="session")
def require_working_cpu_inductor(tmp_path_factory):
    """Skip compile-dependent research tests when Inductor cannot execute."""

    root = tmp_path_factory.mktemp("cpu_inductor_probe")
    cache = root / "torchinductor"
    temporary = root / "tmp"
    cache.mkdir()
    temporary.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCHINDUCTOR_CACHE_DIR": str(cache),
            "TMPDIR": str(temporary),
        }
    )
    source = """
import torch

def pointwise(value):
    return value.sin() + value.square()

compiled = torch.compile(pointwise, backend="inductor", fullgraph=True)
value = torch.arange(8, dtype=torch.float64)
actual = compiled(value)
expected = pointwise(value)
assert torch.equal(actual, expected)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(
            "CPU torch.compile/Inductor toolchain is unavailable in this "
            "environment"
        )
    return {
        "python": sys.executable,
        "torchinductor_cache": str(cache),
        "tmpdir": str(temporary),
    }
