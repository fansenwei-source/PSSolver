"""R5-4 package-surface and installed Plane provenance contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from pssolver.applications import plane_beris_edwards as plane_application
from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
)


ROOT = Path(__file__).resolve().parents[1]


def _setup_keywords() -> dict[str, ast.expr]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    setup_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    ]
    assert len(setup_calls) == 1
    return {keyword.arg: keyword.value for keyword in setup_calls[0].keywords}


def test_distribution_discovers_only_the_supported_pssolver_namespace():
    packages = _setup_keywords()["packages"]
    assert isinstance(packages, ast.Call)
    assert isinstance(packages.func, ast.Name)
    assert packages.func.id == "find_packages"
    assert ast.literal_eval(
        next(item.value for item in packages.keywords if item.arg == "include")
    ) == ["pssolver", "pssolver.*"]


def test_historical_plane_module_is_a_retained_compatibility_shim_only():
    py_modules = _setup_keywords()["py_modules"]
    assert ast.literal_eval(py_modules) == ["Plane_beris_edwards_stokes"]
    source = (ROOT / "Plane_beris_edwards_stokes.py").read_text(
        encoding="utf-8"
    )
    assert "pssolver.applications.plane_beris_edwards import main" in source
    assert "Plane_beris_edwards_stokes.py" not in (
        PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES
    )


def test_plane_implementation_provenance_is_installed_package_internal():
    assert PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES
    for relative in PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES:
        assert relative.startswith("pssolver/")
        source = plane_application._implementation_source_path(relative)
        assert source.is_file()
        assert source.is_relative_to(plane_application._PACKAGE_ROOT)


@pytest.mark.parametrize(
    "relative",
    (
        "Plane_beris_edwards_stokes.py",
        "../Plane_beris_edwards_stokes.py",
        "/tmp/Plane_beris_edwards_stokes.py",
        "benchmarks/profile_periodic_functional.py",
    ),
)
def test_plane_implementation_provenance_rejects_nonpackage_paths(relative):
    with pytest.raises(ValueError, match="package-relative pssolver files"):
        plane_application._implementation_source_path(relative)
