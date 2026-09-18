"""Release-level contracts for the bounded PSSolver 0.1.1rc1 package."""

from __future__ import annotations

import ast
from pathlib import Path

import pssolver


PROJECT_ROOT = Path(__file__).parents[1]


def test_public_version_is_0_1_1_rc1():
    assert pssolver.__version__ == "0.1.1rc1"
    assert "__version__" in pssolver.__all__


def test_setup_uses_the_single_version_source_and_runtime_dependencies():
    source = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    normalized = ast.unparse(tree)

    assert "PROJECT_ROOT / 'pssolver' / '_version.py'" in normalized
    assert "VERSION_NAMESPACE['__version__']" in normalized
    assert "py_modules=['Plane_beris_edwards_stokes']" in normalized
    for dependency in (
        "numpy>=1.22",
        "scipy>=1.8",
        "torch>=2.5",
        "tqdm>=4.64",
    ):
        assert dependency in normalized


def test_standard_build_metadata_and_release_documents_exist():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert 'build-backend = "setuptools.build_meta"' in pyproject
    release_notes = (
        PROJECT_ROOT / "notes" / "pssolver_v0_1_1_rc1.md"
    ).read_text(encoding="utf-8")

    assert "PSSolver 0.1.1rc1" in readme
    assert "## 0.1.1rc1" in changelog
    assert "## 0.1.0" in changelog
    assert "38cd9335b13328922956a46bb0e9dcbc992a5de0" in changelog
    assert "525ba2326de3604e75752364d561af6417063ddc" in changelog
    assert "0.1.1rc1" in release_notes


def test_generated_packaging_artifacts_are_ignored():
    patterns = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "*.egg-info/" in patterns
    assert "build/" in patterns
    assert "dist/" in patterns


def test_source_distribution_includes_release_and_scope_documents():
    manifest = (PROJECT_ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    for path in (
        "CHANGELOG.md",
        "notes/pssolver_v0_1_scope.md",
        "notes/pssolver_v0_1_1_rc1.md",
        "benchmarks/README.md",
        "benchmarks/periodic_fast_path_default_promotion.md",
    ):
        assert (PROJECT_ROOT / path).is_file()
        assert f"include {path}" in manifest
