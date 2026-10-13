"""Installed-venv identity checks for the RC4.5.8 qualification harness.

Virtual environments normally expose ``bin/python`` as a symbolic link to a
base interpreter.  Resolving that link before deriving the environment root
loses the identity of the environment that selected the installed packages.
This module deliberately keeps the lexical invocation path for ownership and
records the resolved interpreter target separately as provenance.
"""

from __future__ import annotations

from pathlib import Path
import sys
import sysconfig


SCHEMA = "pssolver.rc4_5_8.installed_venv_identity.v1"


def lexical_python_path(path: Path) -> Path:
    """Return an absolute interpreter invocation path without resolving it."""

    invocation = path.expanduser().absolute()
    try:
        target = invocation.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as error:
        raise ValueError(
            f"Python invocation must resolve to an ordinary file: {invocation}"
        ) from error
    if not invocation.is_file() or not target.is_file():
        raise ValueError(
            f"Python invocation must resolve to an ordinary file: {invocation}"
        )
    return invocation


def lexical_venv_root(python: Path) -> Path:
    """Derive a venv root from its lexical ``bin/python`` entry point."""

    invocation = lexical_python_path(python)
    if invocation.parent.name != "bin" or invocation.name not in {
        "python",
        "python3",
        "python3.10",
    }:
        raise ValueError(
            "Python invocation must use a venv bin/python entry point: "
            f"{invocation}"
        )
    root = invocation.parent.parent
    marker = root / "pyvenv.cfg"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"venv marker is missing or non-canonical: {marker}")
    return root


def validate_installed_venv_identity(
    *,
    requested_python: Path,
    imported_module: Path,
    runtime_prefix: Path | None = None,
    purelib: Path | None = None,
    reported_executable: Path | None = None,
) -> dict[str, object]:
    """Validate that an import belongs to the requested installed venv.

    ``requested_python`` retains its lexical path.  All filesystem membership
    comparisons use canonical paths after the lexical venv root has been
    selected, so symlinks cannot escape the requested environment unnoticed.
    """

    invocation = lexical_python_path(requested_python)
    root = lexical_venv_root(invocation)
    target = invocation.resolve(strict=True)
    canonical_root = root.resolve(strict=True)
    prefix = Path(sys.prefix if runtime_prefix is None else runtime_prefix).resolve(
        strict=True
    )
    library = Path(
        sysconfig.get_path("purelib") if purelib is None else purelib
    ).resolve(strict=True)
    imported = imported_module.expanduser().resolve(strict=True)
    executable = Path(
        sys.executable if reported_executable is None else reported_executable
    ).expanduser().absolute()

    if prefix != canonical_root:
        raise ValueError(
            "runtime prefix does not match the lexical requested venv: "
            f"prefix={prefix}, requested_venv={canonical_root}"
        )
    if not library.is_relative_to(prefix):
        raise ValueError(
            f"runtime purelib is outside the requested venv: {library}"
        )
    if not imported.is_relative_to(library):
        raise ValueError(
            f"installed module import is outside the requested venv purelib: {imported}"
        )

    return {
        "schema": SCHEMA,
        "requested_python": str(invocation),
        "requested_python_is_symlink": invocation.is_symlink(),
        "resolved_python_target": str(target),
        "lexical_venv_root": str(root),
        "canonical_venv_root": str(canonical_root),
        "reported_executable": str(executable),
        "runtime_prefix": str(prefix),
        "purelib": str(library),
        "imported_module": str(imported),
        "import_is_within_purelib": True,
        "runtime_prefix_matches_requested_venv": True,
        "passed": True,
    }
