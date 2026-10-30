"""Shared, standard-library-only artifact I/O for active benchmark harnesses.

This module deliberately lives outside :mod:`pssolver`.  It is qualification
support, not a public runtime API.  Historical benchmark scripts whose hashes
are frozen by qualification records keep their original local helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Iterable


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool = True,
    temporary_prefix: str | None = None,
) -> None:
    """Atomically publish *payload* beside *path* and fsync it first."""

    if not overwrite and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=temporary_prefix or f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(
    path: Path,
    payload: str,
    *,
    encoding: str = "utf-8",
    overwrite: bool = True,
    temporary_prefix: str | None = None,
) -> None:
    """Encode and atomically publish text without changing its bytes."""

    atomic_write_bytes(
        path,
        payload.encode(encoding),
        overwrite=overwrite,
        temporary_prefix=temporary_prefix,
    )


def atomic_write_json(
    path: Path,
    value: object,
    *,
    allow_nan: bool = False,
    indent: int | None = 2,
    sort_keys: bool = True,
    trailing_newline: bool = True,
    overwrite: bool = True,
) -> None:
    """Serialize JSON with explicit formatting and publish it atomically."""

    payload = json.dumps(
        value,
        allow_nan=allow_nan,
        indent=indent,
        sort_keys=sort_keys,
    )
    if trailing_newline:
        payload += "\n"
    atomic_write_text(path, payload, overwrite=overwrite)


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_identity(repository: Path) -> dict[str, object]:
    """Return a fail-closed Git identity for a benchmark provenance record."""

    root = Path(_git_output(repository, "rev-parse", "--show-toplevel"))
    status = _git_output(root, "status", "--short")
    return {
        "root": str(root.resolve()),
        "head": _git_output(root, "rev-parse", "HEAD"),
        "branch": _git_output(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "clean": status == "",
        "status": status,
    }


def runtime_environment() -> dict[str, object]:
    """Return stable standard-library facts about the executing interpreter."""

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "sys_path": list(sys.path),
    }


def provenance_record(
    repository: Path,
    files: Iterable[Path] = (),
) -> dict[str, Any]:
    """Build a provenance record without importing PSSolver or dependencies."""

    file_records = []
    for raw_path in files:
        path = raw_path.resolve()
        file_records.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "git": git_identity(repository),
        "environment": runtime_environment(),
        "files": file_records,
    }
