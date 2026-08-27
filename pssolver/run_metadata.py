"""Small, model-agnostic helpers for run metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def prepare_new_run_directory(output_directory: str | Path) -> Path:
    """Create an empty output directory for a new run.

    Existing empty directories are accepted, but a nonempty directory is
    rejected so a new run cannot silently mix its outputs with older frames or
    replace their provenance metadata.
    """

    directory = Path(output_directory)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if not directory.is_dir():
            raise FileExistsError(
                f"Run output path exists and is not a directory: {directory}"
            ) from None
        if any(directory.iterdir()):
            raise FileExistsError(
                f"Refusing to mix run outputs in nonempty directory: {directory}"
            ) from None
    return directory


def write_run_metadata(
    output_directory: str | Path,
    metadata: Mapping[str, Any],
    *,
    status: str | None = None,
) -> Path:
    """Write one reproducibility record for a simulation directory.

    The temporary file and final metadata file live in the same directory so
    replacement is atomic on the filesystems used for simulation output.
    """

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    if status is not None:
        payload["status"] = status

    path = directory / "metadata.json"
    temporary_path = directory / "metadata.json.tmp"
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path
