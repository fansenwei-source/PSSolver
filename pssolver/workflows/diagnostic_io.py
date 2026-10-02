"""Shared atomic serialization for workflow diagnostic records."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_structured_diagnostics(
    directory: str | Path,
    rows: list[tuple[object, ...]],
    *,
    dtype: np.dtype,
    header: str,
) -> tuple[Path, Path]:
    """Write matching structured NPY and CSV files without overwriting."""

    directory = Path(directory).expanduser().resolve()
    array = np.array(rows, dtype=dtype)
    npy_path = directory / "diagnostics.npy"
    csv_path = directory / "diagnostics.csv"
    if npy_path.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite workflow diagnostics")
    npy_temporary = npy_path.with_name(f".{npy_path.name}.tmp")
    csv_temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    try:
        with npy_temporary.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
        npy_temporary.replace(npy_path)
        with csv_temporary.open("x", encoding="utf-8") as handle:
            np.savetxt(
                handle,
                array,
                delimiter=",",
                header=header,
                comments="",
            )
        csv_temporary.replace(csv_path)
    except BaseException:
        for temporary in (npy_temporary, csv_temporary):
            if temporary.exists():
                temporary.unlink()
        if npy_path.exists() and not csv_path.exists():
            npy_path.unlink()
        raise
    return npy_path, csv_path


__all__ = ["write_structured_diagnostics"]
