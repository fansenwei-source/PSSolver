"""Shared atomic serialization for workflow diagnostic records."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DIAGNOSTICS_ENVELOPE_SCHEMA_VERSION = 1


def diagnostics_metadata_envelope(
    *,
    application_family: str,
    runtime_path: str,
    enabled: bool,
    steps: list[int] | tuple[int, ...],
    dtype: np.dtype,
) -> dict[str, object]:
    """Describe a diagnostics payload without changing its byte layout."""

    if not isinstance(application_family, str) or not application_family:
        raise ValueError("diagnostics application_family must be non-empty")
    if not isinstance(runtime_path, str) or not runtime_path:
        raise ValueError("diagnostics runtime_path must be non-empty")
    if not isinstance(enabled, bool):
        raise TypeError("diagnostics enabled must be a bool")
    steps = list(steps)
    if any(
        not isinstance(step, int) or isinstance(step, bool) or step < 0
        for step in steps
    ):
        raise ValueError("diagnostics steps must be non-negative integers")
    dtype = np.dtype(dtype)
    if dtype.names is None:
        raise TypeError("diagnostics dtype must be structured")
    if not enabled and steps:
        raise ValueError("disabled diagnostics cannot declare captured steps")
    npy = "diagnostics.npy" if enabled else None
    csv = "diagnostics.csv" if enabled else None
    payload_schema = []
    payload_units = {}
    for name in dtype.names:
        field_dtype = dtype.fields[name][0]
        payload_schema.append({"name": name, "dtype": field_dtype.name})
        payload_units[name] = (
            "step_index"
            if name == "step"
            else "count"
            if name.endswith("_iterations")
            else "solver_native_dimensionless"
        )
    return {
        "schema_version": DIAGNOSTICS_ENVELOPE_SCHEMA_VERSION,
        "application_family": application_family,
        "runtime_path": runtime_path,
        "enabled": enabled,
        "count": len(steps),
        "steps": steps,
        "artifact_paths": {"npy": npy, "csv": csv},
        "payload_schema": payload_schema,
        "payload_units": payload_units,
        # v0.2 compatibility aliases.  New consumers use artifact_paths.
        "npy": npy,
        "csv": csv,
    }


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


__all__ = [
    "DIAGNOSTICS_ENVELOPE_SCHEMA_VERSION",
    "diagnostics_metadata_envelope",
    "write_structured_diagnostics",
]
