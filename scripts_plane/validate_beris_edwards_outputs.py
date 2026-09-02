#!/usr/bin/env python3
"""Read-only post-validation for Beris--Edwards--Stokes run outputs.

The validator deliberately lives in the repository so the post-processing
contract is versioned with the simulation code.  Formal validation requires a
runner plan, and its existing completed-run/metadata check is reused before
the heavier array checks below are performed.
"""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    PROJECT_ROOT / "scripts_plane" / "run_beris_edwards_validation.py"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_plane.run_beris_edwards_validation import (  # noqa: E402
    _canonical_sha256,
    _sha256_file,
    _validate_completed_run,
)


DIAGNOSTIC_FIELDS = (
    "step",
    "div_max",
    "div_rms",
    "div_rel",
    "schur_iterations",
    "schur_abs_residual",
    "schur_rel_residual",
    "wall_normal_momentum_max",
    "wall_normal_momentum_rms",
)
NONNEGATIVE_DIAGNOSTIC_FIELDS = DIAGNOSTIC_FIELDS[1:]
DEFAULT_RESIDUAL_MAX = 1.0e-10
REPORT_SCHEMA_VERSION = 1
MAX_STEP = int(np.iinfo(np.int64).max)
SNAPSHOT_PATTERN = re.compile(r"^(Q|u|p)_([0-9]+)\.npy$")
STEP_PATTERN = re.compile(r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")


def parse_integral_step(
    value: Any,
    *,
    label: str = "step",
    minimum: int = 0,
    maximum: int = MAX_STEP,
) -> int:
    """Parse an integer-valued step without truncating a fractional value.

    Scientific notation such as ``1.000e+00`` is accepted.  Non-finite values,
    booleans, and finite non-integers are rejected rather than passed through
    Python's truncating ``int(float_value)`` conversion.
    """

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, not a boolean")
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError(f"{label} is not an ASCII number") from error
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            raise ValueError(f"{label} must not be empty")
        if len(candidate) > 128 or STEP_PATTERN.fullmatch(candidate) is None:
            raise ValueError(
                f"{label}={value!r} is not a supported numeric representation"
            )
    elif isinstance(value, (int, float)):
        candidate = str(value)
    else:
        raise ValueError(f"{label} has unsupported type {type(value).__name__}")
    try:
        number = Decimal(candidate)
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label}={value!r} is not numeric") from error
    if not number.is_finite():
        raise ValueError(f"{label}={value!r} must be finite")
    integral = number.to_integral_value()
    if number != integral:
        raise ValueError(f"{label}={value!r} is not an integer")
    if integral < minimum or integral > maximum:
        raise ValueError(
            f"{label}={value!r} is outside the supported range "
            f"[{minimum}, {maximum}]"
        )
    return int(integral)


def _file_identity(path: Path) -> tuple[int, int, int, int]:
    if path.is_symlink():
        raise ValueError(f"critical file must not be a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"required regular file is missing: {path}")
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _read_json_object(path: Path) -> dict[str, Any]:
    _file_identity(path)

    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token!r} in {path}")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=reject_pairs,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"metadata is missing {'.'.join(keys)}")
        current = current[key]
    return current


def _parse_shape(value: Any, *, label: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three dimensions")
    shape = tuple(
        parse_integral_step(component, label=f"{label}[{axis}]")
        for axis, component in enumerate(value)
    )
    if any(component <= 0 for component in shape):
        raise ValueError(f"{label} dimensions must be positive: {shape}")
    return shape


def _parse_real_dtype(value: Any, *, label: str) -> np.dtype[Any]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a NumPy dtype name")
    try:
        dtype = np.dtype(value)
    except TypeError as error:
        raise ValueError(f"{label}={value!r} is not a valid dtype") from error
    if dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError(f"{label} must be float32 or float64, got {dtype.name}")
    return dtype


def _array_is_finite(array: np.ndarray, *, target_bytes: int = 64 * 1024**2) -> bool:
    """Scan a possibly memory-mapped array without allocating one huge mask."""

    if array.ndim == 0:
        return bool(np.isfinite(array[()]))
    bytes_per_plane = max(int(array[0:1].size) * array.dtype.itemsize, 1)
    planes_per_chunk = max(target_bytes // bytes_per_plane, 1)
    for start in range(0, array.shape[0], planes_per_chunk):
        if not np.isfinite(array[start : start + planes_per_chunk]).all():
            return False
    return True


def _validate_frame(
    path: Path,
    *,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing final frame: {path.name}")
    _file_identity(path)
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as error:
        raise ValueError(f"cannot load {path.name}: {error}") from error
    if array.shape != expected_shape:
        raise ValueError(
            f"{path.name} shape={array.shape}, expected {expected_shape}"
        )
    if array.dtype != expected_dtype:
        raise ValueError(
            f"{path.name} dtype={array.dtype}, expected {expected_dtype}"
        )
    if not _array_is_finite(array):
        raise ValueError(f"{path.name} contains NaN or Inf")
    return {
        "path": str(path),
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "finite": True,
        "size_bytes": path.stat().st_size,
    }


def _load_npy_diagnostics(path: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    if not path.is_file():
        raise ValueError(f"missing diagnostics file: {path.name}")
    _file_identity(path)
    try:
        values = np.load(path, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"cannot load {path.name}: {error}") from error
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("diagnostics.npy must be a nonempty one-dimensional array")
    if values.dtype.names != DIAGNOSTIC_FIELDS:
        raise ValueError(
            "diagnostics.npy fields={} expected={}".format(
                values.dtype.names, DIAGNOSTIC_FIELDS
            )
        )
    expected_dtypes = {"step": np.dtype("int64")}
    expected_dtypes.update(
        {field: np.dtype("float64") for field in DIAGNOSTIC_FIELDS[1:]}
    )
    for field, expected_dtype in expected_dtypes.items():
        actual_dtype = values.dtype.fields[field][0]
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"diagnostics.npy field {field} dtype={actual_dtype}, "
                f"expected {expected_dtype}"
            )
    steps = [
        parse_integral_step(value, label=f"diagnostics.npy step row {row}")
        for row, value in enumerate(values["step"])
    ]
    columns: dict[str, np.ndarray] = {}
    for field in DIAGNOSTIC_FIELDS[1:]:
        try:
            column = np.asarray(values[field], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"diagnostics.npy field {field} is not numeric") from error
        if not np.isfinite(column).all():
            raise ValueError(f"diagnostics.npy field {field} contains NaN or Inf")
        if field in NONNEGATIVE_DIAGNOSTIC_FIELDS and np.any(column < 0.0):
            raise ValueError(f"diagnostics.npy field {field} contains negatives")
        if field == "schur_iterations":
            for row, value in enumerate(column):
                parse_integral_step(
                    value, label=f"diagnostics.npy schur_iterations row {row}"
                )
        columns[field] = column
    return steps, columns


def _load_csv_diagnostics(path: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    if not path.is_file():
        raise ValueError(f"missing diagnostics file: {path.name}")
    _file_identity(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != DIAGNOSTIC_FIELDS:
            raise ValueError(
                "diagnostics.csv fields={} expected={}".format(
                    tuple(reader.fieldnames or ()),
                    DIAGNOSTIC_FIELDS,
                )
            )
        rows = list(reader)
    if not rows:
        raise ValueError("diagnostics.csv must contain at least one data row")
    steps: list[int] = []
    columns = {field: [] for field in DIAGNOSTIC_FIELDS[1:]}
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise ValueError(f"diagnostics.csv row {row_number} has extra columns")
        steps.append(
            parse_integral_step(
                row["step"],
                label=f"diagnostics.csv step row {row_number}",
            )
        )
        for field in DIAGNOSTIC_FIELDS[1:]:
            text = row.get(field)
            try:
                value = float(text) if text is not None else math.nan
            except ValueError as error:
                raise ValueError(
                    f"diagnostics.csv row {row_number} field {field} is not numeric"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"diagnostics.csv row {row_number} field {field} is not finite"
                )
            if field in NONNEGATIVE_DIAGNOSTIC_FIELDS and value < 0.0:
                raise ValueError(
                    f"diagnostics.csv row {row_number} field {field} is negative"
                )
            if field == "schur_iterations":
                parse_integral_step(
                    text, label=f"diagnostics.csv schur_iterations row {row_number}"
                )
            columns[field].append(value)
    return steps, {
        field: np.asarray(values, dtype=np.float64)
        for field, values in columns.items()
    }


def _validate_diagnostics(
    directory: Path,
    *,
    expected_steps: tuple[int, ...],
    div_rel_max: float,
    schur_rel_residual_max: float,
) -> dict[str, Any]:
    npy_steps, npy_columns = _load_npy_diagnostics(directory / "diagnostics.npy")
    csv_steps, csv_columns = _load_csv_diagnostics(directory / "diagnostics.csv")
    if npy_steps != csv_steps:
        raise ValueError(
            f"diagnostics step mismatch: npy={npy_steps!r}, csv={csv_steps!r}"
        )
    if npy_steps != list(expected_steps):
        raise ValueError(
            "diagnostic step sequence differs from the expected schedule: "
            f"actual={npy_steps!r}, expected={list(expected_steps)!r}"
        )
    for field in DIAGNOSTIC_FIELDS[1:]:
        if not np.array_equal(npy_columns[field], csv_columns[field]):
            raise ValueError(f"diagnostics.npy/csv mismatch in field {field}")
    div_values = npy_columns["div_rel"]
    schur_values = npy_columns["schur_rel_residual"]
    if np.any(div_values > div_rel_max):
        row = int(np.flatnonzero(div_values > div_rel_max)[0])
        raise ValueError(
            f"div_rel={div_values[row]:.17g} at step {npy_steps[row]} "
            f"exceeds {div_rel_max:.17g}"
        )
    if np.any(schur_values > schur_rel_residual_max):
        row = int(np.flatnonzero(schur_values > schur_rel_residual_max)[0])
        raise ValueError(
            "schur_rel_residual={:.17g} at step {} exceeds {:.17g}".format(
                schur_values[row], npy_steps[row], schur_rel_residual_max
            )
        )
    return {
        "row_count": len(npy_steps),
        "first_step": npy_steps[0],
        "final_step": npy_steps[-1],
        "npy_csv_consistent": True,
        "max_div_rel": float(np.max(div_values)),
        "max_schur_rel_residual": float(np.max(schur_values)),
        "final": {
            field: float(npy_columns[field][-1])
            for field in DIAGNOSTIC_FIELDS[1:]
        },
    }


def _validate_source_hashes(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"validation plan {label} must be a nonempty object")
    validated: dict[str, str] = {}
    for relative, expected_digest in value.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"validation plan {label} has an invalid path")
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(character not in "0123456789abcdef" for character in expected_digest)
        ):
            raise ValueError(f"validation plan {label} has an invalid SHA-256")
        relative_path = Path(relative)
        if relative_path.is_absolute():
            raise ValueError(f"validation plan {label} path must be relative: {relative}")
        source_path = (PROJECT_ROOT / relative_path).resolve()
        if not source_path.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"validation plan {label} path escapes project: {relative}")
        _file_identity(source_path)
        actual_digest = _sha256_file(source_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"validation plan {label} SHA-256 differs for {relative}"
            )
        validated[relative] = expected_digest
    return validated


def _validate_plan(plan: dict[str, Any], *, run_dir: Path) -> None:
    if plan.get("schema_version") != 1:
        raise ValueError("validation plan schema_version must equal 1")
    if plan.get("validation") != "beris_edwards_stokes_issue6":
        raise ValueError("unexpected validation plan identity")
    recorded_hash = plan.get("plan_sha256")
    if not isinstance(recorded_hash, str):
        raise ValueError("validation plan is missing plan_sha256")
    unhashed = dict(plan)
    del unhashed["plan_sha256"]
    if _canonical_sha256(unhashed) != recorded_hash:
        raise ValueError("validation plan_sha256 does not match its contents")
    _validate_source_hashes(
        plan.get("implementation_sha256"), label="implementation_sha256"
    )
    _validate_source_hashes(plan.get("validation_tools_sha256"), label="validation_tools_sha256")
    output_root_value = plan.get("output_root")
    if not isinstance(output_root_value, str):
        raise ValueError("validation plan is missing output_root")
    output_root = Path(output_root_value).resolve()
    if not run_dir.is_relative_to(output_root):
        raise ValueError("--run-dir is outside the validation plan output_root")
    rows = plan.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("validation plan must contain a nonempty runs list")
    run_ids: list[str] = []
    output_dirs: list[Path] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"validation plan run {index} is not an object")
        run_id = row.get("run_id")
        output_dir = row.get("output_dir")
        config = row.get("config")
        config_hash = row.get("config_sha256")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"validation plan run {index} has invalid run_id")
        if not isinstance(output_dir, str):
            raise ValueError(f"validation plan run {run_id} has invalid output_dir")
        resolved_output = Path(output_dir).resolve()
        if not resolved_output.is_relative_to(output_root):
            raise ValueError(f"validation plan run {run_id} is outside output_root")
        if not isinstance(config, dict) or _canonical_sha256(config) != config_hash:
            raise ValueError(f"validation plan run {run_id} config_sha256 mismatch")
        run_ids.append(run_id)
        output_dirs.append(resolved_output)
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("validation plan contains duplicate run_id values")
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("validation plan contains duplicate output_dir values")


def _select_plan_run(
    plan: dict[str, Any],
    *,
    run_dir: Path,
    run_id: str | None,
) -> dict[str, Any]:
    _validate_plan(plan, run_dir=run_dir)
    rows = plan["runs"]
    if run_id is not None:
        matches = [row for row in rows if row.get("run_id") == run_id]
    else:
        matches = [
            row for row in rows if Path(row["output_dir"]).resolve() == run_dir
        ]
    if len(matches) != 1:
        selector = f"run_id={run_id!r}" if run_id is not None else str(run_dir)
        raise ValueError(
            f"validation plan must contain exactly one run matching {selector}; "
            f"found {len(matches)}"
        )
    row = matches[0]
    if Path(row["output_dir"]).resolve() != run_dir:
        raise ValueError("selected plan run output_dir does not match --run-dir")
    implementation = plan.get("implementation_sha256")
    if not isinstance(implementation, dict):
        raise ValueError("validation plan is missing implementation_sha256")
    _validate_completed_run(row, implementation_sha256=implementation)
    return row


def _metadata_contract(
    metadata: dict[str, Any],
) -> tuple[int, tuple[int, int, int], np.dtype[Any], int, int, int]:
    if metadata.get("schema_version") != 1:
        raise ValueError("metadata.schema_version must equal 1")
    if metadata.get("script") != "Plane_beris_edwards_stokes.py":
        raise ValueError("metadata.script is not Plane_beris_edwards_stokes.py")
    if metadata.get("status") != "complete":
        raise ValueError("metadata.status must equal 'complete'")

    steps = {
        "completed_steps": parse_integral_step(
            _nested(metadata, "completed_steps"), label="metadata.completed_steps"
        ),
        "steps": parse_integral_step(_nested(metadata, "steps"), label="metadata.steps"),
        "solver.steps": parse_integral_step(
            _nested(metadata, "solver", "steps"), label="metadata.solver.steps"
        ),
    }
    if len(set(steps.values())) != 1:
        raise ValueError(f"metadata step fields disagree: {steps}")
    final_step = next(iter(steps.values()))
    if final_step <= 0:
        raise ValueError("metadata final step must be positive")

    shapes = {
        "shape": _parse_shape(_nested(metadata, "shape"), label="metadata.shape"),
        "solver.shape": _parse_shape(
            _nested(metadata, "solver", "shape"), label="metadata.solver.shape"
        ),
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"metadata shape fields disagree: {shapes}")
    shape = next(iter(shapes.values()))

    dtypes = {
        "dtype": _parse_real_dtype(_nested(metadata, "dtype"), label="metadata.dtype"),
        "solver.real_dtype": _parse_real_dtype(
            _nested(metadata, "solver", "real_dtype"),
            label="metadata.solver.real_dtype",
        ),
    }
    if len(set(dtypes.values())) != 1:
        raise ValueError(
            "metadata dtype fields disagree: {}".format(
                {key: value.name for key, value in dtypes.items()}
            )
        )
    dtype = next(iter(dtypes.values()))

    save_start_step = parse_integral_step(
        _nested(metadata, "save_start_step"),
        label="metadata.save_start_step",
        maximum=final_step,
    )
    save_interval = parse_integral_step(
        _nested(metadata, "save_interval"),
        label="metadata.save_interval",
        minimum=1,
    )
    diagnostic_interval = parse_integral_step(
        _nested(metadata, "diagnostic_interval"),
        label="metadata.diagnostic_interval",
        minimum=1,
        maximum=final_step,
    )
    solver_save_interval = parse_integral_step(
        _nested(metadata, "solver", "save_interval"),
        label="metadata.solver.save_interval",
        minimum=1,
    )
    if solver_save_interval != save_interval:
        raise ValueError(
            "metadata.save_interval and metadata.solver.save_interval disagree"
        )

    files = _nested(metadata, "implementation_provenance", "files")
    if not isinstance(files, dict) or not files:
        raise ValueError("metadata implementation provenance is empty")
    for name, digest in files.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise ValueError("metadata implementation provenance is malformed")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"invalid implementation SHA-256 for {name!r}")
    return (
        final_step,
        shape,
        dtype,
        save_start_step,
        save_interval,
        diagnostic_interval,
    )


def _scheduled_steps(
    *, final_step: int, start_step: int, interval: int
) -> tuple[int, ...]:
    first = ((start_step + interval - 1) // interval) * interval
    return (*range(first, final_step, interval), final_step)


def _validate_snapshot_set(
    directory: Path,
    *,
    expected_steps: tuple[int, ...],
    shape: tuple[int, int, int],
    dtype: np.dtype[Any],
) -> dict[str, Any]:
    actual: dict[str, set[str]] = {"Q": set(), "u": set(), "p": set()}
    for path in directory.iterdir():
        match = SNAPSHOT_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        field, step_text = match.groups()
        step = parse_integral_step(step_text, label=f"snapshot {path.name} step")
        if step_text != str(step):
            raise ValueError(f"noncanonical snapshot filename: {path.name}")
        actual[field].add(path.name)
    for field, names in actual.items():
        expected_names = {f"{field}_{step}.npy" for step in expected_steps}
        if names != expected_names:
            raise ValueError(
                f"{field} snapshot steps/files={sorted(names)}, "
                f"expected={sorted(expected_names)}"
            )

    frame_shapes = {"Q": (*shape, 5), "u": (*shape, 3), "p": shape}
    frames: dict[str, list[dict[str, Any]]] = {"Q": [], "u": [], "p": []}
    for step in expected_steps:
        for field in ("Q", "u", "p"):
            frames[field].append(
                _validate_frame(
                    directory / f"{field}_{step}.npy",
                    expected_shape=frame_shapes[field],
                    expected_dtype=dtype,
                )
            )
    return {
        "expected_steps": list(expected_steps),
        "frame_count_per_field": len(expected_steps),
        "all_shapes_dtypes_and_values_valid": True,
        "frames": frames,
    }


def _collect_identities(paths: tuple[Path, ...]) -> dict[str, tuple[int, int, int, int]]:
    return {str(path): _file_identity(path) for path in paths}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _record_check(
    checks: dict[str, Any],
    errors: list[str],
    name: str,
    operation: Callable[[], Any],
) -> Any | None:
    try:
        details = operation()
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        checks[name] = {"passed": False, "error": message}
        errors.append(f"{name}: {message}")
        return None
    checks[name] = {"passed": True, "details": _jsonable(details)}
    return details


def _validate_stability(
    paths: tuple[Path, ...],
    before: dict[str, tuple[int, int, int, int]],
    *,
    complete_path: Path,
    metadata_path: Path,
    metadata_before: dict[str, Any],
    plan_path: Path,
    plan_before: dict[str, Any],
) -> dict[str, Any]:
    after = _collect_identities(paths)
    if after != before:
        changed = sorted(key for key in before if before.get(key) != after.get(key))
        raise ValueError(f"critical files changed during validation: {changed}")
    _validate_complete_marker(complete_path)
    if _read_json_object(metadata_path) != metadata_before:
        raise ValueError("metadata.json contents changed during validation")
    if _read_json_object(plan_path) != plan_before:
        raise ValueError("validation plan contents changed during validation")
    return {"stable": True, "file_count": len(paths)}


def validate_outputs(
    run_dir: str | Path,
    *,
    plan_path: str | Path | None = None,
    run_id: str | None = None,
    div_rel_max: float | None = None,
    schur_rel_residual_max: float | None = None,
) -> dict[str, Any]:
    """Validate one completed run without writing into its output directory."""

    directory_input = Path(run_dir)
    directory = directory_input.resolve()
    plan_input = None if plan_path is None else Path(plan_path)
    plan_file = None if plan_input is None else plan_input.resolve()
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "validator": "validate_beris_edwards_outputs.py",
        "validator_sha256": _sha256_file(Path(__file__).resolve()),
        "runner_sha256": _sha256_file(RUNNER_PATH),
        "run_dir": str(directory),
        "plan_path": None if plan_file is None else str(plan_file),
        "run_id": run_id,
        "passed": False,
        "checks": {},
        "errors": [],
    }
    checks: dict[str, Any] = report["checks"]
    errors: list[str] = report["errors"]

    if directory_input.is_symlink():
        errors.append("run_directory: --run-dir must not be a symlink")
        return report
    if not directory.is_dir():
        errors.append(f"run_directory: {directory} is not a directory")
        return report
    if plan_file is None:
        errors.append("validation_plan: --plan is required for a formal PASS")
        return report
    if plan_input is not None and plan_input.is_symlink():
        errors.append("validation_plan: --plan must not be a symlink")
        return report

    _record_check(
        checks,
        errors,
        "complete_marker",
        lambda: _validate_complete_marker(directory / "COMPLETE"),
    )
    metadata = _record_check(
        checks,
        errors,
        "metadata_json",
        lambda: _read_json_object(directory / "metadata.json"),
    )
    plan = _record_check(
        checks,
        errors,
        "validation_plan_json",
        lambda: _read_json_object(plan_file),
    )
    plan_row = None
    if plan is not None:
        tool_hashes = plan.get("validation_tools_sha256")
        if not isinstance(tool_hashes, dict):
            errors.append("plan_binding: validation_tools_sha256 is missing or invalid")
            tool_hashes = {}
        report["plan_file_sha256"] = _sha256_file(plan_file)

        def record_binding(label: str, relative: str, actual: str) -> None:
            recorded = tool_hashes.get(relative)
            if recorded is None:
                report[f"{label}_plan_binding"] = "missing"
                errors.append(f"{label}_plan_binding: source SHA-256 is missing from plan")
            elif recorded != actual:
                report[f"{label}_plan_binding"] = "mismatch"
                errors.append(
                    f"{label}_plan_binding: source SHA-256 differs from plan"
                )
            else:
                report[f"{label}_plan_binding"] = "bound"

        record_binding(
            "validator",
            str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            report["validator_sha256"],
        )
        record_binding(
            "runner",
            str(RUNNER_PATH.relative_to(PROJECT_ROOT)),
            report["runner_sha256"],
        )
        plan_row = _record_check(
            checks,
            errors,
            "runner_completed_run_contract",
            lambda: _select_plan_run(plan, run_dir=directory, run_id=run_id),
        )

    planned_gate: float | None = None
    if plan is not None:
        candidate = plan.get("advisory_gates", {}).get(
            "divergence_and_schur_relative_residual_max"
        )
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            candidate = float(candidate)
            if math.isfinite(candidate) and candidate > 0.0:
                planned_gate = candidate
    formal_gate = planned_gate if planned_gate is not None else DEFAULT_RESIDUAL_MAX

    def resolve_limit(value: float | None, label: str) -> float:
        if value is None:
            return formal_gate
        if isinstance(value, bool):
            errors.append(f"thresholds: {label} must be positive and finite")
            return formal_gate
        try:
            result = float(value)
        except (TypeError, ValueError):
            errors.append(f"thresholds: {label} must be positive and finite")
            return formal_gate
        if not math.isfinite(result) or result <= 0.0:
            errors.append(f"thresholds: {label} must be positive and finite")
            return formal_gate
        if result > formal_gate:
            errors.append(
                f"thresholds: {label}={result:.17g} may not relax the formal "
                f"plan gate {formal_gate:.17g}"
            )
            return formal_gate
        return result

    div_limit = resolve_limit(div_rel_max, "div_rel_max")
    schur_limit = resolve_limit(
        schur_rel_residual_max, "schur_rel_residual_max"
    )
    report["thresholds"] = {
        "div_rel_max": div_limit,
        "schur_rel_residual_max": schur_limit,
        "source": (
            "cli"
            if div_rel_max is not None or schur_rel_residual_max is not None
            else "validation_plan" if planned_gate is not None else "validator_default"
        ),
    }

    contract = None
    if metadata is not None:
        contract = _record_check(
            checks, errors, "metadata_contract", lambda: _metadata_contract(metadata)
        )
    if contract is not None:
        (
            final_step,
            shape,
            dtype,
            save_start_step,
            save_interval,
            diagnostic_interval,
        ) = contract
        if plan_row is not None:
            config = plan_row["config"]
            planned_contract = (
                parse_integral_step(config["steps"], label="plan steps"),
                _parse_shape(config["shape"], label="plan shape"),
                _parse_real_dtype(config["dtype"], label="plan dtype"),
                parse_integral_step(
                    config["save_start_step"],
                    label="plan save_start_step",
                    maximum=final_step,
                ),
                parse_integral_step(
                    config["save_interval"],
                    label="plan save_interval",
                    minimum=1,
                ),
                parse_integral_step(
                    config["diagnostic_interval"],
                    label="plan diagnostic_interval",
                    minimum=1,
                    maximum=final_step,
                ),
            )
            if contract != planned_contract:
                message = "metadata contract differs from selected plan run"
                checks["plan_data_contract"] = {"passed": False, "error": message}
                errors.append(f"plan_data_contract: {message}")
            else:
                checks["plan_data_contract"] = {"passed": True}

        snapshot_steps = _scheduled_steps(
            final_step=final_step,
            start_step=save_start_step,
            interval=save_interval,
        )
        diagnostic_steps = _scheduled_steps(
            final_step=final_step,
            start_step=0,
            interval=diagnostic_interval,
        )
        critical_paths = (
            plan_file,
            directory / "COMPLETE",
            directory / "metadata.json",
            directory / "diagnostics.npy",
            directory / "diagnostics.csv",
            *(
                directory / f"{field}_{step}.npy"
                for step in snapshot_steps
                for field in ("Q", "u", "p")
            ),
        )
        identities_before = _record_check(
            checks,
            errors,
            "critical_files_before_scan",
            lambda: _collect_identities(critical_paths),
        )
        report["final_step"] = final_step
        report["shape"] = list(shape)
        report["dtype"] = dtype.name
        snapshots = _record_check(
            checks,
            errors,
            "snapshots",
            lambda: _validate_snapshot_set(
                directory,
                expected_steps=snapshot_steps,
                shape=shape,
                dtype=dtype,
            ),
        )
        if snapshots is not None:
            report["snapshots"] = snapshots
        diagnostics = _record_check(
            checks,
            errors,
            "diagnostics",
            lambda: _validate_diagnostics(
                directory,
                expected_steps=diagnostic_steps,
                div_rel_max=div_limit,
                schur_rel_residual_max=schur_limit,
            ),
        )
        if diagnostics is not None:
            report["diagnostics"] = diagnostics
        if identities_before is not None and metadata is not None and plan is not None:
            _record_check(
                checks,
                errors,
                "critical_files_after_scan",
                lambda: _validate_stability(
                    critical_paths,
                    identities_before,
                    complete_path=directory / "COMPLETE",
                    metadata_path=directory / "metadata.json",
                    metadata_before=metadata,
                    plan_path=plan_file,
                    plan_before=plan,
                ),
            )

    report["passed"] = not errors
    return report


def _validate_complete_marker(path: Path) -> dict[str, Any]:
    _file_identity(path)
    marker = path.read_text(encoding="utf-8").strip()
    if marker != "complete":
        raise ValueError(f"invalid COMPLETE marker {marker!r}")
    return {"path": str(path), "value": marker}


def _positive_finite(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Runner validation plan used for exact metadata/provenance checks.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Plan run_id; otherwise --run-dir must exactly match one plan output_dir.",
    )
    parser.add_argument("--div-rel-max", type=_positive_finite, default=None)
    parser.add_argument(
        "--schur-rel-residual-max",
        type=_positive_finite,
        default=None,
    )
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    report = args.report.resolve()
    if report == run_dir or report.is_relative_to(run_dir):
        parser.error("--report must be outside --run-dir to keep outputs read-only")
    if args.run_id is not None and args.plan is None:
        parser.error("--run-id requires --plan")
    return args


def _write_report_exclusive(path: Path, report: dict[str, Any]) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.report.exists():
        print(f"refusing to overwrite existing report: {args.report}", file=sys.stderr)
        return 2
    try:
        report = validate_outputs(
            args.run_dir,
            plan_path=args.plan,
            run_id=args.run_id,
            div_rel_max=args.div_rel_max,
            schur_rel_residual_max=args.schur_rel_residual_max,
        )
    except Exception as error:  # Preserve an actionable JSON artifact on failure.
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "validator": "validate_beris_edwards_outputs.py",
            "run_dir": str(args.run_dir.resolve()),
            "plan_path": None if args.plan is None else str(args.plan.resolve()),
            "run_id": args.run_id,
            "passed": False,
            "checks": {},
            "errors": [f"internal_validation_error: {type(error).__name__}: {error}"],
        }
    try:
        _write_report_exclusive(args.report, report)
    except FileExistsError:
        print(f"refusing to overwrite existing report: {args.report}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
