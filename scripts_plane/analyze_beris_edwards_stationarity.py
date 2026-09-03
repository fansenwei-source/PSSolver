#!/usr/bin/env python3
"""Read-only stationarity screening for a validated Beris--Edwards long run.

The analyzer is deliberately post-hoc: an already completed simulation plan
cannot bind a tool that did not exist when the simulation was launched.  The
report therefore records the analyzer commit and source hashes independently,
while reusing the version-bound output validator as its input gate.

Twenty-one frames over one finite time window can exclude obvious drift, but
cannot prove strict stationarity or a paper-level benchmark reproduction.  The
strongest positive classification emitted here is consequently
``provisionally_stationary``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Callable, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_plane.run_beris_edwards_validation import (  # noqa: E402
    _sha256_file,
)
from scripts_plane.validate_beris_edwards_outputs import (  # noqa: E402
    _collect_identities,
    _metadata_contract,
    _read_json_object,
    _scheduled_steps,
    _select_plan_run,
    validate_outputs,
)


Q_BC = ("periodic", "periodic", "neumann")
SCHEMA_VERSION = 1
CLASS_PROVISIONAL = "provisionally_stationary"
CLASS_NONSTATIONARY = "non_stationary"
CLASS_INCONCLUSIVE = "inconclusive"
CLASS_NOT_EVALUATED = "not_evaluated"
CORE_FIELD_OBSERVABLES = (
    "u_rms",
    "principal_S_mean_over_S_bulk",
    "Q_magnitude_mean_over_S_bulk",
    "low_principal_S_fraction",
    "biaxiality_fraction_gt_0p5",
    "total_ldg_free_energy_density_mean",
)
TOPOLOGY_OBSERVABLES = (
    "defect_point_density_proxy",
    "defect_line_count",
    "through_line_count",
    "through_line_presence",
)
FIG4_OBSERVABLES = ("mean_sigma_over_h",)


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive_float(value: str) -> float:
    number = _finite_float(value, label="value")
    if number <= 0.0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _unit_interval(value: str) -> float:
    number = _finite_float(value, label="value")
    if number < 0.0 or number > 1.0:
        raise argparse.ArgumentTypeError("must lie in [0, 1]")
    return number


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_stable_bytes(path: Path, *, label: str) -> bytes:
    identities_before = _collect_identities((path,))
    payload = path.read_bytes()
    identities_after = _collect_identities((path,))
    if identities_after != identities_before:
        raise RuntimeError(f"{label} changed while being read")
    return payload


def _strict_json_object_from_bytes(
    payload: bytes, *, path: Path
) -> dict[str, Any]:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token!r} in {path}")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not valid UTF-8 JSON") from error
    document = json.loads(
        text,
        parse_constant=reject_constant,
        object_pairs_hook=reject_pairs,
    )
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return document


def _read_expected_json_object(
    path: Path, *, expected_sha256: str, label: str
) -> tuple[dict[str, Any], str]:
    payload = _read_stable_bytes(path, label=label)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: {digest} != {expected_sha256}"
        )
    return _strict_json_object_from_bytes(payload, path=path), digest


def _validated_sha256(value: str, *, label: str) -> str:
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _git_provenance(
    *, require_clean: bool, expected_head: str | None
) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    if expected_head is not None and head != expected_head:
        raise RuntimeError(
            f"analysis Git HEAD mismatch: expected {expected_head}, got {head}"
        )
    if require_clean and status:
        raise RuntimeError("formal analysis requires a clean Git worktree")
    return {
        "head": head,
        "expected_head": expected_head,
        "expected_head_matched": (
            expected_head is not None and head == expected_head
        ),
        "status_porcelain": status,
        "clean": not bool(status),
        "require_clean_requested": require_clean,
        "git_version": run("--version"),
        "analysis_tool_simulation_plan_binding": "not_applicable_post_hoc",
    }


def _runtime_provenance(*, energy_device: str) -> dict[str, Any]:
    import matplotlib
    import scipy
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_details = None
    if energy_device.startswith("cuda") and cuda_available:
        device = torch.device(energy_device)
        index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        device_details = {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        }
    return {
        "command": [sys.executable, *sys.argv],
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "numpy": {"version": np.__version__, "path": np.__file__},
        "scipy": {"version": scipy.__version__, "path": scipy.__file__},
        "torch": {
            "version": torch.__version__,
            "path": torch.__file__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
        },
        "matplotlib": {
            "version": matplotlib.__version__,
            "path": matplotlib.__file__,
        },
        "energy_device_requested": energy_device,
        "cuda_device": device_details,
    }


def _validate_external_report(
    report: dict[str, Any],
    *,
    run_dir: Path,
    plan_path: Path,
    run_id: str,
    plan_sha256: str,
) -> None:
    required = {
        "passed": True,
        "runner_plan_binding": "bound",
        "validator_plan_binding": "bound",
        "run_dir": str(run_dir),
        "plan_path": str(plan_path),
        "run_id": run_id,
        "plan_file_sha256": plan_sha256,
    }
    mismatches = {
        key: {"expected": expected, "actual": report.get(key)}
        for key, expected in required.items()
        if report.get(key) != expected
    }
    if report.get("errors") != []:
        mismatches["errors"] = {"expected": [], "actual": report.get("errors")}
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or any(
        not isinstance(item, dict) or item.get("passed") is not True
        for item in checks.values()
    ):
        mismatches["checks"] = {
            "expected": "nonempty and all passed",
            "actual": checks,
        }
    if mismatches:
        raise ValueError(f"formal validation report mismatch: {mismatches}")


def _input_paths(
    run_dir: Path,
    plan_path: Path,
    validation_report: Path,
    checksum_manifest: Path,
    steps: Iterable[int],
    *,
    additional_paths: tuple[Path, ...] = (),
) -> tuple[Path, ...]:
    return (
        plan_path,
        validation_report,
        checksum_manifest,
        run_dir / "COMPLETE",
        run_dir / "metadata.json",
        run_dir / "diagnostics.npy",
        run_dir / "diagnostics.csv",
        *(
            run_dir / f"{field}_{step}.npy"
            for step in steps
            for field in ("Q", "u", "p")
        ),
        *additional_paths,
    )

_MANIFEST_PATH_KEYS = (
    "path",
    "absolute_path",
    "relative_path",
    "file",
    "filename",
    "source_path",
)
_MANIFEST_SHA_KEYS = ("sha256", "sha_256", "content_sha256", "digest")
_MANIFEST_SIZE_KEYS = ("size", "size_bytes", "bytes", "st_size")
_MANIFEST_MTIME_NS_KEYS = (
    "mtime_ns",
    "mtime_epoch_ns",
    "st_mtime_ns",
)
_MANIFEST_MTIME_KEYS = ("mtime", "st_mtime")


def _first_manifest_value(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _manifest_records(value: Any) -> list[dict[str, Any]]:
    """Extract file records without assuming one HPCC container layout."""

    records: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            path = _first_manifest_value(item, _MANIFEST_PATH_KEYS)
            digest = _first_manifest_value(item, _MANIFEST_SHA_KEYS)
            if isinstance(path, str) and digest is not None:
                records.append(item)
            for key, child in item.items():
                if isinstance(child, dict):
                    child_digest = _first_manifest_value(
                        child, _MANIFEST_SHA_KEYS
                    )
                    if (
                        child_digest is not None
                        and _first_manifest_value(
                            child, _MANIFEST_PATH_KEYS
                        )
                        is None
                        and isinstance(key, str)
                        and ("/" in key or "\\" in key)
                    ):
                        records.append({"path": key, **child})
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return records


def _match_manifest_path(
    raw_path: str, critical_paths: tuple[Path, ...]
) -> Path | None:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        return resolved if resolved in critical_paths else None
    if ".." in candidate.parts:
        raise ValueError(
            f"checksum manifest relative path must not contain '..': {raw_path!r}"
        )
    normalized = candidate.as_posix().lstrip("./")
    matches = [
        path
        for path in critical_paths
        if path.as_posix() == normalized
        or path.as_posix().endswith("/" + normalized)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"checksum manifest path is ambiguous: {raw_path!r}"
        )
    return matches[0] if matches else None


_EXTERNAL_PAYLOAD_SCOPE = "home_control_payload"
_EXTERNAL_PAYLOAD_PHASE = "stable_payload_before_slurm_log_close"
_FINAL_MANIFEST_PHASE = "final_after_slurm_log_close"
_EXTERNAL_PAYLOAD_STABLE_FIELDS = (
    "schema_version",
    "actual_validation_plan",
    "git_head",
    "job_id",
    "output_root",
    "run_id",
    "scratch_policy",
)


def _manifest_payload_declaration(
    manifest: dict[str, Any],
) -> tuple[str, str] | None:
    declared_items = [
        (key, value)
        for key, value in manifest.items()
        if key in ("payload_sha256", "manifest_payload_sha256")
    ]
    if not declared_items:
        return None
    if len(declared_items) != 1:
        raise ValueError("checksum manifest has ambiguous payload hash fields")
    key, declared = declared_items[0]
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
    ):
        raise ValueError("checksum manifest payload SHA-256 is malformed")
    return key, declared


def _external_payload_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return []
    return [
        record
        for record in entries
        if isinstance(record, dict)
        and record.get("scope") == _EXTERNAL_PAYLOAD_SCOPE
    ]


def _external_payload_record_and_path(
    manifest: dict[str, Any],
    *,
    checksum_manifest_path: Path,
) -> tuple[dict[str, Any], Path] | None:
    records = _external_payload_records(manifest)
    if not records:
        return None
    if len(records) != 1:
        raise ValueError(
            "checksum manifest must contain exactly one external checksum "
            f"payload record with scope {_EXTERNAL_PAYLOAD_SCOPE!r}"
        )
    declaration = _manifest_payload_declaration(manifest)
    if declaration is None:
        raise ValueError(
            "external checksum payload record requires a top-level payload "
            "SHA-256 declaration"
        )
    _, declared = declaration
    record = records[0]
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("external checksum payload path is missing")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        raise ValueError("external checksum payload path must be absolute")
    if ".." in candidate.parts:
        raise ValueError("external checksum payload path must not contain '..'")

    control_root = checksum_manifest_path.resolve().parent
    manifest_resolved = checksum_manifest_path.resolve()
    if candidate == manifest_resolved:
        raise ValueError(
            "external checksum payload must not be the checksum manifest itself"
        )
    try:
        relative = candidate.relative_to(control_root)
    except ValueError as error:
        raise ValueError(
            "external checksum payload is outside checksum manifest control "
            "directory"
        ) from error
    if not relative.parts:
        raise ValueError("external checksum payload path must name a file")

    component = control_root
    for part in relative.parts:
        component = component / part
        if component.is_symlink():
            raise ValueError(
                "external checksum payload path must not contain symlinks"
            )
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("external checksum payload file is missing") from error
    if resolved == control_root or not resolved.is_relative_to(control_root):
        raise ValueError(
            "external checksum payload is outside checksum manifest control "
            "directory"
        )
    if not resolved.is_file():
        raise ValueError("external checksum payload must be a regular file")

    digest = record.get("sha256")
    if digest != declared:
        raise ValueError(
            "external checksum payload record SHA-256 does not match the "
            "top-level declaration"
        )
    size = record.get("size_bytes")
    mtime_ns = record.get("mtime_ns")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("external checksum payload record size_bytes is invalid")
    if (
        isinstance(mtime_ns, bool)
        or not isinstance(mtime_ns, int)
        or mtime_ns < 0
    ):
        raise ValueError("external checksum payload record mtime_ns is invalid")
    return record, resolved


def _strict_external_entries(
    document: dict[str, Any], *, label: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{label} entries must be a list")
    entry_count = document.get("entry_count")
    if (
        isinstance(entry_count, bool)
        or not isinstance(entry_count, int)
        or entry_count != len(entries)
    ):
        raise ValueError(f"{label} entry_count does not match entries")
    by_path: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(entries):
        if not isinstance(record, dict):
            raise ValueError(f"{label} entry {index} must be an object")
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{label} entry {index} has no path")
        if raw_path in by_path:
            raise ValueError(f"{label} contains duplicate path {raw_path!r}")
        by_path[raw_path] = record
    return entries, by_path


def _verify_external_manifest_payload(
    manifest: dict[str, Any],
    *,
    checksum_manifest_path: Path,
    declaration: tuple[str, str],
    record: dict[str, Any],
    payload_path: Path,
) -> dict[str, Any]:
    _, declared = declaration
    stat = payload_path.stat()
    if stat.st_size != record["size_bytes"]:
        raise ValueError("external checksum payload size mismatch")
    if stat.st_mtime_ns != record["mtime_ns"]:
        raise ValueError("external checksum payload mtime_ns mismatch")
    payload_bytes = _read_stable_bytes(
        payload_path, label="external checksum payload"
    )
    actual_digest = hashlib.sha256(payload_bytes).hexdigest()
    if actual_digest != declared:
        raise ValueError("external checksum payload SHA-256 mismatch")
    payload = _strict_json_object_from_bytes(payload_bytes, path=payload_path)

    if manifest.get("phase") != _FINAL_MANIFEST_PHASE:
        raise ValueError(
            "external checksum payload convention requires final manifest "
            f"phase {_FINAL_MANIFEST_PHASE!r}"
        )
    if payload.get("phase") != _EXTERNAL_PAYLOAD_PHASE:
        raise ValueError(
            "external checksum payload phase must be "
            f"{_EXTERNAL_PAYLOAD_PHASE!r}"
        )
    for field in _EXTERNAL_PAYLOAD_STABLE_FIELDS:
        if field not in manifest or field not in payload:
            raise ValueError(
                f"external checksum payload stable field {field!r} is missing"
            )
        if payload[field] != manifest[field]:
            raise ValueError(
                f"external checksum payload stable field mismatch: {field}"
            )

    final_entries, final_by_path = _strict_external_entries(
        manifest, label="final checksum manifest"
    )
    payload_entries, payload_by_path = _strict_external_entries(
        payload, label="external checksum payload"
    )
    payload_path_text = str(payload_path)
    control_root = checksum_manifest_path.resolve().parent
    for raw_path in payload_by_path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = control_root / candidate
        if candidate.resolve(strict=False) == payload_path:
            raise ValueError(
                "external checksum payload must not contain its own file record"
            )
    for raw_path, payload_entry in payload_by_path.items():
        final_entry = final_by_path.get(raw_path)
        if final_entry != payload_entry:
            raise ValueError(
                "external checksum payload entries are not an exact subset "
                f"of the final checksum manifest: {raw_path}"
            )

    return {
        "status": "verified",
        "declared_sha256": declared,
        "convention": "external_raw_bytes_payload",
        "external_payload_path": payload_path_text,
        "external_payload_sha256": actual_digest,
        "external_payload_record_scope": record["scope"],
        "external_payload_record_size_bytes": stat.st_size,
        "external_payload_record_mtime_ns": stat.st_mtime_ns,
        "payload_phase": payload["phase"],
        "manifest_phase": manifest["phase"],
        "payload_entry_count": len(payload_entries),
        "manifest_entry_count": len(final_entries),
        "payload_entries_exact_subset": True,
        "checksum_manifest_path": str(checksum_manifest_path),
    }


def _verify_manifest_payload_hash(
    manifest: dict[str, Any], *, checksum_manifest_path: Path
) -> dict[str, Any]:
    declaration = _manifest_payload_declaration(manifest)
    # The explicit production scope is the convention discriminator.  Never
    # accept a coincidental canonical hash match in place of verifying the
    # external file that the signed final manifest says it created.
    external = _external_payload_record_and_path(
        manifest, checksum_manifest_path=checksum_manifest_path
    )
    if external is not None:
        if declaration is None:  # Defensive; the resolver rejects this first.
            raise ValueError("external checksum payload declaration is missing")
        record, payload_path = external
        return _verify_external_manifest_payload(
            manifest,
            checksum_manifest_path=checksum_manifest_path,
            declaration=declaration,
            record=record,
            payload_path=payload_path,
        )
    if declaration is None:
        return {
            "status": "not_declared",
            "note": (
                "The external manifest did not expose a recognized top-level "
                "payload hash; its own file SHA-256 is still bound separately."
            ),
        }
    key, declared = declaration
    candidates: dict[str, Any] = {
        "top_level_without_declared_hash": {
            name: value for name, value in manifest.items() if name != key
        }
    }
    if "payload" in manifest:
        candidates["payload_object"] = manifest["payload"]
    if "entries" in manifest:
        candidates["entries_object"] = manifest["entries"]
    matches = [
        name
        for name, value in candidates.items()
        if _canonical_sha256(value) == declared
    ]
    if not matches:
        raise ValueError(
            "checksum manifest payload SHA-256 does not match any supported "
            "canonical payload convention"
        )
    return {
        "status": "verified",
        "declared_sha256": declared,
        "canonical_payload": matches[0],
    }


def _verify_checksum_manifest(
    manifest: dict[str, Any],
    *,
    critical_paths: tuple[Path, ...],
    checksum_manifest_path: Path,
) -> dict[str, Any]:
    """Bind every critical input to a unique path/size/SHA/mtime record."""

    targets = tuple(
        path.resolve()
        for path in critical_paths
        if path.resolve() != checksum_manifest_path.resolve()
    )
    records = _manifest_records(manifest)
    matched: dict[Path, dict[str, Any]] = {}
    for record in records:
        raw_path = _first_manifest_value(record, _MANIFEST_PATH_KEYS)
        if not isinstance(raw_path, str):
            continue
        path = _match_manifest_path(raw_path, targets)
        if path is None:
            continue
        if path in matched:
            raise ValueError(
                f"checksum manifest has duplicate records for {path}"
            )
        matched[path] = record
    missing = [str(path) for path in targets if path not in matched]
    if missing:
        raise ValueError(
            "checksum manifest is missing critical inputs: " + ", ".join(missing)
        )

    for path in targets:
        record = matched[path]
        digest = _first_manifest_value(record, _MANIFEST_SHA_KEYS)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                f"checksum manifest SHA-256 is malformed for {path}"
            )
        size = _first_manifest_value(record, _MANIFEST_SIZE_KEYS)
        if isinstance(size, bool) or not isinstance(size, int):
            raise ValueError(f"checksum manifest size is missing for {path}")
        stat = path.stat()
        if size != stat.st_size:
            raise ValueError(
                f"checksum manifest size mismatch for {path}: "
                f"{size} != {stat.st_size}"
            )
        actual_digest = _sha256_file(path)
        if digest != actual_digest:
            raise ValueError(f"checksum manifest SHA-256 mismatch for {path}")
        mtime_ns = _first_manifest_value(record, _MANIFEST_MTIME_NS_KEYS)
        mtime = _first_manifest_value(record, _MANIFEST_MTIME_KEYS)
        if mtime_ns is not None:
            if isinstance(mtime_ns, bool) or int(mtime_ns) != stat.st_mtime_ns:
                raise ValueError(
                    f"checksum manifest mtime_ns mismatch for {path}"
                )
        elif mtime is not None:
            if isinstance(mtime, bool) or not math.isclose(
                float(mtime), stat.st_mtime, rel_tol=0.0, abs_tol=1.0e-6
            ):
                raise ValueError(
                    f"checksum manifest mtime mismatch for {path}"
                )
        else:
            raise ValueError(f"checksum manifest mtime is missing for {path}")

    return {
        "status": "verified",
        "manifest_record_count_discovered": len(records),
        "critical_record_count_verified": len(targets),
        "payload_hash": _verify_manifest_payload_hash(
            manifest, checksum_manifest_path=checksum_manifest_path
        ),
    }



def _compact_q_statistics(
    path: Path,
    *,
    shape: tuple[int, int, int],
    dtype: np.dtype[Any],
    s_bulk: float,
    ldg_a: float,
    ldg_b: float,
    ldg_c: float,
    chunk_x: int,
) -> dict[str, float]:
    q = np.load(path, mmap_mode="r", allow_pickle=False)
    if q.shape != (*shape, 5) or q.dtype != dtype:
        raise ValueError(
            f"{path} has shape/dtype {q.shape}/{q.dtype}; "
            f"expected {(*shape, 5)}/{dtype}"
        )
    count = int(np.prod(shape))
    sums = {
        "principal_S": 0.0,
        "principal_S_sq": 0.0,
        "Q_magnitude": 0.0,
        "Q_magnitude_sq": 0.0,
        "biaxiality": 0.0,
        "bulk_energy": 0.0,
    }
    component_sums = np.zeros(5, dtype=np.float64)
    low_order_count = 0
    biaxial_count = 0
    principal_min = math.inf
    principal_max = -math.inf
    magnitude_min = math.inf
    magnitude_max = -math.inf

    for start in range(0, shape[0], chunk_x):
        slab = np.asarray(q[start : start + chunk_x], dtype=np.float64)
        qxx, qxy, qxz, qyy, qyz = np.moveaxis(slab, -1, 0)
        qzz = -qxx - qyy
        tr_q2 = (
            qxx * qxx
            + qyy * qyy
            + qzz * qzz
            + 2.0 * (qxy * qxy + qxz * qxz + qyz * qyz)
        )
        determinant = (
            qxx * (qyy * qzz - qyz * qyz)
            - qxy * (qxy * qzz - qyz * qxz)
            + qxz * (qxy * qyz - qyy * qxz)
        )
        tr_q3 = 3.0 * determinant
        magnitude = np.sqrt(np.maximum((2.0 / 3.0) * tr_q2, 0.0))

        principal = np.zeros_like(tr_q2)
        nonzero = tr_q2 > np.finfo(np.float64).tiny
        argument = np.zeros_like(tr_q2)
        argument[nonzero] = (
            math.sqrt(6.0)
            * tr_q3[nonzero]
            / np.power(tr_q2[nonzero], 1.5)
        )
        np.clip(argument, -1.0, 1.0, out=argument)
        principal[nonzero] = (
            2.0
            * np.sqrt(tr_q2[nonzero] / 6.0)
            * np.cos(np.arccos(argument[nonzero]) / 3.0)
        )

        biaxiality = np.zeros_like(tr_q2)
        biaxiality[nonzero] = 1.0 - (
            6.0 * tr_q3[nonzero] * tr_q3[nonzero]
            / np.power(tr_q2[nonzero], 3.0)
        )
        np.clip(biaxiality, 0.0, 1.0, out=biaxiality)
        bulk = (
            0.5 * ldg_a * tr_q2
            + (ldg_b / 3.0) * tr_q3
            + 0.25 * ldg_c * tr_q2 * tr_q2
        )

        sums["principal_S"] += float(np.sum(principal, dtype=np.float64))
        sums["principal_S_sq"] += float(
            np.sum(principal * principal, dtype=np.float64)
        )
        sums["Q_magnitude"] += float(np.sum(magnitude, dtype=np.float64))
        sums["Q_magnitude_sq"] += float(
            np.sum(magnitude * magnitude, dtype=np.float64)
        )
        sums["biaxiality"] += float(np.sum(biaxiality, dtype=np.float64))
        sums["bulk_energy"] += float(np.sum(bulk, dtype=np.float64))
        component_sums += np.sum(slab, axis=(0, 1, 2), dtype=np.float64)
        low_order_count += int(np.count_nonzero(principal < 0.5 * s_bulk))
        biaxial_count += int(np.count_nonzero(biaxiality > 0.5))
        principal_min = min(principal_min, float(np.min(principal)))
        principal_max = max(principal_max, float(np.max(principal)))
        magnitude_min = min(magnitude_min, float(np.min(magnitude)))
        magnitude_max = max(magnitude_max, float(np.max(magnitude)))

    principal_mean = sums["principal_S"] / count
    magnitude_mean = sums["Q_magnitude"] / count
    return {
        "principal_S_mean": principal_mean,
        "principal_S_std": math.sqrt(
            max(sums["principal_S_sq"] / count - principal_mean**2, 0.0)
        ),
        "principal_S_min": principal_min,
        "principal_S_max": principal_max,
        "principal_S_mean_over_S_bulk": principal_mean / s_bulk,
        "Q_magnitude_mean": magnitude_mean,
        "Q_magnitude_std": math.sqrt(
            max(sums["Q_magnitude_sq"] / count - magnitude_mean**2, 0.0)
        ),
        "Q_magnitude_min": magnitude_min,
        "Q_magnitude_max": magnitude_max,
        "Q_magnitude_mean_over_S_bulk": magnitude_mean / s_bulk,
        "low_principal_S_fraction": low_order_count / count,
        "biaxiality_mean": sums["biaxiality"] / count,
        "biaxiality_fraction_gt_0p5": biaxial_count / count,
        "bulk_ldg_free_energy_density_mean": sums["bulk_energy"] / count,
        **{
            f"mean_Q_{name}": float(component_sums[index] / count)
            for index, name in enumerate(("xx", "xy", "xz", "yy", "yz"))
        },
    }


def _velocity_statistics(
    path: Path,
    *,
    shape: tuple[int, int, int],
    dtype: np.dtype[Any],
    chunk_x: int,
) -> dict[str, float]:
    velocity = np.load(path, mmap_mode="r", allow_pickle=False)
    if velocity.shape != (*shape, 3) or velocity.dtype != dtype:
        raise ValueError(
            f"{path} has shape/dtype {velocity.shape}/{velocity.dtype}; "
            f"expected {(*shape, 3)}/{dtype}"
        )
    count = int(np.prod(shape))
    component_sum = np.zeros(3, dtype=np.float64)
    sum_speed = 0.0
    sum_speed_sq = 0.0
    max_speed = 0.0
    for start in range(0, shape[0], chunk_x):
        slab = np.asarray(velocity[start : start + chunk_x], dtype=np.float64)
        speed_sq = np.sum(slab * slab, axis=-1, dtype=np.float64)
        speed = np.sqrt(speed_sq)
        component_sum += np.sum(slab, axis=(0, 1, 2), dtype=np.float64)
        sum_speed += float(np.sum(speed, dtype=np.float64))
        sum_speed_sq += float(np.sum(speed_sq, dtype=np.float64))
        max_speed = max(max_speed, float(np.max(speed)))
    mean_speed_sq = sum_speed_sq / count
    return {
        "mean_u_x": float(component_sum[0] / count),
        "mean_u_y": float(component_sum[1] / count),
        "mean_u_z": float(component_sum[2] / count),
        "speed_mean": sum_speed / count,
        "speed_max": max_speed,
        "u_rms": math.sqrt(mean_speed_sq),
        "velocity_energy_density_mean": 0.5 * mean_speed_sq,
    }


class SpectralElasticEnergy:
    """Production-basis elastic energy evaluator, initialized once per run."""

    def __init__(
        self,
        *,
        shape: tuple[int, int, int],
        lengths: tuple[float, float, float],
        dtype: np.dtype[Any],
        device: str,
        ldg_l1: float,
    ) -> None:
        import torch
        from pssolver import TensorProductTransformBackend

        if dtype != np.dtype("float64"):
            raise ValueError("spectral stationarity energy currently requires float64")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA energy analysis requested but CUDA is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.ldg_l1 = ldg_l1
        self.backend = TensorProductTransformBackend(
            shape=shape,
            lengths=lengths,
            device=self.device,
            dtype=torch.float64,
        )

    def evaluate(self, q_path: Path) -> float:
        torch = self.torch
        q_mmap = np.load(q_path, mmap_mode="r", allow_pickle=False)
        q_numpy = np.array(np.moveaxis(q_mmap, -1, 0), copy=True)
        with torch.no_grad():
            q = torch.from_numpy(q_numpy).to(self.device)
            q_hat = self.backend.forward(q, Q_BC)
            gradient_sq_mean = 0.0
            for axis in range(3):
                gradient_hat, gradient_bcs = self.backend.gradient_hat(
                    q_hat, Q_BC, axis
                )
                gradient = self.backend.inverse(gradient_hat, gradient_bcs)
                gxx, gxy, gxz, gyy, gyz = gradient
                gzz = -gxx - gyy
                contraction = (
                    gxx.square()
                    + gyy.square()
                    + gzz.square()
                    + 2.0 * (gxy.square() + gxz.square() + gyz.square())
                )
                gradient_sq_mean += float(contraction.mean().item())
            del q_hat, q
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        return 0.5 * self.ldg_l1 * gradient_sq_mean


def _bind_source_tree(package_root: Path) -> dict[str, Any]:
    package_root = package_root.resolve()
    if package_root.is_symlink() or not package_root.is_dir():
        raise RuntimeError(
            f"nematics3d package root is not a regular directory: {package_root}"
        )
    artifacts: list[dict[str, Any]] = []
    for path in sorted(package_root.rglob("*")):
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise RuntimeError(
                f"nematics3d source tree contains a symlink: {path}"
            )
        if path.is_file():
            artifacts.append(
                {
                    "relative_path": str(path.relative_to(package_root)),
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    if not artifacts:
        raise RuntimeError(
            f"nematics3d package source tree is empty: {package_root}"
        )
    return {
        "package_root": str(package_root),
        "file_count": len(artifacts),
        "tree_sha256": _canonical_sha256(
            [
                {
                    "relative_path": artifact["relative_path"],
                    "size": artifact["size"],
                    "sha256": artifact["sha256"],
                }
                for artifact in artifacts
            ]
        ),
        "artifacts": artifacts,
    }


def _bind_callable_artifacts(
    name: str,
    function: Callable[..., Any],
    *,
    package_root: Path,
) -> list[dict[str, Any]]:
    """Bind both a decorated callable and its unwrapped implementation."""
    try:
        implementation = inspect.unwrap(function)
    except ValueError as error:
        raise RuntimeError(
            f"cannot unwrap nematics3d callable {name}: {error}"
        ) from error
    decorated = implementation is not function

    def bind(source: Callable[..., Any], *, role: str, layer: str):
        source_name = inspect.getsourcefile(source) or inspect.getfile(source)
        unresolved_path = Path(source_name)
        source_path = unresolved_path.resolve()
        if unresolved_path.is_symlink() or not source_path.is_file():
            raise RuntimeError(
                f"cannot bind nematics3d {layer} for {name}: {source_path}"
            )
        try:
            source_path.relative_to(package_root.resolve())
        except ValueError as error:
            raise RuntimeError(
                f"nematics3d {layer} for {name} is outside the package tree: "
                f"{source_path}"
            ) from error
        return {
            "role": role,
            "layer": layer,
            "decorated": decorated,
            "module": getattr(source, "__module__", None),
            "qualname": getattr(source, "__qualname__", role),
            "path": str(source_path),
            "sha256": _sha256_file(source_path),
        }

    artifacts = []
    if decorated:
        artifacts.append(
            bind(
                function,
                role=f"{name}_runtime_wrapper",
                layer="runtime_wrapper",
            )
        )
    artifacts.append(
        bind(
            implementation,
            role=name,
            layer="unwrapped_implementation" if decorated else "implementation",
        )
    )
    return artifacts


def _load_defect_measurement() -> tuple[
    Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]], dict[str, Any]
]:
    try:
        from scripts_plane import analyze_fig4_sigma as sigma
    except Exception as error:
        raise RuntimeError(
            "defect/Fig.4 analysis requires an auditable nematics3d source; "
            f"import failed with {type(error).__name__}: {error}"
        ) from error

    module = sigma.n3d
    artifacts: list[dict[str, Any]] = []
    package_path = Path(module.__file__).resolve()
    if package_path.is_symlink() or not package_path.is_file():
        raise RuntimeError(
            f"nematics3d package artifact is not a regular file: {package_path}"
        )
    source_tree = _bind_source_tree(package_path.parent)
    artifacts.append(
        {
            "role": "package_module",
            "module": module.__name__,
            "path": str(package_path),
            "sha256": _sha256_file(package_path),
        }
    )
    for name in ("defect_detect", "defect_classify_into_lines"):
        function = getattr(module, name, None)
        if not callable(function):
            raise RuntimeError(f"nematics3d is missing callable {name}")
        artifacts.extend(
            _bind_callable_artifacts(
                name, function, package_root=package_path.parent
            )
        )

    adapter_path = (
        PROJECT_ROOT
        / "pssolver/models/active_nematics/nematics3d_adapter.py"
    ).resolve()
    helper_path = Path(sigma.__file__).resolve()
    provenance = {
        "module": module.__name__,
        "version": getattr(module, "__version__", None),
        "package_path": str(package_path),
        "implementation_artifacts": artifacts,
        "source_tree": source_tree,
        "adapter_path": str(adapter_path),
        "adapter_sha256": _sha256_file(adapter_path),
        "geometry_helper_path": str(helper_path),
        "geometry_helper_sha256": _sha256_file(helper_path),
        "boundary_flags": {
            "is_boundary_periodic": list(sigma.PERIODIC),
            "planes": list(sigma.PLANES),
        },
    }
    return sigma.measure_frame, provenance


def _acf_summary(
    values: np.ndarray, *, max_lag: int, frame_spacing: float
) -> dict[str, Any]:
    count = len(values)
    centered = values - np.mean(values)
    variance_sum = float(np.dot(centered, centered))
    signal_scale = max(float(np.dot(values, values)), 1.0)
    if variance_sum <= np.finfo(np.float64).eps * signal_scale:
        return {
            "status": "not_needed_zero_variance",
            "acf": [1.0],
            "statistical_inefficiency_frames": 1.0,
            "integrated_autocorrelation_time_frames": 0.5,
            "integrated_autocorrelation_time_time": 0.5 * frame_spacing,
            "effective_sample_size": float(count),
            "truncated_at_nonpositive": True,
            "reliable": count >= 2,
        }
    acf = [1.0]
    for lag in range(1, min(max_lag, count - 1) + 1):
        acf.append(
            float(np.dot(centered[:-lag], centered[lag:]) / variance_sum)
        )
    positive = []
    truncated = False
    for value in acf[1:]:
        if value <= 0.0:
            truncated = True
            break
        positive.append(value)
    inefficiency = max(1.0, 1.0 + 2.0 * sum(positive))
    tau_integrated = 0.5 * inefficiency
    effective = min(float(count), max(1.0, count / inefficiency))
    return {
        "status": "estimated",
        "acf": acf,
        "statistical_inefficiency_frames": inefficiency,
        "integrated_autocorrelation_time_frames": tau_integrated,
        "integrated_autocorrelation_time_time": (
            tau_integrated * frame_spacing
        ),
        "effective_sample_size": effective,
        "truncated_at_nonpositive": truncated,
        "reliable": truncated and count >= 15,
    }


def summarize_series(
    times: Iterable[float],
    values: Iterable[float | None],
    *,
    pass_relative_drift: float,
    fail_relative_drift: float,
    min_effective_samples: float,
    min_finite_frames: int,
    max_acf_lag: int,
) -> dict[str, Any]:
    time_array = np.asarray(tuple(times), dtype=np.float64)
    raw_values = tuple(values)
    value_array = np.asarray(
        [np.nan if value is None else float(value) for value in raw_values],
        dtype=np.float64,
    )
    if len(time_array) != len(value_array):
        raise ValueError("times and values must have equal lengths")
    if len(time_array) and not np.isfinite(time_array).all():
        raise ValueError("times must all be finite")
    if len(time_array) >= 2:
        spacings = np.diff(time_array)
        if np.any(spacings <= 0.0):
            raise ValueError("times must be strictly increasing")
        tolerance = 32.0 * np.finfo(np.float64).eps * max(
            1.0, float(np.max(np.abs(time_array)))
        )
        if not np.allclose(
            spacings, spacings[0], rtol=1.0e-10, atol=tolerance
        ):
            raise ValueError(
                "times must be equally spaced for frame-lag autocorrelation"
            )
        frame_spacing = float(spacings[0])
    else:
        frame_spacing = math.nan

    finite = np.isfinite(value_array)
    finite_count = int(np.count_nonzero(finite))
    missing_frames = finite_count != len(value_array)
    result: dict[str, Any] = {
        "frame_count": len(value_array),
        "finite_frame_count": finite_count,
        "finite_fraction": (
            finite_count / len(value_array) if len(value_array) else 0.0
        ),
        "frame_spacing_time": _jsonable(frame_spacing),
    }
    if len(value_array) < 3 or finite_count < min_finite_frames:
        result.update(
            {
                "classification": CLASS_INCONCLUSIVE,
                "reason_codes": ["insufficient_finite_frames"],
            }
        )
        return result

    midpoint_index = len(value_array) // 2
    early_indices = np.arange(0, midpoint_index)
    late_indices = np.arange(midpoint_index + 1, len(value_array))
    early = early_indices[finite[early_indices]]
    late = late_indices[finite[late_indices]]
    if len(early) < 2 or len(late) < 2:
        result.update(
            {
                "classification": CLASS_INCONCLUSIVE,
                "reason_codes": ["insufficient_block_coverage"],
            }
        )
        return result

    x = time_array[finite]
    y = value_array[finite]
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    mean = float(np.mean(y))
    std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    scale = max(abs(mean), std, np.finfo(np.float64).eps)
    early_mean = float(np.mean(value_array[early]))
    late_mean = float(np.mean(value_array[late]))
    block_drift = abs(late_mean - early_mean) / scale
    trend_drift = (
        abs(float(slope))
        * float(time_array[-1] - time_array[0])
        / scale
    )
    detrended = y - (slope * x + intercept)
    if missing_frames:
        acf = {
            "status": "not_estimated_missing_frames",
            "acf": None,
            "statistical_inefficiency_frames": None,
            "integrated_autocorrelation_time_frames": None,
            "integrated_autocorrelation_time_time": None,
            "effective_sample_size": None,
            "truncated_at_nonpositive": None,
            "reliable": False,
        }
        neff = None
    else:
        acf = _acf_summary(
            detrended,
            max_lag=max_acf_lag,
            frame_spacing=frame_spacing,
        )
        neff = float(acf["effective_sample_size"])

    reasons: list[str] = []
    if block_drift > fail_relative_drift and trend_drift > fail_relative_drift:
        classification = CLASS_NONSTATIONARY
        reasons.append("concordant_strong_block_and_trend_drift")
    elif (
        not missing_frames
        and block_drift <= pass_relative_drift
        and trend_drift <= pass_relative_drift
        and neff is not None
        and neff >= min_effective_samples
        and bool(acf["reliable"])
    ):
        classification = CLASS_PROVISIONAL
        reasons.append("drift_within_advisory_gates_and_sampling_adequate")
    else:
        classification = CLASS_INCONCLUSIVE
        if block_drift > pass_relative_drift:
            reasons.append("block_drift_in_gray_zone_or_above")
        if trend_drift > pass_relative_drift:
            reasons.append("trend_drift_in_gray_zone_or_above")
        if neff is not None and neff < min_effective_samples:
            reasons.append("effective_sample_size_too_small")
        if not acf["reliable"] and not missing_frames:
            reasons.append("autocorrelation_window_not_resolved")
    if missing_frames:
        reasons.append("missing_frames_break_uniform_lag_acf")

    result.update(
        {
            "classification": classification,
            "reason_codes": reasons,
            "mean": mean,
            "std": std,
            "normalization_scale": scale,
            "slope_per_time": float(slope),
            "normalized_trend_change": trend_drift,
            "normalized_block_mean_difference": block_drift,
            "early_block": {
                "count": len(early),
                "start_time": float(time_array[early[0]]),
                "end_time": float(time_array[early[-1]]),
                "mean": early_mean,
            },
            "excluded_midpoint": {
                "index": midpoint_index,
                "time": float(time_array[midpoint_index]),
                "value": _jsonable(value_array[midpoint_index]),
            },
            "late_block": {
                "count": len(late),
                "start_time": float(time_array[late[0]]),
                "end_time": float(time_array[late[-1]]),
                "mean": late_mean,
            },
            "autocorrelation": acf,
        }
    )
    return result


def _combine_classifications(
    summaries: dict[str, dict[str, Any]], names: Iterable[str]
) -> dict[str, Any]:
    selected = {name: summaries.get(name) for name in names}
    if any(value is None for value in selected.values()):
        return {
            "classification": CLASS_NOT_EVALUATED,
            "observables": selected,
            "reason": "required observable was not evaluated",
        }
    classes = {name: value["classification"] for name, value in selected.items()}
    if CLASS_NONSTATIONARY in classes.values():
        classification = CLASS_NONSTATIONARY
    elif all(value == CLASS_PROVISIONAL for value in classes.values()):
        classification = CLASS_PROVISIONAL
    else:
        classification = CLASS_INCONCLUSIVE
    return {"classification": classification, "observables": classes}


def _plot_png(rows: list[dict[str, Any]], *, include_defects: bool) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.asarray([row["time"] for row in rows])
    panels = [
        ("u_rms", r"$u_{\rm rms}$"),
        ("principal_S_mean_over_S_bulk", r"$\langle S\rangle/S_{\rm bulk}$"),
        ("Q_magnitude_mean_over_S_bulk", r"$\langle Q_{\rm mag}\rangle/S_{\rm bulk}$"),
        ("total_ldg_free_energy_density_mean", "LdG free-energy density"),
    ]
    if include_defects:
        panels.extend(
            [
                ("defect_point_density_proxy", "defect-point density proxy"),
                ("mean_sigma_over_h", r"frame mean $\sigma/H$"),
            ]
        )
    columns = 2
    figure, axes = plt.subplots(
        math.ceil(len(panels) / columns),
        columns,
        figsize=(10.0, 3.2 * math.ceil(len(panels) / columns)),
        dpi=170,
    )
    axes_array = np.atleast_1d(axes).ravel()
    for axis, (name, label) in zip(axes_array, panels):
        values = np.asarray(
            [np.nan if row.get(name) is None else row[name] for row in rows],
            dtype=float,
        )
        axis.plot(time, values, marker="o", linewidth=1.4, markersize=3.2)
        axis.axvline(time[len(time) // 2], color="0.55", linestyle="--", linewidth=0.9)
        axis.set_xlabel("time")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    for axis in axes_array[len(panels) :]:
        axis.set_visible(False)
    figure.suptitle("Beris–Edwards long-pilot stationarity screen")
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return buffer.getvalue()


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: _jsonable(row.get(name)) for name in fieldnames})
    return handle.getvalue().encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_outputs_exclusive(output_dir: Path, payloads: dict[str, bytes]) -> None:
    output_dir.mkdir(parents=False, exist_ok=False)
    for name, payload in payloads.items():
        path = output_dir / name
        with path.open("xb") as handle:
            handle.write(payload)
    with (output_dir / "COMPLETE").open("x", encoding="utf-8") as handle:
        handle.write("complete\n")


def analyze(
    *,
    run_dir: Path,
    plan_path: Path,
    run_id: str,
    validation_report_path: Path,
    checksum_manifest_path: Path,
    expected_validation_report_sha256: str,
    expected_checksum_manifest_sha256: str,
    output_dir: Path,
    include_defects: bool,
    energy_device: str,
    chunk_x: int,
    pass_relative_drift: float,
    fail_relative_drift: float,
    min_effective_samples: float,
    min_finite_frames: int,
    max_acf_lag: int,
    defect_threshold: float,
    wall_tolerance_index: float,
    require_clean_git: bool,
    expected_analysis_git_head: str | None,
) -> dict[str, Any]:
    run_input = run_dir
    plan_input = plan_path
    report_input = validation_report_path
    manifest_input = checksum_manifest_path
    output_input = output_dir
    expected_validation_report_sha256 = _validated_sha256(
        expected_validation_report_sha256,
        label="expected validation report SHA-256",
    )
    expected_checksum_manifest_sha256 = _validated_sha256(
        expected_checksum_manifest_sha256,
        label="expected checksum manifest SHA-256",
    )
    if any(
        path.is_symlink()
        for path in (
            run_input,
            plan_input,
            report_input,
            manifest_input,
            output_input,
        )
    ):
        raise ValueError(
            "run, plan, validation report, checksum manifest, and output "
            "must not be symlinks"
        )
    run_dir = run_input.resolve()
    plan_path = plan_input.resolve()
    validation_report_path = report_input.resolve()
    checksum_manifest_path = manifest_input.resolve()
    try:
        output_parent = output_input.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(
            "analysis output parent directory must already exist"
        ) from error
    if not output_parent.is_dir():
        raise ValueError("analysis output parent must be a directory")
    output_dir = output_parent / output_input.name
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory must be wholly new: {output_dir}")
    if output_dir == PROJECT_ROOT or output_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("analysis output directory must be outside the Git worktree")

    # Bind the manifest itself before trusting any path it names.  In
    # particular, an external payload reference is not inspected unless these
    # exact manifest bytes match the caller-supplied SHA-256.
    checksum_manifest_probe, checksum_manifest_sha = (
        _read_expected_json_object(
            checksum_manifest_path,
            expected_sha256=expected_checksum_manifest_sha256,
            label="checksum manifest",
        )
    )
    if not checksum_manifest_probe:
        raise ValueError("checksum manifest must be a nonempty JSON object")
    external_payload_probe = _external_payload_record_and_path(
        checksum_manifest_probe,
        checksum_manifest_path=checksum_manifest_path,
    )
    additional_paths = (
        (external_payload_probe[1],)
        if external_payload_probe is not None
        else ()
    )

    # These first reads only enumerate the immutable input set.  All values used
    # below are reread after the identity anchor is established.
    plan_probe = _read_json_object(plan_path)
    _select_plan_run(plan_probe, run_dir=run_dir, run_id=run_id)
    metadata_probe = _read_json_object(run_dir / "metadata.json")
    (
        probe_final_step,
        _,
        _,
        probe_save_start,
        probe_save_interval,
        _,
    ) = _metadata_contract(metadata_probe)
    probe_steps = _scheduled_steps(
        final_step=probe_final_step,
        start_step=probe_save_start,
        interval=probe_save_interval,
    )
    paths = _input_paths(
        run_dir,
        plan_path,
        validation_report_path,
        checksum_manifest_path,
        probe_steps,
        additional_paths=additional_paths,
    )
    identities_before = _collect_identities(paths)

    plan = _read_json_object(plan_path)
    plan_row = _select_plan_run(plan, run_dir=run_dir, run_id=run_id)
    output_root = Path(plan["output_root"]).resolve()
    if output_dir == run_dir or output_dir.is_relative_to(run_dir):
        raise ValueError("analysis output directory must be outside the run directory")
    if output_dir == output_root or output_dir.is_relative_to(output_root):
        raise ValueError(
            "analysis output directory must be outside simulation output_root"
        )
    if plan_row.get("purposes") != ["long_pilot"]:
        raise ValueError("selected plan row is not marked as exactly one long_pilot")

    metadata = _read_json_object(run_dir / "metadata.json")
    (
        final_step,
        shape,
        dtype,
        save_start,
        save_interval,
        _,
    ) = _metadata_contract(metadata)
    steps = _scheduled_steps(
        final_step=final_step, start_step=save_start, interval=save_interval
    )
    if steps != probe_steps:
        raise RuntimeError("critical metadata changed while anchoring inputs")

    plan_file_sha = _sha256_file(plan_path)
    validation_report_sha = _sha256_file(validation_report_path)
    checksum_manifest_sha = _sha256_file(checksum_manifest_path)
    if validation_report_sha != expected_validation_report_sha256:
        raise ValueError(
            "validation report SHA-256 mismatch: "
            f"{validation_report_sha} != {expected_validation_report_sha256}"
        )
    if checksum_manifest_sha != expected_checksum_manifest_sha256:
        raise ValueError(
            "checksum manifest SHA-256 mismatch: "
            f"{checksum_manifest_sha} != {expected_checksum_manifest_sha256}"
        )
    external_report = _read_json_object(validation_report_path)
    _validate_external_report(
        external_report,
        run_dir=run_dir,
        plan_path=plan_path,
        run_id=run_id,
        plan_sha256=plan_file_sha,
    )
    checksum_manifest = _read_json_object(checksum_manifest_path)
    if not checksum_manifest:
        raise ValueError("checksum manifest must be a nonempty JSON object")
    if checksum_manifest != checksum_manifest_probe:
        raise RuntimeError("checksum manifest changed while anchoring inputs")
    if _sha256_file(checksum_manifest_path) != checksum_manifest_sha:
        raise RuntimeError("checksum manifest changed while being parsed")
    manifest_verification = _verify_checksum_manifest(
        checksum_manifest,
        critical_paths=paths,
        checksum_manifest_path=checksum_manifest_path,
    )

    fresh_validation = validate_outputs(
        run_dir, plan_path=plan_path, run_id=run_id
    )
    if fresh_validation.get("passed") is not True:
        raise ValueError(
            "fresh read-only output validation failed: "
            f"{fresh_validation['errors']}"
        )
    identities_after_validation = _collect_identities(paths)
    if identities_after_validation != identities_before:
        raise RuntimeError("critical inputs changed during validation")
    git = _git_provenance(
        require_clean=require_clean_git,
        expected_head=expected_analysis_git_head,
    )
    lengths = tuple(float(value) for value in metadata["solver"]["lengths"])
    dt = _positive_metadata_value(metadata["solver"]["dt"], "solver.dt")
    parameters = metadata["model"]["parameters"]
    s_bulk = _positive_metadata_value(parameters["S_bulk"], "S_bulk")
    coefficients = metadata.get("ldg_coefficients", {})
    ldg_a = _finite_float(
        coefficients.get("A", parameters["ldg_A"]), label="ldg A"
    )
    ldg_b = _finite_float(
        coefficients.get("B", parameters["ldg_B"]), label="ldg B"
    )
    ldg_c = _finite_float(
        coefficients.get("C", parameters["ldg_C"]), label="ldg C"
    )
    ldg_l1 = _finite_float(
        metadata.get("ldg_l1", parameters["ldg_L1"]), label="ldg L1"
    )
    if ldg_l1 < 0.0:
        raise ValueError("ldg L1 must be nonnegative")
    if tuple(metadata["boundary_conditions"]["Q"]) != Q_BC:
        raise ValueError("spectral energy requires periodic/periodic/Neumann Q")
    energy = SpectralElasticEnergy(
        shape=shape,
        lengths=lengths,
        dtype=dtype,
        device=energy_device,
        ldg_l1=ldg_l1,
    )
    runtime = _runtime_provenance(energy_device=energy_device)

    defect_measure: (
        Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]] | None
    ) = None
    defect_provenance = None
    if include_defects:
        defect_measure, defect_provenance = _load_defect_measurement()

    frame_rows: list[dict[str, Any]] = []
    line_rows: list[dict[str, Any]] = []
    volume = math.prod(lengths)
    for index, step in enumerate(steps, start=1):
        q_path = run_dir / f"Q_{step}.npy"
        u_path = run_dir / f"u_{step}.npy"
        q_stats = _compact_q_statistics(
            q_path,
            shape=shape,
            dtype=dtype,
            s_bulk=s_bulk,
            ldg_a=ldg_a,
            ldg_b=ldg_b,
            ldg_c=ldg_c,
            chunk_x=chunk_x,
        )
        u_stats = _velocity_statistics(
            u_path, shape=shape, dtype=dtype, chunk_x=chunk_x
        )
        elastic = energy.evaluate(q_path)
        row: dict[str, Any] = {
            "step": step,
            "time": step * dt,
            **q_stats,
            **u_stats,
            "elastic_ldg_free_energy_density_mean": elastic,
            "total_ldg_free_energy_density_mean": (
                q_stats["bulk_ldg_free_energy_density_mean"] + elastic
            ),
        }
        if defect_measure is not None:
            q = np.load(q_path, mmap_mode="r", allow_pickle=False)
            defect_frame, measured_lines = defect_measure(
                np.asarray(q),
                step=step,
                threshold=defect_threshold,
                wall_tolerance=wall_tolerance_index,
                lengths=lengths,
            )
            through_count = int(defect_frame["through_count"])
            row.update(
                {
                    "defect_point_count": defect_frame["defect_point_count"],
                    "defect_point_density_proxy": (
                        defect_frame["defect_point_count"] / volume
                    ),
                    "defect_line_count": defect_frame["line_count"],
                    "through_line_count": through_count,
                    "through_line_presence": float(through_count > 0),
                    "bottom_only_line_count": defect_frame["bottom_only_count"],
                    "top_only_line_count": defect_frame["top_only_count"],
                    "interior_line_count": defect_frame["interior_count"],
                    "loop_count": defect_frame["loop_count"],
                    "mean_sigma_over_h": _jsonable(defect_frame["mean_sigma_over_h"]),
                    "std_sigma_over_h": _jsonable(defect_frame["std_sigma_over_h"]),
                }
            )
            line_rows.extend(_jsonable(measured_lines))
        frame_rows.append(row)
        print(f"[{index}/{len(steps)}] step={step} time={step * dt:.8g}", flush=True)

    identities_after = _collect_identities(paths)
    if identities_after != identities_before:
        changed = sorted(
            name
            for name in set(identities_before) | set(identities_after)
            if identities_before.get(name) != identities_after.get(name)
        )
        raise RuntimeError(f"critical inputs changed during analysis: {changed}")

    thresholds = {
        "pass_relative_drift": pass_relative_drift,
        "fail_relative_drift": fail_relative_drift,
        "min_effective_samples": min_effective_samples,
        "min_finite_frames": min_finite_frames,
        "fig4_min_finite_frames": max(15, min_finite_frames),
        "max_acf_lag": max_acf_lag,
        "defect_threshold": defect_threshold if include_defects else None,
        "wall_tolerance_index": (
            wall_tolerance_index if include_defects else None
        ),
        "autocorrelation_method": {
            "sampling": "strictly equally spaced saved frames",
            "detrending": "ordinary least squares in physical time",
            "truncation": "initial positive ACF sequence",
            "statistical_inefficiency": "g = 1 + 2 * sum(rho_k)",
            "integrated_autocorrelation_time": "tau_int = g / 2",
            "effective_sample_size": "N_eff = N / g",
            "missing_frame_policy": (
                "do not estimate ACF/ESS; classification is at most "
                "inconclusive unless both drift tests are strongly positive"
            ),
        },
        "classification_semantics": (
            "advisory screening thresholds, not universal physical laws"
        ),
    }
    times = [float(row["time"]) for row in frame_rows]
    names = list(CORE_FIELD_OBSERVABLES)
    if include_defects:
        names.extend(TOPOLOGY_OBSERVABLES)
        names.extend(FIG4_OBSERVABLES)
    series = {}
    for name in names:
        minimum = (
            thresholds["fig4_min_finite_frames"]
            if name in FIG4_OBSERVABLES
            else min_finite_frames
        )
        series[name] = summarize_series(
            times,
            [row.get(name) for row in frame_rows],
            pass_relative_drift=pass_relative_drift,
            fail_relative_drift=fail_relative_drift,
            min_effective_samples=min_effective_samples,
            min_finite_frames=minimum,
            max_acf_lag=max_acf_lag,
        )
    field = _combine_classifications(series, CORE_FIELD_OBSERVABLES)
    topology = (
        _combine_classifications(series, TOPOLOGY_OBSERVABLES)
        if include_defects
        else {"classification": CLASS_NOT_EVALUATED, "reason": "defects not requested"}
    )
    fig4 = (
        _combine_classifications(series, FIG4_OBSERVABLES)
        if include_defects
        else {"classification": CLASS_NOT_EVALUATED, "reason": "defects not requested"}
    )
    layers = [
        field["classification"],
        topology["classification"],
        fig4["classification"],
    ]
    if CLASS_NONSTATIONARY in layers:
        overall = CLASS_NONSTATIONARY
    elif all(value == CLASS_PROVISIONAL for value in layers):
        overall = CLASS_PROVISIONAL
    else:
        overall = CLASS_INCONCLUSIVE

    formal_provenance_gate = bool(
        git["clean"] and git["expected_head_matched"]
    )
    through_frame_count = (
        sum(float(row.get("through_line_presence", 0.0)) > 0.0 for row in frame_rows)
        if include_defects
        else 0
    )
    analyzer_path = Path(__file__).resolve()
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "analysis": "beris_edwards_long_pilot_stationarity_screen",
        "classification": {
            "overall": overall,
            "field_stationarity": field,
            "topology_stationarity": topology,
            "fig4_observable_stationarity": fig4,
            "formal_provenance_gate_passed": formal_provenance_gate,
            "eligible_to_consider_multiseed": (
                overall == CLASS_PROVISIONAL and formal_provenance_gate
            ),
        },
        "claim_scope": (
            "tests for detectable drift on the saved "
            f"t={times[0]:.8g}..{times[-1]:.8g} sampling window"
        ),
        "limitations": [
            "A finite 21-frame window cannot prove strict stationarity.",
            (
                f"Frame spacing {times[1] - times[0]:.8g} cannot resolve "
                "faster temporal correlations."
            ),
            "One seed is not an ensemble and cannot establish benchmark statistics.",
            (
                "Frame mean sigma/H is conditional on at least one through-line; "
                "through-line count and presence are screened separately."
            ),
            "The initial condition is PSSolver-constructed and not paper-identical.",
            (
                "Defect-point density is a fixed-grid discrete proxy, not "
                "physical line length density."
            ),
            (
                "Velocity-energy density is a velocity-squared diagnostic, not "
                "inertial kinetic energy in the quasistatic Stokes model."
            ),
            "No result in this report alone establishes Shendruk Fig. 4 reproduction.",
        ],
        "thresholds": thresholds,
        "run": {
            "run_dir": str(run_dir),
            "run_id": run_id,
            "steps": list(steps),
            "times": times,
            "shape": list(shape),
            "lengths": list(lengths),
            "dt": dt,
            "activity_number": metadata["activity_number"],
            "seed": metadata["seed"],
            "frames_with_through_lines": (
                through_frame_count if include_defects else None
            ),
            "fraction_frames_with_through_lines": (
                through_frame_count / len(frame_rows)
                if include_defects
                else None
            ),
            "paper_identical_initial_condition": (
                metadata["initial_condition"]["paper_identical"]
            ),
        },
        "series": series,
        "input_provenance": {
            "plan_path": str(plan_path),
            "plan_file_sha256": plan_file_sha,
            "plan_internal_sha256": plan["plan_sha256"],
            "simulation_git": plan.get("git"),
            "validation_report_path": str(validation_report_path),
            "validation_report_sha256": validation_report_sha,
            "expected_validation_report_sha256": (
                expected_validation_report_sha256
            ),
            "validation_report_expected_sha256_matched": True,
            "checksum_manifest_path": str(checksum_manifest_path),
            "checksum_manifest_sha256": checksum_manifest_sha,
            "expected_checksum_manifest_sha256": (
                expected_checksum_manifest_sha256
            ),
            "checksum_manifest_expected_sha256_matched": True,
            "checksum_manifest_verification": manifest_verification,
            "fresh_validation_passed": True,
            "critical_input_file_count": len(paths),
            "critical_inputs_stable": True,
            "checksum_manifest_reverified_before_output_commit": True,
        },
        "analysis_provenance": {
            "git": git,
            "analyzer_path": str(analyzer_path.relative_to(PROJECT_ROOT)),
            "analyzer_sha256": _sha256_file(analyzer_path),
            "validator_sha256": _sha256_file(
                PROJECT_ROOT / "scripts_plane/validate_beris_edwards_outputs.py"
            ),
            "runner_sha256": _sha256_file(
                PROJECT_ROOT / "scripts_plane/run_beris_edwards_validation.py"
            ),
            "nematics3d": defect_provenance,
            "runtime": runtime,
            "energy_device": str(energy.device),
            "analysis_tool_simulation_plan_binding": "not_applicable_post_hoc",
        },
        "frame_observables": frame_rows,
    }
    report["report_payload_sha256"] = _canonical_sha256(report)

    frame_fields = list(frame_rows[0])
    series_rows = [
        {"observable": name, **_jsonable(summary)}
        for name, summary in series.items()
    ]
    series_fields = sorted({key for row in series_rows for key in row})
    line_fields = sorted({key for row in line_rows for key in row}) if line_rows else [
        "step", "line_index", "line_kind", "sigma", "sigma_over_h"
    ]
    payloads = {
        "frame_observables.csv": _csv_bytes(frame_rows, frame_fields),
        "observable_stationarity.csv": _csv_bytes(series_rows, series_fields),
        "defect_lines.csv": _csv_bytes(line_rows, line_fields),
        "stationarity_report.json": _json_bytes(report),
        "stationarity_timeseries.png": _plot_png(
            frame_rows, include_defects=include_defects
        ),
    }
    output_hashes = {
        name: {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        for name, payload in payloads.items()
    }
    analysis_manifest = {
        "schema_version": 1,
        "status": "complete",
        "analysis_report_payload_sha256": report["report_payload_sha256"],
        "outputs": output_hashes,
        "input_provenance": report["input_provenance"],
        "analysis_provenance": report["analysis_provenance"],
    }
    payloads["analysis_manifest.json"] = _json_bytes(analysis_manifest)

    identities_final = _collect_identities(paths)
    if identities_final != identities_before:
        changed = sorted(
            name
            for name in set(identities_before) | set(identities_final)
            if identities_before.get(name) != identities_final.get(name)
        )
        raise RuntimeError(
            f"critical inputs changed before output commit: {changed}"
        )
    final_manifest_verification = _verify_checksum_manifest(
        checksum_manifest,
        critical_paths=paths,
        checksum_manifest_path=checksum_manifest_path,
    )
    if final_manifest_verification != manifest_verification:
        raise RuntimeError(
            "checksum manifest verification changed during analysis"
        )
    if _sha256_file(checksum_manifest_path) != checksum_manifest_sha:
        raise RuntimeError("checksum manifest content changed during analysis")
    git_final = _git_provenance(
        require_clean=require_clean_git,
        expected_head=expected_analysis_git_head,
    )
    if git_final != git:
        raise RuntimeError("analysis Git provenance changed during analysis")
    if defect_provenance is not None:
        source_tree_final = _bind_source_tree(
            Path(defect_provenance["source_tree"]["package_root"])
        )
        if source_tree_final != defect_provenance["source_tree"]:
            raise RuntimeError(
                "nematics3d source tree changed during analysis"
            )
        bound_files = [
            *defect_provenance["implementation_artifacts"],
            *defect_provenance["source_tree"]["artifacts"],
            {
                "path": defect_provenance["adapter_path"],
                "sha256": defect_provenance["adapter_sha256"],
            },
            {
                "path": defect_provenance["geometry_helper_path"],
                "sha256": defect_provenance["geometry_helper_sha256"],
            },
        ]
        for artifact in bound_files:
            if _sha256_file(Path(artifact["path"])) != artifact["sha256"]:
                raise RuntimeError(
                    "bound defect-analysis source changed during analysis: "
                    f"{artifact['path']}"
                )

    _write_outputs_exclusive(output_dir, payloads)
    return report


def _positive_metadata_value(value: Any, label: str) -> float:
    number = _finite_float(value, label=label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument(
        "--expected-validation-report-sha256", required=True
    )
    parser.add_argument("--checksum-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-checksum-manifest-sha256", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-defects", action="store_true")
    parser.add_argument("--energy-device", default="cpu")
    parser.add_argument("--chunk-x", type=int, default=16)
    parser.add_argument("--pass-relative-drift", type=_unit_interval, default=0.10)
    parser.add_argument("--fail-relative-drift", type=_unit_interval, default=0.20)
    parser.add_argument("--min-effective-samples", type=_positive_float, default=5.0)
    parser.add_argument("--min-finite-frames", type=int, default=15)
    parser.add_argument("--max-acf-lag", type=int, default=7)
    parser.add_argument("--defect-threshold", type=float, default=0.0)
    parser.add_argument("--wall-tolerance-index", type=_positive_float, default=0.51)
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--expected-analysis-git-head")
    args = parser.parse_args(argv)
    if args.chunk_x <= 0:
        parser.error("--chunk-x must be positive")
    if args.min_finite_frames < 3:
        parser.error("--min-finite-frames must be at least 3")
    if args.max_acf_lag <= 0:
        parser.error("--max-acf-lag must be positive")
    if args.fail_relative_drift <= args.pass_relative_drift:
        parser.error("--fail-relative-drift must exceed --pass-relative-drift")
    if not math.isfinite(args.defect_threshold):
        parser.error("--defect-threshold must be finite")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = analyze(
            run_dir=args.run_dir,
            plan_path=args.plan,
            run_id=args.run_id,
            validation_report_path=args.validation_report,
            checksum_manifest_path=args.checksum_manifest,
            expected_validation_report_sha256=(
                args.expected_validation_report_sha256
            ),
            expected_checksum_manifest_sha256=(
                args.expected_checksum_manifest_sha256
            ),
            output_dir=args.output_dir,
            include_defects=args.include_defects,
            energy_device=args.energy_device,
            chunk_x=args.chunk_x,
            pass_relative_drift=args.pass_relative_drift,
            fail_relative_drift=args.fail_relative_drift,
            min_effective_samples=args.min_effective_samples,
            min_finite_frames=args.min_finite_frames,
            max_acf_lag=args.max_acf_lag,
            defect_threshold=args.defect_threshold,
            wall_tolerance_index=args.wall_tolerance_index,
            require_clean_git=args.require_clean_git,
            expected_analysis_git_head=args.expected_analysis_git_head,
        )
    except Exception as error:
        print(
            f"stationarity analysis failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(_jsonable(report["classification"]), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
