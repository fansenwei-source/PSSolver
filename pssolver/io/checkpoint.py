"""Canonical metadata integrity for versioned checkpoints.

The digest is deliberately part of the checkpoint schema rather than an
authentication mechanism.  It detects incomplete edits, interrupted metadata
updates, and accidental schema downgrades before any tensor payload is opened.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
from typing import Iterator

import numpy as np


METADATA_SHA256_KEY = "metadata_sha256"
FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE = "functional_checkpoint.json"
_FUNCTIONAL_PROVENANCE_KIND = "pssolver_functional_checkpoint"
_FUNCTIONAL_PROVENANCE_VERSION = 1


class CheckpointError(Exception):
    """Base class for durable checkpoint failures across every runtime."""

    code = "checkpoint_error"

    def __init__(
        self,
        message: str,
        *,
        operation: str | None = None,
        failure_domain: str = "checkpoint",
        legacy_exception_type: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.failure_domain = failure_domain
        self.legacy_exception_type = legacy_exception_type
        self.details = dict(details or {})

    def to_metadata(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "operation": self.operation,
            "failure_domain": self.failure_domain,
            "legacy_exception_type": self.legacy_exception_type,
            "details": dict(self.details),
        }


class CheckpointNotFoundError(CheckpointError, FileNotFoundError):
    """A required checkpoint path is missing."""

    code = "checkpoint_not_found"


class CheckpointExistsError(CheckpointError, FileExistsError):
    """A checkpoint publication target already exists."""

    code = "checkpoint_exists"


class CheckpointIntegrityError(CheckpointError, ValueError):
    """Checkpoint bytes or exact current-format schema are invalid."""

    code = "checkpoint_integrity_error"


class CheckpointCompatibilityError(CheckpointError, ValueError):
    """A valid checkpoint cannot be consumed by the requested runtime."""

    code = "checkpoint_compatibility_error"


class CheckpointIOError(CheckpointError, OSError):
    """The filesystem prevented checkpoint input or publication."""

    code = "checkpoint_io_error"


def _io_error(
    exception: OSError,
    *,
    operation: str,
    message: str,
) -> CheckpointIOError:
    return CheckpointIOError(
        message,
        operation=operation,
        failure_domain="filesystem",
        legacy_exception_type=type(exception).__name__,
    )


def require_exact_checkpoint_keys(
    value: object,
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
    description: str,
    operation: str,
    require_all: bool = True,
) -> Mapping[str, object]:
    """Reject missing and unregistered keys in one current-format object."""

    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError(
            f"{description} must be a mapping",
            operation=operation,
        )
    keys = set(value)
    missing = set(required) - keys
    unknown = keys - set(required) - set(optional)
    if (require_all and missing) or unknown:
        raise CheckpointIntegrityError(
            f"{description} schema differs",
            operation=operation,
            details={
                "missing_keys": sorted(missing),
                "unknown_keys": sorted(unknown),
            },
        )
    return value


def read_checkpoint_json(
    path: Path,
    *,
    operation: str,
) -> Mapping[str, object]:
    """Read strict JSON from one regular, non-symlink checkpoint file."""

    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointNotFoundError(
            f"checkpoint metadata is missing: {path}",
            operation=operation,
            legacy_exception_type=type(exc).__name__,
        ) from exc
    except OSError as exc:
        raise _io_error(
            exc,
            operation=operation,
            message=f"checkpoint metadata cannot be inspected: {path}",
        ) from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise CheckpointIntegrityError(
            f"checkpoint metadata must be a regular non-symlink file: {path}",
            operation=operation,
        )

    def unique_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise CheckpointIntegrityError(
                    f"checkpoint metadata contains duplicate key: {key}",
                    operation=operation,
                )
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CheckpointIntegrityError(
                    f"checkpoint metadata contains non-finite JSON: {token}",
                    operation=operation,
                )
            ),
        )
    except CheckpointError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointIntegrityError(
            "checkpoint metadata is not valid JSON",
            operation=operation,
            legacy_exception_type=type(exc).__name__,
        ) from exc
    except OSError as exc:
        raise _io_error(
            exc,
            operation=operation,
            message=f"checkpoint metadata cannot be read: {path}",
        ) from exc
    if not isinstance(value, Mapping):
        raise CheckpointIntegrityError(
            "checkpoint metadata must be a JSON object",
            operation=operation,
        )
    return value


def resolve_checkpoint_payload(
    directory: Path,
    filename: object,
    *,
    operation: str,
    description: str,
    expected_filename: str | None = None,
) -> Path:
    """Resolve one schema basename to a confined regular payload file."""

    if (
        not isinstance(filename, str)
        or filename in {"", ".", ".."}
        or "/" in filename
        or "\\" in filename
        or Path(filename).is_absolute()
        or Path(filename).name != filename
        or (expected_filename is not None and filename != expected_filename)
    ):
        raise CheckpointIntegrityError(
            f"{description} filename is invalid",
            operation=operation,
            details={"filename": filename},
        )
    root = directory.resolve()
    path = root / filename
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise CheckpointNotFoundError(
            f"checkpoint payload is missing: {path}",
            operation=operation,
            legacy_exception_type=type(exc).__name__,
        ) from exc
    except OSError as exc:
        raise _io_error(
            exc,
            operation=operation,
            message=f"checkpoint payload cannot be inspected: {path}",
        ) from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise CheckpointIntegrityError(
            f"{description} must be a regular non-symlink file",
            operation=operation,
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _io_error(
            exc,
            operation=operation,
            message=f"checkpoint payload cannot be resolved: {path}",
        ) from exc
    if not resolved.is_relative_to(root):
        raise CheckpointIntegrityError(
            f"{description} escapes the checkpoint root",
            operation=operation,
        )
    return resolved


def checkpoint_file_sha256(path: Path, *, operation: str) -> str:
    """Hash one checkpoint file with a stable filesystem error category."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _io_error(
            exc,
            operation=operation,
            message=f"checkpoint file cannot be hashed: {path}",
        ) from exc
    return digest.hexdigest()


def load_checkpoint_numpy(path: Path, *, operation: str) -> np.ndarray:
    """Load one NumPy payload with stable integrity/I/O error categories."""

    try:
        return np.load(path, allow_pickle=False)
    except OSError as exc:
        raise _io_error(
            exc,
            operation=operation,
            message=f"checkpoint payload cannot be read: {path}",
        ) from exc
    except ValueError as exc:
        raise CheckpointIntegrityError(
            f"checkpoint tensor file is invalid: {path}",
            operation=operation,
            legacy_exception_type=type(exc).__name__,
        ) from exc


def _new_staging_directory(target: Path, *, operation: str) -> Path:
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise _io_error(
            exc,
            operation=operation,
            message=f"checkpoint parent cannot be created: {target.parent}",
        ) from exc
    if target.exists() or target.is_symlink():
        raise CheckpointExistsError(
            f"checkpoint directory already exists: {target}",
            operation=operation,
        )
    for _attempt in range(128):
        staging = target.parent / (
            f".{target.name}.tmp-{secrets.token_hex(12)}"
        )
        try:
            os.mkdir(staging, 0o777)
        except FileExistsError:
            continue
        except OSError as exc:
            raise _io_error(
                exc,
                operation=operation,
                message=f"checkpoint staging directory cannot be created: {staging}",
            ) from exc
        return staging
    raise CheckpointIOError(
        "checkpoint staging name allocation was exhausted",
        operation=operation,
        failure_domain="filesystem",
    )


@contextmanager
def checkpoint_directory_transaction(
    directory: str | Path,
    *,
    operation: str,
) -> Iterator[tuple[Path, Path]]:
    """Stage beside a target and publish once, preserving the caller umask."""

    # Normalize ``.`` and ``..`` without following a pre-existing target
    # symlink; a dangling symlink is still an occupied no-overwrite target.
    target = Path(os.path.abspath(Path(directory).expanduser()))
    staging = _new_staging_directory(target, operation=operation)
    try:
        yield staging, target
        if target.exists() or target.is_symlink():
            raise CheckpointExistsError(
                f"checkpoint directory already exists: {target}",
                operation=operation,
            )
        try:
            os.replace(staging, target)
        except OSError as exc:
            raise _io_error(
                exc,
                operation=operation,
                message=f"checkpoint cannot be published: {target}",
            ) from exc
    except BaseException as exc:
        if staging.exists() or staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, OSError) and not isinstance(exc, CheckpointError):
            raise _io_error(
                exc,
                operation=operation,
                message=f"checkpoint staging write failed: {target}",
            ) from exc
        raise


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def checkpoint_metadata_sha256(metadata: Mapping[str, object]) -> str:
    """Return the canonical digest while excluding the digest field itself."""

    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    payload = dict(metadata)
    payload.pop(METADATA_SHA256_KEY, None)
    return _canonical_sha256(payload)


def seal_checkpoint_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Return a JSON-compatible copy carrying its canonical metadata digest."""

    if METADATA_SHA256_KEY in metadata:
        raise ValueError("checkpoint metadata is already sealed")
    try:
        sealed = json.loads(
            json.dumps(
                dict(metadata),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be JSON-compatible") from exc
    sealed[METADATA_SHA256_KEY] = checkpoint_metadata_sha256(sealed)
    return sealed


def verify_checkpoint_metadata(
    metadata: Mapping[str, object],
    *,
    required: bool,
) -> bool:
    """Validate a canonical metadata digest and report whether one was present."""

    if not isinstance(metadata, Mapping):
        raise TypeError("checkpoint metadata must be a mapping")
    digest = metadata.get(METADATA_SHA256_KEY)
    if digest is None:
        if required:
            raise ValueError("checkpoint metadata SHA-256 is missing")
        return False
    expected = _require_sha256(digest, "checkpoint metadata SHA-256")
    if checkpoint_metadata_sha256(metadata) != expected:
        raise ValueError("checkpoint metadata checksum mismatch")
    return True


def _functional_provenance(
    metadata: Mapping[str, object],
    functional_bridge: Mapping[str, object],
) -> dict[str, object]:
    return {
        "format_kind": _FUNCTIONAL_PROVENANCE_KIND,
        "format_version": _FUNCTIONAL_PROVENANCE_VERSION,
        "checkpoint_metadata_sha256": checkpoint_metadata_sha256(metadata),
        "functional_bridge_sha256": _canonical_sha256(dict(functional_bridge)),
    }


def write_functional_checkpoint_provenance(
    directory: Path,
    metadata: Mapping[str, object],
    functional_bridge: Mapping[str, object],
) -> Path:
    """Write the durable marker that prevents functional-to-production downgrade."""

    path = directory / FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _functional_provenance(metadata, functional_bridge),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return path


def verify_functional_checkpoint_provenance(
    directory: Path,
    metadata: Mapping[str, object],
    functional_bridge: object,
    *,
    required: bool,
) -> bool:
    """Validate the functional-origin marker without opening tensor payloads."""

    path = directory / FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE
    if not path.exists() and not path.is_symlink():
        if required:
            raise CheckpointIntegrityError(
                "functional checkpoint provenance is missing",
                operation="functional_checkpoint_import",
            )
        return False
    if not isinstance(functional_bridge, Mapping):
        raise ValueError(
            "functional checkpoint provenance forbids production-path import"
        )
    value = read_checkpoint_json(
        path,
        operation="functional_checkpoint_import",
    )
    require_exact_checkpoint_keys(
        value,
        required={
            "format_kind",
            "format_version",
            "checkpoint_metadata_sha256",
            "functional_bridge_sha256",
        },
        description="functional checkpoint provenance",
        operation="functional_checkpoint_import",
    )
    if value != _functional_provenance(metadata, functional_bridge):
        raise CheckpointIntegrityError(
            "functional checkpoint provenance checksum mismatch",
            operation="functional_checkpoint_import",
        )
    return True


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointExistsError",
    "CheckpointIOError",
    "CheckpointIntegrityError",
    "CheckpointNotFoundError",
    "FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE",
    "checkpoint_file_sha256",
    "checkpoint_directory_transaction",
    "checkpoint_metadata_sha256",
    "load_checkpoint_numpy",
    "read_checkpoint_json",
    "require_exact_checkpoint_keys",
    "resolve_checkpoint_payload",
    "seal_checkpoint_metadata",
    "verify_checkpoint_metadata",
    "verify_functional_checkpoint_provenance",
    "write_functional_checkpoint_provenance",
]
