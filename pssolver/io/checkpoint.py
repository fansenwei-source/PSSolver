"""Canonical metadata integrity for versioned checkpoints.

The digest is deliberately part of the checkpoint schema rather than an
authentication mechanism.  It detects incomplete edits, interrupted metadata
updates, and accidental schema downgrades before any tensor payload is opened.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path


METADATA_SHA256_KEY = "metadata_sha256"
FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE = "functional_checkpoint.json"
_FUNCTIONAL_PROVENANCE_KIND = "pssolver_functional_checkpoint"
_FUNCTIONAL_PROVENANCE_VERSION = 1


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
    path.write_text(
        json.dumps(
            _functional_provenance(metadata, functional_bridge),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
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
    if not path.is_file():
        if required:
            raise ValueError("functional checkpoint provenance is missing")
        return False
    if not isinstance(functional_bridge, Mapping):
        raise ValueError(
            "functional checkpoint provenance forbids production-path import"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("functional checkpoint provenance is not valid JSON") from exc
    if value != _functional_provenance(metadata, functional_bridge):
        raise ValueError("functional checkpoint provenance checksum mismatch")
    return True


__all__ = [
    "FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE",
    "checkpoint_metadata_sha256",
    "seal_checkpoint_metadata",
    "verify_checkpoint_metadata",
    "verify_functional_checkpoint_provenance",
    "write_functional_checkpoint_provenance",
]
