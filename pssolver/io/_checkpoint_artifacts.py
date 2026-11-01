"""Private byte-preserving checkpoint artifact helpers.

This module owns only payload serialization and digest mechanics shared by
checkpoint families.  Directory transactions, publication, format schemas,
and family-specific exception translation remain with their existing owners.
"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


FileHasher = Callable[[Path], str]


def canonical_indented_json_bytes(value: object) -> bytes:
    """Return the historical indented checkpoint identity representation."""

    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of *value*."""

    return hashlib.sha256(value).hexdigest()


def raw_file_sha256(path: Path) -> str:
    """Hash a file while preserving the caller's native ``OSError`` surface."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def indented_identity_sha256(value: object) -> str:
    """Hash the historical indented JSON identity representation."""

    return sha256_bytes(canonical_indented_json_bytes(value))


def write_standard_tensor_record(
    path: Path,
    value: torch.Tensor,
    *,
    hash_file: FileHasher,
) -> dict[str, object]:
    """Write the common exclusive NumPy payload and its four-field record."""

    array = value.detach().to(device="cpu").contiguous().numpy()
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hash_file(path),
    }


def write_reference_tensor_record(
    path: Path,
    value: torch.Tensor,
    *,
    hash_file: FileHasher = raw_file_sha256,
) -> dict[str, object]:
    """Write the provisional reference format without changing its bytes."""

    if value.device.type != "cpu":
        raise ValueError("reference checkpoint tensors must be on CPU")
    array = value.detach().contiguous().numpy()
    np.save(path, array, allow_pickle=False)
    return {
        "filename": path.name,
        "shape": list(value.shape),
        "numpy_dtype": str(array.dtype),
        "torch_dtype": str(value.dtype),
        "finite": bool(torch.isfinite(value).all()),
        "sha256": hash_file(path),
        "size_bytes": path.stat().st_size,
    }


def read_reference_tensor_record(
    root: Path,
    name: str,
    record: object,
    *,
    filename: str,
    hash_file: FileHasher = raw_file_sha256,
) -> torch.Tensor:
    """Read one provisional reference tensor with its historical errors."""

    if not isinstance(record, dict):
        raise ValueError(f"tensor record {name!r} is invalid")
    if record.get("filename") != filename:
        raise ValueError(f"tensor filename for {name!r} is invalid")
    path = root / filename
    if not path.is_file():
        raise ValueError(f"checkpoint tensor {name!r} is missing")
    if path.stat().st_size != record.get("size_bytes"):
        raise ValueError(f"checkpoint tensor {name!r} size mismatch")
    if hash_file(path) != record.get("sha256"):
        raise ValueError(f"checkpoint tensor {name!r} SHA-256 mismatch")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"checkpoint tensor {name!r} cannot be loaded") from exc
    if list(array.shape) != record.get("shape"):
        raise ValueError(f"checkpoint tensor {name!r} shape mismatch")
    if str(array.dtype) != record.get("numpy_dtype"):
        raise ValueError(f"checkpoint tensor {name!r} dtype mismatch")
    if not bool(np.isfinite(array).all()) or record.get("finite") is not True:
        raise ValueError(f"checkpoint tensor {name!r} is non-finite")
    tensor = torch.from_numpy(np.array(array, copy=True))
    if str(tensor.dtype) != record.get("torch_dtype"):
        raise ValueError(f"checkpoint tensor {name!r} torch dtype mismatch")
    return tensor


def write_sized_tensor_record(
    path: Path,
    value: torch.Tensor,
    *,
    hash_file: FileHasher,
) -> dict[str, object]:
    """Write the finite-Q payload and its historical size-bearing record."""

    materialized = value.detach().cpu().contiguous().numpy()
    with path.open("wb") as handle:
        np.save(handle, materialized, allow_pickle=False)
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": hash_file(path),
        "shape": list(materialized.shape),
        "dtype": str(materialized.dtype),
    }
