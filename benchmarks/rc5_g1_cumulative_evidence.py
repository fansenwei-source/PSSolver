"""Dependency-free archive primitives for the rc5 G-1 cumulative closure."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


EXPECTED_ARCHIVES = {
    "cpu": {
        "entries": 1712,
        "manifest_sha256": (
            "6a10414a20d3c5f7c12e21e98afb594fe3008be3010388355a16e6fdb9fbe832"
        ),
    },
    "checkpoint": {
        "entries": 123,
        "manifest_sha256": (
            "6bc84435b3dab444156a87ba7ddc5981f93989c38098dfa3f9e56d8dcff9d627"
        ),
    },
    "v5": {
        "entries": 26,
        "manifest_sha256": (
            "049f7422b62618416745ca665f82ab2370f6a5f2e662ce296f332f461e3b7f71"
        ),
    },
    "v6": {
        "entries": 15,
        "manifest_sha256": (
            "74ef98002f8bd6b37f0d9e6cc001d17d9653cd6cb274afbd726273bd02eaeac2"
        ),
    },
    "v16": {
        "entries": 1814,
        "manifest_sha256": (
            "da50cbda196ace1e7a60c6974dfea40533575493096ed1971a91ef959e002de0"
        ),
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("cumulative evidence JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def verify_archive(
    root: Path,
    *,
    expected_entries: int,
    expected_manifest_sha256: str,
) -> dict[str, object]:
    """Verify an immutable evidence archive without rewriting its manifest."""

    root = root.expanduser().resolve()
    manifest = root / "checksums.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "checksum manifest missing")
    actual_manifest_sha256 = sha256(manifest)
    require(
        actual_manifest_sha256 == expected_manifest_sha256,
        "checksum manifest identity differs",
    )
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    require(len(lines) == expected_entries, "checksum entry count differs")
    verified = 0
    for line in lines:
        digest, separator, name = line.partition("  ")
        require(bool(separator) and len(digest) == 64 and bool(name), "invalid checksum line")
        path = Path(name)
        if not path.is_absolute():
            path = root / path
        require(path.is_file() and not path.is_symlink(), f"archive entry missing: {name}")
        require(sha256(path) == digest, f"archive checksum mismatch: {name}")
        verified += 1
    return {
        "root": str(root),
        "entries": expected_entries,
        "verified": verified,
        "failed": 0,
        "manifest_sha256": actual_manifest_sha256,
        "passed": True,
    }
