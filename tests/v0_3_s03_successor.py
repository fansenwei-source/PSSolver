"""Forward-only source-hash resolution for the S-03 successor chain."""

from __future__ import annotations

import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_public_errors_metadata.json"
SUCCESSOR_PATTERN = "PSSolver_v0_3_0_s03_successor_*.json"


def _records() -> list[Path]:
    successors = list((ROOT / "notes").glob(SUCCESSOR_PATTERN))
    successors.sort(
        key=lambda path: int(
            re.search(r"_successor_(\d+)", path.name).group(1)
        )
    )
    return [BASE_RECORD, *successors]


def resolve_s03_successor(relative: str, historical: str) -> str:
    """Resolve one hash only when the S-03 record names its predecessor."""

    resolved = historical
    for record_path in _records():
        registration = json.loads(record_path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        if registration is None:
            continue
        accepted = registration.get(
            "accepted_predecessor_sha256",
            [registration["historical_sha256"]],
        )
        if resolved in accepted:
            resolved = registration["current_sha256"]
    return resolved


def read_registered_source(relative: str, root: Path = ROOT) -> bytes:
    """Read a live source or verify the precise Git bytes of a registered retirement."""
    path = root / relative
    if path.is_file():
        return path.read_bytes()
    for record_path in reversed(_records()):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        retired = record.get("retired_sources", {}).get(relative)
        if retired is None:
            continue
        try:
            contents = subprocess.check_output(
                ["git", "show", f"{retired['source_commit']}:{relative}"], cwd=root,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                f"PSSOLVER_TEST_PREREQUISITE: registered Git source is unavailable: {relative}"
            ) from error
        if sha256(contents).hexdigest() != retired["sha256"]:
            raise ValueError(f"Retired source hash does not match: {relative}")
        return contents
    raise FileNotFoundError(path)


def registered_source_sha256(relative: str, root: Path = ROOT) -> str:
    """Return the current or recorded terminal digest, not a substitute file."""
    return sha256(read_registered_source(relative, root)).hexdigest()


__all__ = ["read_registered_source", "registered_source_sha256", "resolve_s03_successor"]
