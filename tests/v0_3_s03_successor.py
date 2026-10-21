"""Forward-only source-hash resolution for the S-03 successor chain."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_public_errors_metadata.json"
SUCCESSOR_PATTERN = "PSSolver_v0_3_0_s03_successor_*.json"


def _records() -> list[Path]:
    return [BASE_RECORD, *sorted((ROOT / "notes").glob(SUCCESSOR_PATTERN))]


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


__all__ = ["resolve_s03_successor"]
