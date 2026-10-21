"""Forward-only source-hash resolution for the S-03 successor record."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_public_errors_metadata.json"


def resolve_s03_successor(relative: str, historical: str) -> str:
    """Resolve one hash only when the S-03 record names its predecessor."""

    registration = json.loads(RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is None:
        return historical
    accepted = registration.get(
        "accepted_predecessor_sha256",
        [registration["historical_sha256"]],
    )
    if historical in accepted:
        return registration["current_sha256"]
    return historical


__all__ = ["resolve_s03_successor"]
