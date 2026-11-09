"""P8.2 local-candidate evidence and source-identity contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p82_periodic_complete_stress.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p82_record_is_local_candidate_and_does_not_overclaim_h100_closure():
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert record["phase"] == "P8.2"
    assert record["status"] == "local_candidate"
    assert record["combination"]["public_registry_connected"] is True
    assert record["combination"]["h100_qualified"] is False
    assert record["verification"]["h100_required"] is True
    assert record["verification"]["h100_completed"] is False
    assert record["eligibility"]["p8_2_complete"] is False
    assert record["scope"]["existing_plane_defaults_changed"] is False
    assert record["scope"]["existing_channel_defaults_changed"] is False


def test_p82_record_preserves_well_formed_frozen_source_identities():
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    # P8.3 is allowed to extend shared registry/lowering sources.  P8.2 keeps
    # the hashes of the then-qualified files as historical evidence rather
    # than silently rewriting them to the current worktree.
    for relative, frozen in record["source_sha256"].items():
        assert (ROOT / relative).is_file()
        assert len(frozen) == 64
        int(frozen, 16)
