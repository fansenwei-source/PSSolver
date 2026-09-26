"""P8.3 local-candidate evidence and source-identity contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pssolver import available_combinations


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/architecture_v0_2/phase_8_p83_channel_complete_stress.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p83_record_is_local_candidate_without_h100_overclaim():
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert record["phase"] == "P8.3"
    assert record["status"] == "local_candidate"
    assert record["combination"]["public_registry_connected"] is True
    assert record["combination"]["h100_qualified"] is False
    assert record["verification"]["h100_required"] is True
    assert record["verification"]["h100_completed"] is False
    assert record["eligibility"]["p8_3_complete"] is False
    assert record["scope"]["existing_channel_defaults_changed"] is False


def test_p83_record_binds_reviewed_implementation_sources():
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected


def test_complete_stress_channel_is_distinct_registered_combination():
    combinations = {
        (value.equation_variant, value.geometry_name): value
        for value in available_combinations()
    }
    channel = combinations[
        ("complete_stress_beris_edwards", "rectangular_channel")
    ]

    assert channel.application == "channel_complete_stress_beris_edwards"
    assert channel.runtime_paths == ("channel_complete_stress",)
