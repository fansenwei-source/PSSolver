"""P8.2 local-candidate evidence and source-identity contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pssolver import available_combinations


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


def test_p82_record_binds_the_current_reviewed_implementation_sources():
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    for relative, expected in record["source_sha256"].items():
        assert _sha256(ROOT / relative) == expected


def test_periodic_pair_is_a_distinct_public_registry_entry():
    combinations = {
        (value.equation_variant, value.geometry_name): value
        for value in available_combinations()
    }
    periodic = combinations[
        ("complete_stress_beris_edwards", "periodic_box")
    ]

    assert periodic.application == "periodic_complete_stress_beris_edwards"
    assert periodic.runtime_paths == ("periodic_spectral",)
    assert (
        "complete_stress_beris_edwards",
        "plane_slab",
    ) in combinations
    assert (
        "legacy_active_force_active_nematics",
        "rectangular_channel",
    ) in combinations
