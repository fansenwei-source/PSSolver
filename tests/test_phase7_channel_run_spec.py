"""P7.1 contracts for the disconnected Channel run specification."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


NOTES = ROOT / "notes" / "architecture_v0_2"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_p74_keeps_the_frozen_configuration_package_root_surface():
    import pssolver.configuration as configuration

    assert "ChannelActiveNematicRunSpec" not in configuration.__all__
    assert not hasattr(configuration, "ChannelActiveNematicRunSpec")


def test_p71_preserves_the_historical_oracle_source_identities():
    inventory = json.loads(
        (NOTES / "phase_7_channel_inventory.json").read_text(encoding="utf-8")
    )

    assert _sha256("Channel.py") == inventory["legacy_oracle"][
        "entry_point_sha256"
    ]
    assert inventory["legacy_oracle"]["reusable_module_sha256"] == (
        "9be8c7fd97e6167c44df4f38f2481683d108ac4e191ef1e0562d616fb4406223"
    )


def test_p71_result_record_keeps_the_next_implementation_unauthorized():
    value = json.loads(
        (NOTES / "phase_7_p71_run_spec_decomposition.json").read_text(
            encoding="utf-8"
        )
    )

    assert value["classification"] == (
        "PASS_P7_1_CHANNEL_RUN_SPEC_DECOMPOSITION"
    )
    assert value["facade"]["public_initialization_field_count"] == 37
    assert value["facade"]["production_connection"] is False
    assert value["source_isolation"]["channel_entry_point_modified"] is False
    assert value["source_isolation"]["pssolver_channel_modified"] is False
    assert value["authorization"] == {
        "p7_1_complete": True,
        "p7_2_planning_eligible": True,
        "p7_2_implementation_authorized": False,
        "phase_7_h100_authorized": False,
        "new_boundary_law_authorized": False,
        "production_default_changed": False,
    }
