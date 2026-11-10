"""Frozen S-25 wiring assertions still execute against their actual v0.4 source."""

from pathlib import Path

from qualification.historical_stage_audit import run_audit


def test_s25_historical_assertions_are_executed_without_skipping(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run_audit(
        root,
        tmp_path / "historical_s25",
        manifest_path=root / "qualification/historical_s25_tests.json",
    )
