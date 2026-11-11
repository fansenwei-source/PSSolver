"""Mandatory isolated installed replay of retired Plane wiring assertions."""

from pathlib import Path

from qualification.historical_s27_audit import run_audit


def test_s27_historical_assertions_are_executed_without_skipping(tmp_path):
    run_audit(Path(__file__).resolve().parents[1], tmp_path / "historical_s27")
