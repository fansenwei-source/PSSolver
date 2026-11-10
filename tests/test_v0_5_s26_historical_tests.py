"""Execute retired S-26 planning assertions against their reviewed S-25 source."""

from pathlib import Path

from qualification.historical_s26_audit import run_audit


def test_s26_historical_assertions_are_executed_without_skipping(tmp_path):
    run_audit(Path(__file__).resolve().parents[1], tmp_path / "historical_s26")
