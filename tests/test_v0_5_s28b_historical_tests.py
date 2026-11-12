"""Run the frozen tests of retired entry points on their accepted Provider."""

from pathlib import Path

from qualification import historical_s28b_audit


def test_retired_public_and_v1_contracts_on_the_accepted_predecessor(tmp_path):
    root = Path(__file__).resolve().parents[1]
    historical_s28b_audit.run_audit(root, tmp_path / "s28b_history")
