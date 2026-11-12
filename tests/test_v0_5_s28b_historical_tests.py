"""Run the frozen tests of retired entry points on their accepted Provider wheel."""

import json
from pathlib import Path

from qualification import historical_s27_audit


def test_retired_public_and_v1_contracts_on_the_accepted_predecessor(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "qualification/s28b_historical_tests.json"
    source_commit = json.loads(manifest.read_bytes())["source_commit"]
    monkeypatch.setattr(historical_s27_audit, "SOURCE_COMMIT", source_commit)
    historical_s27_audit.run_audit(root, tmp_path / "s28b_history", manifest_path=manifest)
