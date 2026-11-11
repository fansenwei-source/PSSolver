"""Run retired private warm-start semantics against the installed reference."""

import json
from pathlib import Path

from qualification import historical_s27_audit


def test_s28a_historical_private_cache_invalidation(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    manifest = root / "qualification/s28a_historical_tests.json"
    source_commit = json.loads(manifest.read_bytes())["source_commit"]
    monkeypatch.setattr(historical_s27_audit, "SOURCE_COMMIT", source_commit)
    historical_s27_audit.run_audit(root, tmp_path / "s28a_history", manifest_path=manifest)
