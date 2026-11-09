"""One explicit historical prerequisite, retaining every B-group audit case."""

import importlib.util
from pathlib import Path


def test_frozen_v040_stage_audits(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_s24_historical_stage_audit", root / "qualification/historical_stage_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_audit(root, tmp_path / "historical_stage_audit")
