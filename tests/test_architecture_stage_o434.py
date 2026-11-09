"""CPU contracts for Stage O.4.3.4 materialization attribution."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]


def test_stage_o434_instrumentation_remains_experimental_and_plane_only():
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "stage_o434" not in source
        assert "source_attribution" not in source
