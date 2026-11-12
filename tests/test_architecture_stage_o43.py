"""CPU contracts for Stage O.4.3 zero-copy projected batch assembly."""

from __future__ import annotations

from pathlib import Path

from tests.v0_3_s03_successor import read_registered_source


PROJECT_ROOT = Path(__file__).parents[1]


def test_stage_o43_remains_outside_production_and_channel_paths():
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
    ):
        source = read_registered_source(relative).decode("utf-8")
        assert "AlgebraicOutputPublicationPolicy" not in source
        assert "ProjectedBatchAssemblyPolicy" not in source
