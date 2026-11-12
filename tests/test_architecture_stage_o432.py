"""CPU contracts for Stage O.4.3.2 producer-owned packed values."""

from __future__ import annotations

from pathlib import Path

import pssolver
from tests.v0_3_s03_successor import read_registered_source


PROJECT_ROOT = Path(__file__).parents[1]


def test_stage_o432_contract_remains_outside_production_paths():
    assert not hasattr(pssolver, "ProducerPackedComponentValues")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
    ):
        source = read_registered_source(relative).decode("utf-8")
        assert "ProducerPackedComponentValues" not in source
