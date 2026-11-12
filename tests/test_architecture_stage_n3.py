"""CPU qualification for Stage N.3 batched physical computation islands."""

from __future__ import annotations

from pathlib import Path

import torch

import pssolver
from pssolver.core import HomogeneousNeumannBC, PeriodicBC
from pssolver.experimental.representations import (
    AlgebraicGenerationState,
    BatchedPhysicalMaterialization,
)
from tests.v0_3_s03_successor import read_registered_source


P = PeriodicBC()


N = HomogeneousNeumannBC()


PROJECT_ROOT = Path(__file__).parents[1]


def test_generation_state_prefetches_one_declared_physical_island():
    physical = torch.zeros((1, 3), dtype=torch.float64)
    spectral = torch.complex(physical, torch.zeros_like(physical))

    def materialize(name, value):
        return value.real.clone()

    def materialize_many(names, values):
        return BatchedPhysicalMaterialization(
            {
                name: value.real.clone()
                for name, value in zip(names, values, strict=True)
            },
            (2, 1),
        )

    generation = AlgebraicGenerationState(
        1,
        {"q": physical},
        {"q": spectral},
        materialize=materialize,
        materialize_many=materialize_many,
    )
    for name, offset in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
        generation.publish_spectral(name, spectral + offset)
    view = generation.view(("q", "a", "b", "c"))
    view.prefetch_physical(("a", "b", "c"))

    assert torch.equal(view["a"], (spectral + 1.0).real)
    snapshot = generation.snapshot()
    assert snapshot["physical_materializations"] == 3
    assert snapshot["physical_materialization_batches"] == 2
    assert snapshot["batched_physical_components"] == 2
    assert snapshot["singleton_materialization_batches"] == 1
    assert snapshot["maximum_materialization_batch_size"] == 2
    assert snapshot["physical_island_prefetches"] == 1
    assert snapshot["physical_island_requested_components"] == 3

    final = generation.invalidate()
    assert final["retained_physical_components"] == 0
    assert final["retained_spectral_components"] == 0


def test_stage_n3_remains_outside_production_and_generic_solver_paths():
    assert not hasattr(pssolver, "BatchedPhysicalMaterialization")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
    ):
        text = read_registered_source(relative).decode("utf-8")
        assert "enable_batched_physical_islands" not in text
        assert "BatchedPhysicalMaterialization" not in text
