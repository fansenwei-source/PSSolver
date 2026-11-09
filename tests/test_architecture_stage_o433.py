"""CPU contracts for Stage O.4.3.3 producer-owned H/stress packing."""

from __future__ import annotations

from pathlib import Path

import torch

from pssolver.core import BoundarySet, HomogeneousDirichletBC, HomogeneousNeumannBC, PeriodicBC


PROJECT_ROOT = Path(__file__).parents[1]


P = PeriodicBC()


N = HomogeneousNeumannBC()


D = HomogeneousDirichletBC()


EVEN = BoundarySet((P, P, N))


ODD = BoundarySet((P, P, D))


def _random_components(count: int, *, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn((1, 6, 5, 4), generator=generator, dtype=torch.float64)
        for _ in range(count)
    )
