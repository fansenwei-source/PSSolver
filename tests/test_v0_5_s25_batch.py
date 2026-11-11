"""R-0085: CPU batch members retain the exact singleton transform path."""

import pytest
import torch

from pssolver.engine import Engine
from qualification.identity.records import tensor_record
from tests.test_v0_5_s25_engine_core import _setup


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("batch_size,steps", [(2, 16), (3, 8)], ids=["T14", "T14b"])
def test_members_match_singleton_bytes(tmp_path, dtype, batch_size, steps):
    _, single, initial = _setup(tmp_path, dtype)
    batch = Engine(single.plan, device="cpu", dtype=single.dtype, batch=batch_size)
    values = [
        {name: value * (1.0 - 0.25 * index) + 0.001 * index
         for name, value in initial.items()}
        for index in range(batch_size)
    ]
    states = [single.initial_state(value) for value in values]
    combined = batch.initial_state({
        name: torch.stack([value[name] for value in values]) for name in initial
    })
    for _ in range(steps):
        batch.step_(combined)
        for index, state in enumerate(states):
            single.step_(state)
            assert tensor_record(combined.spatial[:, index:index + 1]) == tensor_record(state.spatial)
            assert tensor_record(combined.spectral[:, index:index + 1]) == tensor_record(state.spectral)
    batch.prepare_constraints_(combined)
    assert all(value.shape[1] == batch_size for value in batch.observe(combined).values())
