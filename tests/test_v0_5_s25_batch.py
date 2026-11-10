"""R-0085: CPU batch members retain the exact singleton transform path."""

from unittest.mock import patch

import pytest
import torch

from pssolver.backends.tensor_product import TensorProductTransformBackend
from pssolver.engine import Engine
from pssolver.engine.memberwise_transforms import _MemberwiseTransforms
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


def test_singleton_keeps_original_backend_object(tmp_path):
    _, original, _ = _setup(tmp_path)
    backend = original.transform_backend
    assert type(backend) is TensorProductTransformBackend
    with patch("pssolver.engine.engine.TensorProductTransformBackend", return_value=backend):
        engine = Engine(original.plan, device="cpu", dtype=original.dtype, batch=1)
    assert engine.transform_backend is backend
    assert engine.projector.transform_backend is backend


@pytest.mark.parametrize("method", ["forward", "inverse"])
def test_memberwise_rejects_wrong_batch_without_calling_backend(tmp_path, method):
    _, engine, _ = _setup(tmp_path)
    backend = engine.transform_backend
    adapter = _MemberwiseTransforms(backend, batch=2)
    tensor = torch.zeros((3, *engine.shape), dtype=engine.dtype)
    with patch.object(backend, method, side_effect=AssertionError("must reject first")):
        with pytest.raises(RuntimeError, match="batch differs from engine"):
            getattr(adapter, method)(tensor, engine.boundaries)


@pytest.mark.parametrize("method", ["forward", "inverse"])
def test_memberwise_preserves_arguments_views_and_other_ranks(tmp_path, method):
    _, engine, _ = _setup(tmp_path)
    backend = engine.transform_backend
    adapter = _MemberwiseTransforms(backend, batch=2)
    counts = (3, 3, 3)
    tensor = torch.zeros((2, *engine.shape), dtype=engine.dtype).transpose(-1, -2)
    with patch.object(backend, method, side_effect=lambda value, *args, **kwargs: value) as call:
        result = getattr(adapter, method)(tensor, engine.boundaries, retained_axis_counts=counts)
        assert call.call_count == 2
        for index, arguments in enumerate(call.call_args_list):
            value = arguments.args[0]
            assert value.untyped_storage()._cdata == tensor.untyped_storage()._cdata
            assert value.storage_offset() == tensor.narrow(0, index, 1).storage_offset()
            assert value.stride() == tensor.stride()
            assert arguments.args[1] is engine.boundaries
            assert arguments.kwargs == {"retained_axis_counts": counts}
        assert torch.equal(result, tensor)
        for shape in [engine.shape, (5, 2, *engine.shape)]:
            value = torch.zeros(shape, dtype=engine.dtype)
            call.reset_mock()
            assert getattr(adapter, method)(value, engine.boundaries) is value
            assert call.call_count == 1
    assert adapter.get_q2 == backend.get_q2
