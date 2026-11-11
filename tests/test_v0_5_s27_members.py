"""R-0094: singleton-sized kernels, storage and diagnostics for every member."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pssolver.engine import Engine
from tests.test_v0_5_s25_engine_core import _setup
from tests.test_v0_5_s26_allen_cahn import build, initial
from tests.test_v0_5_s27_engine import plane_engine


def setup(tmp_path, family, dtype, batch):
    if family == "plane":
        _, single, values = plane_engine(tmp_path, dtype)
        values = {key: value[0] for key, value in values.items()}
    elif family == "periodic":
        _, single, values = _setup(tmp_path, str(dtype).split('.')[-1])
    else:
        single = build(dtype=dtype)
        values = {"phi": initial(8, dtype)}
    combined = (build(dtype=dtype, batch=batch) if family == "allen_cahn" else
                Engine(single.plan, device="cpu", dtype=dtype, batch=batch))
    data = [{key: value * (1 + .1 * i) + .001 * i for key, value in values.items()}
            for i in range(batch)]
    return single, combined, data


class Kernels(TorchDispatchMode):
    def __init__(self):
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if any(key in name for key in ("fft", "mm.", "bmm.", "sum.", "mean.", "norm", "linalg")):
            self.calls.append((name, tuple((tuple(x.shape), tuple(x.stride()))
                                          for x in torch.utils._pytree.tree_leaves(args)
                                          if isinstance(x, torch.Tensor))))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("family", ["periodic", "plane", "allen_cahn"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batch_kernels_have_exact_singleton_shapes(tmp_path, family, dtype):
    single, combined, values = setup(tmp_path, family, dtype, 2)
    assert not hasattr(single, "_member_engine")
    assert not hasattr(single, "_member_buffer")
    state = combined.initial_state({k: torch.stack([v[k] for v in values]) for k in values[0]})
    trace = Kernels()
    reports = []
    with trace:
        combined.step_(state, diagnostics=reports)
    individual = []
    for index in range(2):
        other = single.initial_state(values[index])
        expected = {}
        own = Kernels()
        with own:
            single.step_(other, diagnostics=expected)
        individual.extend(own.calls)
        assert torch.equal(state.spatial[:, index:index+1], other.spatial)
        assert reports[index].keys() == expected.keys()
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(reports[index][key], value)
            else:
                assert reports[index][key] == value
    assert trace.calls == individual
    template = single.initial_state(values[0])
    for name in ("spatial", "spectral"):
        left, right = getattr(combined._member_buffer, name), getattr(template, name)
        assert left.shape == right.shape
        assert all(a == b for size, a, b in zip(left.shape, left.stride(), right.stride(), strict=True) if size != 1)


@pytest.mark.parametrize("family", ["periodic", "plane", "allen_cahn"])
def test_pure_batch_is_differentiable_and_does_not_mutate_input(tmp_path, family):
    single, combined, values = setup(tmp_path, family, torch.float64, 2)
    raw = {k: torch.stack([v[k] for v in values]) for k in values[0]}
    state = combined.initial_state(raw)
    # Exercise the pure State API, not the legacy in-place initial conditioning.
    state.spatial = state.spatial.detach().requires_grad_()
    state.spectral = state.spectral.detach().requires_grad_()
    before = state.spatial.clone()
    result = combined.step(state)
    assert torch.equal(state.spatial, before)
    gradients = torch.autograd.grad(result.spatial.square().sum(), (state.spatial, state.spectral),
                                    allow_unused=family == "allen_cahn")
    assert all(torch.isfinite(value).all() for value in gradients if value is not None)
    for index in range(2):
        expected = single.step(single.initial_state(values[index]))
        assert torch.equal(result.spatial[:, index:index+1], expected.spatial)
