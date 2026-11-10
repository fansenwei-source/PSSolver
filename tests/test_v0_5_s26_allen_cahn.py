"""Preregistered Allen--Cahn checks exercise the reusable model template."""

import inspect
import json
import math
from pathlib import Path

import pytest
import torch

from examples.allen_cahn import AllenCahn, build
from examples.model_template.checks import (
    energy_monotone,
    fitted_order,
    raw_equal,
    restart_check,
    spatial_mms,
    taylor_check,
)
from tests.mms.allen_cahn_mms import ForcedAllenCahn


def initial(size, dtype):
    generator = torch.Generator().manual_seed(26)
    return .03 * torch.randn((size,) * 3, generator=generator, dtype=dtype)


def test_manufactured_spatial_convergence(tmp_path):
    reports = []
    def measure(size):
        model = ForcedAllenCahn()
        engine = build(size=size, dt=.01, dtype=torch.float64, model=model)
        axis = torch.arange(size, dtype=torch.float64) / size
        xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"))
        state = engine.initial_state({"phi": model.exact(*xyz, 0.)})
        for _ in range(10):
            engine.step_(state)
        error = float((state.spatial[0,0] - model.exact(*xyz, .1)).abs().max())
        reports.append(dict(size=size, error=error, provenance=model.provenance))
        (tmp_path / "mms.json").write_text(json.dumps(reports, indent=2) + "\n")
        return error
    spatial_mms((8,16,32), measure, ratio=100, bound=1e-9)


def test_temporal_euler_order(tmp_path):
    steps, errors = [], []
    exact = .2 * math.exp(.5) / math.sqrt(1 + .2**2 * (math.exp(1.)-1))
    for k in range(4):
        dt = .05 * 2**(-k)
        engine = build(dt=dt, dtype=torch.float64)
        state = engine.initial_state({"phi": torch.full((8,8,8), .2, dtype=torch.float64)})
        for _ in range(round(.5/dt)):
            engine.step_(state)
        steps.append(dt)
        errors.append(float((state.spatial-exact).abs().max()))
    order = fitted_order(steps, errors)
    (tmp_path / "time.json").write_text(json.dumps(dict(steps=steps, errors=errors, order=order)) + "\n")
    assert .9 <= order <= 1.1


def test_energy_is_monotone_for_every_step(tmp_path):
    engine = build(size=16, dt=1e-3, dtype=torch.float64)
    state = engine.initial_state({"phi": initial(16,torch.float64)})
    energies = []
    for step in range(51):
        phi = state.spatial[0,0]
        grad2 = torch.zeros_like(phi)
        for axis in range(3):
            hat, bc = engine.transform_backend.gradient_hat(state.spectral[0,0], engine.boundaries, axis)
            grad2 += engine.transform_backend.inverse(hat, bc).square()
        energies.append(float((.05**2/2 * grad2 + (phi.square()-1).square()/4).mean()))
        if step < 50:
            engine.step_(state)
    (tmp_path / "energy.json").write_text(json.dumps(energies) + "\n")
    energy_monotone(energies)


def test_ten_step_restart_is_bitwise(tmp_path):
    engine = build(dtype=torch.float64)
    restart_check(engine, {"phi":initial(8,torch.float64)}, tmp_path / "checkpoint")


def test_pure_step_taylor_order(tmp_path):
    engine = build(dtype=torch.float64)
    value = initial(8,torch.float64)
    direction = torch.sin(torch.arange(value.numel(),dtype=value.dtype)).reshape(value.shape)
    weights = torch.cos(torch.arange(value.numel(),dtype=value.dtype)*.31).reshape(value.shape)
    def objective(phi):
        state = engine.initial_state({"phi":phi})
        return (weights * engine.step(state).spatial[0,0]).sum()
    steps = [1e-2,1e-3,1e-4,1e-5]
    order, errors = taylor_check(objective, value, direction, steps)
    (tmp_path / "taylor.json").write_text(json.dumps(dict(steps=steps, errors=errors, order=order)) + "\n")
    assert 1.9 <= order <= 2.1


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_pure_and_inplace_match_at_all_eight_steps(dtype):
    engine = build(dtype=dtype)
    pure = engine.initial_state({"phi":initial(8,dtype)})
    inplace = engine.initial_state({"phi":initial(8,dtype)})
    for _ in range(8):
        pure = engine.step(pure)
        engine.step_(inplace)
        assert raw_equal(pure.spatial, inplace.spatial)
        assert raw_equal(pure.spectral, inplace.spectral)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("batch", [2,3])
def test_batch_members_match_singletons(dtype,batch):
    engines = [build(dtype=dtype) for _ in range(batch)]
    batched = build(dtype=dtype,batch=batch)
    values = [initial(8,dtype)*(1+.2*i)+.001*i for i in range(batch)]
    states = [engine.initial_state({"phi":value}) for engine,value in zip(engines,values,strict=True)]
    combined = batched.initial_state({"phi":torch.stack(values)})
    for _ in range(8):
        batched.step_(combined)
        for i, (engine,state) in enumerate(zip(engines,states,strict=True)):
            engine.step_(state)
            assert raw_equal(combined.spatial[:,i:i+1],state.spatial)
            assert raw_equal(combined.spectral[:,i:i+1],state.spectral)


def test_example_is_small_and_documentation_is_the_same_model():
    source = inspect.getsource(AllenCahn)
    lines = [line for line in source.splitlines() if line.strip() and not line.strip().startswith(('"""','#'))]
    assert len(lines) <= 30
    doc = (Path(__file__).resolve().parents[1] / "docs/adding_a_model.md").read_text()
    assert source.strip() in doc
