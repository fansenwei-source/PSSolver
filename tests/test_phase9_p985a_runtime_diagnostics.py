"""P9.8.5a public runtime diagnostics without private-state exposure."""

from __future__ import annotations

import json

import pytest
import torch

from pssolver import SpectralSolver
import pssolver.functional.api as stable
from pssolver.functional.channel_activity_runtime import (
    ChannelActivityFunctionalRuntime,
)
from pssolver.functional.channel_pressure_adjoint import (
    ChannelImplicitPressureAdjoint,
)
from pssolver.functional.periodic_activity import (
    PeriodicActivityFunctionalRuntime,
)
from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver


class _DiagnosticsDelegate:
    checkpoint_bridge = None

    def __init__(self, diagnostics):
        self.value = diagnostics

    def diagnostics(self):
        return self.value


def _pressure_operator():
    spectral = SpectralSolver(
        (3, 2, 2),
        L=(3.0, 2.0, 2.0),
        device="cpu",
        dtype=torch.float64,
    )
    solver = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=1.0e-30,
        pressure_max_iterations=100,
        pressure_fixed_iterations=2,
    )
    return ChannelImplicitPressureAdjoint(solver)


def test_concrete_runtimes_publish_one_json_diagnostic_shape():
    periodic = PeriodicActivityFunctionalRuntime.__new__(
        PeriodicActivityFunctionalRuntime
    )
    assert periodic.diagnostics() == {
        "schema_version": 1,
        "runtime_kind": "periodic_activity_batch_one",
        "pressure": None,
    }

    channel = ChannelActivityFunctionalRuntime.__new__(
        ChannelActivityFunctionalRuntime
    )
    channel._pressure = _pressure_operator()
    before = channel.diagnostics()
    assert before["runtime_kind"] == "channel_activity_batch_one"
    assert before["pressure"]["primal"]["termination_reason"] == "not_run"
    assert before["pressure"]["transpose"]["termination_reason"] == "not_run"

    shape = channel._pressure._solver.pressure_null_mask.shape
    generator = torch.Generator().manual_seed(20260930)
    rhs = torch.complex(
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    ).to(torch.complex128).requires_grad_(True)
    pressure = channel._pressure.solve(rhs)
    torch.autograd.grad(pressure.real.square().sum(), rhs)
    after = channel.diagnostics()
    assert after["pressure"]["primal"]["acceptable"] is True
    assert after["pressure"]["transpose"]["acceptable"] is True
    assert json.loads(json.dumps(after, allow_nan=False)) == after


def test_stable_runtime_diagnostics_are_owned_finite_json_snapshots(monkeypatch):
    source = {
        "schema_version": 1,
        "runtime_kind": "channel_activity_batch_one",
        "pressure": {"primal": {"acceptable": True}},
    }
    delegate = _DiagnosticsDelegate(source)
    monkeypatch.setattr(
        stable._channel_runtime,
        "build_channel_activity_functional_runtime",
        lambda request: delegate,
    )
    runtime = stable.build_channel_activity_functional_runtime(object())

    first = runtime.diagnostics()
    assert first == source
    assert first is not source
    assert first["pressure"] is not source["pressure"]
    first["pressure"]["primal"]["acceptable"] = False
    assert runtime.diagnostics()["pressure"]["primal"]["acceptable"] is True
    assert not hasattr(runtime, "_flow_model")

    delegate.value = {"bad": torch.tensor(1.0)}
    with pytest.raises(stable.FunctionalValueError, match="finite JSON"):
        runtime.diagnostics()

    delegate.value = {"bad": float("nan")}
    with pytest.raises(stable.FunctionalValueError, match="finite JSON"):
        runtime.diagnostics()


def test_stable_runtime_diagnostics_require_a_json_object(monkeypatch):
    delegate = _DiagnosticsDelegate(["not", "an", "object"])
    monkeypatch.setattr(
        stable._periodic_runtime,
        "build_functional_runtime",
        lambda request: delegate,
    )
    runtime = stable.build_functional_runtime(object())
    with pytest.raises(stable.FunctionalTypeError, match="must be a dict"):
        runtime.diagnostics()
