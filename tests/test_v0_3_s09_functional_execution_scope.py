"""CPU contract for the public functional forward/backward execution scope."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import pssolver.functional.api as stable
from pssolver.configuration.execution_device import (
    snapshot_tf32_execution_state,
)
from tests.test_phase9_p92_periodic_functional_runtime import (
    _control as periodic_control,
    _simulation as periodic_simulation,
)
from tests.test_phase9_p974_channel_functional_runtime import (
    _control as channel_control,
    _simulation as channel_simulation,
)


CALLER_STATES = (
    ("highest", False, False),
    ("highest", False, True),
    ("high", True, False),
    ("high", True, True),
    ("medium", True, False),
    ("medium", True, True),
)


def _set_tf32_state(precision: str, matmul: bool, cudnn: bool) -> None:
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


@pytest.fixture(autouse=True)
def _restore_tf32_state():
    before = snapshot_tf32_execution_state()
    yield
    _set_tf32_state(
        before.torch_float32_matmul_precision,
        before.cuda_matmul_allow_tf32,
        before.cudnn_allow_tf32,
    )


def _periodic_runtime(tmp_path):
    simulation = periodic_simulation(tmp_path, name="s09_periodic")
    request = stable.periodic_activity_functional_request(
        simulation.specification
    )
    runtime = stable.build_functional_runtime(request)
    return runtime, periodic_control(runtime, requires_grad=True)


def _channel_runtime(tmp_path):
    simulation = channel_simulation(tmp_path, name="s09_channel")
    request = stable.channel_activity_functional_request(
        simulation.specification
    )
    runtime = stable.build_channel_activity_functional_runtime(request)
    return runtime, channel_control(runtime, requires_grad=True)


RUNTIME_FACTORIES: tuple[
    tuple[str, Callable], ...
] = (
    ("periodic", _periodic_runtime),
    ("channel", _channel_runtime),
)


def _weighted_loss(state) -> torch.Tensor:
    value = state[0]
    weights = torch.linspace(
        0.5,
        1.5,
        value.numel(),
        dtype=value.dtype,
        device=value.device,
    ).reshape(value.shape)
    return (value * weights).sum()


@pytest.mark.parametrize(("family", "factory"), RUNTIME_FACTORIES)
@pytest.mark.parametrize(
    ("precision", "matmul", "cudnn"),
    CALLER_STATES,
)
def test_public_scope_owns_forward_and_backward_and_restores_caller(
    tmp_path,
    family,
    factory,
    precision,
    matmul,
    cudnn,
):
    runtime, controls = factory(tmp_path)
    assert isinstance(runtime, stable.FunctionalRuntimeProtocol)
    _set_tf32_state(precision, matmul, cudnn)
    caller = snapshot_tf32_execution_state()
    observed = []
    controls["activity"].register_hook(
        lambda gradient: (
            observed.append(snapshot_tf32_execution_state()),
            gradient,
        )[1]
    )

    with runtime.execution_scope() as policy:
        assert policy == {
            "tf32_requested": "off",
            "tf32_effective": False,
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        next_state = runtime.step(
            runtime.initial_state(), controls, step_index=0
        )
        _weighted_loss(next_state).backward()
        assert snapshot_tf32_execution_state().to_metadata() == {
            "torch_float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }

    assert observed
    assert all(
        value.to_metadata()
        == {
            "torch_float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        for value in observed
    ), family
    assert snapshot_tf32_execution_state() == caller


@pytest.mark.parametrize(("family", "factory"), RUNTIME_FACTORIES)
def test_backward_outside_public_scope_uses_restored_caller_state(
    tmp_path,
    family,
    factory,
):
    runtime, controls = factory(tmp_path)
    _set_tf32_state("high", True, True)
    caller = snapshot_tf32_execution_state()
    observed = []
    controls["activity"].register_hook(
        lambda gradient: (
            observed.append(snapshot_tf32_execution_state()),
            gradient,
        )[1]
    )

    next_state = runtime.step(runtime.initial_state(), controls, 0)
    assert snapshot_tf32_execution_state() == caller
    _weighted_loss(next_state).backward()

    assert observed and all(value == caller for value in observed), family
    assert snapshot_tf32_execution_state() == caller


def test_public_scope_is_nested_exception_safe_and_identity_is_api_1_1(
    tmp_path,
):
    runtime, _controls = _periodic_runtime(tmp_path)
    _set_tf32_state("medium", True, True)
    caller = snapshot_tf32_execution_state()

    with pytest.raises(RuntimeError, match="owned failure"):
        with runtime.execution_scope():
            outer = snapshot_tf32_execution_state()
            with runtime.execution_scope():
                assert snapshot_tf32_execution_state() == outer
            assert snapshot_tf32_execution_state() == outer
            raise RuntimeError("owned failure")

    assert snapshot_tf32_execution_state() == caller
    assert stable.FUNCTIONAL_API_VERSION == "1.1"
    assert runtime.api_version == "1.1"
    assert runtime.identity().to_metadata()["api_version"] == "1.1"
