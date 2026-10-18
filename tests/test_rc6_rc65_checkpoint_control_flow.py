"""RC6-5 regression tests for functional checkpoint control flow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.functional.api import (
    FunctionalError,
    _invoke,
    build_channel_activity_functional_runtime,
    build_functional_runtime,
    channel_activity_functional_request,
    periodic_activity_functional_request,
)
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.models.active_nematics import CompleteStressBerisEdwards


def _model(*, periodic: bool) -> CompleteStressBerisEdwards:
    options = (
        {
            "friction": 0.0,
            "tangential_zero_mode_policy": "zero_mean",
        }
        if periodic
        else {}
    )
    return CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
        **options,
    )


def _runtime(tmp_path: Path, family: str):
    periodic = family == "periodic"
    model = _model(periodic=periodic)
    shape = (6, 6, 4) if periodic else (8, 6, 4)
    lengths = tuple(float(value) for value in shape)
    geometry = (
        PeriodicBox(shape=shape, lengths=lengths)
        if periodic
        else RectangularChannel(shape=shape, lengths=lengths)
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": (
                free_slip_velocity() if periodic else no_slip_velocity()
            ),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    source = tmp_path / f"{family}_source"
    source.mkdir()
    rng = np.random.default_rng(20261017)
    values = rng.normal(scale=0.01, size=(*shape, 5)).astype(np.float64)
    values[..., 0] += 0.2
    values[..., 3] -= 0.1
    np.save(source / "Q_0.npy", values, allow_pickle=False)
    simulation = Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage=("hermitian_half" if periodic else "full_complex"),
            **({"hermitian_axis": 1} if periodic else {}),
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(source, step=0),
        execution=TorchSpectralExecution(
            runtime_path=(
                "periodic_spectral" if periodic else "channel_complete_stress"
            ),
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": (
                    "spectral" if periodic else "physical"
                ),
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": periodic,
            },
        ),
        output=Output(
            directory=tmp_path / f"{family}_output",
            steps=1,
            save_interval=10,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization=(
            {}
            if periodic
            else {
                "pressure_solver": {
                    "algorithm": "preconditioned_conjugate_gradient",
                    "relative_tolerance": 1.0e-10,
                    "max_iterations": 40,
                    "fixed_iterations": 12,
                    "warm_start": True,
                }
            }
        ),
    )
    if periodic:
        request = periodic_activity_functional_request(simulation.specification)
        return build_functional_runtime(request)
    request = channel_activity_functional_request(simulation.specification)
    return build_channel_activity_functional_runtime(request)


def _activity_gradient(runtime, *, checkpointed: bool) -> torch.Tensor:
    state = runtime.initial_state()
    control_spec = runtime.control_specs[0].tensor
    activity = torch.full(
        control_spec.shape,
        0.01,
        dtype=torch.float64,
        device=control_spec.device,
        requires_grad=True,
    )
    weight = torch.randn(
        state[0].shape,
        dtype=torch.float64,
        device=state[0].device,
        generator=torch.Generator().manual_seed(3),
    )

    def one_step(value: torch.Tensor) -> torch.Tensor:
        return runtime.step(state, {"activity": value}, 0)[0]

    evolved = (
        checkpoint(one_step, activity, use_reentrant=False)
        if checkpointed
        else one_step(activity)
    )
    return torch.autograd.grad((weight * evolved).sum(), activity)[0]


@pytest.mark.parametrize("family", ["periodic", "channel"])
def test_stable_runtime_preserves_default_checkpoint_early_stop(
    tmp_path: Path,
    family: str,
):
    runtime = _runtime(tmp_path, family)

    reference = _activity_gradient(runtime, checkpointed=False)
    checkpointed = _activity_gradient(runtime, checkpointed=True)

    assert torch.equal(checkpointed, reference)
    assert torch.isfinite(checkpointed).all()


def test_ordinary_empty_exception_is_still_translated_by_stable_facade():
    def raise_empty(*_args, **_kwargs):
        raise Exception()

    with pytest.raises(FunctionalError) as caught:
        _invoke("step", raise_empty)

    assert type(caught.value) is FunctionalError
    assert caught.value.operation == "step"
    assert caught.value.details["legacy_exception_type"] == "Exception"
