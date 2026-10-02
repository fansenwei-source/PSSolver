"""P9.4 gradient, hidden-detach, replay-gradient, and R12 tests."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

import pssolver.functional.validation as functional_validation

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from pssolver.applications.periodic_beris_edwards import load_periodic_initial_q
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.public_simulation_runner import compile_public_simulation
from pssolver.functional import (
    FunctionalValidationError,
    build_functional_runtime,
    evaluate_periodic_activity_gradients,
    evaluate_periodic_production_consistency,
    periodic_activity_functional_request,
    validate_periodic_activity_gradients,
    validate_periodic_production_consistency,
)
from pssolver.geometries import PeriodicBox
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.runtime.periodic_beris_edwards import (
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)


def _simulation(tmp_path: Path) -> Simulation:
    model = CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
        friction=0.0,
        tangential_zero_mode_policy="zero_mean",
    )
    shape = (6, 6, 4)
    geometry = PeriodicBox(shape=shape, lengths=(6.0, 6.0, 4.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    rng = np.random.default_rng(20260928)
    q = rng.normal(scale=0.02, size=(*shape, 5)).astype(np.float64)
    q[..., 0] += 0.2
    q[..., 3] -= 0.1
    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "Q_0.npy", q, allow_pickle=False)
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(source, step=0),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": True,
            },
        ),
        output=Output(
            directory=tmp_path / "output",
            steps=1,
            save_interval=10,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
    )


def _runtime(tmp_path: Path):
    simulation = _simulation(tmp_path)
    request = periodic_activity_functional_request(simulation.specification)
    return simulation, build_functional_runtime(request)


def _control(runtime, value: float = 0.013):
    spec = runtime.control_specs[0].tensor
    indices = torch.arange(
        int(np.prod(spec.shape)), dtype=torch.float64
    ).reshape(spec.shape)
    return {"activity": value + 0.002 * torch.sin(indices * 0.071)}


def _production(simulation):
    compiled = compile_public_simulation(simulation.specification)
    initial_values, _, _ = load_periodic_initial_q(compiled.run_spec)
    return build_periodic_beris_edwards_runtime(
        PeriodicRuntimeBuildRequest(
            run_spec=compiled.run_spec,
            initial_values=initial_values,
            device="cpu",
        )
    )


class _DetachedInputProxy:
    def __init__(self, runtime, *, detach_state=False, detach_control=False):
        self._runtime = runtime
        self._detach_state = detach_state
        self._detach_control = detach_control

    @property
    def state_spec(self):
        return self._runtime.state_spec

    def step_and_observe(self, state, controls, step_index):
        if self._detach_state:
            state = tuple(value.detach() for value in state)
        if self._detach_control:
            controls = {"activity": controls["activity"].detach()}
        return self._runtime.step_and_observe(state, controls, step_index)


def _terminal_value(state):
    return state[0].square().mean() + state[1].abs().square().mean()


def _checkpointed_terminal_gradients(runtime, initial, controls, stride):
    """Independent consumer-style segmented reverse replay reference."""

    checkpoints = {0: tuple(value.detach().clone() for value in initial)}
    state = checkpoints[0]
    with torch.no_grad():
        for step, control in enumerate(controls):
            state = runtime.step(state, control, step)
            if (step + 1) % stride == 0 or step + 1 == len(controls):
                checkpoints[step + 1] = tuple(value.clone() for value in state)

    terminal = tuple(value.requires_grad_(True) for value in checkpoints[len(controls)])
    cotangent = torch.autograd.grad(_terminal_value(terminal), terminal)
    control_gradients = [None] * len(controls)
    stops = sorted(checkpoints)
    for start, stop in reversed(tuple(zip(stops[:-1], stops[1:], strict=True))):
        segment_state = tuple(
            value.detach().clone().requires_grad_(True)
            for value in checkpoints[start]
        )
        segment_controls = [
            controls[index]["activity"].detach().clone().requires_grad_(True)
            for index in range(start, stop)
        ]
        end_state = segment_state
        for local, activity in enumerate(segment_controls, start=start):
            end_state = runtime.step(end_state, {"activity": activity}, local)
        gradients = torch.autograd.grad(
            end_state,
            (*segment_state, *segment_controls),
            grad_outputs=cotangent,
            allow_unused=False,
        )
        cotangent = gradients[: len(segment_state)]
        for offset, value in enumerate(gradients[len(segment_state) :], start=start):
            control_gradients[offset] = value
    return (*cotangent, *control_gradients), checkpoints[len(controls)]


def test_p94_finite_difference_and_hidden_detach_gates_pass(tmp_path):
    _, runtime = _runtime(tmp_path)
    report = validate_periodic_activity_gradients(
        runtime, runtime.initial_state(), _control(runtime)
    )

    assert report.passed is True
    assert {value.input_name for value in report.directional_derivatives} == {
        "state",
        "activity",
    }
    assert all(value.passed for value in report.directional_derivatives)
    assert all(value.passed for value in report.gradient_paths)
    metadata = report.to_metadata()
    assert report.format_version == 2
    assert metadata["passed"] is True
    directional = {item["input"]: item for item in metadata["directional_derivatives"]}
    assert directional["state"]["direction"] == "low_mode"
    assert directional["state"]["finite_difference_method"] == "paired_quadratic"
    assert directional["state"]["epsilon"] == 1.0e-5
    assert directional["activity"]["epsilon"] == 1.0e-2
    assert all(
        item["relative_tolerance"] == 2.0e-5
        for item in directional.values()
    )
    assert all(
        item["absolute_direction_cosine"] > 0.0
        for item in directional.values()
    )


def test_p94_paired_quadratic_derivative_matches_direct_complex_difference():
    plus = (
        torch.tensor([1.0 + 2.0j, -0.5 + 0.25j], dtype=torch.complex128),
        torch.tensor([0.75 - 0.5j], dtype=torch.complex128),
    )
    minus = (
        torch.tensor([0.9 + 1.8j, -0.45 + 0.2j], dtype=torch.complex128),
        torch.tensor([0.7 - 0.45j], dtype=torch.complex128),
    )
    epsilon = 0.01
    expected = sum(
        value.abs().square().mean() for value in plus
    ) - sum(value.abs().square().mean() for value in minus)
    expected = float((expected / (2.0 * epsilon)).item())

    observed = functional_validation._paired_quadratic_derivative(
        plus, minus, epsilon
    )

    assert observed == pytest.approx(expected, rel=1.0e-14, abs=1.0e-14)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_p94_low_mode_direction_is_finite_normalized_and_shape_preserving(dtype):
    value = torch.zeros((2, 3, 4), dtype=dtype)

    direction = functional_validation._low_mode_direction(value, 0.31)

    assert direction.shape == value.shape
    assert direction.dtype == dtype
    assert torch.isfinite(direction).all()
    assert direction.abs().amax().item() == pytest.approx(1.0)


@pytest.mark.parametrize("detached", ["state", "control"])
def test_p94_hidden_detach_fails_even_when_observations_remain_differentiable(
    tmp_path,
    detached,
):
    _, runtime = _runtime(tmp_path)
    proxy = _DetachedInputProxy(
        runtime,
        detach_state=detached == "state",
        detach_control=detached == "control",
    )
    state = runtime.initial_state()
    controls = _control(runtime)

    report = evaluate_periodic_activity_gradients(proxy, state, controls)
    assert report.passed is False
    failed = {value.path for value in report.gradient_paths if not value.passed}
    if detached == "state":
        assert "dynamics<-q_physical" in failed
        assert "dynamics<-q_spectral" in failed
    else:
        assert "dynamics<-activity" in failed
    with pytest.raises(FunctionalValidationError) as error:
        validate_periodic_activity_gradients(proxy, state, controls)
    assert error.value.report.passed is False


def test_p94_checkpoint_stride_gradients_are_bitwise_identical(tmp_path):
    _, runtime = _runtime(tmp_path)
    initial = runtime.initial_state()
    controls = [_control(runtime, value) for value in (0.011, 0.014, 0.018)]

    reference_gradients, reference_final = _checkpointed_terminal_gradients(
        runtime, initial, controls, 1
    )
    for stride in (2, 3):
        gradients, final = _checkpointed_terminal_gradients(
            runtime, initial, controls, stride
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(reference_gradients, gradients, strict=True)
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(reference_final, final, strict=True)
        )


def test_p94_r12_production_functional_consistency_uses_frozen_tolerances(tmp_path):
    simulation, runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    controls = _control(runtime)
    report = validate_periodic_production_consistency(
        runtime, _production(simulation), state, controls
    )

    assert report.passed is True
    assert report.tolerances == {"atol": 5.0e-13, "rtol": 5.0e-12}
    assert {value.name for value in report.checks} == {
        "input.q_physical",
        "input.q_spectral",
        "observation.Q",
        "observation.velocity",
        "observation.pressure",
        "next.q_physical",
        "next.q_spectral",
    }
    assert all(value.passed for value in report.checks)
    signature = inspect.signature(evaluate_periodic_production_consistency)
    assert "atol" not in signature.parameters
    assert "rtol" not in signature.parameters


def test_p94_r12_mismatch_fails_closed_without_tolerance_override(tmp_path):
    simulation, runtime = _runtime(tmp_path)
    production = _production(simulation)
    production.fields.spatial[0].add_(1.0e-6)
    before_step = production.completed_steps
    state = runtime.initial_state()
    controls = _control(runtime)

    report = evaluate_periodic_production_consistency(
        runtime, production, state, controls
    )
    assert report.passed is False
    assert report.checks[0].passed is False
    assert production.completed_steps == before_step

    production = _production(simulation)
    production.fields.spatial[0].add_(1.0e-6)
    with pytest.raises(FunctionalValidationError) as error:
        validate_periodic_production_consistency(
            runtime, production, state, controls
        )
    assert error.value.report.passed is False
