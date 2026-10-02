"""P9.2 pure periodic activity-control runtime qualification on CPU."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from pssolver.applications.periodic_beris_edwards import (
    load_periodic_initial_q,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.public_simulation_runner import (
    compile_public_simulation,
)
from pssolver.functional import (
    FUNCTIONAL_API_VERSION,
    FunctionalRuntimeProtocol,
    build_functional_runtime,
    periodic_activity_functional_request,
)
from pssolver.geometries import PeriodicBox, PlaneSlab
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.runtime.periodic_beris_edwards import (
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)


def _model():
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
        friction=0.0,
        tangential_zero_mode_policy="zero_mean",
    )


def _simulation(
    tmp_path: Path,
    *,
    name: str = "functional",
    device: str = "cpu",
) -> Simulation:
    model = _model()
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
    source = tmp_path / f"{name}_source"
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
            device=device,
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": True,
            },
        ),
        output=Output(
            directory=tmp_path / name,
            steps=1,
            save_interval=10,
            diagnostic_interval=1,
        ),
    )


def _runtime(tmp_path: Path):
    simulation = _simulation(tmp_path)
    request = periodic_activity_functional_request(simulation.specification)
    return simulation, request, build_functional_runtime(request)


def test_non_cuda_index_is_normalized_to_tensor_device_type(tmp_path):
    simulation = _simulation(tmp_path, name="cpu_index", device="cpu:0")
    request = periodic_activity_functional_request(simulation.specification)
    runtime = build_functional_runtime(request)

    assert runtime.state_spec.components[0].device == "cpu"
    runtime.state_spec.validate(runtime.initial_state())


def _control(runtime, value=0.01, *, requires_grad=False):
    spec = runtime.control_specs[0].tensor
    result = torch.full(
        spec.shape,
        value,
        dtype=torch.float64,
        device=spec.device,
    )
    result.requires_grad_(requires_grad)
    return {"activity": result}


def test_p92_constructs_explicit_state_identity_and_capabilities(tmp_path):
    _, request, runtime = _runtime(tmp_path)

    assert isinstance(runtime, FunctionalRuntimeProtocol)
    assert runtime.api_version == FUNCTIONAL_API_VERSION
    assert runtime.state_spec.batch_size == 1
    assert [item.name for item in runtime.state_spec.components] == [
        "q_physical",
        "q_spectral",
    ]
    state = runtime.initial_state()
    runtime.state_spec.validate(state)
    assert state[0].shape == (5, 1, 6, 6, 4)
    assert state[1].shape == (5, 1, 6, 4, 4)
    assert state[0].dtype is torch.float64
    assert state[1].dtype is torch.complex128
    assert runtime.control_specs == request.control_fields
    assert runtime.control_field_schema == request.control_fields
    assert [value.name for value in runtime.observation_specs] == [
        "Q",
        "velocity",
        "pressure",
    ]
    capabilities = runtime.capabilities
    assert capabilities.pure_step is True
    assert capabilities.combined_step_and_observe is True
    # P9.3 subsequently qualifies these capabilities on this same runtime.
    assert capabilities.deterministic_replay == "bitwise"
    assert capabilities.differentiability == "torch_autograd"
    assert capabilities.durable_checkpoint_bridge is True
    assert capabilities.differentiable_inputs == ("state", "activity")
    identity = runtime.identity().to_metadata()
    assert identity["execution"]["functional_runtime"]["kind"] == (
        "periodic_activity_batch_one"
    )
    assert identity["execution"]["functional_runtime"][
        "activity_product_before_divergence"
    ] is True
    assert len(runtime.identity().canonical_sha256()) == 64


def test_p92_step_is_input_pure_and_combined_observation_uses_input_state(
    tmp_path,
):
    _, _, runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    before = tuple(value.clone() for value in state)
    controls = _control(runtime)
    production_alpha = runtime._solver.model.parameters["alpha"]

    separate = runtime.observe(state, controls)
    next_state, combined = runtime.step_and_observe(state, controls, 0)
    direct_next = runtime.step(state, controls, 0)

    assert all(torch.equal(old, new) for old, new in zip(before, state))
    assert runtime._solver.model.parameters["alpha"] is production_alpha
    assert all(
        torch.equal(separate[name], combined[name])
        for name in ("Q", "velocity", "pressure")
    )
    assert combined["Q"] is state[0]
    assert all(
        torch.equal(left, right)
        for left, right in zip(next_state, direct_next)
    )
    assert not torch.equal(next_state[0], state[0])
    runtime.state_spec.validate(next_state)

    terminal = runtime.observe(next_state, None)
    assert set(terminal) == {"Q"}
    assert terminal["Q"] is next_state[0]


def test_p92_activity_is_a_spatial_product_before_divergence(tmp_path):
    _, _, runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    uniform = _control(runtime, 0.01)
    varying = _control(runtime, 0.01)
    varying["activity"][:, :3] = 0.03

    uniform_observation = runtime.observe(state, uniform)
    varying_observation = runtime.observe(state, varying)
    uniform_next = runtime.step(state, uniform, 0)
    varying_next = runtime.step(state, varying, 0)

    assert not torch.equal(
        uniform_observation["velocity"],
        varying_observation["velocity"],
    )
    assert not torch.equal(uniform_next[0], varying_next[0])


def test_p92_torch_autograd_reaches_state_and_activity(tmp_path):
    _, _, runtime = _runtime(tmp_path)
    initial = runtime.initial_state()
    state = tuple(value.detach().clone().requires_grad_(True) for value in initial)
    controls = _control(runtime, requires_grad=True)

    next_state, observations = runtime.step_and_observe(state, controls, 0)
    loss = (
        next_state[0].square().mean()
        + observations["velocity"].square().mean()
        + observations["pressure"].square().mean()
    )
    gradients = torch.autograd.grad(
        loss,
        (*state, controls["activity"]),
        allow_unused=False,
    )

    assert len(gradients) == 3
    assert all(value is not None for value in gradients)
    assert all(torch.isfinite(value).all() for value in gradients)
    assert torch.count_nonzero(gradients[-1]) > 0


def test_p92_matches_one_qualified_production_step(tmp_path):
    simulation, _, runtime = _runtime(tmp_path)
    compiled = compile_public_simulation(simulation.specification)
    initial_values, _, _ = load_periodic_initial_q(compiled.run_spec)
    production = build_periodic_beris_edwards_runtime(
        PeriodicRuntimeBuildRequest(
            run_spec=compiled.run_spec,
            initial_values=initial_values,
            device="cpu",
        )
    )
    controls = _control(runtime)
    state = runtime.initial_state()

    assert torch.equal(state[0], production.fields.spatial[:5])
    assert torch.equal(state[1], production.fields.spectral[:5])
    production.solver.model.parameters["alpha"] = controls["activity"]
    production.synchronize_for_observation()
    functional_observation = runtime.observe(state, controls)
    assert torch.equal(
        functional_observation["velocity"],
        production.fields.spatial[5:8],
    )
    assert torch.equal(
        functional_observation["pressure"],
        production.fields.spatial[8],
    )

    expected = runtime.step(state, controls, 0)
    production.advance(1)
    assert torch.equal(expected[0], production.fields.spatial[:5])
    assert torch.equal(expected[1], production.fields.spectral[:5])


def test_p92_rejects_invalid_inputs_without_conversion_or_clipping(tmp_path):
    _, _, runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    controls = _control(runtime)

    with pytest.raises(ValueError, match="only activity"):
        runtime.step(state, {"wrong": controls["activity"]}, 0)
    with pytest.raises(ValueError, match="lower bound"):
        runtime.step(state, _control(runtime, -0.01), 0)
    with pytest.raises(TypeError, match="dtype must be float64"):
        runtime.step(
            state,
            {"activity": controls["activity"].to(torch.float32)},
            0,
        )
    with pytest.raises(ValueError, match="shape must be"):
        runtime.step(state, {"activity": controls["activity"][..., :-1]}, 0)
    with pytest.raises(TypeError, match="step_index"):
        runtime.step(state, controls, True)
    with pytest.raises(ValueError, match="non-negative"):
        runtime.step(state, controls, -1)

    broken_state = (state[0].clone(), state[1].clone())
    broken_state[0][0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        runtime.observe(broken_state, None)


def test_p92_factory_rejects_unqualified_geometry_and_compiled_pointwise(
    tmp_path,
):
    periodic = _simulation(tmp_path, name="compile_rejection")
    execution = TorchSpectralExecution(
        runtime_path="periodic_spectral",
        device="cpu",
        options={
            **{
                key: value
                for key, value in periodic.specification.execution.options.items()
                if key not in {"device", "fallback_allowed"}
            },
            "pointwise_execution": "compile",
        },
    )
    with pytest.raises(ValueError, match="eager pointwise"):
        periodic_activity_functional_request(
            replace(periodic, execution=execution).specification
        )

    model = _model()
    geometry = PlaneSlab(shape=(6, 6, 4), lengths=(6.0, 6.0, 4.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    plane = replace(
        periodic,
        geometry=geometry,
        boundaries=boundaries,
    )
    with pytest.raises(ValueError, match="Plane execution"):
        periodic_activity_functional_request(plane.specification)
