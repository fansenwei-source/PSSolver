"""P9.7.4 pure Channel functional runtime and checkpoint bridge tests."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

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
from pssolver.applications.channel_beris_edwards import load_channel_initial_q
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.public_simulation_runner import compile_public_simulation
from pssolver.functional import (
    CHANNEL_ACTIVITY_RUNTIME_STAGE,
    ChannelActivityCheckpointBridge,
    ChannelActivityFunctionalRuntime,
    FunctionalRuntimeProtocol,
    build_channel_activity_functional_runtime,
    channel_activity_functional_request,
)
from pssolver.geometries import RectangularChannel
from pssolver.io.checkpoint import seal_checkpoint_metadata
from pssolver.models.active_nematics import CompleteStressBerisEdwards, Q_COMPONENTS
from pssolver.runtime.channel_beris_edwards import (
    ChannelBerisEdwardsRuntimeBuildRequest,
    build_channel_beris_edwards_runtime,
)
from pssolver.workflows.channel_beris_edwards import (
    write_channel_beris_edwards_checkpoint,
)


def _simulation(tmp_path: Path, *, name: str = "p974") -> Simulation:
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
    )
    shape = (8, 6, 4)
    geometry = RectangularChannel(shape=shape, lengths=(8.0, 6.0, 4.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": no_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    source = tmp_path / f"{name}_snapshot"
    source.mkdir(parents=True)
    rng = np.random.default_rng(20260929)
    values = rng.normal(scale=0.01, size=(*shape, 5)).astype(np.float64)
    values[..., 0] += 0.2
    values[..., 3] -= 0.1
    np.save(source / "Q_0.npy", values, allow_pickle=False)
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(source, step=0),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / name,
            steps=1,
            save_interval=1,
            diagnostic_interval=1,
        ),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        },
    )


def _runtime(tmp_path: Path):
    simulation = _simulation(tmp_path)
    request = channel_activity_functional_request(simulation.specification)
    return simulation, request, build_channel_activity_functional_runtime(request)


def _control(runtime, value: float = 0.01, *, requires_grad=False):
    spec = runtime.control_specs[0].tensor
    control = torch.full(
        spec.shape,
        value,
        dtype=torch.float64,
        device=spec.device,
    )
    control[:, :4] *= 1.15
    control.requires_grad_(requires_grad)
    return {"activity": control}


def _assert_state_equal(left, right):
    assert all(
        torch.equal(a, b) for a, b in zip(left, right, strict=True)
    )


def _channel_full_complex_violation(spectral):
    reverse_x = torch.remainder(
        -torch.arange(spectral.shape[-3], device=spectral.device),
        spectral.shape[-3],
    )
    reflected = spectral.index_select(-3, reverse_x)
    return (spectral - reflected.conj()).abs().max()


def _production_adapter(simulation):
    compiled = compile_public_simulation(simulation.specification)
    values, _, _ = load_channel_initial_q(compiled.run_spec)
    adapter = build_channel_beris_edwards_runtime(
        ChannelBerisEdwardsRuntimeBuildRequest(
            run_spec=compiled.run_spec,
            initial_values=values,
            device="cpu",
        )
    )
    return compiled.run_spec, adapter


def _adapter_state(adapter):
    fields = adapter.fields
    return (
        torch.stack(tuple(fields[name] for name in Q_COMPONENTS)),
        torch.stack(tuple(fields[f"{name}.hat"] for name in Q_COMPONENTS)),
    )


def test_p974_constructs_executable_pure_channel_runtime(tmp_path):
    _, request, runtime = _runtime(tmp_path)

    assert isinstance(runtime, FunctionalRuntimeProtocol)
    assert isinstance(runtime, ChannelActivityFunctionalRuntime)
    assert runtime.control_specs == request.control_fields
    assert [item.name for item in runtime.observation_specs] == [
        "Q",
        "velocity",
        "pressure",
    ]
    runtime.state_spec.validate(runtime.initial_state())
    capabilities = runtime.capabilities
    assert capabilities.pure_step is True
    assert capabilities.combined_step_and_observe is True
    assert capabilities.deterministic_replay == "bitwise"
    assert capabilities.differentiability == "validated_custom_adjoint"
    assert capabilities.inner_solve_gradient == (
        "custom_implicit_pressure_adjoint_v1"
    )
    assert capabilities.durable_checkpoint_bridge is True
    assert capabilities.differentiable_inputs == ("state", "activity")
    identity = runtime.identity().to_metadata()["execution"][
        "functional_runtime"
    ]
    assert identity["stage"] == CHANNEL_ACTIVITY_RUNTIME_STAGE
    assert identity["pressure_solver"]["initial_guess"] == "zero_every_call"
    assert identity["pressure_solver"]["production_warm_start_read"] is False
    assert runtime._flow_model.pressure_guess is None


def test_p974_step_is_input_pure_and_replay_is_bitwise(tmp_path):
    _, request, runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    before = tuple(value.clone() for value in state)
    control = _control(runtime)

    first_next, first_observations = runtime.step_and_observe(state, control, 0)
    unrelated = tuple(value + 1.0e-6 for value in state)
    runtime.step_and_observe(unrelated, _control(runtime, 0.02), 17)
    second_next, second_observations = runtime.step_and_observe(state, control, 0)
    fresh = build_channel_activity_functional_runtime(request)
    fresh_next, fresh_observations = fresh.step_and_observe(
        fresh.initial_state(), control, 0
    )

    _assert_state_equal(state, before)
    _assert_state_equal(first_next, second_next)
    _assert_state_equal(first_next, fresh_next)
    assert all(
        torch.equal(first_observations[name], second_observations[name])
        for name in first_observations
    )
    assert all(
        torch.equal(first_observations[name], fresh_observations[name])
        for name in first_observations
    )
    assert runtime._flow_model.pressure_guess is None
    assert runtime._pressure.last_primal_diagnostics.iterations == 12
    assert np.isfinite(runtime._pressure.last_primal_diagnostics.residual)
    assert runtime._solver.model.nlmodel.q_gradient_cache is None
    assert runtime._flow_model.q_gradient_cache is None


def test_p974_projects_injected_anti_hermitian_state_after_each_step(tmp_path):
    _, _, runtime = _runtime(tmp_path)
    physical, spectral = runtime.initial_state()
    perturbed = spectral.clone()
    perturbed[0, 0, 1, 2, 1] += 1.0e-6j
    assert _channel_full_complex_violation(perturbed).item() > 0.0

    next_state = runtime.step(
        (physical, perturbed),
        _control(runtime),
        0,
    )

    assert _channel_full_complex_violation(next_state[1]).item() == 0.0
    assert all(bool(torch.isfinite(value).all()) for value in next_state)


def test_p974_activity_and_state_have_finite_first_order_gradients(tmp_path):
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
    assert all(torch.isfinite(value).all() for value in gradients)
    assert torch.count_nonzero(gradients[-1]) > 0


def test_p974_matches_production_first_step_under_zero_pressure_start(tmp_path):
    simulation, _, runtime = _runtime(tmp_path)
    run_spec, production = _production_adapter(simulation)
    del run_spec
    state = runtime.initial_state()
    controls = _control(runtime)

    production.solver.model.parameters["alpha"] = controls["activity"]
    production.solver.model.static_model.pressure_guess = None
    production.synchronize_for_observation()
    observations = runtime.observe(state, controls)
    assert torch.equal(observations["velocity"], production.fields.spatial[5:8])
    assert torch.equal(observations["pressure"], production.fields.spatial[8])

    expected = runtime.step(state, controls, 0)
    production.advance(1)
    _assert_state_equal(expected, _adapter_state(production))


def test_p974_checkpoint_round_trip_and_production_import(tmp_path):
    simulation, _, runtime = _runtime(tmp_path)
    assert isinstance(runtime.checkpoint_bridge, ChannelActivityCheckpointBridge)
    state = runtime.step(runtime.initial_state(), _control(runtime), 0)
    before = tuple(value.clone() for value in state)
    functional_directory = tmp_path / "functional_checkpoint"
    runtime.checkpoint_bridge.export_checkpoint(
        functional_directory,
        state,
        completed_steps=1,
    )
    imported = runtime.checkpoint_bridge.import_checkpoint(functional_directory)

    _assert_state_equal(state, before)
    _assert_state_equal(state, imported.state)
    assert imported.completed_steps == 1
    assert imported.source_format == "channel_functional_bridge_v2"
    metadata = json.loads(
        (functional_directory / "checkpoint.json").read_text()
    )
    assert metadata["functional_bridge"]["pressure_warm_start"] == (
        "absent_functional_zero_start"
    )

    run_spec, production = _production_adapter(simulation)
    production.synchronize_for_observation()
    production_directory = tmp_path / "production_checkpoint"
    write_channel_beris_edwards_checkpoint(
        production_directory,
        production,
        runtime_identity_sha256=run_spec.runtime_identity_sha256(),
    )
    imported_production = runtime.checkpoint_bridge.import_checkpoint(
        production_directory
    )
    _assert_state_equal(imported_production.state, _adapter_state(production))
    assert imported_production.source_format == "channel_production_v2"


def test_p974_checkpoint_rejects_identity_checksum_and_existing_target(
    tmp_path,
):
    _, _, runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    target = tmp_path / "checkpoint"
    runtime.checkpoint_bridge.export_checkpoint(
        target, state, completed_steps=0
    )
    with pytest.raises(FileExistsError):
        runtime.checkpoint_bridge.export_checkpoint(
            target, state, completed_steps=0
        )

    metadata_path = target / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["functional_bridge"][
        "functional_runtime_identity_sha256"
    ] = "0" * 64
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="metadata checksum mismatch"):
        runtime.checkpoint_bridge.import_checkpoint(target)

    checksum = tmp_path / "checksum"
    runtime.checkpoint_bridge.export_checkpoint(
        checksum, state, completed_steps=0
    )
    tensor_path = checksum / "state__q_physical.npy"
    data = bytearray(tensor_path.read_bytes())
    data[-1] ^= 1
    tensor_path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum mismatch"):
        runtime.checkpoint_bridge.import_checkpoint(checksum)


def test_p974_functional_completed_step_tamper_is_rejected_before_loading(
    tmp_path,
    monkeypatch,
):
    _, _, runtime = _runtime(tmp_path)
    target = tmp_path / "functional_progress_tamper"
    runtime.checkpoint_bridge.export_checkpoint(
        target,
        runtime.initial_state(),
        completed_steps=0,
    )
    metadata_path = target / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["completed_steps"] = 999
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("np.load must not run before metadata preflight")

    monkeypatch.setattr(
        "pssolver.functional.channel_checkpoint.np.load",
        forbidden,
    )
    with pytest.raises(ValueError, match="metadata checksum mismatch"):
        runtime.checkpoint_bridge.import_checkpoint(target)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("format_version", "unsupported complete-stress Channel checkpoint"),
        ("completed_steps", "progress counters differ"),
        ("backend_files", "backend manifest is incomplete"),
    ),
)
def test_p974_production_import_applies_the_complete_checkpoint_contract(
    tmp_path,
    mutation,
    message,
):
    simulation, _, runtime = _runtime(tmp_path)
    run_spec, production = _production_adapter(simulation)
    source = tmp_path / "strict_production_source"
    write_channel_beris_edwards_checkpoint(
        source,
        production,
        runtime_identity_sha256=run_spec.runtime_identity_sha256(),
    )
    target = tmp_path / f"strict_production_{mutation}"
    shutil.copytree(source, target)
    metadata_path = target / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mutation == "format_version":
        metadata["format_version"] = 99
    elif mutation == "completed_steps":
        metadata["completed_steps"] = 7
    else:
        del metadata["backend_files"]
    metadata.pop("metadata_sha256")
    metadata = seal_checkpoint_metadata(metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        runtime.checkpoint_bridge.import_checkpoint(target)


def test_p974_rejects_invalid_control_and_terminal_observation_is_q_only(
    tmp_path,
):
    _, _, runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    terminal = runtime.observe(state, None)
    assert set(terminal) == {"Q"}
    assert terminal["Q"] is state[0]

    with pytest.raises(ValueError, match="only activity"):
        runtime.step(state, {"wrong": _control(runtime)["activity"]}, 0)
    with pytest.raises(ValueError, match="lower bound"):
        runtime.step(state, _control(runtime, -0.01), 0)
    with pytest.raises(TypeError, match="step_index"):
        runtime.step(state, _control(runtime), True)


def test_p974_activity_gradient_matches_central_difference(tmp_path):
    _, _, runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    control = _control(runtime, requires_grad=True)["activity"]
    generator = torch.Generator().manual_seed(974)
    direction = torch.randn(
        control.shape,
        generator=generator,
        dtype=control.dtype,
    )
    direction /= torch.linalg.vector_norm(direction)

    def objective(activity):
        next_state, observations = runtime.step_and_observe(
            state, {"activity": activity}, 0
        )
        return (
            next_state[0].square().mean()
            + 0.1 * observations["velocity"].square().mean()
            + 0.01 * observations["pressure"].square().mean()
        )

    value = objective(control)
    gradient = torch.autograd.grad(value, control)[0]
    predicted = torch.sum(gradient * direction)
    epsilon = 2.0e-5
    measured = (
        objective((control + epsilon * direction).detach())
        - objective((control - epsilon * direction).detach())
    ) / (2.0 * epsilon)
    torch.testing.assert_close(predicted, measured, rtol=2.0e-5, atol=2.0e-10)
