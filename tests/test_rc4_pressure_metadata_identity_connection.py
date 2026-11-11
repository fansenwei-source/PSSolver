"""RC4.4.4 public pressure metadata, identity, and restart connection."""

from __future__ import annotations

import hashlib
import json
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
    compile_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.legacy_channel import (
    build_channel_checkpoint_compatibility_identity,
)
from pssolver.functional import (
    FunctionalCheckpointCompatibilityError,
    build_channel_activity_functional_runtime,
    channel_activity_functional_declaration,
    channel_activity_functional_request,
)
from pssolver.geometries import RectangularChannel
from pssolver.io.checkpoint import (
    seal_checkpoint_metadata,
    write_functional_checkpoint_provenance,
)
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from tests.v0_5_s28a_channel_v04_checkpoint import (
    build_channel_production_fixture,
    write_channel_beris_edwards_checkpoint,
)
from tests.v0_5_s28a_channel_v04_checkpoint import (
    restore_channel_checkpoint as _load_checkpoint,
)


def _simulation(
    tmp_path: Path,
    *,
    name: str = "rc444",
    fixed_iterations: int | None = 2,
    pressure_extra: dict[str, object] | None = None,
) -> Simulation:
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
    shape = (4, 4, 4)
    geometry = RectangularChannel(shape=shape, lengths=(4.0, 4.0, 4.0))
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
    values = np.zeros((*shape, 5), dtype=np.float64)
    values[..., 0] = 0.2
    values[..., 3] = -0.1
    values += np.random.default_rng(444).normal(
        scale=1.0e-3,
        size=values.shape,
    )
    np.save(source / "Q_0.npy", values, allow_pickle=False)
    pressure = {
        "algorithm": "preconditioned_conjugate_gradient",
        "relative_tolerance": 1.0e-10,
        "max_iterations": 20,
        "fixed_iterations": fixed_iterations,
        "warm_start": True,
    }
    pressure.update({} if pressure_extra is None else pressure_extra)
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
            steps=2,
            save_interval=2,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization={"pressure_solver": pressure},
    )


def _production(simulation: Simulation):
    compiled = compile_simulation(simulation)
    adapter = build_channel_production_fixture(compiled.application_request)
    return compiled, adapter


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _tamper_production_identity(
    metadata: dict[str, object],
    *,
    mutation: str,
) -> dict[str, object]:
    identity = dict(metadata["compatibility_identity"])
    layer = dict(identity["forward_dynamics"])
    if mutation == "version":
        layer["version"] = "channel.production.forward.v3"
    elif mutation == "policy":
        payload = dict(layer["payload"])
        policy = dict(payload["pressure_solver_policy"])
        policy["relative_tolerance_role"] = "tampered"
        payload["pressure_solver_policy"] = policy
        layer["payload"] = payload
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unsupported mutation: {mutation}")
    identity["forward_dynamics"] = layer
    metadata["compatibility_identity"] = identity
    metadata["compatibility_identity_sha256"] = _canonical_sha256(identity)
    metadata.pop("metadata_sha256")
    return seal_checkpoint_metadata(metadata)


def _tamper_functional_identity_version(
    directory: Path,
    *,
    field: str,
    version: str | None,
) -> None:
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    bridge = dict(metadata["functional_bridge"])
    identity = dict(bridge["checkpoint_compatibility_identity"])
    layer = dict(identity[field])
    if version is None:
        payload = dict(layer["payload"])
        execution = dict(payload["execution"])
        functional_runtime = dict(execution["functional_runtime"])
        pressure = dict(functional_runtime["pressure_solver"])
        policy = dict(pressure["derived_policy"])
        policy["relative_tolerance_role"] = "tampered"
        pressure["derived_policy"] = policy
        functional_runtime["pressure_solver"] = pressure
        execution["functional_runtime"] = functional_runtime
        payload["execution"] = execution
        layer["payload"] = payload
    else:
        layer["version"] = version
    identity[field] = layer
    bridge["checkpoint_compatibility_identity"] = identity
    bridge["checkpoint_compatibility_sha256"] = _canonical_sha256(identity)
    metadata["functional_bridge"] = bridge
    metadata.pop("metadata_sha256")
    sealed = seal_checkpoint_metadata(metadata)
    path.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_functional_checkpoint_provenance(directory, sealed, bridge)


def test_declaration_runtime_capabilities_identity_and_diagnostics_agree(
    tmp_path,
):
    simulation = _simulation(tmp_path)
    declaration = channel_activity_functional_declaration(
        simulation.specification
    )
    request = channel_activity_functional_request(simulation.specification)
    runtime = build_channel_activity_functional_runtime(request)

    assert declaration.capabilities.to_metadata() == (
        runtime.capabilities.to_metadata()
    )
    assert runtime.capabilities.explicit_vjp is False
    assert runtime.capabilities.differentiability == (
        "validated_custom_adjoint"
    )
    assert runtime.capabilities.inner_solve_gradient == (
        "custom_implicit_pressure_adjoint_v2"
    )
    declared_pressure = declaration.identity.to_metadata()["execution"][
        "functional_runtime_declaration"
    ]["pressure_solver"]
    runtime_pressure = runtime.identity().to_metadata()["execution"][
        "functional_runtime"
    ]["pressure_solver"]
    assert declared_pressure == runtime_pressure
    assert runtime_pressure["derived_policy"]["initial_guess_policy"] == (
        "zero_every_call"
    )
    assert runtime_pressure["pressure_transpose_protocol"] == "2"
    assert runtime_pressure["implicit_adjoint"] == "2"
    assert runtime_pressure["diagnostics_schema_version"] == 2

    diagnostics = runtime.diagnostics()
    assert diagnostics["schema_version"] == 2
    assert diagnostics["pressure"]["schema_version"] == 2
    diagnostics["pressure"]["primal"]["termination_reason"] = "tampered"
    assert runtime.diagnostics()["pressure"]["primal"][
        "termination_reason"
    ] == "not_run"


def test_production_runtime_diagnostics_are_public_json_snapshots(tmp_path):
    _, adapter = _production(_simulation(tmp_path))
    before = adapter.flow_diagnostics()
    assert before["schema_version"] == 2
    assert before["pressure"]["schema_version"] == 2
    original_termination = before["pressure"]["termination_reason"]
    before["pressure"]["termination_reason"] = "tampered"
    assert adapter.flow_diagnostics()["pressure"]["termination_reason"] == (
        original_termination
    )

    adapter.advance(1)
    after = adapter.flow_diagnostics()
    assert after["pressure"]["solve_phase"] == "production_primal"
    assert after["pressure"]["initial_guess_policy"] == (
        "persistent_production_warm_start"
    )
    assert json.loads(json.dumps(after, allow_nan=False)) == after


def test_v4_identities_bind_pressure_policy_without_layout_version_changes(
    tmp_path,
):
    simulation = _simulation(tmp_path)
    compiled, adapter = _production(simulation)
    production = build_channel_checkpoint_compatibility_identity(
        compiled.application_request,
        fields=adapter.fields,
        pressure_guess=adapter.capture_pressure_guess(),
        backend_restart=adapter.backend_restart_metadata(),
    )
    assert production.forward_dynamics.version == "channel.production.forward.v4"
    assert production.state_layout.version == "channel.production.state.v3"
    assert production.backend_restart.version == "channel.production.backend.v1"
    production_payload = production.forward_dynamics.to_metadata()["payload"]
    assert production_payload["pressure_solver_policy"] == (
        adapter.engine.constraint.term.pressure_solver.pressure_convergence_policy()
    )
    assert production_payload["pressure_convergence_schema_version"] == 2

    runtime = build_channel_activity_functional_runtime(
        channel_activity_functional_request(simulation.specification)
    )
    functional = runtime.checkpoint_bridge._compatibility_identity
    assert functional.forward_dynamics.version == "channel.functional.forward.v4"
    assert functional.derivative_dynamics.version == (
        "channel.functional.derivative.v4"
    )
    assert functional.state_layout.version == "channel.functional.state.v2"
    assert functional.backend_restart.version == (
        "channel.functional.backend.v2"
    )
    derivative = functional.derivative_dynamics.to_metadata()["payload"]
    assert derivative["pressure_transpose_protocol"] == "2"
    assert derivative["implicit_adjoint"] == (
        "channel_pressure_implicit_adjoint.v2"
    )
    assert derivative["pressure_convergence_schema_version"] == 2
    assert derivative["transpose_initial_guess"] == "zero_every_call"


@pytest.mark.parametrize("mutation", ("version", "policy"))
def test_production_stale_or_tampered_identity_rejects_before_mutation(
    tmp_path,
    monkeypatch,
    mutation,
):
    simulation = _simulation(tmp_path)
    compiled, source = _production(simulation)
    source.advance(1)
    directory = tmp_path / "production_v3"
    write_channel_beris_edwards_checkpoint(
        directory,
        source,
        run_spec=compiled.application_request,
    )
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata = _tamper_production_identity(
        metadata,
        mutation=mutation,
    )
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _, target = _production(simulation)
    before_fields = {
        f"{name}{suffix}": target.fields[f"{name}{suffix}"].clone()
        for name in target.engine.components
        for suffix in ("", ".hat")
    }
    before_guess = target.capture_pressure_guess().clone()
    before_steps = target.completed_steps

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tensor payload opened before identity rejection")

    monkeypatch.setattr(
        "pssolver.configuration.legacy_channel._validated_array",
        forbidden,
    )
    with pytest.raises(ValueError, match="identity does not match target"):
        _load_checkpoint(
            directory,
            target,
            run_spec=compiled.application_request,
        )
    assert target.completed_steps == before_steps
    assert torch.equal(target.capture_pressure_guess(), before_guess)
    assert all(
        torch.equal(target.fields[name], value)
        for name, value in before_fields.items()
    )


@pytest.mark.parametrize(
    ("field", "version"),
    (
        ("forward_dynamics", "channel.functional.forward.v3"),
        ("derivative_dynamics", "channel.functional.derivative.v3"),
        ("forward_dynamics", None),
    ),
)
def test_functional_stale_or_tampered_identity_rejects_before_tensor_load(
    tmp_path,
    monkeypatch,
    field,
    version,
):
    runtime = build_channel_activity_functional_runtime(
        channel_activity_functional_request(
            _simulation(tmp_path, name=field).specification
        )
    )
    directory = tmp_path / f"functional_{field}"
    runtime.checkpoint_bridge.export_checkpoint(
        directory,
        runtime.initial_state(),
        completed_steps=0,
    )
    _tamper_functional_identity_version(
        directory,
        field=field,
        version=version,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tensor payload opened before identity rejection")

    monkeypatch.setattr(
        "pssolver.functional.channel_checkpoint.ChannelActivityCheckpointBridge._load_tensor",
        forbidden,
    )
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="forward compatibility differs",
    ):
        runtime.checkpoint_bridge.import_checkpoint(directory)


def test_production_v3_import_to_functional_rejects_before_tensor_load(
    tmp_path,
    monkeypatch,
):
    simulation = _simulation(tmp_path)
    compiled, production = _production(simulation)
    directory = tmp_path / "production_to_functional_v3"
    write_channel_beris_edwards_checkpoint(
        directory,
        production,
        run_spec=compiled.application_request,
    )
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata = _tamper_production_identity(
        metadata,
        mutation="version",
    )
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    runtime = build_channel_activity_functional_runtime(
        channel_activity_functional_request(simulation.specification)
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tensor payload opened before identity rejection")

    monkeypatch.setattr(
        "pssolver.functional.channel_checkpoint.ChannelActivityCheckpointBridge._load_tensor",
        forbidden,
    )
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="forward dynamics differs",
    ):
        runtime.checkpoint_bridge.import_checkpoint(directory)
