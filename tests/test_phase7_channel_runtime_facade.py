"""P7.4 opt-in Channel runtime, output, checkpoint, and restart contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest
import torch

from pssolver.channel import Q_COMPONENTS, build_active_nematic_channel
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    ChannelRuntimePath,
)
from pssolver.experimental.channel_compiled_v2 import (
    build_channel_compiled_v2_runtime,
)
from pssolver.runtime import (
    ChannelOutputViews,
    ChannelRuntimeAdapterProtocol,
    ChannelRuntimeBuildRequest,
    LegacyChannelRuntimeAdapter,
    build_channel_active_nematic_runtime,
)
from pssolver.workflows import (
    CHANNEL_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    capture_channel_checkpoint,
    load_channel_checkpoint,
    read_channel_checkpoint_header,
    restore_channel_checkpoint,
    write_channel_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
SHAPE = (8, 6, 5)
LENGTHS = (8.0, 6.0, 5.0)


def _spec(path="legacy_channel", **overrides):
    values = {
        "shape": SHAPE,
        "lengths": LENGTHS,
        "dt": 1.0e-3,
        "steps": 5,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "activity": 0.2,
        "pressure_relative_tolerance": 1.0e-8,
        "pressure_max_iterations": 100,
        "runtime_path": path,
    }
    values.update(overrides)
    return ChannelActiveNematicRunSpec(**values)


def _initial_q():
    coordinate = torch.arange(
        SHAPE[0] * SHAPE[1] * SHAPE[2],
        dtype=torch.float32,
    ).reshape(SHAPE)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


def _request(spec, initial_q=None):
    return ChannelRuntimeBuildRequest(
        run_spec=spec,
        production_metadata={
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        },
        initial_q=_initial_q() if initial_q is None else initial_q,
        device="cpu",
    )


def _build(path="legacy_channel"):
    spec = _spec(path)
    values = _initial_q()
    compiled_builder = None
    legacy_builder = None
    if spec.runtime_path is ChannelRuntimePath.LEGACY_CHANNEL:
        material = spec.components.material
        pressure = spec.components.pressure_solver

        def legacy_builder():
            solver = build_active_nematic_channel(
                spec.shape,
                spec.lengths,
                spec.dt,
                values,
                device="cpu",
                batchsize=spec.batch_size,
                rho=material.rho,
                elastic_constant=material.elastic_constant,
                beta=material.beta,
                friction=material.friction,
                viscosity=material.viscosity,
                pressure_rel_tol=pressure.relative_tolerance,
                pressure_max_iter=pressure.max_iterations,
                pressure_fixed_iterations=pressure.fixed_iterations,
            )
            solver.parameters["alpha"] = torch.full(
                (spec.batch_size, *spec.shape),
                material.activity,
                dtype=solver.dtype,
                device=solver.device,
            )
            return solver

    if spec.runtime_path is ChannelRuntimePath.COMPILED_CHANNEL_V2:
        compiled_builder = lambda: build_channel_compiled_v2_runtime(
            spec,
            initial_q=values,
            device="cpu",
        )
    adapter = build_channel_active_nematic_runtime(
        _request(spec, values),
        legacy_builder=legacy_builder,
        compiled_builder=compiled_builder,
    )
    return spec, adapter


def _state_clone(adapter):
    return (
        adapter.fields.spatial.detach().clone(),
        adapter.fields.spectral.detach().clone(),
        adapter.completed_steps,
    )


def _assert_state_unchanged(adapter, before):
    physical, spectral, completed = before
    assert torch.equal(adapter.fields.spatial, physical)
    assert torch.equal(adapter.fields.spectral, spectral)
    assert adapter.completed_steps == completed


def test_runtime_selection_is_explicit_and_legacy_remains_default():
    default = ChannelActiveNematicRunSpec()
    compiled = ChannelActiveNematicRunSpec(
        runtime_path="compiled_channel_v2"
    )

    assert default.runtime_path is ChannelRuntimePath.LEGACY_CHANNEL
    assert compiled.runtime_path is ChannelRuntimePath.COMPILED_CHANNEL_V2
    assert default.runtime_selection_metadata() == {
        "authority": (
            "pssolver.configuration.channel_active_nematics."
            "ChannelActiveNematicRunSpec.runtime_path"
        ),
        "default": "legacy_channel",
        "requested": "legacy_channel",
        "effective": "legacy_channel",
        "fallback_allowed": False,
    }
    assert compiled.runtime_selection_metadata()["effective"] == (
        "compiled_channel_v2"
    )


def test_runtime_identity_excludes_workflow_but_binds_numerics_and_path():
    baseline = _spec()

    assert baseline.runtime_identity_sha256() == _spec(
        steps=17,
        generated_output_directory=Path("elsewhere"),
    ).runtime_identity_sha256()
    assert baseline.runtime_identity_sha256() != _spec(
        activity=0.3
    ).runtime_identity_sha256()
    assert baseline.runtime_identity_sha256() != _spec(
        "compiled_channel_v2"
    ).runtime_identity_sha256()
    assert baseline.runtime_identity_metadata()["dynamics_versions"] == {
        "forward_stokes_nyquist": (
            "channel_no_slip.periodic_nyquist_zero.v2"
        )
    }
    assert baseline.pre_rc5_runtime_identity_sha256() == (
        "14e578c48897d3595c610d8f6f9dbff3573bb9a7e8fd9b5483e46b1acdc90fec"
    )
    assert (
        baseline.runtime_identity_sha256()
        != baseline.pre_rc5_runtime_identity_sha256()
    )
    json.dumps(baseline.identity_metadata(), allow_nan=False, sort_keys=True)


def test_build_request_rejects_mixed_authority_and_unordered_fields():
    spec = _spec()
    metadata = {
        "configuration": spec.identity_metadata(),
        "runtime_selection": spec.runtime_selection_metadata(),
    }
    bad = dict(metadata)
    bad["runtime_selection"] = dict(spec.runtime_selection_metadata())
    bad["runtime_selection"]["effective"] = "compiled_channel_v2"
    with pytest.raises(ValueError, match="resolved run spec"):
        ChannelRuntimeBuildRequest(spec, bad, _initial_q(), "cpu")
    reversed_q = dict(reversed(tuple(_initial_q().items())))
    with pytest.raises(ValueError, match="ordered Q components"):
        ChannelRuntimeBuildRequest(spec, metadata, reversed_q, "cpu")


def test_factory_builds_selected_runtime_without_fallback():
    legacy_spec, legacy = _build()
    compiled_spec, compiled = _build("compiled_channel_v2")

    assert isinstance(legacy, LegacyChannelRuntimeAdapter)
    assert isinstance(legacy, ChannelRuntimeAdapterProtocol)
    assert isinstance(compiled, ChannelRuntimeAdapterProtocol)
    assert legacy.runtime_path is ChannelRuntimePath.LEGACY_CHANNEL
    assert compiled.runtime_path is ChannelRuntimePath.COMPILED_CHANNEL_V2
    assert legacy_spec.runtime_path is ChannelRuntimePath.LEGACY_CHANNEL
    assert compiled_spec.runtime_path is ChannelRuntimePath.COMPILED_CHANNEL_V2
    assert legacy.to_metadata()["fallback_used"] is False
    assert compiled.to_metadata()["fallback_used"] is False

    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        build_channel_active_nematic_runtime(_request(legacy_spec))
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        build_channel_active_nematic_runtime(_request(compiled_spec))

    def fail():
        raise RuntimeError("sentinel construction failure")

    with pytest.raises(RuntimeError, match="sentinel construction failure"):
        build_channel_active_nematic_runtime(
            _request(compiled_spec),
            compiled_builder=fail,
        )


@pytest.mark.parametrize("path", ("legacy_channel", "compiled_channel_v2"))
def test_output_ownership_is_zero_copy_and_stable(path):
    _, adapter = _build(path)
    views = adapter.output_views

    assert isinstance(views, ChannelOutputViews)
    assert views.owner is adapter.fields.spatial
    assert views.q.untyped_storage().data_ptr() == (
        views.owner.untyped_storage().data_ptr()
    )
    assert views.velocity.untyped_storage().data_ptr() == (
        views.owner.untyped_storage().data_ptr()
    )
    assert views.pressure.untyped_storage().data_ptr() == (
        views.owner.untyped_storage().data_ptr()
    )
    assert views.to_metadata()["zero_copy"] is True
    adapter.advance(2)
    assert adapter.output_views is views


def test_legacy_and_compiled_facades_are_byte_identical_with_callbacks():
    _, legacy = _build()
    _, compiled = _build("compiled_channel_v2")
    legacy_callbacks = []
    compiled_callbacks = []

    for step in range(5):
        legacy.advance(
            1,
            pre_update_callback=lambda solver, local: legacy_callbacks.append(
                (solver is legacy.solver, step, local)
            ),
        )
        compiled.advance(
            1,
            pre_update_callback=lambda solver, local: compiled_callbacks.append(
                (solver is compiled.solver, step, local)
            ),
        )
        assert torch.equal(legacy.fields.spatial, compiled.fields.spatial)
        assert torch.equal(legacy.fields.spectral, compiled.fields.spectral)

    assert legacy_callbacks == [(True, step, 0) for step in range(5)]
    assert compiled_callbacks == [(True, step, 0) for step in range(5)]
    legacy.synchronize_for_observation()
    compiled.synchronize_for_observation()
    assert torch.equal(legacy.fields.spatial, compiled.fields.spatial)
    assert torch.equal(
        legacy.capture_pressure_guess(),
        compiled.capture_pressure_guess(),
    )


@pytest.mark.parametrize("path", ("legacy_channel", "compiled_channel_v2"))
def test_checkpoint_disk_round_trip_preserves_q_pressure_and_header(
    tmp_path,
    path,
):
    spec, adapter = _build(path)
    adapter.advance(2)
    checkpoint = capture_channel_checkpoint(
        adapter,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    directory = write_channel_checkpoint(tmp_path / path, checkpoint)
    header = read_channel_checkpoint_header(directory)
    loaded = load_channel_checkpoint(directory)

    current_umask = os.umask(0)
    os.umask(current_umask)
    assert stat.S_IMODE(directory.stat().st_mode) == (0o777 & ~current_umask)
    assert checkpoint.format_version == (
        CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    )
    assert header.runtime_path is spec.runtime_path
    assert header.format_version == CHANNEL_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    assert header.completed_steps == 2
    assert loaded.to_metadata() == checkpoint.to_metadata()
    for name in Q_COMPONENTS:
        assert torch.equal(
            loaded.evolved_spatial[name],
            checkpoint.evolved_spatial[name].cpu(),
        )
        assert torch.equal(
            loaded.evolved_spectral[name],
            checkpoint.evolved_spectral[name].cpu(),
        )
    assert torch.equal(loaded.pressure_guess, checkpoint.pressure_guess.cpu())


@pytest.mark.parametrize("path", ("legacy_channel", "compiled_channel_v2"))
def test_same_runtime_split_restart_is_byte_identical(path):
    spec, continuous = _build(path)
    _, segment = _build(path)
    _, resumed = _build(path)

    continuous.advance(5)
    segment.advance(2)
    checkpoint = capture_channel_checkpoint(
        segment,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    assert restore_channel_checkpoint(
        resumed,
        checkpoint,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    ) == 2
    resumed.advance(3)

    assert continuous.completed_steps == resumed.completed_steps == 5
    assert torch.equal(continuous.fields.spatial, resumed.fields.spatial)
    assert torch.equal(continuous.fields.spectral, resumed.fields.spectral)
    continuous.synchronize_for_observation()
    resumed.synchronize_for_observation()
    assert torch.equal(continuous.fields.spatial, resumed.fields.spatial)
    assert torch.equal(
        continuous.capture_pressure_guess(),
        resumed.capture_pressure_guess(),
    )


def test_pre_rc5_even_checkpoint_with_repaired_nyquist_state_restores_exactly():
    spec, source = _build()
    _, target = _build()
    source.advance(2)
    current = capture_channel_checkpoint(
        source,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    legacy = dataclasses.replace(
        current,
        format_version=CHANNEL_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_identity_sha256=spec.pre_rc5_runtime_identity_sha256(),
        dynamics_versions=None,
    )

    assert restore_channel_checkpoint(
        target,
        legacy,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
        pre_rc5_runtime_identity_sha256=(
            spec.pre_rc5_runtime_identity_sha256()
        ),
    ) == 2
    _assert_state_unchanged(target, _state_clone(source))


def test_pre_repair_even_nyquist_state_rejects_before_target_mutation():
    spec, source = _build()
    _, target = _build()
    source.advance(2)
    current = capture_channel_checkpoint(
        source,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    algebraic = {
        name: value.clone()
        for name, value in current.algebraic_spectral.items()
    }
    algebraic["ux"][:, SHAPE[0] // 2, 0, 0] = 1.0
    legacy = dataclasses.replace(
        current,
        format_version=CHANNEL_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_identity_sha256=spec.pre_rc5_runtime_identity_sha256(),
        algebraic_spectral=algebraic,
        dynamics_versions=None,
    )
    before = _state_clone(target)

    with pytest.raises(ValueError, match="pre-RC4.1 forward Nyquist dynamics"):
        restore_channel_checkpoint(
            target,
            legacy,
            runtime_identity_sha256=spec.runtime_identity_sha256(),
            pre_rc5_runtime_identity_sha256=(
                spec.pre_rc5_runtime_identity_sha256()
            ),
        )
    _assert_state_unchanged(target, before)


def test_cross_runtime_checkpoint_is_rejected_before_mutation():
    legacy_spec, legacy = _build()
    _, compiled = _build("compiled_channel_v2")
    legacy.advance(2)
    checkpoint = capture_channel_checkpoint(
        legacy,
        runtime_identity_sha256=legacy_spec.runtime_identity_sha256(),
    )
    before = _state_clone(compiled)

    with pytest.raises(ValueError, match="cross-runtime"):
        restore_channel_checkpoint(
            compiled,
            checkpoint,
            runtime_identity_sha256=_spec(
                "compiled_channel_v2"
            ).runtime_identity_sha256(),
        )
    _assert_state_unchanged(compiled, before)


@pytest.mark.parametrize("case", ("identity", "backend", "shape", "nan"))
def test_checkpoint_tamper_is_rejected_before_target_mutation(case):
    spec, source = _build("compiled_channel_v2")
    _, target = _build("compiled_channel_v2")
    source.advance(2)
    checkpoint = capture_channel_checkpoint(
        source,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    if case == "identity":
        runtime_identity = "0" * 64
    else:
        runtime_identity = spec.runtime_identity_sha256()
    if case == "backend":
        checkpoint = dataclasses.replace(
            checkpoint,
            backend_restart={"kind": "tampered", "state_keys": []},
        )
    elif case == "shape":
        values = dict(checkpoint.evolved_spatial)
        values["Qxx"] = values["Qxx"][..., :-1]
        checkpoint = dataclasses.replace(checkpoint, evolved_spatial=values)
    elif case == "nan":
        checkpoint.pressure_guess.reshape(-1)[0] = torch.nan
    before = _state_clone(target)

    with pytest.raises(ValueError):
        restore_channel_checkpoint(
            target,
            checkpoint,
            runtime_identity_sha256=runtime_identity,
        )
    _assert_state_unchanged(target, before)


def test_disk_checksum_tamper_is_rejected(tmp_path):
    spec, adapter = _build("compiled_channel_v2")
    adapter.advance(1)
    checkpoint = capture_channel_checkpoint(
        adapter,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    directory = write_channel_checkpoint(tmp_path / "checkpoint", checkpoint)
    path = directory / "evolved_spatial__Qxx.npy"
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_channel_checkpoint(directory)


def test_p74_does_not_connect_channel_entry_point_or_change_default():
    assert _spec().runtime_path is ChannelRuntimePath.LEGACY_CHANNEL
    assert "channel_active_nematics" not in (
        ROOT / "Channel.py"
    ).read_text(encoding="utf-8")
    assert hashlib.sha256((ROOT / "Channel.py").read_bytes()).hexdigest() == (
        "ea7087c4a22796bcefb77407505d2971cee948a9d2b643ced03d6dcb0a5d34b2"
    )


def test_p74_record_binds_runtime_and_checkpoint_sources():
    record = json.loads(
        (NOTES / "phase_7_p74_channel_runtime_facade.json").read_text(
            encoding="utf-8"
        )
    )

    def sha256(relative):
        return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()

    assert record["classification"] == "PASS_P7_4_CHANNEL_RUNTIME_FACADE"
    assert sha256("pssolver/runtime/channel_active_nematics.py") == record[
        "implementation"
    ]["runtime_facade_sha256"]
    historical_checkpoint = subprocess.run(
        [
            "git",
            "show",
            (
                "19b241d62d2ea04455201e3321e86e3702563bb1:"
                "pssolver/workflows/channel_checkpoint.py"
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(historical_checkpoint).hexdigest() == record[
        "implementation"
    ]["checkpoint_module_sha256"]
    successor = json.loads(
        (
            ROOT
            / "notes/PSSolver_v0_2_0rc4_rc453_checkpoint_transactionality.json"
        ).read_text(encoding="utf-8")
    )
    historical = successor[
        "source_sha256"
    ]["pssolver/workflows/channel_checkpoint.py"]
    r5_2 = json.loads(
        (
            ROOT
            / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
        ).read_text(encoding="utf-8")
    )["historical_successor_sources"][
        "pssolver/workflows/channel_checkpoint.py"
    ]
    assert r5_2["historical_sha256"] == historical
    assert sha256("pssolver/workflows/channel_checkpoint.py") == r5_2[
        "current_sha256"
    ]
    assert record["authorization"] == {
        "p7_4_complete": True,
        "p7_5_planning_eligible": True,
        "p7_5_implementation_authorized": False,
        "phase_7_h100_authorized": False,
        "channel_entry_point_connection_authorized": False,
        "production_default_changed": False,
        "new_boundary_law_authorized": False,
        "plane_evidence_modified": False,
    }
