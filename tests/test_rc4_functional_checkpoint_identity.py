"""RC4.2.4 functional derivative identity and migration contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver.functional import (
    ChannelActivityCheckpointBridge,
    FunctionalCheckpointCompatibilityError,
    FunctionalRuntimeIdentity,
    FunctionalStateSpec,
    FunctionalTensorSpec,
    PeriodicActivityCheckpointBridge,
    build_channel_functional_checkpoint_identity,
    build_periodic_functional_checkpoint_identity,
)
from pssolver.functional.versioning import (
    runtime_identity_sha256_for_api_version,
)
from pssolver.io.checkpoint import (
    FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE,
    seal_checkpoint_metadata,
    write_functional_checkpoint_provenance,
)


def _state_spec(*, periodic_size: int, device: str = "cpu"):
    physical_shape = (5, 1, periodic_size, 2, 2)
    return FunctionalStateSpec(
        (
            FunctionalTensorSpec(
                name="q_physical",
                shape=physical_shape,
                dtype="float64",
                device=device,
                batch_axis=1,
                layout="component_batch_xyz",
                meaning="test physical Q",
            ),
            FunctionalTensorSpec(
                name="q_spectral",
                shape=physical_shape,
                dtype="complex128",
                device=device,
                batch_axis=1,
                layout="component_batch_native_spectral_xyz",
                meaning="test spectral Q",
            ),
        )
    )


def _periodic_identity(state_spec, *, device: str = "cpu"):
    return FunctionalRuntimeIdentity(
        scientific={"model": "complete_stress_beris_edwards"},
        discretization={"grid": list(state_spec.components[0].shape[-3:])},
        execution={
            "functional_runtime": {
                "kind": "periodic_activity_batch_one",
                "device": device,
                "snapshot_sha256": "a" * 64,
                "fallback_allowed": False,
                "fallback_used": False,
                "spectral_refresh": "disabled",
                "hermitian_state_projection": (
                    "self_conjugate_planes_each_step"
                ),
            }
        },
        state_layout=state_spec.to_metadata(),
    )


def _channel_identity(state_spec, *, device: str = "cpu"):
    return FunctionalRuntimeIdentity(
        scientific={"model": "complete_stress_beris_edwards"},
        discretization={"grid": list(state_spec.components[0].shape[-3:])},
        execution={
            "functional_runtime": {
                "kind": "channel_activity_batch_one",
                "device": device,
                "snapshot_sha256": "b" * 64,
                "fallback_allowed": False,
                "fallback_used": False,
                "pressure_solver": {
                    "kind": "channel_no_slip_modal_stokes_pcg",
                    "gradient": "custom_implicit_pressure_adjoint_v1",
                    "initial_guess": "zero_every_call",
                },
            }
        },
        state_layout=state_spec.to_metadata(),
    )


def _identity_record(metadata):
    return {
        "metadata": metadata,
        "sha256": hashlib.sha256(
            json.dumps(
                metadata,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _layered_identity(state_spec, *, device: str, channel: bool):
    execution = _identity_record(
        {
            "backend": "torch_spectral",
            "device": device,
            "options": {"tf32": "off"},
            "runtime_path": (
                "channel_complete_stress" if channel else "periodic_spectral"
            ),
            "stage": "test_provenance",
        }
    )
    execution["functional_runtime"] = {
        "kind": (
            "channel_activity_batch_one"
            if channel
            else "periodic_activity_batch_one"
        ),
        "device": device,
        "snapshot_sha256": "b" * 64 if channel else "a" * 64,
        "fallback_allowed": False,
        "fallback_used": False,
        "spectral_refresh": "disabled",
        **(
            {
                "pressure_solver": {
                    "kind": "channel_no_slip_modal_stokes_pcg",
                    "gradient": "custom_implicit_pressure_adjoint_v1",
                    "initial_guess": "zero_every_call",
                }
            }
            if channel
            else {
                "hermitian_state_projection": (
                    "self_conjugate_planes_each_step"
                )
            }
        ),
    }
    return FunctionalRuntimeIdentity(
        scientific=_identity_record(
            {"model": "complete_stress_beris_edwards"}
        ),
        discretization=_identity_record(
            {"grid": list(state_spec.components[0].shape[-3:])}
        ),
        execution=execution,
        state_layout=state_spec.to_metadata(),
    )


def _state(state_spec):
    physical = torch.arange(
        torch.tensor(state_spec.components[0].shape).prod().item(),
        dtype=torch.float64,
    ).reshape(state_spec.components[0].shape)
    return physical, torch.zeros_like(physical, dtype=torch.complex128)


class _Projector:
    def __init__(self):
        self.calls = 0

    def forward_transform(self, physical, boundary_conditions):
        self.calls += 1
        assert boundary_conditions == ("periodic", "periodic", "periodic")
        return physical.to(torch.complex128)


def _tree_sha256(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _legacy_bridge(identity, current, *, api_version: str, format_version: int):
    result = dict(current)
    provenance = result.pop("run_provenance")
    result.pop("checkpoint_compatibility_identity")
    result.pop("checkpoint_compatibility_sha256")
    result.update(
        {
            "api_version": api_version,
            "format_version": format_version,
            "functional_runtime_identity_sha256": (
                runtime_identity_sha256_for_api_version(
                    identity.to_metadata(),
                    api_version,
                )
            ),
            "production_runtime_identity_sha256": provenance[
                "production_runtime_identity_sha256"
            ],
            "state_layout": identity.to_metadata()["state_layout"],
        }
    )
    result["state_layout_sha256"] = hashlib.sha256(
        json.dumps(
            result["state_layout"],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return result


def _rewrite_periodic_rc1(directory: Path, identity) -> None:
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["format_version"] = 1
    metadata["functional_bridge"] = _legacy_bridge(
        identity,
        metadata["functional_bridge"],
        api_version="0.1-provisional",
        format_version=1,
    )
    metadata.pop("metadata_sha256", None)
    (directory / FUNCTIONAL_CHECKPOINT_PROVENANCE_FILE).unlink()
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_periodic_v2(directory: Path, identity) -> None:
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["functional_bridge"] = _legacy_bridge(
        identity,
        metadata["functional_bridge"],
        api_version="1.0",
        format_version=2,
    )
    metadata.pop("metadata_sha256", None)
    sealed = seal_checkpoint_metadata(metadata)
    path.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_functional_checkpoint_provenance(
        directory,
        sealed,
        sealed["functional_bridge"],
    )


def _rewrite_channel_legacy(
    directory: Path,
    identity,
    *,
    api_version: str,
    format_version: int,
) -> None:
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["format_version"] = format_version
    metadata["functional_bridge"] = _legacy_bridge(
        identity,
        metadata["functional_bridge"],
        api_version=api_version,
        format_version=format_version,
    )
    metadata.pop("metadata_sha256", None)
    sealed = seal_checkpoint_metadata(metadata)
    path.write_text(
        json.dumps(sealed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_functional_checkpoint_provenance(
        directory,
        sealed,
        sealed["functional_bridge"],
    )


def test_functional_layered_identity_excludes_device_and_run_provenance():
    cpu = _state_spec(periodic_size=4, device="cpu")
    cuda = _state_spec(periodic_size=4, device="cuda:0")

    periodic_cpu = build_periodic_functional_checkpoint_identity(
        functional_identity=_periodic_identity(cpu, device="cpu"),
        state_spec=cpu,
    )
    periodic_cuda = build_periodic_functional_checkpoint_identity(
        functional_identity=_periodic_identity(cuda, device="cuda:0"),
        state_spec=cuda,
    )
    channel_cpu = build_channel_functional_checkpoint_identity(
        functional_identity=_channel_identity(cpu, device="cpu"),
        state_spec=cpu,
    )
    channel_cuda = build_channel_functional_checkpoint_identity(
        functional_identity=_channel_identity(cuda, device="cuda:0"),
        state_spec=cuda,
    )

    assert periodic_cpu == periodic_cuda
    assert channel_cpu == channel_cuda
    assert periodic_cpu.derivative_dynamics is not None
    assert channel_cpu.derivative_dynamics is not None


@pytest.mark.parametrize("channel", (False, True))
def test_layered_identity_rehashes_filtered_execution_metadata(channel):
    state_spec = _state_spec(periodic_size=4)
    cpu_runtime = _layered_identity(
        state_spec,
        device="cpu",
        channel=channel,
    )
    cuda_runtime = _layered_identity(
        state_spec,
        device="cuda:0",
        channel=channel,
    )
    builder = (
        build_channel_functional_checkpoint_identity
        if channel
        else build_periodic_functional_checkpoint_identity
    )
    cpu = builder(functional_identity=cpu_runtime, state_spec=state_spec)
    cuda = builder(functional_identity=cuda_runtime, state_spec=state_spec)

    assert cpu == cuda
    execution = cpu.forward_dynamics.to_metadata()["payload"]["execution"]
    assert "device" not in execution["metadata"]
    assert "stage" not in execution["metadata"]
    assert execution["sha256"] == hashlib.sha256(
        json.dumps(
            execution["metadata"],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("channel", (False, True))
def test_current_functional_checkpoint_crosses_device_identity_tokens(
    tmp_path,
    channel,
):
    state_spec = _state_spec(periodic_size=4)
    source_identity = _layered_identity(
        state_spec,
        device="cpu",
        channel=channel,
    )
    target_identity = _layered_identity(
        state_spec,
        device="cuda:0",
        channel=channel,
    )
    if channel:
        source_bridge = ChannelActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=source_identity,
            production_runtime_identity_sha256="c" * 64,
            production_backend_restart={},
        )
        target_bridge = ChannelActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=target_identity,
            production_runtime_identity_sha256="d" * 64,
            production_backend_restart={},
        )
    else:
        source_bridge = PeriodicActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=source_identity,
            production_runtime_identity_sha256="c" * 64,
            backend_restart={},
            projector=_Projector(),
        )
        target_bridge = PeriodicActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=target_identity,
            production_runtime_identity_sha256="d" * 64,
            backend_restart={},
            projector=_Projector(),
        )

    source = source_bridge.export_checkpoint(
        tmp_path / "source",
        _state(state_spec),
        completed_steps=4,
    )
    target = target_bridge.export_checkpoint(
        tmp_path / "target",
        _state(state_spec),
        completed_steps=0,
    )
    source_metadata = json.loads((source / "checkpoint.json").read_text())
    target_metadata = json.loads((target / "checkpoint.json").read_text())
    assert source_metadata["functional_bridge"][
        "checkpoint_compatibility_sha256"
    ] == target_metadata["functional_bridge"][
        "checkpoint_compatibility_sha256"
    ]

    restored = target_bridge.import_checkpoint(source)
    assert restored.completed_steps == 4
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            restored.state,
            _state(state_spec),
            strict=True,
        )
    )


@pytest.mark.parametrize("channel", (False, True))
def test_pre_recovery_stale_execution_digest_remains_readable(tmp_path, channel):
    state_spec = _state_spec(periodic_size=4)
    identity = _layered_identity(state_spec, device="cpu", channel=channel)
    if channel:
        bridge = ChannelActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=identity,
            production_runtime_identity_sha256="e" * 64,
            production_backend_restart={},
        )
    else:
        bridge = PeriodicActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=identity,
            production_runtime_identity_sha256="e" * 64,
            backend_restart={},
            projector=_Projector(),
        )
    directory = bridge.export_checkpoint(
        tmp_path / "checkpoint",
        _state(state_spec),
        completed_steps=2,
    )
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text())
    functional_bridge = metadata["functional_bridge"]
    compatibility = functional_bridge["checkpoint_compatibility_identity"]
    compatibility["forward_dynamics"]["payload"]["execution"]["sha256"] = (
        "f" * 64
    )
    functional_bridge["checkpoint_compatibility_sha256"] = hashlib.sha256(
        json.dumps(
            compatibility,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    metadata.pop("metadata_sha256", None)
    sealed = seal_checkpoint_metadata(metadata)
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    write_functional_checkpoint_provenance(
        directory,
        sealed,
        sealed["functional_bridge"],
    )

    restored = bridge.import_checkpoint(directory)
    assert restored.completed_steps == 2


def test_m14_periodic_provisional_checkpoint_requires_reprojection_migration(
    tmp_path,
    monkeypatch,
):
    state_spec = _state_spec(periodic_size=4)
    identity = _periodic_identity(state_spec)
    projector = _Projector()
    bridge = PeriodicActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity,
        production_runtime_identity_sha256="c" * 64,
        backend_restart={},
        projector=projector,
    )
    source = bridge.export_checkpoint(
        tmp_path / "periodic_rc1",
        _state(state_spec),
        completed_steps=7,
    )
    _rewrite_periodic_rc1(source, identity)
    before = _tree_sha256(source)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tensor payload must not open during direct rejection")

    monkeypatch.setattr("pssolver.workflows.periodic_checkpoint.np.load", forbidden)
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="explicit state migration",
    ):
        bridge.import_checkpoint(source)
    monkeypatch.undo()

    target = bridge.migrate_legacy_checkpoint(
        source,
        tmp_path / "periodic_v3",
        source_release_generation="rc1",
    )
    restored = bridge.import_checkpoint(target)
    metadata = json.loads((target / "checkpoint.json").read_text())

    assert projector.calls == 1
    assert restored.completed_steps == 7
    assert restored.compatibility.exact_current_protocol is False
    assert restored.compatibility.reader == (
        "periodic_reprojection_state_migration_v3_reader"
    )
    assert torch.equal(restored.state[1], restored.state[0].to(torch.complex128))
    assert metadata["migration_provenance"]["migration_kind"] == (
        "state_only_migration"
    )
    assert metadata["migration_provenance"]["trajectory_equivalent"] is False
    assert _tree_sha256(source) == before


def test_periodic_post_repair_v2_retains_qualified_reader_and_exact_upgrade(
    tmp_path,
):
    state_spec = _state_spec(periodic_size=4)
    identity = _periodic_identity(state_spec)
    bridge = PeriodicActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity,
        production_runtime_identity_sha256="3" * 64,
        backend_restart={},
        projector=_Projector(),
    )
    source = bridge.export_checkpoint(
        tmp_path / "periodic_rc2",
        _state(state_spec),
        completed_steps=9,
    )
    _rewrite_periodic_v2(source, identity)
    before = _tree_sha256(source)

    direct = bridge.import_checkpoint(source)
    assert direct.completed_steps == 9
    assert direct.source_format == "periodic_functional_bridge_v2"
    assert direct.compatibility.reader == (
        "qualified_periodic_functional_bridge_v2_reader"
    )

    target = bridge.migrate_legacy_checkpoint(
        source,
        tmp_path / "periodic_v3",
        source_release_generation="rc2",
    )
    upgraded = bridge.import_checkpoint(target)
    metadata = json.loads((target / "checkpoint.json").read_text())
    assert upgraded.compatibility.exact_current_protocol is True
    assert metadata["migration_provenance"]["migration_kind"] == (
        "exact_legacy_upgrade"
    )
    assert metadata["migration_provenance"]["trajectory_equivalent"] is True
    assert _tree_sha256(source) == before


def test_m13_channel_pre_rc3_even_grid_is_state_only_migration(
    tmp_path,
    monkeypatch,
):
    state_spec = _state_spec(periodic_size=4)
    identity = _channel_identity(state_spec)
    bridge = ChannelActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity,
        production_runtime_identity_sha256="d" * 64,
        production_backend_restart={},
    )
    original_state = _state(state_spec)
    source = bridge.export_checkpoint(
        tmp_path / "channel_rc2",
        original_state,
        completed_steps=5,
    )
    _rewrite_channel_legacy(
        source,
        identity,
        api_version="1.0",
        format_version=2,
    )
    before = _tree_sha256(source)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tensor payload must not open during direct rejection")

    monkeypatch.setattr("pssolver.functional.channel_checkpoint.np.load", forbidden)
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="requires explicit migration",
    ):
        bridge.import_checkpoint(source)
    monkeypatch.undo()

    target = bridge.migrate_legacy_checkpoint(
        source,
        tmp_path / "channel_v3",
        source_release_generation="rc2",
    )
    restored = bridge.import_checkpoint(target)
    metadata = json.loads((target / "checkpoint.json").read_text())

    assert restored.completed_steps == 5
    assert restored.compatibility.exact_current_protocol is False
    assert restored.compatibility.reader == "channel_state_only_migration_v3_reader"
    assert all(
        torch.equal(left, right)
        for left, right in zip(restored.state, original_state, strict=True)
    )
    assert metadata["migration_provenance"]["derivative_dynamics_change"] == (
        "pre_rc3_pressure_transpose_nyquist"
    )
    assert metadata["migration_provenance"]["trajectory_equivalent"] is False
    assert _tree_sha256(source) == before


def test_pre_rc3_channel_odd_periodic_grid_is_exact_legacy_upgrade(tmp_path):
    state_spec = _state_spec(periodic_size=3)
    identity = _channel_identity(state_spec)
    bridge = ChannelActivityCheckpointBridge(
        state_spec=state_spec,
        functional_identity=identity,
        production_runtime_identity_sha256="e" * 64,
        production_backend_restart={},
    )
    source = bridge.export_checkpoint(
        tmp_path / "channel_rc2_odd",
        _state(state_spec),
        completed_steps=3,
    )
    _rewrite_channel_legacy(
        source,
        identity,
        api_version="1.0",
        format_version=2,
    )
    target = bridge.migrate_legacy_checkpoint(
        source,
        tmp_path / "channel_v3_odd",
        source_release_generation="rc2",
    )
    restored = bridge.import_checkpoint(target)
    metadata = json.loads((target / "checkpoint.json").read_text())

    assert restored.compatibility.exact_current_protocol is True
    assert metadata["migration_provenance"]["migration_kind"] == (
        "exact_legacy_upgrade"
    )
    assert metadata["migration_provenance"]["trajectory_equivalent"] is True


@pytest.mark.parametrize("family", ("periodic", "channel"))
def test_unknown_v3_identity_rejects_before_tensor_load(tmp_path, monkeypatch, family):
    state_spec = _state_spec(periodic_size=4)
    if family == "periodic":
        identity = _periodic_identity(state_spec)
        bridge = PeriodicActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=identity,
            production_runtime_identity_sha256="f" * 64,
            backend_restart={},
            projector=_Projector(),
        )
        module = "pssolver.workflows.periodic_checkpoint.np.load"
    else:
        identity = _channel_identity(state_spec)
        bridge = ChannelActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=identity,
            production_runtime_identity_sha256="f" * 64,
            production_backend_restart={},
        )
        module = "pssolver.functional.channel_checkpoint.np.load"
    directory = bridge.export_checkpoint(
        tmp_path / family,
        _state(state_spec),
        completed_steps=0,
    )
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text())
    metadata["functional_bridge"]["checkpoint_compatibility_sha256"] = "0" * 64
    metadata.pop("metadata_sha256", None)
    sealed = seal_checkpoint_metadata(metadata)
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    write_functional_checkpoint_provenance(
        directory,
        sealed,
        sealed["functional_bridge"],
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tensor payload must not open before identity rejection")

    monkeypatch.setattr(module, forbidden)
    with pytest.raises(FunctionalCheckpointCompatibilityError):
        bridge.import_checkpoint(directory)


@pytest.mark.parametrize("family", ("periodic", "channel"))
def test_incomplete_migration_provenance_rejects_before_tensor_load(
    tmp_path,
    monkeypatch,
    family,
):
    state_spec = _state_spec(periodic_size=4)
    if family == "periodic":
        bridge = PeriodicActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=_periodic_identity(state_spec),
            production_runtime_identity_sha256="1" * 64,
            backend_restart={},
            projector=_Projector(),
        )
        module = "pssolver.workflows.periodic_checkpoint.np.load"
    else:
        bridge = ChannelActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=_channel_identity(state_spec),
            production_runtime_identity_sha256="2" * 64,
            production_backend_restart={},
        )
        module = "pssolver.functional.channel_checkpoint.np.load"
    directory = bridge.export_checkpoint(
        tmp_path / family,
        _state(state_spec),
        completed_steps=0,
    )
    path = directory / "checkpoint.json"
    metadata = json.loads(path.read_text())
    metadata["migration_provenance"] = {
        "migration_kind": "state_only_migration"
    }
    metadata.pop("metadata_sha256", None)
    sealed = seal_checkpoint_metadata(metadata)
    path.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    write_functional_checkpoint_provenance(
        directory,
        sealed,
        sealed["functional_bridge"],
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("tensor payload must not open before provenance rejection")

    monkeypatch.setattr(module, forbidden)
    with pytest.raises(
        FunctionalCheckpointCompatibilityError,
        match="migration provenance schema",
    ):
        bridge.import_checkpoint(directory)
