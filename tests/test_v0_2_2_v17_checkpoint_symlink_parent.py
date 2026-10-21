"""V-17 checkpoint publication through a cross-device parent symlink."""

from __future__ import annotations

import errno
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pssolver.functional import (
    ChannelActivityCheckpointBridge,
    FunctionalRuntimeIdentity,
    FunctionalStateSpec,
    FunctionalTensorSpec,
    PeriodicActivityCheckpointBridge,
)
import pssolver.io.checkpoint as checkpoint_io


def _force_nfs_compatibility(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported(_source, _target, *, operation):
        raise checkpoint_io._RenameNoReplaceUnsupported(
            "atomic no-replace checkpoint publication is unsupported on NFS",
            operation=operation,
            failure_domain="filesystem",
            legacy_exception_type="OSError",
            details={
                "errno": errno.EINVAL,
                "filesystem": "nfs",
                "filesystem_type": "0x6969",
            },
        )

    monkeypatch.setattr(
        checkpoint_io,
        "_rename_directory_no_replace",
        unsupported,
    )


def _state_contract(kind: str):
    physical = FunctionalTensorSpec(
        name="q_physical",
        shape=(5, 1, 2, 2),
        dtype="float64",
        device="cpu",
        batch_axis=1,
        layout="component_batch_xyz",
        meaning="V-17 physical Q",
    )
    spectral = FunctionalTensorSpec(
        name="q_spectral",
        shape=(5, 1, 2, 2),
        dtype="complex128",
        device="cpu",
        batch_axis=1,
        layout="component_batch_kxyz",
        meaning="V-17 spectral Q",
    )
    state_spec = FunctionalStateSpec(components=(physical, spectral))
    runtime = {
        "kind": kind,
    }
    if kind == "periodic_activity_batch_one":
        runtime["hermitian_state_projection"] = (
            "self_conjugate_planes_each_step"
        )
    else:
        runtime["pressure_solver"] = {
            "gradient": "custom_implicit_pressure_adjoint_v2",
        }
    identity = FunctionalRuntimeIdentity(
        scientific={"model": "v17-test"},
        discretization={"grid": [1, 2, 2]},
        execution={
            "runtime": "v17-test",
            "functional_runtime": runtime,
        },
        state_layout=state_spec.to_metadata(),
    )
    state = (
        torch.arange(20, dtype=torch.float64).reshape(5, 1, 2, 2),
        torch.arange(20, dtype=torch.float64)
        .reshape(5, 1, 2, 2)
        .to(torch.complex128),
    )
    return state_spec, identity, state


def _bridge(kind: str):
    state_spec, identity, state = _state_contract(kind)
    if kind == "periodic_activity_batch_one":
        bridge = PeriodicActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=identity,
            production_runtime_identity_sha256="a" * 64,
            backend_restart={},
        )
    else:
        bridge = ChannelActivityCheckpointBridge(
            state_spec=state_spec,
            functional_identity=identity,
            production_runtime_identity_sha256="b" * 64,
            production_backend_restart={},
        )
    return bridge, state


def _model_cross_device_parent_symlink(
    linked_parent: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make the link inode device differ without requiring two real mounts."""

    original_lstat = Path.lstat
    target_device = linked_parent.stat().st_dev
    link_device = target_device + 1

    def lstat(path: Path):
        if path == linked_parent:
            return SimpleNamespace(st_dev=link_device)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", lstat)
    assert linked_parent.lstat().st_dev != linked_parent.stat().st_dev


@pytest.mark.parametrize(
    "kind",
    ("periodic_activity_batch_one", "channel_activity_batch_one"),
)
def test_functional_export_follows_cross_device_parent_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Periodic and Channel exports compare the resolved parent filesystem."""

    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    _model_cross_device_parent_symlink(linked_parent, monkeypatch)
    _force_nfs_compatibility(monkeypatch)

    bridge, state = _bridge(kind)
    requested = linked_parent / f"{kind}_checkpoint"
    written = bridge.export_checkpoint(
        requested,
        state,
        completed_steps=0,
    )

    assert written == requested.absolute()
    assert written.resolve().parent == real_parent.resolve()
    restored = bridge.import_checkpoint(written)
    assert restored.completed_steps == 0
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(restored.state, state, strict=True)
    )
    assert not list(real_parent.glob(f".{requested.name}.tmp-*"))


def test_fallback_still_rejects_distinct_resolved_filesystems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Following the parent link must not weaken the true device guard."""

    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"payload")
    target_parent = tmp_path / "target_parent"
    target_parent.mkdir()
    target = target_parent / "checkpoint"
    source_device = source.lstat().st_dev
    original_stat = Path.stat

    def stat_with_distinct_target_device(path: Path, *, follow_symlinks=True):
        if path == target_parent:
            return SimpleNamespace(st_dev=source_device + 1)
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", stat_with_distinct_target_device)

    with pytest.raises(
        checkpoint_io.CheckpointIOError,
        match="staging and destination are on different filesystems",
    ) as captured:
        checkpoint_io._publish_directory_over_owned_placeholder(
            source,
            target,
            operation="checkpoint_export",
        )

    assert captured.value.details == {
        "source_device": source_device,
        "destination_device": source_device + 1,
    }
    assert source.is_dir()
    assert not target.exists()
