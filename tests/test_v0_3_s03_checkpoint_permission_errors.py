"""S-03 public checkpoint permission-error classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from pssolver.functional import (
    FunctionalCheckpointIOError,
    PeriodicActivityCheckpointBridge,
)
from pssolver.io.checkpoint import (
    CheckpointIOError,
    checkpoint_directory_transaction,
)
import pssolver.functional.periodic_checkpoint as periodic_checkpoint


def test_checkpoint_target_probe_maps_permission_error_to_public_io_error(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "checkpoint"
    original = Path.lstat

    def guarded_lstat(path):
        if path == target:
            raise PermissionError("parent cannot be searched")
        return original(path)

    monkeypatch.setattr(Path, "lstat", guarded_lstat)

    with pytest.raises(CheckpointIOError) as captured:
        with checkpoint_directory_transaction(
            target,
            operation="s03_permission_test",
        ):
            raise AssertionError("transaction body must not start")

    assert captured.value.code == "checkpoint_io_error"
    assert captured.value.failure_domain == "filesystem"
    assert captured.value.legacy_exception_type == "PermissionError"
    assert not tuple(tmp_path.glob(".checkpoint.tmp-*"))


def test_periodic_functional_reader_maps_placeholder_permission_error(
    tmp_path,
    monkeypatch,
):
    def unreadable(*_args, **_kwargs):
        raise PermissionError("mode-000 placeholder")

    monkeypatch.setattr(periodic_checkpoint, "read_checkpoint_json", unreadable)
    bridge = object.__new__(PeriodicActivityCheckpointBridge)

    with pytest.raises(FunctionalCheckpointIOError) as captured:
        bridge.import_checkpoint(tmp_path / "checkpoint")

    assert captured.value.code == "functional_checkpoint_io_error"
    assert captured.value.failure_domain == "filesystem"
    assert captured.value.legacy_exception_type == "PermissionError"
