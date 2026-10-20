"""Executable NFS publication contract for the v0.2.1 N-1 fix."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import errno
import hashlib
import os
from pathlib import Path
import stat
import threading

import pytest

import pssolver.io.checkpoint as checkpoint_io
from pssolver.io.checkpoint import (
    CheckpointExistsError,
    CheckpointIOError,
    checkpoint_directory_transaction,
)


def _nfs_unsupported(operation: str) -> checkpoint_io._RenameNoReplaceUnsupported:
    return checkpoint_io._RenameNoReplaceUnsupported(
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


def _force_nfs_compatibility(monkeypatch) -> None:
    def unsupported(_source, _target, *, operation):
        raise _nfs_unsupported(operation)

    monkeypatch.setattr(
        checkpoint_io,
        "_rename_directory_no_replace",
        unsupported,
    )


def _write_tree(staging: Path, payload: bytes = b"payload") -> None:
    (staging / "nested").mkdir()
    (staging / "nested" / "state.bin").write_bytes(payload)
    (staging / "checkpoint.json").write_bytes(b'{"complete":true}\n')


def _tree_bytes(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class _FailingRenameAt2:
    argtypes = None
    restype = None

    def __init__(self, error_number: int) -> None:
        self.error_number = error_number
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *arguments: object) -> int:
        self.calls.append(arguments)
        ctypes.set_errno(self.error_number)
        return -1


class _LibcWithRenameAt2:
    def __init__(self, renameat2: _FailingRenameAt2) -> None:
        self.renameat2 = renameat2


@pytest.mark.parametrize(
    "error_number",
    [errno.EINVAL, errno.EOPNOTSUPP],
)
def test_raw_renameat2_unsupported_falls_back_only_with_nfs_statfs(
    tmp_path,
    monkeypatch,
    error_number,
):
    renameat2 = _FailingRenameAt2(error_number)
    monkeypatch.setattr(
        checkpoint_io.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: _LibcWithRenameAt2(renameat2),
    )
    monkeypatch.setattr(
        checkpoint_io,
        "_linux_filesystem_type",
        lambda _path, *, operation: checkpoint_io._LINUX_NFS_SUPER_MAGIC,
    )
    target = tmp_path / "checkpoint"

    with checkpoint_directory_transaction(
        target,
        operation="checkpoint_export",
    ) as (staging, _published):
        _write_tree(staging)

    assert _tree_bytes(target) == {
        "checkpoint.json": b'{"complete":true}\n',
        "nested/state.bin": b"payload",
    }
    assert len(renameat2.calls) == 1
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_renameat2_einval_does_not_fallback_without_nfs_statfs(
    tmp_path,
    monkeypatch,
):
    renameat2 = _FailingRenameAt2(errno.EINVAL)
    monkeypatch.setattr(
        checkpoint_io.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: _LibcWithRenameAt2(renameat2),
    )
    monkeypatch.setattr(
        checkpoint_io,
        "_linux_filesystem_type",
        lambda _path, *, operation: 0xEF53,
    )
    target = tmp_path / "checkpoint"

    with pytest.raises(CheckpointIOError) as captured:
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            _write_tree(staging)

    assert captured.value.details == {
        "errno": errno.EINVAL,
        "filesystem": "unrecognized",
        "filesystem_type": "0xef53",
    }
    assert not target.exists()
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EIO, errno.ENOSPC])
def test_non_capability_errors_never_enter_compatibility_path(
    tmp_path,
    monkeypatch,
    error_number,
):
    entered = False

    def fail_fast(_source, target, *, operation):
        exception = OSError(error_number, os.strerror(error_number), target)
        raise checkpoint_io._io_error(
            exception,
            operation=operation,
            message=f"checkpoint cannot be published: {target}",
        ) from exception

    def observe_compatibility(*_args, **_kwargs):
        nonlocal entered
        entered = True

    monkeypatch.setattr(checkpoint_io, "_rename_directory_no_replace", fail_fast)
    monkeypatch.setattr(
        checkpoint_io,
        "_publish_directory_over_owned_placeholder",
        observe_compatibility,
    )
    target = tmp_path / "checkpoint"
    with pytest.raises(CheckpointIOError):
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            _write_tree(staging)
    assert entered is False
    assert not target.exists()


def test_compatibility_path_exposes_only_an_inaccessible_empty_claim(
    tmp_path,
    monkeypatch,
):
    _force_nfs_compatibility(monkeypatch)
    real_rename = checkpoint_io.os.rename
    observations: dict[str, object] = {}

    def inspect_claim(source, target):
        target_stat = Path(target).lstat()
        observations["mode"] = stat.S_IMODE(target_stat.st_mode)
        observations["source_digest"] = checkpoint_io._checkpoint_tree_digest(
            Path(source),
            operation="checkpoint_export",
        )
        real_rename(source, target)

    monkeypatch.setattr(checkpoint_io.os, "rename", inspect_claim)
    target = tmp_path / "checkpoint"
    with checkpoint_directory_transaction(
        target,
        operation="checkpoint_export",
    ) as (staging, _published):
        _write_tree(staging)

    assert observations["mode"] == 0
    assert checkpoint_io._checkpoint_tree_digest(
        target,
        operation="checkpoint_export",
    ) == observations["source_digest"]


def test_compatibility_path_confirms_a_lost_rename_acknowledgement(
    tmp_path,
    monkeypatch,
):
    _force_nfs_compatibility(monkeypatch)
    real_rename = checkpoint_io.os.rename

    def rename_then_lose_ack(source, target):
        real_rename(source, target)
        raise OSError(errno.EIO, "injected lost acknowledgement")

    monkeypatch.setattr(checkpoint_io.os, "rename", rename_then_lose_ack)
    target = tmp_path / "checkpoint"
    with checkpoint_directory_transaction(
        target,
        operation="checkpoint_export",
    ) as (staging, _published):
        _write_tree(staging)

    assert (target / "nested" / "state.bin").read_bytes() == b"payload"
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_unconfirmed_rename_failure_is_fail_closed_and_preserves_evidence(
    tmp_path,
    monkeypatch,
):
    _force_nfs_compatibility(monkeypatch)
    real_rename = checkpoint_io.os.rename

    def rename_corrupt_then_lose_ack(source, target):
        real_rename(source, target)
        (Path(target) / "nested" / "state.bin").write_bytes(b"different")
        raise OSError(errno.EIO, "injected ambiguous acknowledgement")

    monkeypatch.setattr(
        checkpoint_io.os,
        "rename",
        rename_corrupt_then_lose_ack,
    )
    target = tmp_path / "checkpoint"
    with pytest.raises(CheckpointIOError) as captured:
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            _write_tree(staging)

    assert captured.value.details["publication_outcome"] == "ambiguous"
    assert (target / "nested" / "state.bin").read_bytes() == b"different"


def test_prepublication_rename_failure_removes_only_the_owned_placeholder(
    tmp_path,
    monkeypatch,
):
    _force_nfs_compatibility(monkeypatch)

    def fail_before_rename(_source, _target):
        raise OSError(errno.EIO, "injected prepublication failure")

    monkeypatch.setattr(checkpoint_io.os, "rename", fail_before_rename)
    target = tmp_path / "checkpoint"
    with pytest.raises(CheckpointIOError):
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            _write_tree(staging)

    assert not target.exists()
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_fast_and_compatibility_paths_publish_identical_checkpoint_bytes(
    tmp_path,
    monkeypatch,
):
    fast = tmp_path / "fast"
    with checkpoint_directory_transaction(
        fast,
        operation="checkpoint_export",
    ) as (staging, _published):
        _write_tree(staging, payload=b"identical")

    _force_nfs_compatibility(monkeypatch)
    compatible = tmp_path / "compatible"
    with checkpoint_directory_transaction(
        compatible,
        operation="checkpoint_export",
    ) as (staging, _published):
        _write_tree(staging, payload=b"identical")

    assert _tree_bytes(fast) == _tree_bytes(compatible)
    assert checkpoint_io._checkpoint_tree_digest(
        fast,
        operation="checkpoint_export",
    ) == checkpoint_io._checkpoint_tree_digest(
        compatible,
        operation="checkpoint_export",
    )


def test_concurrent_compatibility_writers_have_exactly_one_winner(
    tmp_path,
    monkeypatch,
):
    _force_nfs_compatibility(monkeypatch)
    writers = 8
    barrier = threading.Barrier(writers)
    target = tmp_path / "checkpoint"

    def publish(index: int) -> tuple[str, int]:
        try:
            with checkpoint_directory_transaction(
                target,
                operation="checkpoint_export",
            ) as (staging, _published):
                (staging / "writer.txt").write_text(str(index), encoding="utf-8")
                barrier.wait(timeout=10)
            return "success", index
        except CheckpointExistsError:
            return "collision", index

    with ThreadPoolExecutor(max_workers=writers) as executor:
        results = list(executor.map(publish, range(writers)))

    successes = [index for outcome, index in results if outcome == "success"]
    collisions = [index for outcome, index in results if outcome == "collision"]
    assert len(successes) == 1
    assert len(collisions) == writers - 1
    assert (target / "writer.txt").read_text(encoding="utf-8") == str(successes[0])
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_compatibility_path_adds_no_backend_marker_to_checkpoint_tree(
    tmp_path,
    monkeypatch,
):
    _force_nfs_compatibility(monkeypatch)
    target = tmp_path / "checkpoint"
    with checkpoint_directory_transaction(
        target,
        operation="checkpoint_export",
    ) as (staging, _published):
        _write_tree(staging)

    assert set(_tree_bytes(target)) == {
        "checkpoint.json",
        "nested/state.bin",
    }
    manifest = hashlib.sha256((target / "checkpoint.json").read_bytes()).hexdigest()
    assert len(manifest) == 64
