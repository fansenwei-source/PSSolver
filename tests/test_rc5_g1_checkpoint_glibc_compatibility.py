"""glibc 2.17 compatibility for atomic checkpoint publication."""

from __future__ import annotations

import ctypes
import errno
import os

import pytest

import pssolver.io.checkpoint as checkpoint_io
from pssolver.io.checkpoint import (
    CheckpointExistsError,
    CheckpointIOError,
    checkpoint_directory_transaction,
)


class _FakeSyscall:
    restype = None

    def __init__(self, *, error_number: int | None = None) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.error_number = error_number

    def __call__(self, *arguments: object) -> int:
        self.calls.append(arguments)
        if self.error_number is not None:
            ctypes.set_errno(self.error_number)
            return -1
        source = os.fsdecode(arguments[2].value)
        target = os.fsdecode(arguments[4].value)
        if os.path.lexists(target):
            ctypes.set_errno(errno.EEXIST)
            return -1
        os.rename(source, target)
        return 0


class _OldGlibc:
    def __init__(self, syscall: _FakeSyscall) -> None:
        self.syscall = syscall

    @property
    def renameat2(self):
        raise AttributeError("glibc does not export renameat2")


def _emulate_glibc_2_17(monkeypatch, syscall: _FakeSyscall) -> None:
    monkeypatch.setattr(
        checkpoint_io.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: _OldGlibc(syscall),
    )
    monkeypatch.setattr(checkpoint_io.platform, "machine", lambda: "x86_64")


def test_old_glibc_uses_x86_64_renameat2_syscall_without_replacing(
    tmp_path,
    monkeypatch,
):
    syscall = _FakeSyscall()
    _emulate_glibc_2_17(monkeypatch, syscall)
    target = tmp_path / "checkpoint"

    with checkpoint_directory_transaction(
        target,
        operation="checkpoint_export",
    ) as (staging, _published):
        (staging / "payload.bin").write_bytes(b"complete")

    assert (target / "payload.bin").read_bytes() == b"complete"
    assert len(syscall.calls) == 1
    arguments = syscall.calls[0]
    assert arguments[0].value == 316
    assert arguments[1].value == -100
    assert arguments[3].value == -100
    assert arguments[5].value == 1


def test_old_glibc_syscall_still_rejects_a_racing_empty_target(tmp_path, monkeypatch):
    syscall = _FakeSyscall()
    _emulate_glibc_2_17(monkeypatch, syscall)
    target = tmp_path / "checkpoint"
    real_publish = checkpoint_io._rename_directory_no_replace

    def publish_after_competitor(source, occupied, *, operation):
        occupied.mkdir()
        real_publish(source, occupied, operation=operation)

    monkeypatch.setattr(
        checkpoint_io,
        "_rename_directory_no_replace",
        publish_after_competitor,
    )
    with pytest.raises(CheckpointExistsError):
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            (staging / "payload.bin").write_bytes(b"complete")

    assert target.is_dir()
    assert list(target.iterdir()) == []
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_old_glibc_kernel_without_renameat2_fails_closed(tmp_path, monkeypatch):
    syscall = _FakeSyscall(error_number=errno.ENOSYS)
    _emulate_glibc_2_17(monkeypatch, syscall)
    target = tmp_path / "checkpoint"

    with pytest.raises(CheckpointIOError, match="atomic no-replace") as captured:
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            (staging / "payload.bin").write_bytes(b"complete")

    assert captured.value.details == {"errno": errno.ENOSYS}
    assert not target.exists()
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_old_glibc_fallback_is_not_guessed_for_unknown_architecture(
    tmp_path,
    monkeypatch,
):
    syscall = _FakeSyscall()
    _emulate_glibc_2_17(monkeypatch, syscall)
    monkeypatch.setattr(checkpoint_io.platform, "machine", lambda: "unknown")
    target = tmp_path / "checkpoint"

    with pytest.raises(CheckpointIOError, match="atomic no-replace") as captured:
        with checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            (staging / "payload.bin").write_bytes(b"complete")

    assert captured.value.details == {"machine": "unknown"}
    assert syscall.calls == []
    assert not target.exists()
