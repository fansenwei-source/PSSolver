"""S-04 checkpoint publication robustness regressions."""

from __future__ import annotations

import errno
import os
from pathlib import Path

import pytest

import pssolver.io.checkpoint as checkpoint_io
from pssolver.functional import FunctionalCheckpointDurabilityError
from pssolver.functional.errors import translate_functional_exception
from pssolver.io import CheckpointDurabilityError, CheckpointIOError


def _force_nfs_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported(_source, _target, *, operation):
        raise checkpoint_io._RenameNoReplaceUnsupported(
            "injected NFS capability gap",
            operation=operation,
            failure_domain="filesystem",
            details={"errno": errno.EINVAL, "filesystem": "nfs"},
        )

    monkeypatch.setattr(
        checkpoint_io,
        "_rename_directory_no_replace",
        unsupported,
    )


def _write_payload(staging: Path, value: bytes = b"payload") -> None:
    (staging / "payload.bin").write_bytes(value)


def test_v22_parent_retarget_fails_closed_and_cleans_the_pinned_staging(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    replacement = tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(original, target_is_directory=True)
    target = link / "checkpoint"
    replacement_target = replacement / "checkpoint"
    replacement_target.mkdir()
    (replacement_target / "sentinel").write_bytes(b"unchanged")

    with pytest.raises(CheckpointIOError, match="parent changed"):
        with checkpoint_io.checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, published):
            assert staging.parent == original.resolve()
            assert published == target.absolute()
            _write_payload(staging)
            link.unlink()
            link.symlink_to(replacement, target_is_directory=True)

    assert not (original / "checkpoint").exists()
    assert not list(original.glob(".checkpoint.tmp-*"))
    assert not list(replacement.glob(".checkpoint.tmp-*"))
    assert (replacement_target / "sentinel").read_bytes() == b"unchanged"


def test_v13_lost_placeholder_mkdir_reply_is_recovered_without_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_nfs_fallback(monkeypatch)
    target = tmp_path / "checkpoint"
    real_mkdir = checkpoint_io.os.mkdir
    injected = False

    def mkdir_then_lose_reply(path, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if Path(path) == target and not injected and dir_fd is None:
            injected = True
            real_mkdir(path, mode)
            raise OSError(errno.EIO, "injected lost mkdir acknowledgement")
        return real_mkdir(path, mode, dir_fd=dir_fd)

    with checkpoint_io.checkpoint_directory_transaction(
        target,
        operation="checkpoint_export",
    ) as (staging, _published):
        _write_payload(staging)
        monkeypatch.setattr(checkpoint_io.os, "mkdir", mkdir_then_lose_reply)

    assert injected is True
    assert (target / "payload.bin").read_bytes() == b"payload"
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))
    assert not (target / checkpoint_io._PLACEHOLDER_OWNER_MARKER).exists()


@pytest.mark.parametrize(
    "mkdir_errno",
    [errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES, errno.EIO],
    ids=["ENOSPC", "EDQUOT", "EROFS", "EACCES", "EIO"],
)
def test_v25_definite_placeholder_mkdir_failure_removes_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mkdir_errno: int,
) -> None:
    _force_nfs_fallback(monkeypatch)
    target = tmp_path / "checkpoint"
    real_mkdir = checkpoint_io.os.mkdir
    staged: Path | None = None

    def fail_target_mkdir(path, mode=0o777, *, dir_fd=None):
        if Path(path) == target and dir_fd is None:
            raise OSError(mkdir_errno, "injected definite mkdir failure")
        return real_mkdir(path, mode, dir_fd=dir_fd)

    with pytest.raises(
        CheckpointIOError,
        match="placeholder was not created",
    ) as captured:
        with checkpoint_io.checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            staged = staging
            _write_payload(staging)
            monkeypatch.setattr(checkpoint_io.os, "mkdir", fail_target_mkdir)

    error = captured.value
    assert not isinstance(error, checkpoint_io._AmbiguousCheckpointPublication)
    assert error.details == {
        "publication_outcome": "not_published",
        "mkdir_errno": mkdir_errno,
        "claim_error": "CheckpointIOError",
        "target_state": "absent",
        "staging_removed": True,
    }
    assert staged is not None
    assert not staged.exists()
    assert not target.exists()
    assert not list(tmp_path.glob(".checkpoint.tmp-*"))


def test_v14_replaced_placeholder_is_never_removed_by_the_losing_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_nfs_fallback(monkeypatch)
    target = tmp_path / "checkpoint"
    real_rmdir = checkpoint_io.os.rmdir
    real_mkdir = checkpoint_io.os.mkdir

    def replace_claim_then_fail(_source, occupied):
        real_rmdir(occupied)
        real_mkdir(occupied, 0o700)
        (Path(occupied) / "competitor").write_bytes(b"keep")
        raise OSError(errno.EIO, "injected rename failure after replacement")

    monkeypatch.setattr(checkpoint_io.os, "rename", replace_claim_then_fail)
    with pytest.raises(CheckpointIOError) as captured:
        with checkpoint_io.checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            _write_payload(staging)

    assert captured.value.details["publication_outcome"] == "ambiguous"
    assert captured.value.details["staging_path"]
    assert captured.value.details["target_state"] == "present"
    assert (target / "competitor").read_bytes() == b"keep"
    preserved = list(tmp_path.glob(".checkpoint.tmp-*"))
    assert len(preserved) == 1
    assert (preserved[0] / "payload.bin").read_bytes() == b"payload"


def test_v16_post_publish_parent_fsync_has_a_distinct_public_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint"
    real_fsync_directory = checkpoint_io._fsync_directory

    def fail_final_parent_sync(path, *, operation):
        if Path(path) == tmp_path and target.exists():
            cause = OSError(errno.EIO, "injected parent fsync failure")
            raise checkpoint_io._io_error(
                cause,
                operation=operation,
                message="injected parent fsync failure",
            ) from cause
        return real_fsync_directory(path, operation=operation)

    monkeypatch.setattr(
        checkpoint_io,
        "_fsync_directory",
        fail_final_parent_sync,
    )
    with pytest.raises(CheckpointDurabilityError) as captured:
        with checkpoint_io.checkpoint_directory_transaction(
            target,
            operation="checkpoint_export",
        ) as (staging, _published):
            _write_payload(staging)

    assert isinstance(captured.value, CheckpointIOError)
    assert captured.value.code == "checkpoint_durability_error"
    assert captured.value.details["publication_outcome"] == "published"
    assert captured.value.details["durability_stage"] == "parent_directory_fsync"
    assert (target / "payload.bin").read_bytes() == b"payload"

    translated = translate_functional_exception(
        captured.value,
        operation="checkpoint_export",
    )
    assert isinstance(translated, FunctionalCheckpointDurabilityError)
    assert isinstance(translated, CheckpointDurabilityError)
    assert translated.code == "functional_checkpoint_durability_error"
