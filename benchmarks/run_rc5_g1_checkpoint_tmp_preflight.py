#!/usr/bin/env python3
"""Verify installed-wheel RENAME_NOREPLACE on an isolated node-local /tmp root."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import pssolver
from pssolver.io.checkpoint import (
    CheckpointExistsError,
    checkpoint_directory_transaction,
)


SCHEMA = "pssolver.rc5.g1.checkpoint_tmp_preflight.v1"


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("preflight JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _mount(path: Path) -> dict[str, str | None]:
    resolved = path.resolve()
    matches: list[tuple[int, str, str, str]] = []
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        device, mountpoint, filesystem, options = fields[:4]
        mount = Path(mountpoint)
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        matches.append((len(mount.parts), device, mountpoint, filesystem))
    if not matches:
        return {"device": None, "mountpoint": None, "filesystem": None}
    _, device, mountpoint, filesystem = max(matches)
    return {"device": device, "mountpoint": mountpoint, "filesystem": filesystem}


def _run(root: Path, expected_tmpdir: Path) -> dict[str, object]:
    root = root.expanduser().absolute()
    expected_tmpdir = expected_tmpdir.expanduser().absolute()
    actual_tmpdir = Path(os.environ.get("TMPDIR", "")).absolute()
    root_parent = root.parent
    tmp_parent = expected_tmpdir.parent
    import_path = Path(pssolver.__file__).resolve()
    prefix = Path(sys.prefix).resolve()

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        libc.renameat2
    except AttributeError:
        wrapper_present = False
    else:
        wrapper_present = True

    report: dict[str, object] = {
        "schema": SCHEMA,
        "passed": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "python": sys.version.split()[0],
        "platform_machine": platform.machine(),
        "glibc": os.confstr("CS_GNU_LIBC_VERSION"),
        "renameat2_wrapper_present": wrapper_present,
        "pssolver_version": pssolver.__version__,
        "pssolver_import": str(import_path),
        "sys_prefix": str(prefix),
        "installed_wheel_import": import_path.is_relative_to(prefix),
        "root": str(root),
        "root_parent": str(root_parent),
        "expected_tmpdir": str(expected_tmpdir),
        "actual_tmpdir": str(actual_tmpdir),
        "tmpdir_matches": actual_tmpdir == expected_tmpdir,
        "independent_preflight_and_pytest_directories": root != expected_tmpdir,
        "shared_node_local_parent": root_parent == tmp_parent,
        "root_is_node_local_tmp": root.is_relative_to(Path("/tmp")),
        "tmpdir_is_node_local_tmp": expected_tmpdir.is_relative_to(Path("/tmp")),
        "mount": _mount(root_parent),
        "publication": {"passed": False},
        "race": {"passed": False},
        "staging_residue_count": None,
    }

    try:
        if root.exists() or root.is_symlink():
            raise FileExistsError(f"preflight root already exists: {root}")
        root.mkdir(parents=False)
        target = root / "checkpoint"
        payload = b"rc5-g1-node-local-rename-noreplace"
        with checkpoint_directory_transaction(
            target,
            operation="rc5_g1_node_local_tmp_preflight",
        ) as (staging, published):
            if published != target:
                raise RuntimeError("published target differs")
            (staging / "payload.bin").write_bytes(payload)
        actual = (target / "payload.bin").read_bytes()
        report["publication"] = {
            "passed": actual == payload,
            "target": str(target),
            "payload_sha256": hashlib.sha256(actual).hexdigest(),
        }

        race_target = root / "racing-checkpoint"
        try:
            with checkpoint_directory_transaction(
                race_target,
                operation="rc5_g1_node_local_tmp_race",
            ) as (staging, _published):
                (staging / "payload.bin").write_bytes(b"must-not-publish")
                race_target.mkdir()
        except CheckpointExistsError as error:
            race_rejected = True
            race_exception = type(error).__name__
            race_message = str(error)
        else:
            race_rejected = False
            race_exception = None
            race_message = None
        race_empty = race_target.is_dir() and not list(race_target.iterdir())
        report["race"] = {
            "passed": race_rejected and race_empty,
            "rejected": race_rejected,
            "exception_type": race_exception,
            "message": race_message,
            "target_empty": race_empty,
        }
        residue = sorted(path.name for path in root.glob(".*.tmp-*"))
        report["staging_residue_count"] = len(residue)
        report["staging_residue"] = residue
    except Exception as error:
        report["exception"] = {
            "type": type(error).__name__,
            "message": str(error),
        }

    report["passed"] = all(
        (
            report["installed_wheel_import"],
            report["tmpdir_matches"],
            report["independent_preflight_and_pytest_directories"],
            report["shared_node_local_parent"],
            report["root_is_node_local_tmp"],
            report["tmpdir_is_node_local_tmp"],
            report["publication"].get("passed") is True,
            report["race"].get("passed") is True,
            report["staging_residue_count"] == 0,
        )
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-tmpdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = _run(args.root, args.expected_tmpdir)
    _atomic_json(args.output, report)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    if report["passed"] is not True:
        raise RuntimeError("node-local RENAME_NOREPLACE preflight failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
