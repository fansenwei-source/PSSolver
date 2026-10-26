#!/usr/bin/env python3
"""Persist TF32-off policy before one S-09 A-p cache diagnostic profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from benchmarks import profile_v0_3_s09_ap_cache_diagnostic as profile


SCHEMA = "pssolver.v0_3.s09.ap_cache_diagnostic_child.v1"
TF32_OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(payload):
            raise RuntimeError("child-bootstrap JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _tf32() -> dict[str, object]:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tf32-bootstrap-output", type=Path, required=True)
    parser.add_argument("--profile-script", type=Path, required=True)
    args, profile_argv = parser.parse_known_args(argv)
    expected = Path(profile.__file__).resolve()
    requested = args.profile_script.resolve()
    if requested != expected or not requested.is_file() or requested.is_symlink():
        raise ValueError(f"diagnostic profile script identity differs: {requested} != {expected}")

    before = _tf32()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    after = _tf32()
    report = {
        "schema": SCHEMA,
        "profile_script": str(requested),
        "profile_script_sha256": _sha256(requested),
        "policy": TF32_OFF,
        "before": before,
        "after": after,
        "passed": after == TF32_OFF,
        "profile_start_authorized_after_persisted_enforcement": True,
    }
    _atomic_json(args.tf32_bootstrap_output, report)
    if not report["passed"]:
        raise RuntimeError(f"TF32-off bootstrap failed: {after!r}")
    return profile.main(profile_argv)


if __name__ == "__main__":
    raise SystemExit(main())
