#!/usr/bin/env python3
"""Capture the D-0008 local S-09 qualification environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


SCHEMA = "pssolver.v0_3.s09.local_environment.v1"


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
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def capture(args: argparse.Namespace) -> dict[str, object]:
    import numpy
    import pssolver
    import scipy
    import torch

    root = args.repository.expanduser().resolve()
    python = args.python.expanduser().absolute()
    clone = args.baseline_clone.expanduser().resolve()
    wheel = args.wheel.expanduser().resolve()
    if Path(sys.executable).absolute() != python:
        raise RuntimeError("reported Python executable differs")
    if not clone.is_dir() or not wheel.is_file() or wheel.is_symlink():
        raise FileNotFoundError("ordinary baseline clone and candidate wheel required")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"], check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    distribution = importlib.metadata.distribution("pssolver")
    record = Path(distribution.locate_file("pssolver-0.3.0.dev0.dist-info/RECORD")).resolve()
    tracked = (
        root / "pssolver/functional/api.py",
        root / "pssolver/functional/periodic_runtime.py",
        root / "pssolver/functional/channel_runtime.py",
        root / "pssolver/linear_solvers/stokes/channel_no_slip.py",
    )
    usage = shutil.disk_usage(root)
    return {
        "schema": SCHEMA,
        "repository": str(root),
        "git_commit": head,
        "baseline_clone_source": str(args.baseline_source.expanduser().resolve()),
        "baseline_clone": str(clone),
        "clone_method": "cp -al hard-link clone",
        "python": platform.python_version(),
        "python_executable": str(python),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pssolver_version": importlib.metadata.version("pssolver"),
        "pssolver_import": str(Path(pssolver.__file__).resolve()),
        "sys_path": list(sys.path),
        "candidate_wheel": str(wheel),
        "candidate_wheel_sha256": _sha256(wheel),
        "installed_record": str(record),
        "installed_record_sha256": _sha256(record),
        "runtime_file_sha256": {
            path.relative_to(root).as_posix(): _sha256(path) for path in tracked
        },
        "pip_freeze": freeze,
        "disk": {
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_at_least_50_gib": usage.free >= 50 * 1024**3,
        },
        "cuda_visible": bool(torch.cuda.is_available()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-clone", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = capture(args)
    if result["pssolver_version"] != "0.3.0.dev0":
        raise RuntimeError("candidate package version differs")
    if result["disk"]["free_at_least_50_gib"] is not True:
        raise RuntimeError("D-0008 minimum free-space gate failed")
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
