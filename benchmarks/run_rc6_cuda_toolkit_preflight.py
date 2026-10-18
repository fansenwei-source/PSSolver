#!/usr/bin/env python3
"""Persist and enforce the RC6 CUDA-toolkit bootstrap identity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


SCHEMA = "pssolver.rc6.cuda_toolkit_preflight.v1"


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
            raise RuntimeError("CUDA-toolkit preflight JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def observe(expected_module: str, expected_release: str) -> dict[str, object]:
    loaded_modules = [value for value in os.environ.get("LOADEDMODULES", "").split(":") if value]
    nvcc_lexical = shutil.which("nvcc")
    nvcc_resolved: str | None = None
    nvcc_returncode: int | None = None
    nvcc_stdout = ""
    nvcc_stderr = ""
    cuda_root: str | None = None
    cuda_header: str | None = None
    cuda_header_regular = False
    if nvcc_lexical is not None:
        nvcc_path = Path(nvcc_lexical)
        resolved = nvcc_path.resolve()
        nvcc_resolved = str(resolved)
        completed = subprocess.run(
            [str(nvcc_path), "--version"], check=False, capture_output=True, text=True,
        )
        nvcc_returncode = completed.returncode
        nvcc_stdout = completed.stdout
        nvcc_stderr = completed.stderr
        root = resolved.parent.parent
        header = root / "include" / "cuda.h"
        cuda_root = str(root)
        cuda_header = str(header)
        cuda_header_regular = header.is_file() and not header.is_symlink()

    torch_version: str | None = None
    torch_cuda_runtime: str | None = None
    torch_exception: str | None = None
    try:
        import torch

        torch_version = torch.__version__
        torch_cuda_runtime = torch.version.cuda
    except Exception as exc:  # pragma: no cover - exercised only in broken environments
        torch_exception = f"{type(exc).__name__}: {exc}"

    failures: list[str] = []
    if expected_module not in loaded_modules:
        failures.append("expected CUDA module is not loaded")
    if nvcc_lexical is None:
        failures.append("nvcc is not discoverable through PATH")
    if nvcc_returncode != 0:
        failures.append("nvcc --version did not succeed")
    if f"release {expected_release}" not in nvcc_stdout:
        failures.append("nvcc release differs")
    if not cuda_header_regular:
        failures.append("ordinary CUDA driver header is unavailable")
    if torch_exception is not None:
        failures.append("PyTorch import failed")
    if torch_cuda_runtime != expected_release:
        failures.append("PyTorch CUDA runtime differs")

    return {
        "schema": SCHEMA,
        "expected_module": expected_module,
        "expected_release": expected_release,
        "loaded_modules": loaded_modules,
        "nvcc_lexical": nvcc_lexical,
        "nvcc_resolved": nvcc_resolved,
        "nvcc_returncode": nvcc_returncode,
        "nvcc_stdout": nvcc_stdout,
        "nvcc_stderr": nvcc_stderr,
        "cuda_path_environment": os.environ.get("CUDA_PATH"),
        "cuda_root": cuda_root,
        "cuda_header": cuda_header,
        "cuda_header_regular": cuda_header_regular,
        "torch": torch_version,
        "torch_cuda_runtime": torch_cuda_runtime,
        "torch_exception": torch_exception,
        "failures": failures,
        "passed": not failures,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-module", required=True)
    parser.add_argument("--expected-release", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = observe(args.expected_module, args.expected_release)
    _atomic_json(args.output, result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
