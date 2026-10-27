#!/usr/bin/env python3
"""Run the S-09 diagnostic H100 compiler and CUDA preflight.

This diagnostic reuses the cumulative S-09 host-compiler checks so the
``channel_compiled_v2`` profiles are never started with an unverified
toolchain.  It is qualification support only and does not adjudicate S-09.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.run_v0_3_s09_environment import (
    _host_compiler_and_compile_smokes,
)


SCHEMA = "pssolver.v0_3.s09.ap_cache_diagnostic_preflight.v1"


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


def run(work_root: Path) -> dict[str, object]:
    import pssolver
    import torch

    before = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    after = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    available = bool(torch.cuda.is_available())
    visible = int(torch.cuda.device_count())
    allocated = str(torch.empty((), device="cuda").device) if available else None
    device_name = torch.cuda.get_device_name(0) if available and visible else None
    if not available or visible != 1 or allocated != "cuda:0":
        raise RuntimeError("diagnostic preflight requires exactly one cuda:0 device")
    if not isinstance(device_name, str) or "H100" not in device_name:
        raise RuntimeError("diagnostic preflight requires an NVIDIA H100")
    if after != {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }:
        raise RuntimeError("TF32-off policy was not enforced")

    compiler = _host_compiler_and_compile_smokes(work_root)
    for name in (
        "stdatomic_pic_compile",
        "cuda_driver_shared_library_compile_link_load",
        "cuda_driver_load",
        "empty_triton_cache_compile",
        "empty_torchinductor_cache_compile",
    ):
        if compiler.get(name) is not True:
            raise RuntimeError(f"compiler preflight failed: {name}")
    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "qualification_complete": False,
        "passed": True,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "pssolver_import": str(Path(pssolver.__file__).resolve()),
        "cuda": {
            "available": available,
            "visible_device_count": visible,
            "allocated_device": allocated,
            "device_name": device_name,
        },
        "tf32_before": before,
        "tf32_after": after,
        "compiler": compiler,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, run(args.work_root.expanduser().absolute()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
