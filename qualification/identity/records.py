"""Byte-preserving tensor records and explicit numerical prerequisites."""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
from pathlib import Path
import platform

import numpy as np
import torch


def tensor_record(value: torch.Tensor) -> dict:
    """Hash bytes without value conversion, NaN normalization or tolerance."""
    array = value.detach().cpu().contiguous().numpy()
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "finite": bool(np.isfinite(array).all())}


def fingerprint() -> dict:
    """Numerical identity excludes install paths, package version and CUDA visibility."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.show_config()
    cpu = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    flags = sorted({word for line in cpu.splitlines() if line.startswith("flags")
                    for word in line.partition(":")[2].split()})
    settings = [line.strip() for line in torch.__config__.show().splitlines()
                if "Build settings:" in line]
    return {
        "schema": 1, "python": platform.python_version(), "machine": platform.machine(),
        "torch": str(torch.__version__), "torch_cuda_build": torch.version.cuda,
        "torch_git": torch.version.git_version, "torch_build_settings": settings,
        "numpy": np.__version__, "numpy_build": stream.getvalue(), "cpu_flags": flags,
        "torch_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "thread_environment": {key: os.environ.get(key) for key in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }


def require_fingerprint(actual: dict, expected: dict) -> None:
    """Fail explicitly when stored bytes cannot be compared in this environment."""
    if actual != expected:
        raise RuntimeError(f"PSSOLVER_TEST_PREREQUISITE: baseline fingerprint differs: "
                           f"actual={actual!r}; expected={expected!r}")
