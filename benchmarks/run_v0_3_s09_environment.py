#!/usr/bin/env python3
"""Capture and enforce the v0.3 S-09 installed-wheel H100 environment."""

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
import sysconfig
import tempfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_v0_3_s09_h100 import (
    CANDIDATE_COMMIT,
    CONSUMER_WHEEL_SHA256,
    BASELINE_WHEEL_SHA256,
    CANDIDATE_WHEEL_SHA256,
)
from benchmarks.rc458_venv_identity import lexical_python_path, validate_installed_venv_identity


SCHEMA = "pssolver.v0_3.s09.environment.v1"


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
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("environment JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _internal_observation(role: str) -> dict[str, object]:
    import numpy
    import pssolver
    import scipy
    import torch

    provider_distribution = importlib.metadata.distribution("pssolver")
    provider_record = Path(provider_distribution._path) / "RECORD"  # type: ignore[attr-defined]
    if not provider_record.is_file() or provider_record.is_symlink():
        raise FileNotFoundError(f"ordinary Provider RECORD required: {provider_record}")
    result: dict[str, object] = {
        "role": role,
        "python": platform.python_version(),
        "reported_executable": sys.executable,
        "runtime_prefix": sys.prefix,
        "purelib": sysconfig.get_path("purelib"),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pssolver_version": importlib.metadata.version("pssolver"),
        "pssolver_import": str(Path(pssolver.__file__).resolve()),
        "pssolver_record": {
            "path": str(provider_record.resolve()),
            "sha256": _sha256(provider_record),
            "entry_count": sum(1 for _ in provider_record.open(encoding="utf-8")),
        },
    }
    if role == "consumer":
        import pssolver_control
        consumer_distribution = importlib.metadata.distribution("pssolver-control")
        consumer_record = Path(consumer_distribution._path) / "RECORD"  # type: ignore[attr-defined]
        if not consumer_record.is_file() or consumer_record.is_symlink():
            raise FileNotFoundError(f"ordinary Consumer RECORD required: {consumer_record}")

        result.update(
            {
                "pssolver_control_version": importlib.metadata.version("pssolver-control"),
                "pssolver_control_import": str(Path(pssolver_control.__file__).resolve()),
                "pssolver_control_record": {
                    "path": str(consumer_record.resolve()),
                    "sha256": _sha256(consumer_record),
                    "entry_count": sum(
                        1 for _ in consumer_record.open(encoding="utf-8")
                    ),
                },
            }
        )
    return result


def _inspect(python: Path, role: str, script: Path) -> dict[str, object]:
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    completed = subprocess.run(
        [str(python), str(script), "--internal-inspect-role", role],
        check=False, capture_output=True, text=True, env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{role} inspection failed: {completed.stderr}")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError(f"{role} observation must be an object")
    return value


def _pip_check(python: Path) -> dict[str, object]:
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    completed = subprocess.run(
        [str(python), "-m", "pip", "check"], check=False,
        capture_output=True, text=True, env=environment,
    )
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _identity(python: Path, observation: dict[str, object], key: str) -> dict[str, object]:
    return validate_installed_venv_identity(
        requested_python=python,
        imported_module=Path(str(observation[key])),
        runtime_prefix=Path(str(observation["runtime_prefix"])),
        purelib=Path(str(observation["purelib"])),
        reported_executable=Path(str(observation["reported_executable"])),
    )


def _host_compiler_and_compile_smokes(work_root: Path) -> dict[str, object]:
    import torch

    cc = Path(os.environ.get("CC", ""))
    cxx = Path(os.environ.get("CXX", ""))
    if not cc.is_file() or not cxx.is_file():
        raise FileNotFoundError("ordinary CC/CXX files are required")
    work_root.mkdir(parents=True, exist_ok=False)
    header = subprocess.run(
        [str(cc), "-print-file-name=include/stdatomic.h"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    header_path = Path(header)
    if not header_path.is_file() or header_path.is_symlink():
        raise FileNotFoundError(f"ordinary stdatomic.h required: {header_path}")
    source = work_root / "stdatomic.c"
    source.write_text(
        "#include <stdatomic.h>\n_Atomic int value = 0;\nint load_value(void){return atomic_load(&value);}\n",
        encoding="utf-8",
    )
    library = work_root / "libstdatomic_smoke.so"
    subprocess.run(
        [str(cc), "-std=c11", "-fPIC", "-shared", str(source), "-o", str(library)],
        check=True, capture_output=True, text=True,
    )
    import ctypes

    ctypes.CDLL(str(library)).load_value()
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise FileNotFoundError("nvcc is required for CUDA header discovery")
    cuda_root = Path(nvcc).resolve().parent.parent
    cuda_header = cuda_root / "include" / "cuda.h"
    if not cuda_header.is_file() or cuda_header.is_symlink():
        raise FileNotFoundError(f"ordinary CUDA driver header required: {cuda_header}")
    cuda_source = work_root / "cuda_driver.c"
    cuda_source.write_text(
        "#include <cuda.h>\nint initialize_cuda(void){return (int)cuInit(0);}\n",
        encoding="utf-8",
    )
    cuda_library = work_root / "libcuda_driver_smoke.so"
    subprocess.run(
        [
            str(cc), "-fPIC", "-shared", str(cuda_source),
            "-I", str(cuda_header.parent), "-Wl,-l:libcuda.so.1",
            "-o", str(cuda_library),
        ],
        check=True, capture_output=True, text=True,
    )
    compiled_driver = ctypes.CDLL(str(cuda_library))
    compiled_driver.initialize_cuda.restype = ctypes.c_int
    if compiled_driver.initialize_cuda() != 0:
        raise RuntimeError("compiled CUDA driver smoke failed to initialize CUDA")

    triton_cache = work_root / "triton"
    inductor_cache = work_root / "torchinductor"
    triton_cache.mkdir()
    inductor_cache.mkdir()
    if list(triton_cache.iterdir()) or list(inductor_cache.iterdir()):
        raise RuntimeError("compiler caches must begin empty")
    previous_triton = os.environ.get("TRITON_CACHE_DIR")
    previous_inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
    try:
        from triton.runtime.driver import driver

        _ = driver.active.utils
        compiled = torch.compile(lambda value: value.sin() + 1.0)
        value = torch.ones(16, device="cuda")
        observed = compiled(value)
        torch.cuda.synchronize()
        if not bool(torch.isfinite(observed).all().item()):
            raise RuntimeError("TorchInductor smoke produced non-finite output")
    finally:
        if previous_triton is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = previous_triton
        if previous_inductor is None:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = previous_inductor
    return {
        "cc": str(cc.resolve()),
        "cxx": str(cxx.resolve()),
        "stdatomic_header": str(header_path.resolve()),
        "cuda_header": str(cuda_header.resolve()),
        "stdatomic_pic_compile": True,
        "cuda_driver_shared_library_compile_link_load": True,
        "cuda_driver_load": True,
        "empty_triton_cache_compile": True,
        "empty_torchinductor_cache_compile": True,
    }


def validate_observations(
    *,
    observations: dict[str, dict[str, object]],
    identities: dict[str, dict[str, object]],
    pip_checks: dict[str, dict[str, object]],
    wheel_hashes: dict[str, str],
    support_contains_pssolver: bool,
    cuda: dict[str, object],
    tf32_after: dict[str, object],
    tmp_preflight: dict[str, object],
    compiler: dict[str, object],
) -> None:
    if set(observations) != {"baseline", "candidate", "consumer"}:
        raise ValueError("installed environment role set differs")
    if observations["baseline"].get("pssolver_version") != "0.2.2":
        raise ValueError("baseline package version differs")
    for role in ("candidate", "consumer"):
        if observations[role].get("pssolver_version") != "0.3.0.dev0":
            raise ValueError(f"{role} Provider version differs")
    if observations["consumer"].get("pssolver_control_version") != "0.0.1.dev0":
        raise ValueError("Consumer package version differs")
    if set(identities) != {"baseline", "candidate", "consumer_provider", "consumer"} or not all(value.get("passed") is True for value in identities.values()):
        raise ValueError("installed import identity failed")
    if set(pip_checks) != {"baseline", "candidate", "consumer"} or not all(value.get("status") == "PASS" for value in pip_checks.values()):
        raise ValueError("pip check failed")
    if wheel_hashes != {
        "baseline": BASELINE_WHEEL_SHA256,
        "candidate": CANDIDATE_WHEEL_SHA256,
        "consumer": CONSUMER_WHEEL_SHA256,
    }:
        raise ValueError("installed wheel identity differs")
    if support_contains_pssolver:
        raise ValueError("qualification support shadows PSSolver")
    if cuda != {"available": True, "visible_device_count": 1, "allocated_device": "cuda:0", "device_name_contains_h100": True}:
        raise ValueError("allocated H100 identity differs")
    if tf32_after != {"float32_matmul_precision": "highest", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}:
        raise ValueError("TF32-off policy was not enforced")
    if tmp_preflight.get("passed") is not True or tmp_preflight.get("tmpdir_matches") is not True:
        raise ValueError("node-local tmp preflight failed")
    for name in (
        "stdatomic_pic_compile",
        "cuda_driver_shared_library_compile_link_load",
        "cuda_driver_load",
        "empty_triton_cache_compile",
        "empty_torchinductor_cache_compile",
    ):
        if compiler.get(name) is not True:
            raise ValueError(f"compiler preflight failed: {name}")


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    script = Path(__file__).resolve()
    pythons = {
        "baseline": lexical_python_path(args.baseline_python),
        "candidate": lexical_python_path(args.candidate_python),
        "consumer": lexical_python_path(args.consumer_python),
    }
    wheels = {name: Path(value).expanduser().resolve() for name, value in {
        "baseline": args.baseline_wheel,
        "candidate": args.candidate_wheel,
        "consumer": args.consumer_wheel,
    }.items()}
    for role, path in wheels.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"regular {role} wheel required: {path}")
    wheel_hashes = {role: _sha256(path) for role, path in wheels.items()}
    support = args.provider_support.expanduser().resolve()
    if not support.is_dir() or support.is_symlink():
        raise FileNotFoundError(f"ordinary Provider support root required: {support}")
    support_contains_pssolver = (support / "pssolver").exists() or (support / "pssolver").is_symlink()
    observations = {role: _inspect(python, role, script) for role, python in pythons.items()}
    identities = {
        "baseline": _identity(pythons["baseline"], observations["baseline"], "pssolver_import"),
        "candidate": _identity(pythons["candidate"], observations["candidate"], "pssolver_import"),
        "consumer_provider": _identity(pythons["consumer"], observations["consumer"], "pssolver_import"),
        "consumer": _identity(pythons["consumer"], observations["consumer"], "pssolver_control_import"),
    }
    pip_checks = {role: _pip_check(python) for role, python in pythons.items()}
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
    cuda = {
        "available": available,
        "visible_device_count": visible,
        "allocated_device": allocated,
        "device_name_contains_h100": isinstance(device_name, str) and "H100" in device_name,
    }
    tmp_preflight = json.loads(args.tmp_preflight.read_text(encoding="utf-8"))
    compiler = _host_compiler_and_compile_smokes(args.compiler_work_root.expanduser().absolute())
    validate_observations(
        observations=observations, identities=identities, pip_checks=pip_checks,
        wheel_hashes=wheel_hashes, support_contains_pssolver=support_contains_pssolver,
        cuda=cuda, tf32_after=after, tmp_preflight=tmp_preflight, compiler=compiler,
    )
    return {
        "schema": SCHEMA,
        "name": "node_local_tmp_environment_and_host_compiler",
        "provider_candidate_commit": CANDIDATE_COMMIT,
        "candidate_wheel_sha256": CANDIDATE_WHEEL_SHA256,
        "baseline_wheel_sha256": BASELINE_WHEEL_SHA256,
        "consumer_wheel_sha256": CONSUMER_WHEEL_SHA256,
        "passed": True,
        "python": observations["candidate"]["python"],
        "torch": observations["candidate"]["torch"],
        "torch_cuda_runtime": observations["candidate"]["torch_cuda_runtime"],
        "numpy": observations["candidate"]["numpy"],
        "scipy": observations["candidate"]["scipy"],
        "cuda_available": available, "visible_device_count": visible,
        "allocated_device": allocated, "device_name": device_name,
        "tf32_before": before, "tf32_after": after,
        "tf32_matmul": after["cuda_matmul_allow_tf32"],
        "tf32_cudnn": after["cudnn_allow_tf32"],
        "source_shadow_import": False,
        "pip_check": True,
        "pip_check_status": "PASS",
        "single_visible_h100": visible == 1 and isinstance(device_name, str) and "H100" in device_name,
        "allocated_cuda_0": allocated == "cuda:0",
        "installed_wheels_only": True,
        "source_shadow_import_false": True,
        "node_local_tmp": tmp_preflight["passed"] is True,
        "host_compiler": True,
        "empty_compile_caches": (
            compiler["empty_triton_cache_compile"]
            and compiler["empty_torchinductor_cache_compile"]
        ),
        "imports_from_site_packages": True, "installed_wheels": wheel_hashes,
        "installed_environments": observations,
        "installed_import_identities": identities, "pip_checks": pip_checks,
        "compiler": compiler,
        "preflight": {
            "node_local_tmp_rename_noreplace": tmp_preflight["passed"],
            "pytest_tmpdir_is_unique_node_local_tmp": tmp_preflight["tmpdir_matches"],
            "host_compiler_stdatomic_and_cuda_compile": (
                compiler["stdatomic_pic_compile"]
                and compiler["cuda_driver_shared_library_compile_link_load"]
                and compiler["cuda_driver_load"]
            ),
            "empty_triton_cache_compile": compiler["empty_triton_cache_compile"],
            "empty_torchinductor_cache_compile": compiler["empty_torchinductor_cache_compile"],
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline-python", "candidate-python", "consumer-python", "baseline-wheel", "candidate-wheel", "consumer-wheel"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--provider-support", type=Path)
    parser.add_argument("--tmp-preflight", type=Path)
    parser.add_argument("--compiler-work-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--internal-inspect-role", choices=("baseline", "candidate", "consumer"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.internal_inspect_role is not None:
        print(json.dumps(_internal_observation(args.internal_inspect_role), allow_nan=False, sort_keys=True))
        return 0
    missing = [name for name in ("baseline_python", "candidate_python", "consumer_python", "baseline_wheel", "candidate_wheel", "consumer_wheel", "provider_support", "tmp_preflight", "compiler_work_root", "output") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")
    _atomic_json(args.output, run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
