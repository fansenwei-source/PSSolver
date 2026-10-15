#!/usr/bin/env python3
"""Capture the installed-wheel rc5 G-1 H100 environment gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc5_g1_h100 import (
    CONSUMER_WHEEL_SHA256,
    PROVIDER_BASELINE_WHEEL_SHA256,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
)
from benchmarks.rc458_venv_identity import (
    lexical_python_path,
    validate_installed_venv_identity,
)


SCHEMA = "pssolver.rc5.g1.environment.v1"


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
    }
    if role == "consumer":
        import pssolver_control

        result.update(
            {
                "pssolver_control_version": importlib.metadata.version(
                    "pssolver-control"
                ),
                "pssolver_control_import": str(
                    Path(pssolver_control.__file__).resolve()
                ),
            }
        )
    return result


def _inspect(python: Path, role: str, script: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(python), str(script), "--internal-inspect-role", role],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{role} environment inspection failed with {completed.returncode}: "
            f"{completed.stderr}"
        )
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise TypeError(f"{role} environment observation must be an object")
    return value


def _pip_check(python: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(python), "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
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
) -> None:
    if set(observations) != {"baseline", "candidate", "consumer"}:
        raise ValueError("installed environment role set differs")
    if set(identities) != {"baseline", "candidate", "consumer_provider", "consumer"}:
        raise ValueError("installed import identity set differs")
    if set(pip_checks) != {"baseline", "candidate", "consumer"}:
        raise ValueError("pip-check role set differs")
    if observations["baseline"].get("pssolver_version") != "0.2.0rc4":
        raise ValueError("baseline package version differs")
    for role in ("candidate", "consumer"):
        if observations[role].get("pssolver_version") != "0.2.0rc5":
            raise ValueError(f"{role} Provider version differs")
    if observations["consumer"].get("pssolver_control_version") != "0.0.1.dev0":
        raise ValueError("Consumer package version differs")
    if not all(value.get("passed") is True for value in identities.values()):
        raise ValueError("installed import identity failed")
    if not all(value.get("status") == "PASS" for value in pip_checks.values()):
        raise ValueError("pip check failed")
    if wheel_hashes != {
        "baseline": PROVIDER_BASELINE_WHEEL_SHA256,
        "candidate": PROVIDER_CANDIDATE_WHEEL_SHA256,
        "consumer": CONSUMER_WHEEL_SHA256,
    }:
        raise ValueError("installed wheel SHA-256 differs")
    if support_contains_pssolver:
        raise ValueError("qualification support shadows PSSolver")
    if cuda != {
        "available": True,
        "visible_device_count": 1,
        "allocated_device": "cuda:0",
        "device_name_contains_h100": True,
    }:
        raise ValueError("allocated H100 identity differs")
    if tf32_after != {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }:
        raise ValueError("TF32-off policy was not enforced")


def _identity(
    requested_python: Path,
    observation: dict[str, object],
    import_key: str,
) -> dict[str, object]:
    return validate_installed_venv_identity(
        requested_python=requested_python,
        imported_module=Path(str(observation[import_key])),
        runtime_prefix=Path(str(observation["runtime_prefix"])),
        purelib=Path(str(observation["purelib"])),
        reported_executable=Path(str(observation["reported_executable"])),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    script = Path(__file__).resolve()
    pythons = {
        "baseline": lexical_python_path(args.baseline_python),
        "candidate": lexical_python_path(args.candidate_python),
        "consumer": lexical_python_path(args.consumer_python),
    }
    wheels = {
        "baseline": args.baseline_wheel.expanduser().resolve(),
        "candidate": args.candidate_wheel.expanduser().resolve(),
        "consumer": args.consumer_wheel.expanduser().resolve(),
    }
    expected_hashes = {
        "baseline": PROVIDER_BASELINE_WHEEL_SHA256,
        "candidate": PROVIDER_CANDIDATE_WHEEL_SHA256,
        "consumer": CONSUMER_WHEEL_SHA256,
    }
    for role, path in wheels.items():
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"regular {role} wheel required: {path}")
    wheel_hashes = {role: _sha256(path) for role, path in wheels.items()}
    if wheel_hashes != expected_hashes:
        raise ValueError("wheel identity differs before environment inspection")
    support = args.provider_support.expanduser().resolve()
    if not support.is_dir() or support.is_symlink():
        raise FileNotFoundError(f"ordinary Provider support root required: {support}")
    support_contains_pssolver = (support / "pssolver").exists() or (
        support / "pssolver"
    ).is_symlink()

    observations = {
        role: _inspect(python, role, script) for role, python in pythons.items()
    }
    identities = {
        "baseline": _identity(
            pythons["baseline"], observations["baseline"], "pssolver_import"
        ),
        "candidate": _identity(
            pythons["candidate"], observations["candidate"], "pssolver_import"
        ),
        "consumer_provider": _identity(
            pythons["consumer"], observations["consumer"], "pssolver_import"
        ),
        "consumer": _identity(
            pythons["consumer"],
            observations["consumer"],
            "pssolver_control_import",
        ),
    }
    pip_checks = {role: _pip_check(python) for role, python in pythons.items()}

    tf32_before = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tf32_after = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    cuda_available = bool(torch.cuda.is_available())
    visible = int(torch.cuda.device_count())
    allocated = str(torch.empty((), device="cuda").device) if cuda_available else None
    device_name = torch.cuda.get_device_name(0) if cuda_available and visible else None
    cuda = {
        "available": cuda_available,
        "visible_device_count": visible,
        "allocated_device": allocated,
        "device_name_contains_h100": isinstance(device_name, str)
        and "H100" in device_name,
    }
    validate_observations(
        observations=observations,
        identities=identities,
        pip_checks=pip_checks,
        wheel_hashes=wheel_hashes,
        support_contains_pssolver=support_contains_pssolver,
        cuda=cuda,
        tf32_after=tf32_after,
    )
    return {
        "schema": SCHEMA,
        "passed": True,
        "python": observations["candidate"]["python"],
        "torch": observations["candidate"]["torch"],
        "torch_cuda_runtime": observations["candidate"]["torch_cuda_runtime"],
        "numpy": observations["candidate"]["numpy"],
        "scipy": observations["candidate"]["scipy"],
        "cuda_available": cuda_available,
        "visible_device_count": visible,
        "allocated_device": allocated,
        "device_name": device_name,
        "tf32_before": tf32_before,
        "tf32_after": tf32_after,
        "tf32_matmul": tf32_after["cuda_matmul_allow_tf32"],
        "tf32_cudnn": tf32_after["cudnn_allow_tf32"],
        "source_shadow_import": False,
        "pip_check": "PASS",
        "imports_from_site_packages": True,
        "installed_wheels": wheel_hashes,
        "installed_environments": observations,
        "installed_import_identities": identities,
        "pip_checks": pip_checks,
        "provider_support_root": str(support),
        "provider_support_contains_pssolver": support_contains_pssolver,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--consumer-python", type=Path)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--consumer-wheel", type=Path)
    parser.add_argument("--provider-support", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--internal-inspect-role",
        choices=("baseline", "candidate", "consumer"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.internal_inspect_role is not None:
        print(
            json.dumps(
                _internal_observation(args.internal_inspect_role),
                allow_nan=False,
                sort_keys=True,
            )
        )
        return 0
    required = (
        "baseline_python",
        "candidate_python",
        "consumer_python",
        "baseline_wheel",
        "candidate_wheel",
        "consumer_wheel",
        "provider_support",
        "output",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")
    _atomic_json(args.output, run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
