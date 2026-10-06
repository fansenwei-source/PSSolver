#!/usr/bin/env python3
"""Run one frozen rc2 installed-wheel continuous/restart G5 smoke.

This qualification executable owns the complete public ``Simulation``
declaration instead of relying on an external helper to reconstruct it.  In
particular, both qualified applications explicitly request disabled spectral
refresh, as required by the P8.2/P8.3 public compiler contracts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import sysconfig
import time

import numpy as np
import torch

import pssolver
from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
    run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.models.active_nematics import CompleteStressBerisEdwards


SCHEMA_VERSION = 1
KIND = "pssolver_v0_2_0rc2_g5_installed_smoke"
APPLICATIONS = ("periodic", "channel")
VARIANTS = ("baseline", "candidate")
FROZEN_SHAPE = (128, 128, 32)
FROZEN_LENGTHS = (100.0, 100.0, 20.0)
FROZEN_INPUT_SHA256 = (
    "28c70b72118c6d55b7646919ef2157959f40f52a1585753bb0732565cc6895c6"
)
FROZEN_DT = 0.001
FROZEN_TOTAL_STEPS = 100
FROZEN_SPLIT_STEPS = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _model() -> CompleteStressBerisEdwards:
    return CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
    )


def build_simulation(
    *,
    application: str,
    initial_q_path: Path,
    output_directory: Path,
    steps: int,
    device: str,
    checkpoint_interval: int | None = None,
    restart_from: Path | None = None,
    shape: tuple[int, int, int] = FROZEN_SHAPE,
    lengths: tuple[float, float, float] = FROZEN_LENGTHS,
) -> Simulation:
    """Build the exact public declaration used by the installed-wheel smoke."""

    if application not in APPLICATIONS:
        raise ValueError(f"application must be one of {APPLICATIONS}")
    model = _model()
    if application == "periodic":
        geometry = PeriodicBox(shape=shape, lengths=lengths)
        velocity = free_slip_velocity()
        storage = "hermitian_half"
        hermitian_axis = 1
        runtime_path = "periodic_spectral"
        stress_sum = "spectral"
        discretization: dict[str, object] = {}
    else:
        geometry = RectangularChannel(shape=shape, lengths=lengths)
        velocity = no_slip_velocity()
        storage = "full_complex"
        hermitian_axis = None
        runtime_path = "channel_complete_stress"
        stress_sum = "physical"
        discretization = {
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        }
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": velocity,
            "pressure": neumann_pressure_compatibility(),
        },
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            projected_transform_execution="truncated",
            spectral_storage=storage,
            hermitian_axis=hermitian_axis,
        ),
        # Keep this explicit even though TimeStepping currently normalizes a
        # missing refresh declaration to disabled.  G5 must not depend on a
        # helper or constructor default to satisfy the P8.2/P8.3 contract.
        time=TimeStepping(dt=FROZEN_DT, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(
            initial_q_path.parent,
            step=0,
        ),
        execution=TorchSpectralExecution(
            runtime_path=runtime_path,
            device=device,
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": stress_sum,
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=output_directory,
            steps=steps,
            save_start_step=0,
            save_interval=(
                FROZEN_SPLIT_STEPS
                if checkpoint_interval is not None
                else FROZEN_TOTAL_STEPS
            ),
            diagnostic_interval=10,
            diagnostics=True,
            save_hydrodynamics=True,
            checkpoint_interval=checkpoint_interval,
            restart_from=restart_from,
        ),
        discretization=discretization,
        invocation=InvocationSpec({}),
    )


def _run(simulation: Simulation):
    compiled = compile_simulation(simulation)
    metadata = simulation.to_metadata()
    if metadata["time_integration"]["refresh"] != {"mode": "disabled"}:
        raise RuntimeError("G5 simulation does not explicitly disable spectral refresh")
    steps = int(simulation.output.options["steps"])
    result = run_simulation(compiled, progress=range(steps))
    return compiled, result


def _artifact(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"application artifact is absent: {path}")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "size_bytes": path.stat().st_size,
    }


def _result_metadata(result) -> dict[str, object]:
    metadata_path = result.output_directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    runtime = metadata["runtime_selection"]
    if runtime["requested"] != runtime["effective"]:
        raise RuntimeError("runtime requested/effective identity differs")
    if runtime["fallback_used"] is not False:
        raise RuntimeError("runtime fallback occurred")
    if not (result.output_directory / "COMPLETE").is_file():
        raise RuntimeError("workflow COMPLETE marker is absent")
    return {
        "start_step": int(result.start_step),
        "final_step": int(result.final_step),
        "saved_steps": list(result.saved_steps),
        "checkpoint_steps": list(result.checkpoint_steps),
        "elapsed_seconds": float(result.elapsed_seconds),
        "runtime_selection": runtime,
        "metadata_sha256": _sha256(metadata_path),
        "complete": True,
    }


def _git_identity(repository_root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", "-C", str(repository_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "status": git("status", "--porcelain=v1", "--untracked-files=all"),
    }


def _installed_wheel_identity() -> dict[str, object]:
    prefix = Path(sys.prefix).resolve()
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    imported_pssolver = Path(pssolver.__file__).resolve()
    if not purelib.is_relative_to(prefix):
        raise RuntimeError("G5 purelib is outside the active Python environment")
    if not imported_pssolver.is_relative_to(purelib):
        raise RuntimeError("G5 pssolver import is not from installed-wheel purelib")
    return {
        "python_executable": str(Path(sys.executable).absolute()),
        "python_prefix": str(prefix),
        "purelib": str(purelib),
        "pssolver_file": str(imported_pssolver),
        "source_shadow_import": False,
    }


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    input_path = args.initial_q_path.expanduser().resolve()
    if not input_path.is_file() or input_path.is_symlink():
        raise FileNotFoundError("frozen Q input is not a regular file")
    if _sha256(input_path) != FROZEN_INPUT_SHA256:
        raise ValueError("frozen R128 input SHA-256 differs")
    initial = np.load(input_path, allow_pickle=False, mmap_mode="r")
    if initial.shape != (*FROZEN_SHAPE, 5) or initial.dtype != np.float64:
        raise ValueError("frozen R128 input shape or dtype differs")
    if not bool(np.isfinite(initial).all()):
        raise ValueError("frozen R128 input is non-finite")
    output_root = args.output_root.expanduser().absolute()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse output root {output_root}")
    repository_root = args.repository_root.expanduser().resolve()
    installed_wheel = _installed_wheel_identity()
    git = _git_identity(repository_root)
    if git["head"] != args.expected_commit or git["status"]:
        raise RuntimeError("G5 repository identity differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    continuous_simulation = build_simulation(
        application=args.application,
        initial_q_path=input_path,
        output_directory=output_root / "continuous",
        steps=FROZEN_TOTAL_STEPS,
        device=args.device,
    )
    compiled, continuous = _run(continuous_simulation)
    segment_simulation = build_simulation(
        application=args.application,
        initial_q_path=input_path,
        output_directory=output_root / "segment",
        steps=FROZEN_SPLIT_STEPS,
        checkpoint_interval=FROZEN_SPLIT_STEPS,
        device=args.device,
    )
    _segment_compiled, segment = _run(segment_simulation)
    checkpoint = segment.output_directory / f"checkpoint_{FROZEN_SPLIT_STEPS}"
    resumed_simulation = build_simulation(
        application=args.application,
        initial_q_path=input_path,
        output_directory=output_root / "resumed",
        steps=FROZEN_SPLIT_STEPS,
        restart_from=checkpoint,
        device=args.device,
    )
    _resumed_compiled, resumed = _run(resumed_simulation)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    if continuous.final_step != FROZEN_TOTAL_STEPS:
        raise RuntimeError("continuous G5 trajectory did not reach step 100")
    if resumed.final_step != FROZEN_TOTAL_STEPS:
        raise RuntimeError("resumed G5 trajectory did not reach step 100")
    comparisons: dict[str, object] = {}
    all_identical = True
    all_finite = True
    for field in ("Q", "u", "p"):
        left = continuous.output_directory / f"{field}_{FROZEN_TOTAL_STEPS}.npy"
        right = resumed.output_directory / f"{field}_{FROZEN_TOTAL_STEPS}.npy"
        left_record = _artifact(left)
        right_record = _artifact(right)
        identical = left.read_bytes() == right.read_bytes()
        finite = left_record["finite"] is True and right_record["finite"] is True
        all_identical = all_identical and identical
        all_finite = all_finite and finite
        comparisons[field] = {
            "continuous": left_record,
            "resumed": right_record,
            "byte_identical": identical,
            "finite": finite,
        }
    if not all_identical:
        raise RuntimeError("continuous/restart G5 state is not byte-identical")
    if not all_finite:
        raise RuntimeError("G5 state contains NaN or Inf")
    elapsed = float(continuous.elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError("continuous workflow timer is invalid")
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "variant": args.variant,
        "application": args.application,
        "repository": git,
        "environment": {
            "python": platform.python_version(),
            **installed_wheel,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "contract": {
            "shape": list(FROZEN_SHAPE),
            "lengths": list(FROZEN_LENGTHS),
            "dt": FROZEN_DT,
            "total_steps": FROZEN_TOTAL_STEPS,
            "split_steps": FROZEN_SPLIT_STEPS,
            "spectral_refresh": "disabled",
            "initial_q_path": str(input_path),
            "initial_q_sha256": FROZEN_INPUT_SHA256,
        },
        "compiled": compiled.to_metadata(),
        "continuous": _result_metadata(continuous),
        "segment": _result_metadata(segment),
        "resumed": _result_metadata(resumed),
        "comparisons": comparisons,
        "all_byte_identical": all_identical,
        "all_finite": all_finite,
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "timing": {
            "continuous_mean_timestep_seconds": elapsed / FROZEN_TOTAL_STEPS,
            "all_workflows_wall_seconds": wall_seconds,
        },
        "passed": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--application", choices=APPLICATIONS, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--initial-q-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_smoke(args)
    _atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
