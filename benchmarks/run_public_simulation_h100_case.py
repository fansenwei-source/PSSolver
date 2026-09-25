#!/usr/bin/env python3
"""Run one auditable P7.7.12 direct/public CUDA application case.

The executable deliberately measures an existing application entry and the
public ``run_simulation`` entry with the same declaration.  It does not add a
new runtime, change a production default, or dispatch inside the timestep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
    run_simulation,
)
from pssolver.applications.channel_active_nematics import (
    run_channel_active_nematics,
)
from pssolver.applications.plane_beris_edwards import (
    run_plane_beris_edwards,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_APPLICATION,
    PUBLIC_PLANE_APPLICATION,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import CompleteStressBerisEdwards


APPLICATIONS = ("plane", "channel")
ENTRIES = ("direct", "public")
RUNTIME_PATHS = {
    "plane": ("legacy_production", "compiled_v2"),
    "channel": ("legacy_channel", "compiled_channel_v2"),
}


def _triplet(value: str, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: list[str]) -> str:
    completed = subprocess.run(
        ["git", *command],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_provenance() -> dict[str, object]:
    return {
        "head": _git(["rev-parse", "HEAD"]),
        "status_porcelain": _git(
            ["status", "--porcelain=v1", "--untracked-files=all"]
        ),
    }


def _plane_simulation(args: argparse.Namespace) -> Simulation:
    model = CompleteStressBerisEdwards(
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
    geometry = PlaneSlab(shape=args.shape, lengths=args.lengths)
    hermitian_axis = 1 if args.spectral_storage == "hermitian_half" else None
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": free_slip_velocity(),
                "pressure": neumann_pressure_compatibility(),
            },
        ),
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage=args.spectral_storage,
            hermitian_axis=hermitian_axis,
        ),
        time=TimeStepping(
            dt=args.dt,
            refresh={"mode": "disabled"},
        ),
        initial_condition=GeneratedInitialCondition(
            "extruded_defect_gas",
            parameters={
                "seed": args.seed,
                "num_defect_pairs": 6,
                "defect_min_separation": 10.0,
                "defect_core_radius": 1.5,
                "background_angle": 0.0,
                "twist_amplitude": 0.01,
                "twist_modes": [1, 2, 3],
                "initial_s": 1.0 / 3.0,
            },
        ),
        execution=TorchSpectralExecution(
            runtime_path=args.runtime_path,
            device=args.device,
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": args.pointwise_execution,
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=args.output_directory,
            steps=args.steps,
            save_interval=args.save_interval,
            diagnostic_interval=args.diagnostic_interval,
            checkpoint_interval=args.checkpoint_interval,
            restart_from=args.restart_from,
        ),
        invocation=InvocationSpec({}),
    )


def _channel_simulation(args: argparse.Namespace) -> Simulation:
    request = ChannelActiveNematicRunSpec(
        shape=args.shape,
        lengths=args.lengths,
        dt=args.dt,
        steps=args.steps,
        save_interval=args.save_interval,
        diagnostic_interval=args.diagnostic_interval,
        checkpoint_interval=args.checkpoint_interval,
        restart_from=args.restart_from,
        generated_output_directory=args.output_directory,
        snapshot_output_directory=args.output_directory.with_name(
            f"{args.output_directory.name}_snapshot"
        ),
        seed=args.seed,
        device=args.device,
        dtype="float32",
        activity=args.activity,
        pressure_relative_tolerance=1.0e-6,
        pressure_max_iterations=80,
        runtime_path=args.runtime_path,
    )
    source = compose_channel_active_nematics_simulation(request.components)
    return Simulation(
        model=source.equation_system,
        geometry=source.geometry,
        boundaries=source.boundaries,
        numerics=source.numerics,
        time=source.time_integration,
        discretization=source.discretization_parameters,
        initial_condition=source.initial_condition,
        execution=source.execution,
        output=source.workflow,
        invocation=source.invocation,
    )


def build_simulation(args: argparse.Namespace) -> Simulation:
    """Build the exact public declaration selected by one CLI case."""

    if args.application == "plane":
        return _plane_simulation(args)
    return _channel_simulation(args)


def _run_direct(compiled, progress):
    if compiled.application == PUBLIC_PLANE_APPLICATION:
        return run_plane_beris_edwards(
            compiled.application_request,
            progress=progress,
        )
    if compiled.application == PUBLIC_CHANNEL_APPLICATION:
        return run_channel_active_nematics(
            compiled.application_request,
            progress=progress,
        )
    raise RuntimeError("unexpected qualified application")


def _artifact_manifest(directory: Path, final_step: int) -> dict[str, object]:
    required = (
        f"Q_{final_step}.npy",
        f"u_{final_step}.npy",
        f"p_{final_step}.npy",
        "diagnostics.npy",
        "diagnostics.csv",
        "COMPLETE",
    )
    records: dict[str, object] = {}
    finite = True
    for name in required:
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"required application artifact is absent: {path}")
        record: dict[str, object] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".npy":
            array = np.load(path, allow_pickle=False)
            record.update(shape=list(array.shape), dtype=str(array.dtype))
            if name.startswith(("Q_", "u_", "p_")):
                finite = finite and bool(np.isfinite(array).all())
        records[name] = record
    metadata = json.loads((directory / "metadata.json").read_text())
    return {
        "records": records,
        "finite": finite,
        "runtime_selection": metadata["runtime_selection"],
        "completed_steps": metadata["completed_steps"],
    }


def _checkpoint_manifest(
    directory: Path,
    checkpoint_steps: tuple[int, ...],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for step in checkpoint_steps:
        root = directory / f"checkpoint_{step}"
        if not root.is_dir():
            raise RuntimeError(f"reported checkpoint is absent: {root}")
        result[str(step)] = {
            str(path.relative_to(root)): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
    return result


def run_case(args: argparse.Namespace) -> dict[str, object]:
    if args.runtime_path not in RUNTIME_PATHS[args.application]:
        raise ValueError(
            f"runtime {args.runtime_path!r} is invalid for {args.application}"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.output_directory.exists():
        raise FileExistsError(f"output already exists: {args.output_directory}")
    if args.report.exists():
        raise FileExistsError(f"report already exists: {args.report}")
    if args.application == "plane" and args.activity != 5.0:
        raise ValueError("--activity is a Channel-only qualification control")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    simulation = build_simulation(args)
    compiled = compile_simulation(simulation)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.perf_counter()
    progress = range(args.steps)
    if args.entry == "direct":
        application_result = _run_direct(compiled, progress)
        result = application_result
        public_metadata = None
    else:
        result = run_simulation(compiled, progress=progress)
        application_result = result
        public_metadata = result.to_metadata()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_started
    final_step = int(result.final_step)
    output = args.output_directory.expanduser().resolve()
    artifacts = _artifact_manifest(output, final_step)
    runtime = artifacts["runtime_selection"]
    if runtime["requested"] != args.runtime_path:
        raise RuntimeError("requested runtime differs from output metadata")
    if runtime["effective"] != args.runtime_path:
        raise RuntimeError("effective runtime differs from requested runtime")
    if runtime["fallback_used"] is not False:
        raise RuntimeError("runtime fallback occurred")
    if artifacts["finite"] is not True:
        raise RuntimeError("non-finite application output")
    if int(artifacts["completed_steps"]) != final_step:
        raise RuntimeError("metadata completion clock differs from result")
    elapsed_seconds = float(result.elapsed_seconds)
    completed_this_call = final_step - int(result.start_step)
    if completed_this_call <= 0 or not math.isfinite(elapsed_seconds):
        raise RuntimeError("invalid workflow timer or completion clock")
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0
    )
    return {
        "schema_version": 1,
        "phase": "P7.7.12",
        "case": {
            "case_id": args.case_id,
            "kind": args.kind,
            "trial": args.trial,
            "application": args.application,
            "runtime_path": args.runtime_path,
            "entry": args.entry,
            "shape": list(args.shape),
            "lengths": list(args.lengths),
            "steps_requested": args.steps,
            "spectral_storage": (
                args.spectral_storage if args.application == "plane" else None
            ),
        },
        "environment": {
            "git": _git_provenance(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "compiled_simulation": compiled.to_metadata(),
        "result": {
            "start_step": int(result.start_step),
            "final_step": final_step,
            "saved_steps": list(result.saved_steps),
            "checkpoint_steps": list(result.checkpoint_steps),
            "elapsed_seconds": elapsed_seconds,
            "mean_timestep_seconds": elapsed_seconds / completed_this_call,
            "wall_seconds": wall_seconds,
            "public_metadata": public_metadata,
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "artifacts": artifacts,
        "checkpoints": _checkpoint_manifest(
            output,
            tuple(result.checkpoint_steps),
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--kind",
        choices=("performance", "continuous", "segment", "resume"),
        required=True,
    )
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--application", choices=APPLICATIONS, required=True)
    parser.add_argument("--runtime-path", required=True)
    parser.add_argument("--entry", choices=ENTRIES, required=True)
    parser.add_argument("--shape", type=lambda value: _triplet(value, int), required=True)
    parser.add_argument("--lengths", type=lambda value: _triplet(value, float), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--activity", type=float, default=5.0)
    parser.add_argument("--save-interval", type=int, required=True)
    parser.add_argument("--diagnostic-interval", type=int, required=True)
    parser.add_argument("--checkpoint-interval", type=int)
    parser.add_argument("--restart-from", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--spectral-storage",
        choices=("full_complex", "hermitian_half"),
        default="hermitian_half",
    )
    parser.add_argument(
        "--pointwise-execution",
        choices=("eager", "compile"),
        default="compile",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.trial < 0:
        parser.error("--trial must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_case(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
