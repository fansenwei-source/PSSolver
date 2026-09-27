#!/usr/bin/env python3
"""Profile the P8.5 finite-Q anchoring relaxation runtime."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
import torch

import pssolver
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.finite_q_anchoring import (
    apply_plane_finite_q_anchoring_pilot,
    lower_plane_finite_q_anchoring_pilot,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    quadratic_finite_q_anchoring,
    uniaxial_Q,
)
from pssolver.runtime.finite_q_anchoring import PlaneFiniteQAnchoringRuntime


PROFILE_SCHEMA_VERSION = 1
PROFILE_MODES = ("scalar_reference", "aggregate_runtime")


@dataclass(frozen=True, slots=True)
class FiniteQAnchoringProfileConfig:
    mode: str = "aggregate_runtime"
    trial: int = 1
    shape: tuple[int, int, int] = (128, 128, 32)
    lengths: tuple[float, float, float] = (100.0, 100.0, 20.0)
    device: str = "cuda"
    dt: float = 0.005
    warmup_steps: int = 5
    profile_steps: int = 30
    seed: int = 20260927
    lower_strength: float = 0.04
    upper_strength: float = 0.07

    def __post_init__(self) -> None:
        if self.mode not in PROFILE_MODES:
            raise ValueError(f"mode must be one of {PROFILE_MODES!r}")
        if not isinstance(self.trial, int) or self.trial <= 0:
            raise ValueError("trial must be a positive integer")
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise ValueError("shape must contain three positive integers")
        if len(self.lengths) != 3 or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.lengths
        ):
            raise ValueError("lengths must be positive and finite")
        if self.device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("device must be cpu, cuda, or cuda:0")
        if self.warmup_steps < 0 or self.profile_steps <= 0:
            raise ValueError("profile window is invalid")
        if self.dt <= 0.0 or not math.isfinite(self.dt):
            raise ValueError("dt must be positive and finite")


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git_provenance(root: Path) -> dict[str, object]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain=v1", "--untracked-files=no")),
    }


def build_runtime(
    config: FiniteQAnchoringProfileConfig,
) -> PlaneFiniteQAnchoringRuntime:
    nx, ny, nz = config.shape
    lx, ly, height = config.lengths
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=Path("p856_unused_output"),
        device=config.device,
        dtype="float64",
        pointwise_execution="eager",
        nx=nx,
        ny=ny,
        nz=nz,
        lx=lx,
        ly=ly,
        height=height,
        steps=config.warmup_steps + config.profile_steps,
        save_start_step=0,
        save_interval=max(1, config.profile_steps),
        diagnostic_interval=max(1, config.profile_steps),
        defect_min_separation=2.0,
        defect_core_radius=0.5,
        twist_modes=(1, 2, 3),
    )
    base = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )
    anchoring = quadratic_finite_q_anchoring(
        k_q=base.equation_system.parameters["ldg_l1"],
        wall_strengths={
            (2, "lower"): config.lower_strength,
            (2, "upper"): config.upper_strength,
        },
        target_q={
            (2, "lower"): uniaxial_Q(np.asarray((0.0, 0.0, 1.0)), 0.6),
            (2, "upper"): uniaxial_Q(np.asarray((1.0, 0.0, 0.0)), 0.5),
        },
    )
    simulation = apply_plane_finite_q_anchoring_pilot(base, anchoring)
    plan = lower_plane_finite_q_anchoring_pilot(simulation, anchoring)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    initial = {
        component: (
            0.03
            * torch.randn(
                config.shape,
                generator=generator,
                dtype=torch.float64,
            )
        ).to(config.device)
        for component in Q_COMPONENTS
    }
    return PlaneFiniteQAnchoringRuntime(
        plan,
        initial,
        dt=config.dt,
    )


def _advance_scalar_reference(
    runtime: PlaneFiniteQAnchoringRuntime,
    steps: int,
) -> None:
    mass = 1.0 / (runtime.dt * runtime.k_q)
    for _ in range(steps):
        candidates = {}
        for component, scalar in runtime.components.items():
            physical = scalar.physical_observation()
            candidates[component] = scalar.solve_bounded_helmholtz(
                physical * mass,
                mass=mass,
            ).contiguous()
        for component, scalar in runtime.components.items():
            scalar.replace_physical_observation(candidates[component])
        for scalar in runtime.components.values():
            scalar.state.progress.commit_step(refreshed=False)


def _advance(runtime, mode: str, steps: int) -> None:
    if mode == "aggregate_runtime":
        runtime.advance(steps)
    else:
        _advance_scalar_reference(runtime, steps)


def profile(config: FiniteQAnchoringProfileConfig) -> dict[str, object]:
    runtime = build_runtime(config)
    device = runtime.device
    initial_sha256 = _tensor_sha256(runtime.physical_q())
    _advance(runtime, config.mode, config.warmup_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    runtime.reset_operation_counts()
    samples = []
    for _ in range(config.profile_steps):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            _advance(runtime, config.mode, 1)
            stop.record()
            stop.synchronize()
            samples.append(float(start.elapsed_time(stop)) / 1000.0)
        else:
            start_time = time.perf_counter()
            _advance(runtime, config.mode, 1)
            samples.append(time.perf_counter() - start_time)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        memory_stats = torch.cuda.memory_stats(device)
        memory = {
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_active_bytes": int(
                memory_stats.get("active_bytes.all.peak", 0)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
            "device_total_bytes": int(
                torch.cuda.get_device_properties(device).total_memory
            ),
        }
        cuda = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "allocated_device": str(device),
            "name": torch.cuda.get_device_name(device),
        }
    else:
        memory = {
            "peak_allocated_bytes": 0,
            "peak_active_bytes": 0,
            "peak_reserved_bytes": 0,
            "device_total_bytes": 0,
        }
        cuda = {"available": torch.cuda.is_available()}
    counts = runtime.operation_counts()
    per_step = {
        name: value / config.profile_steps for name, value in counts.items()
    }
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    standard_deviation = (
        statistics.stdev(samples) if len(samples) > 1 else 0.0
    )
    root = Path(__file__).resolve().parents[1]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "phase": "P8.5.6",
        "kind": "finite_q_anchoring_profile",
        "config": asdict(config),
        "timing": {
            "samples_seconds": samples,
            "mean_timestep_seconds": mean,
            "median_timestep_seconds": median,
            "sample_standard_deviation_seconds": standard_deviation,
            "coefficient_of_variation": standard_deviation / mean,
            "timesteps_per_second": 1.0 / mean,
        },
        "memory": memory,
        "operation_counts": {
            "raw": counts,
            "per_step": per_step,
        },
        "conditioning": {
            "maximum_basis_condition_number": (
                runtime.maximum_condition_number()
            ),
            "limit": 1.0e10,
        },
        "runtime": {
            "requested": "finite_q_anchoring_relaxation_pilot",
            "effective": "finite_q_anchoring_relaxation_pilot",
            "fallback_allowed": False,
            "fallback_used": False,
            "graph_breaks": 0,
            "construction_in_timed_loop": False,
            "root_solve_in_timed_loop": False,
            "matrix_factorization_in_timed_loop": False,
        },
        "initial_q_sha256": initial_sha256,
        "final_q_sha256": _tensor_sha256(runtime.physical_q()),
        "completed_steps": runtime.completed_steps,
        "finite": bool(torch.isfinite(runtime.physical_q()).all().item()),
        "cuda": cuda,
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "pssolver_import": str(Path(pssolver.__file__).resolve()),
        "git": _git_provenance(root),
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=PROFILE_MODES, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", type=int, nargs=3, required=True)
    parser.add_argument("--lengths", type=float, nargs=3, default=(100, 100, 20))
    parser.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), required=True)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FiniteQAnchoringProfileConfig(
        mode=args.mode,
        trial=args.trial,
        shape=tuple(args.shape),
        lengths=tuple(args.lengths),
        device=args.device,
        dt=args.dt,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        seed=args.seed,
    )
    write_report(args.output, profile(config))


if __name__ == "__main__":
    main()
