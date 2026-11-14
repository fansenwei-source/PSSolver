#!/usr/bin/env python3
"""Persist a scale-aware P9.5 activity-gradient finite-difference diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

import numpy as np
import torch

from benchmarks.profile_periodic_functional import (
    PeriodicFunctionalProfileConfig,
    _activity,
    _allocated_device,
    _simulation,
)
from pssolver.functional import (
    build_functional_runtime,
    evaluate_periodic_activity_gradients,
    periodic_activity_functional_request,
)


DIAGNOSTIC_SCHEMA_VERSION = 1
DIAGNOSTIC_KIND = "p95_activity_gradient_epsilon_diagnostic"
ACTIVITY_EPSILONS = (
    1.0e-4,
    3.0e-4,
    1.0e-3,
    2.0e-3,
    3.0e-3,
    4.0e-3,
    5.0e-3,
    8.0e-3,
    1.0e-2,
)
FROZEN_RELATIVE_TOLERANCE = 2.0e-5


@dataclass(frozen=True, slots=True)
class GradientEpsilonDiagnosticConfig:
    grid_id: str
    trial: int
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    initial_q_path: str
    device: str = "cuda"
    dtype: str = "float64"
    dt: float = 0.001
    base_activity: float = 0.013


def _validate_config(config: GradientEpsilonDiagnosticConfig) -> None:
    if config.grid_id not in {"R128", "R320"}:
        raise ValueError("grid_id must be R128 or R320")
    if (
        not isinstance(config.trial, int)
        or isinstance(config.trial, bool)
        or config.trial <= 0
    ):
        raise ValueError("trial must be a positive integer")
    if len(config.shape) != 3 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 1
        for value in config.shape
    ):
        raise ValueError("shape must contain three integer values greater than one")
    if len(config.lengths) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in config.lengths
    ):
        raise ValueError("lengths must contain three positive finite values")
    if config.dtype != "float64":
        raise ValueError("P9.5 diagnostic is frozen to float64")
    if not math.isfinite(config.dt) or config.dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not math.isfinite(config.base_activity) or config.base_activity <= 0.012:
        raise ValueError("base_activity does not keep the sweep inside its bounds")
    path = Path(config.initial_q_path).expanduser()
    if not path.is_file() or path.name != "Q_0.npy":
        raise FileNotFoundError("initial_q_path must identify an existing Q_0.npy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "root": str(root.resolve()),
        "head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ),
    }


def _activity_direction(value: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(
        value.numel(), dtype=value.dtype, device=value.device
    ).reshape(value.shape)
    result = torch.sin(indices * 0.173 + 2.71)
    scale = result.abs().amax()
    if not bool(torch.isfinite(scale).item()) or float(scale.item()) == 0.0:
        raise RuntimeError("failed to construct the frozen activity direction")
    return result / scale


def _audit_loss(runtime, state, activity: torch.Tensor) -> torch.Tensor:
    next_state, observations = runtime.step_and_observe(
        state, {"activity": activity}, 0
    )
    return (
        next_state[0].square().mean()
        + next_state[1].abs().square().mean()
        + observations["velocity"].square().mean()
        + observations["pressure"].square().mean()
    )


def _objective(runtime, state, activity: torch.Tensor) -> float:
    with torch.no_grad():
        return float(_audit_loss(runtime, state, activity).detach().cpu().item())


def _memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "device_total_bytes": None,
        }
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_total_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
    }


def run_diagnostic(
    config: GradientEpsilonDiagnosticConfig,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    _validate_config(config)
    device = _allocated_device(config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    initial_path = Path(config.initial_q_path).expanduser().resolve()
    values = np.load(initial_path, allow_pickle=False, mmap_mode="r")
    if values.shape != (*config.shape, 5) or values.dtype != np.float64:
        raise ValueError("initial Q shape or dtype differs from diagnostic contract")
    if not bool(np.isfinite(values).all()):
        raise ValueError("initial Q contains NaN or Inf")

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pssolver-p95-gradient-diag-") as td:
        profile_config = PeriodicFunctionalProfileConfig(
            role="functional_vjp",
            trial=config.trial,
            shape=config.shape,
            lengths=config.lengths,
            initial_q_path=config.initial_q_path,
            device=str(device),
            dtype=config.dtype,
            dt=config.dt,
            base_activity=config.base_activity,
        )
        simulation = _simulation(
            profile_config,
            device=device,
            output_directory=Path(td) / "unused-output",
        )
        runtime = build_functional_runtime(
            periodic_activity_functional_request(simulation.specification)
        )
        state = runtime.initial_state()
        controls = _activity(runtime, config.base_activity)
        frozen = evaluate_periodic_activity_gradients(
            runtime, state, controls
        ).to_metadata()

        activity = controls["activity"].detach().clone().requires_grad_(True)
        loss = _audit_loss(runtime, state, activity)
        gradient = torch.autograd.grad(loss, activity, allow_unused=False)[0]
        direction = _activity_direction(activity.detach())
        autograd_value = float(
            (gradient.conj() * direction).real.sum().detach().cpu().item()
        )
        rows = []
        for epsilon in ACTIVITY_EPSILONS:
            lower = activity.detach() - epsilon * direction
            upper = activity.detach() + epsilon * direction
            if bool((lower < 0.0).any().item()):
                raise ValueError("diagnostic epsilon leaves the admissible control set")
            minus_values = [_objective(runtime, state, lower) for _ in range(2)]
            plus_values = [_objective(runtime, state, upper) for _ in range(2)]
            finite_difference = (plus_values[0] - minus_values[0]) / (
                2.0 * epsilon
            )
            absolute_error = abs(autograd_value - finite_difference)
            relative_error = absolute_error / max(
                abs(autograd_value),
                abs(finite_difference),
                torch.finfo(activity.dtype).tiny,
            )
            rows.append(
                {
                    "epsilon": epsilon,
                    "autograd": autograd_value,
                    "objective_minus": minus_values,
                    "objective_plus": plus_values,
                    "finite_difference": finite_difference,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "finite": all(
                        math.isfinite(value)
                        for value in (
                            autograd_value,
                            finite_difference,
                            absolute_error,
                            relative_error,
                            *minus_values,
                            *plus_values,
                        )
                    ),
                    "objective_replay_bitwise": (
                        minus_values[0] == minus_values[1]
                        and plus_values[0] == plus_values[1]
                    ),
                    "passes_frozen_relative_tolerance": (
                        relative_error <= FROZEN_RELATIVE_TOLERANCE
                    ),
                }
            )
        del loss, gradient
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "kind": DIAGNOSTIC_KIND,
        "phase": "P9.5-gradient-diagnostic",
        "config": {
            **asdict(config),
            "shape": list(config.shape),
            "lengths": list(config.lengths),
            "device": str(device),
            "initial_q_path": str(initial_path),
            "initial_q_sha256": _sha256(initial_path),
            "activity_epsilons": list(ACTIVITY_EPSILONS),
            "relative_tolerance": FROZEN_RELATIVE_TOLERANCE,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "git": _git_provenance(root),
        },
        "frozen_validation": frozen,
        "activity_sweep": rows,
        "memory": _memory(device),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "qualification_changed": False,
        "p9_5_pass_claimed": False,
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
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


def _tuple(values: list[str], cast):
    return tuple(cast(value) for value in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-id", choices=("R128", "R320"), required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", nargs=3, required=True)
    parser.add_argument("--lengths", nargs=3, required=True)
    parser.add_argument("--initial-q-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = GradientEpsilonDiagnosticConfig(
        grid_id=args.grid_id,
        trial=args.trial,
        shape=_tuple(args.shape, int),
        lengths=_tuple(args.lengths, float),
        initial_q_path=args.initial_q_path,
        device=args.device,
    )
    _atomic_json(args.output, run_diagnostic(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
