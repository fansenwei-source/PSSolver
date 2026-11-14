#!/usr/bin/env python3
"""Diagnose numerically stable P9.5 directional-gradient contracts.

This is an analysis-only tool.  It does not replace the frozen validator in
``pssolver.functional.validation`` and cannot qualify P9.5.
"""

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
DIAGNOSTIC_KIND = "p95_gradient_validator_v2_diagnostic"
DIRECTION_KINDS = (
    "frozen_oscillatory",
    "low_mode",
    "gradient_aligned",
)
METHODS = ("legacy_scalar", "paired_quadratic")
STATE_EPSILONS = (1.0e-7, 3.0e-7, 1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5, 1.0e-4)
ACTIVITY_EPSILONS = (
    1.0e-5,
    3.0e-5,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    3.0e-3,
    1.0e-2,
)
REFERENCE_RELATIVE_TOLERANCE = 2.0e-5


@dataclass(frozen=True, slots=True)
class GradientValidatorV2DiagnosticConfig:
    grid_id: str
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    initial_q_path: str
    device: str = "cuda"
    dtype: str = "float64"
    dt: float = 0.001
    base_activity: float = 0.013


def _validate_config(config: GradientValidatorV2DiagnosticConfig) -> None:
    if config.grid_id not in {"R128", "R320"}:
        raise ValueError("grid_id must be R128 or R320")
    # Small shapes remain useful for CPU contract tests, while formal evidence
    # is constrained by the fail-closed analyzer.
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
        raise ValueError("P9.5 validator-v2 diagnostic is frozen to float64")
    if not math.isfinite(config.dt) or config.dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not math.isfinite(config.base_activity) or config.base_activity <= 0.01:
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


def _normalize_direction(value: torch.Tensor) -> torch.Tensor:
    scale = value.abs().amax()
    if not bool(torch.isfinite(scale).item()) or float(scale.item()) == 0.0:
        raise RuntimeError("failed to construct a finite nonzero direction")
    return value / scale


def _frozen_direction(value: torch.Tensor, phase: float) -> torch.Tensor:
    real_dtype = value.real.dtype if value.is_complex() else value.dtype
    indices = torch.arange(
        value.numel(), dtype=real_dtype, device=value.device
    ).reshape(value.shape)
    real = torch.sin(indices * 0.173 + phase)
    if value.is_complex():
        result = torch.complex(real, torch.cos(indices * 0.119 + phase))
    else:
        result = real
    return _normalize_direction(result)


def _low_mode_direction(value: torch.Tensor, phase: float) -> torch.Tensor:
    real_dtype = value.real.dtype if value.is_complex() else value.dtype
    real = torch.zeros(value.shape, dtype=real_dtype, device=value.device)
    imaginary = torch.zeros(value.shape, dtype=real_dtype, device=value.device)
    for axis, size in enumerate(value.shape):
        coordinate = torch.arange(size, dtype=real_dtype, device=value.device)
        view = [1] * value.ndim
        view[axis] = size
        coordinate = coordinate.reshape(view)
        angle = (
            (2.0 * torch.pi * coordinate / float(size))
            + phase
            + 0.37 * axis
        )
        real = real + torch.cos(angle)
        imaginary = imaginary + 0.5 * torch.sin(angle + 0.19)
    if value.is_complex():
        return _normalize_direction(torch.complex(real, imaginary))
    return _normalize_direction(real)


def _gradient_aligned_direction(gradient: torch.Tensor | None) -> torch.Tensor:
    if gradient is None:
        raise RuntimeError("gradient-aligned direction requires a gradient")
    return _normalize_direction(gradient.detach().clone())


def _audit_outputs(runtime, state, activity: torch.Tensor) -> tuple[torch.Tensor, ...]:
    next_state, observations = runtime.step_and_observe(
        state, {"activity": activity}, 0
    )
    return (
        next_state[0],
        next_state[1],
        observations["velocity"],
        observations["pressure"],
    )


def _quadratic_objective(outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return sum(value.abs().square().mean() for value in outputs)


def _paired_quadratic_derivative(
    plus: tuple[torch.Tensor, ...],
    minus: tuple[torch.Tensor, ...],
    epsilon: float,
) -> torch.Tensor:
    """Evaluate the same quadratic central difference before global reduction."""

    if len(plus) != len(minus) or not plus:
        raise ValueError("plus/minus output collections must be nonempty and aligned")
    terms = []
    for plus_value, minus_value in zip(plus, minus, strict=True):
        if plus_value.shape != minus_value.shape:
            raise ValueError("plus/minus output shapes differ")
        terms.append(
            ((plus_value - minus_value).conj() * (plus_value + minus_value))
            .real.mean()
        )
    return sum(terms) / (2.0 * epsilon)


def _directional_product(
    gradients: tuple[torch.Tensor | None, ...],
    directions: tuple[torch.Tensor, ...],
) -> float:
    if len(gradients) != len(directions):
        raise ValueError("gradient/direction collections differ")
    total = 0.0
    for gradient, direction in zip(gradients, directions, strict=True):
        if gradient is None:
            raise RuntimeError("directional product encountered a missing gradient")
        total += float(
            (gradient.conj() * direction).real.sum().detach().cpu().item()
        )
    return total


def _directional_conditioning(
    gradients: tuple[torch.Tensor | None, ...],
    directions: tuple[torch.Tensor, ...],
) -> float:
    numerator = abs(_directional_product(gradients, directions))
    gradient_norm_squared = 0.0
    direction_norm_squared = 0.0
    for gradient, direction in zip(gradients, directions, strict=True):
        if gradient is None:
            raise RuntimeError(
                "directional conditioning encountered a missing gradient"
            )
        gradient_norm_squared += float(
            gradient.abs().square().sum().detach().cpu().item()
        )
        direction_norm_squared += float(
            direction.abs().square().sum().detach().cpu().item()
        )
    denominator = math.sqrt(gradient_norm_squared * direction_norm_squared)
    if denominator == 0.0:
        return 0.0
    return min(1.0, numerator / denominator)


def _relative_error(reference: float, estimate: float, dtype: torch.dtype) -> float:
    return abs(reference - estimate) / max(
        abs(reference), abs(estimate), torch.finfo(dtype).tiny
    )


def _evaluate_sweep(
    runtime,
    state,
    activity: torch.Tensor,
    *,
    target: str,
    direction_kind: str,
    directions: tuple[torch.Tensor, ...],
    autograd_value: float,
    direction_cosine: float,
    epsilons: tuple[float, ...],
    base_objective: float,
) -> dict[str, object]:
    rows = []
    for epsilon in epsilons:
        if target == "state":
            plus_state = tuple(
                value + epsilon * direction
                for value, direction in zip(state, directions, strict=True)
            )
            minus_state = tuple(
                value - epsilon * direction
                for value, direction in zip(state, directions, strict=True)
            )
            plus_activity = minus_activity = activity
        elif target == "activity":
            if len(directions) != 1:
                raise ValueError("activity sweep requires exactly one direction")
            plus_state = minus_state = state
            plus_activity = activity + epsilon * directions[0]
            minus_activity = activity - epsilon * directions[0]
            if bool((minus_activity < 0.0).any().item()):
                raise ValueError("epsilon leaves the admissible activity set")
        else:
            raise ValueError("target must be state or activity")

        with torch.no_grad():
            plus_outputs = tuple(
                value.detach()
                for value in _audit_outputs(runtime, plus_state, plus_activity)
            )
            minus_outputs = tuple(
                value.detach()
                for value in _audit_outputs(runtime, minus_state, minus_activity)
            )
            plus_objective = float(
                _quadratic_objective(plus_outputs).detach().cpu().item()
            )
            minus_objective = float(
                _quadratic_objective(minus_outputs).detach().cpu().item()
            )
            legacy = (plus_objective - minus_objective) / (2.0 * epsilon)
            paired = float(
                _paired_quadratic_derivative(
                    plus_outputs, minus_outputs, epsilon
                ).detach().cpu().item()
            )
        methods = {}
        for name, estimate in (("legacy_scalar", legacy), ("paired_quadratic", paired)):
            absolute = abs(autograd_value - estimate)
            relative = _relative_error(autograd_value, estimate, activity.dtype)
            methods[name] = {
                "finite_difference": estimate,
                "absolute_error": absolute,
                "relative_error": relative,
                "finite": all(
                    math.isfinite(value)
                    for value in (estimate, absolute, relative)
                ),
                "passes_reference_tolerance": (
                    relative <= REFERENCE_RELATIVE_TOLERANCE
                ),
            }
        plus_remainder = abs(
            plus_objective - base_objective - epsilon * autograd_value
        )
        minus_remainder = abs(
            minus_objective - base_objective + epsilon * autograd_value
        )
        rows.append(
            {
                "epsilon": epsilon,
                "objective_plus": plus_objective,
                "objective_minus": minus_objective,
                "one_sided_linearization_remainder": {
                    "plus": plus_remainder,
                    "minus": minus_remainder,
                    "normalized_max": max(plus_remainder, minus_remainder)
                    / max(
                        epsilon * abs(autograd_value),
                        torch.finfo(activity.dtype).tiny,
                    ),
                },
                "methods": methods,
            }
        )
    return {
        "target": target,
        "direction": direction_kind,
        "autograd": autograd_value,
        "autograd_nonzero": autograd_value != 0.0,
        "absolute_direction_cosine": direction_cosine,
        "direction_max_abs": [
            float(value.abs().amax().detach().cpu().item()) for value in directions
        ],
        "rows": rows,
    }


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
    config: GradientValidatorV2DiagnosticConfig,
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
    with tempfile.TemporaryDirectory(prefix="pssolver-p95-validator-v2-") as td:
        profile_config = PeriodicFunctionalProfileConfig(
            role="functional_vjp",
            trial=1,
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
        activity = controls["activity"].detach()
        frozen = evaluate_periodic_activity_gradients(
            runtime, state, controls
        ).to_metadata()

        state_for_ad = tuple(
            value.detach().clone().requires_grad_(True) for value in state
        )
        activity_for_ad = activity.detach().clone().requires_grad_(True)
        outputs = _audit_outputs(runtime, state_for_ad, activity_for_ad)
        loss = _quadratic_objective(outputs)
        gradients = torch.autograd.grad(
            loss, (*state_for_ad, activity_for_ad), allow_unused=True
        )
        base_objective = float(loss.detach().cpu().item())

        state_directions = {
            "frozen_oscillatory": tuple(
                _frozen_direction(value, 0.31 + index)
                for index, value in enumerate(state)
            ),
            "low_mode": tuple(
                _low_mode_direction(value, 0.31 + index)
                for index, value in enumerate(state)
            ),
            "gradient_aligned": tuple(
                _gradient_aligned_direction(value) for value in gradients[:2]
            ),
        }
        activity_directions = {
            "frozen_oscillatory": (_frozen_direction(activity, 2.71),),
            "low_mode": (_low_mode_direction(activity, 2.71),),
            "gradient_aligned": (
                _gradient_aligned_direction(gradients[2]),
            ),
        }
        sweeps = []
        for direction_kind in DIRECTION_KINDS:
            directions = state_directions[direction_kind]
            sweeps.append(
                _evaluate_sweep(
                    runtime,
                    state,
                    activity,
                    target="state",
                    direction_kind=direction_kind,
                    directions=directions,
                    autograd_value=_directional_product(
                        gradients[:2], directions
                    ),
                    direction_cosine=_directional_conditioning(
                        gradients[:2], directions
                    ),
                    epsilons=STATE_EPSILONS,
                    base_objective=base_objective,
                )
            )
            directions = activity_directions[direction_kind]
            sweeps.append(
                _evaluate_sweep(
                    runtime,
                    state,
                    activity,
                    target="activity",
                    direction_kind=direction_kind,
                    directions=directions,
                    autograd_value=_directional_product(
                        (gradients[2],), directions
                    ),
                    direction_cosine=_directional_conditioning(
                        (gradients[2],), directions
                    ),
                    epsilons=ACTIVITY_EPSILONS,
                    base_objective=base_objective,
                )
            )

        with torch.no_grad():
            replay_a = tuple(
                value.detach() for value in _audit_outputs(runtime, state, activity)
            )
            replay_b = tuple(
                value.detach() for value in _audit_outputs(runtime, state, activity)
            )
        replay_bitwise = all(
            torch.equal(left, right)
            for left, right in zip(replay_a, replay_b, strict=True)
        )
        all_finite = all(value is not None for value in gradients) and all(
            bool(torch.isfinite(value).all().item())
            for value in (*outputs, *gradients)
            if value is not None
        )
        del outputs, loss, gradients, replay_a, replay_b
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "kind": DIAGNOSTIC_KIND,
        "phase": "P9.5-gradient-validator-v2-diagnostic",
        "config": {
            **asdict(config),
            "shape": list(config.shape),
            "lengths": list(config.lengths),
            "device": str(device),
            "initial_q_path": str(initial_path),
            "initial_q_sha256": _sha256(initial_path),
            "state_epsilons": list(STATE_EPSILONS),
            "activity_epsilons": list(ACTIVITY_EPSILONS),
            "direction_kinds": list(DIRECTION_KINDS),
            "methods": list(METHODS),
            "reference_relative_tolerance": REFERENCE_RELATIVE_TOLERANCE,
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
        "base_objective": base_objective,
        "sweeps": sweeps,
        "all_finite": all_finite,
        "objective_replay_bitwise": replay_bitwise,
        "memory": _memory(device),
        "elapsed_wall_seconds": time.perf_counter() - started,
        "qualification_changed": False,
        "p9_5_pass_claimed": False,
        "p9_6_authorized": False,
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
    parser.add_argument("--shape", nargs=3, required=True)
    parser.add_argument("--lengths", nargs=3, required=True)
    parser.add_argument("--initial-q-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = GradientValidatorV2DiagnosticConfig(
        grid_id=args.grid_id,
        shape=_tuple(args.shape, int),
        lengths=_tuple(args.lengths, float),
        initial_q_path=args.initial_q_path,
        device=args.device,
    )
    _atomic_json(args.output, run_diagnostic(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
