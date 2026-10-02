#!/usr/bin/env python3
"""Audit long-horizon Hermitian consistency of the periodic functional state."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import torch

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.functional.api import (
    build_functional_runtime,
    periodic_activity_functional_request,
)
from pssolver.geometries import PeriodicBox
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.models.active_nematics.q_tensor import positive_equilibrium_S


CHECK_SCHEMA_VERSION = 1
CHECK_KIND = "periodic_functional_hermitian_stability"
CASES = {
    "loop3d": {
        "material": {
            "ldg_a": -1.0,
            "ldg_b": -6.0,
            "ldg_c": 6.0,
            "ldg_l1": 1.0,
            "gamma": 1.0,
            "flow_alignment": 1.0,
            "viscosity": 1.0,
            "beta": -1.0,
            "friction": 0.0,
            "tangential_zero_mode_policy": "zero_mean",
        },
        "shape": (32, 32, 8),
        "lengths": (32.0, 32.0, 8.0),
        "director": (1.0, 0.0, 0.0),
        "activity": 0.05,
    },
    "r1": {
        "material": {
            "ldg_a": -0.545651,
            "ldg_b": -0.3,
            "ldg_c": 0.830717,
            "ldg_l1": 2.0,
            "gamma": 1.0,
            "flow_alignment": 1.0,
            "viscosity": 1.0,
            "beta": -1.0,
            "friction": 0.01,
            "tangential_zero_mode_policy": "friction",
        },
        "shape": (200, 100, 1),
        "lengths": (100.0, 50.0, 1.0),
        "director": (1.0, 1.0, 0.0),
        "activity": 0.1,
    },
}


@dataclass(frozen=True, slots=True)
class HermitianStabilityConfig:
    case: str = "loop3d"
    device: str = "cpu"
    spectral_storage: str = "hermitian_half"
    dt: float = 0.02
    horizon: float = 200.0
    noise: float = 1.0e-3
    sample_interval: float = 5.0
    tolerance_factor: float = 64.0


def _validate_config(config: HermitianStabilityConfig) -> None:
    if config.case not in CASES:
        raise ValueError(f"case must be one of {tuple(CASES)}")
    if config.spectral_storage not in ("full_complex", "hermitian_half"):
        raise ValueError(
            "spectral_storage must be full_complex or hermitian_half"
        )
    for name in ("dt", "horizon", "sample_interval", "tolerance_factor"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not math.isfinite(config.noise) or config.noise < 0.0:
        raise ValueError("noise must be non-negative and finite")
    steps = config.horizon / config.dt
    interval = config.sample_interval / config.dt
    if not math.isclose(steps, round(steps), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("horizon must be an integer number of timesteps")
    if not math.isclose(interval, round(interval), rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("sample_interval must be an integer number of timesteps")


def _allocated_device(value: str) -> torch.device:
    requested = torch.device(value)
    if requested.type != "cuda":
        return requested
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device is unavailable")
    return torch.device("cuda", index)


def _uniform_q(order: float, director) -> np.ndarray:
    unit = np.asarray(director, dtype=np.float64)
    unit /= np.linalg.norm(unit)
    tensor = 1.5 * order * (
        np.outer(unit, unit) - np.eye(3, dtype=np.float64) / 3.0
    )
    return np.array(
        (tensor[0, 0], tensor[0, 1], tensor[0, 2], tensor[1, 1], tensor[1, 2])
    )


def _initial_q(case, noise: float) -> np.ndarray:
    shape = case["shape"]
    material = case["material"]
    order = float(
        positive_equilibrium_S(
            material["ldg_a"],
            material["ldg_b"],
            material["ldg_c"],
        )
    )
    values = np.broadcast_to(
        _uniform_q(order, case["director"])[:, None, None, None],
        (5, *shape),
    ).copy()
    values += np.random.default_rng(0).normal(scale=noise, size=values.shape)
    if shape[2] == 1:
        values[[2, 4]] = 0.0
    return values


def _runtime(config, case, device, root: Path):
    material = case["material"]
    model = CompleteStressBerisEdwards(
        **material,
        activity=case["activity"],
    )
    geometry = PeriodicBox(shape=case["shape"], lengths=case["lengths"])
    np.save(root / "Q_0.npy", _initial_q(case, config.noise), allow_pickle=False)
    simulation = Simulation(
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
            spectral_storage=config.spectral_storage,
            hermitian_axis=(
                1 if config.spectral_storage == "hermitian_half" else None
            ),
        ),
        time=TimeStepping(dt=config.dt, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(root, step=0),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device=str(device),
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": True,
            },
        ),
        output=Output(
            directory=root / "unused-output",
            steps=1,
            save_interval=1,
            diagnostic_interval=1,
        ),
    )
    return build_functional_runtime(
        periodic_activity_functional_request(simulation.specification)
    )


def _reverse_index(size: int, device: torch.device) -> torch.Tensor:
    return torch.remainder(-torch.arange(size, device=device), size)


def _packed_plane_violation(spectral, shape) -> torch.Tensor:
    indices = [0]
    if shape[1] % 2 == 0:
        indices.append(shape[1] // 2)
    planes = spectral.index_select(
        -2,
        torch.tensor(indices, device=spectral.device),
    )
    reflected = planes.index_select(
        -3,
        _reverse_index(shape[0], spectral.device),
    ).index_select(
        -1,
        _reverse_index(shape[2], spectral.device),
    )
    return (planes - reflected.conj()).abs().max()


def _full_spectrum_violation(spectral, shape) -> torch.Tensor:
    reflected = spectral
    for tensor_axis, size in zip((-3, -2, -1), shape):
        reflected = reflected.index_select(
            tensor_axis,
            _reverse_index(size, spectral.device),
        )
    return (spectral - reflected.conj()).abs().max()


def _hermitian_violation(spectral, shape, storage) -> torch.Tensor:
    if storage == "hermitian_half":
        return _packed_plane_violation(spectral, shape)
    return _full_spectrum_violation(spectral, shape)


def run_stability_check(config: HermitianStabilityConfig) -> dict[str, object]:
    """Run the frozen case and return a JSON-compatible audit report."""

    _validate_config(config)
    case = CASES[config.case]
    device = _allocated_device(config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    report: dict[str, object] = {
        "schema_version": CHECK_SCHEMA_VERSION,
        "kind": CHECK_KIND,
        "config": asdict(config),
        "case": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in case.items()
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "samples": [],
        "failure": None,
        "passed": False,
    }
    steps = round(config.horizon / config.dt)
    sample_steps = round(config.sample_interval / config.dt)
    completed_steps = 0
    with tempfile.TemporaryDirectory(prefix="pssolver-hermitian-") as temporary:
        runtime = _runtime(config, case, device, Path(temporary))
        identity = runtime.identity()
        report["functional_runtime_identity_sha256"] = identity.canonical_sha256()
        report["hermitian_state_projection"] = identity.to_metadata()[
            "execution"
        ]["functional_runtime"].get("hermitian_state_projection")
        state = runtime.initial_state()
        control_spec = runtime.control_specs[0]
        controls = {
            "activity": torch.full(
                control_spec.tensor.shape,
                case["activity"],
                device=state[0].device,
                dtype=torch.float64,
            )
        }
        with torch.no_grad():
            for step in range(steps):
                try:
                    state = runtime.step(state, controls, step)
                except Exception as error:  # preserve an auditable failure record
                    report["failure"] = {
                        "step": step,
                        "time": step * config.dt,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    break
                completed_steps = step + 1
                if completed_steps % sample_steps != 0 and completed_steps != steps:
                    continue
                finite = all(bool(torch.isfinite(value).all()) for value in state)
                violation = float(
                    _hermitian_violation(
                        state[1],
                        case["shape"],
                        config.spectral_storage,
                    )
                )
                maximum = float(state[1].abs().max())
                bound = config.tolerance_factor * torch.finfo(
                    state[0].dtype
                ).eps * max(1.0, maximum)
                sample = {
                    "step": completed_steps,
                    "time": completed_steps * config.dt,
                    "finite": finite,
                    "hermitian_violation": violation,
                    "violation_bound": bound,
                    "q_physical_max_abs": float(state[0].abs().max()),
                    "q_spectral_max_abs": maximum,
                }
                report["samples"].append(sample)
                if not finite or violation > bound:
                    report["failure"] = {
                        "step": completed_steps,
                        "time": completed_steps * config.dt,
                        "error": "Hermitian consistency or finite-state gate failed",
                        "sample": sample,
                    }
                    break

    report["completed_steps"] = completed_steps
    report["completed_time"] = completed_steps * config.dt
    report["passed"] = report["failure"] is None and completed_steps == steps
    return report


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="loop3d")
    parser.add_argument(
        "--spectral-storage",
        choices=("full_complex", "hermitian_half"),
        default="hermitian_half",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--horizon", type=float, default=200.0)
    parser.add_argument("--noise", type=float, default=1.0e-3)
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--tolerance-factor", type=float, default=64.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_stability_check(
        HermitianStabilityConfig(
            case=args.case,
            device=args.device,
            spectral_storage=args.spectral_storage,
            dt=args.dt,
            horizon=args.horizon,
            noise=args.noise,
            sample_interval=args.sample_interval,
            tolerance_factor=args.tolerance_factor,
        )
    )
    _atomic_json(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
