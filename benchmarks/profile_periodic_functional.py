#!/usr/bin/env python3
"""Profile one P9.5 periodic production/functional/VJP case.

Each invocation is one fresh-process role, grid, and trial.  Scientific
correctness gates run outside the timed window.  Results are committed
atomically as one JSON object for the fail-closed P9.5 analyzer.
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
import statistics
import subprocess
import tempfile
import time

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
from pssolver.applications.periodic_beris_edwards import load_periodic_initial_q
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.public_simulation_runner import compile_public_simulation
from pssolver.functional import (
    build_functional_runtime,
    periodic_activity_functional_request,
    validate_periodic_activity_gradients,
    validate_periodic_production_consistency,
)
from pssolver.geometries import PeriodicBox
from pssolver.models.active_nematics import CompleteStressBerisEdwards, Q_COMPONENTS
from pssolver.runtime.periodic_beris_edwards import (
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)


PROFILE_SCHEMA_VERSION = 1
PROFILE_KIND = "p95_periodic_functional_profile"
PROFILE_ROLES = ("production_forward", "functional_forward", "functional_vjp")


@dataclass(frozen=True, slots=True)
class PeriodicFunctionalProfileConfig:
    role: str
    trial: int
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    initial_q_path: str
    device: str = "cuda"
    dtype: str = "float64"
    dt: float = 0.001
    base_activity: float = 0.013
    warmup_steps: int = 5
    profile_steps: int = 20


def _validate_config(config: PeriodicFunctionalProfileConfig) -> None:
    if config.role not in PROFILE_ROLES:
        raise ValueError(f"role must be one of {PROFILE_ROLES}")
    if (
        not isinstance(config.trial, int)
        or isinstance(config.trial, bool)
        or config.trial <= 0
    ):
        raise ValueError("trial must be a positive integer")
    if len(config.shape) != 3 or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 1
        for value in config.shape
    ):
        raise ValueError("shape must contain three integer values greater than one")
    if len(config.lengths) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in config.lengths
    ):
        raise ValueError("lengths must contain three positive finite values")
    if config.dtype != "float64":
        raise ValueError("P9.5 qualification is frozen to float64")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not math.isfinite(config.dt) or config.dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not math.isfinite(config.base_activity) or config.base_activity <= 0.002:
        raise ValueError("base_activity must exceed the fixed control modulation")
    if config.warmup_steps < 0 or config.profile_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and profile_steps positive")
    path = Path(config.initial_q_path).expanduser()
    if not path.is_file() or path.name != "Q_0.npy":
        raise FileNotFoundError("initial_q_path must identify an existing Q_0.npy")


def _allocated_device(value: str) -> torch.device:
    requested = torch.device(value)
    if requested.type != "cuda":
        return requested
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    index = requested.index
    if index is None:
        index = torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device is unavailable")
    return torch.device("cuda", index)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(values) -> str:
    digest = hashlib.sha256()
    for index, value in enumerate(values):
        array = value.detach().to(device="cpu").contiguous().numpy()
        digest.update(str(index).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
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


def _simulation(config, *, device: torch.device, output_directory: Path) -> Simulation:
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
        friction=0.0,
        tangential_zero_mode_policy="zero_mean",
    )
    geometry = PeriodicBox(shape=config.shape, lengths=config.lengths)
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype=config.dtype,
            dealias_rule="cubic_half",
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        ),
        time=TimeStepping(dt=config.dt, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(
            Path(config.initial_q_path).expanduser().resolve().parent,
            step=0,
        ),
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
            directory=output_directory,
            steps=1,
            save_interval=2,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
    )


def _production(simulation: Simulation, device: torch.device):
    compiled = compile_public_simulation(simulation.specification)
    initial_values, _, _ = load_periodic_initial_q(compiled.run_spec)
    return build_periodic_beris_edwards_runtime(
        PeriodicRuntimeBuildRequest(
            run_spec=compiled.run_spec,
            initial_values=initial_values,
            device=str(device),
        )
    )


def _activity(runtime, base: float) -> dict[str, torch.Tensor]:
    spec = runtime.control_specs[0].tensor
    count = math.prod(spec.shape)
    indices = torch.arange(
        count,
        device=spec.device,
        dtype=torch.float64,
    ).reshape(spec.shape)
    return {"activity": base + 0.002 * torch.sin(indices * 0.071)}


def _production_state(adapter):
    fields = adapter.fields
    return (
        fields.spatial[: len(Q_COMPONENTS)],
        fields.spectral[: len(Q_COMPONENTS)],
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_samples(operation, count: int, device: torch.device) -> list[float]:
    if device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            operation()
            end.record()
        torch.cuda.synchronize(device)
        return [
            float(start.elapsed_time(end)) / 1000.0
            for start, end in zip(starts, ends, strict=True)
        ]
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return samples


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        _synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": None,
            "peak_active_bytes": None,
            "peak_reserved_bytes": None,
            "device_total_bytes": None,
        }
    stats = torch.cuda.memory_stats(device)
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_active_bytes": int(stats.get("active_bytes.all.peak", 0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device_total_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
    }


def _sample_summary(samples: list[float]) -> dict[str, object]:
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "samples_seconds": samples,
        "mean_seconds": mean,
        "median_seconds": statistics.median(samples),
        "sample_standard_deviation_seconds": deviation,
        "coefficient_of_variation": deviation / mean if mean else 0.0,
        "operations_per_second": 1.0 / mean,
    }


def _audit_loss(next_state, observations):
    return (
        next_state[0].square().mean()
        + next_state[1].abs().square().mean()
        + observations["velocity"].square().mean()
        + observations["pressure"].square().mean()
    )


def _one_vjp(runtime, base_state, base_controls):
    state = tuple(
        value.detach().clone().requires_grad_(True) for value in base_state
    )
    activity = (
        base_controls["activity"].detach().clone().requires_grad_(True)
    )
    next_state, observations = runtime.step_and_observe(
        state, {"activity": activity}, 0
    )
    gradients = torch.autograd.grad(
        _audit_loss(next_state, observations),
        (*state, activity),
        allow_unused=False,
    )
    return tuple(value.detach() for value in gradients)


def _vjp_evidence(gradients) -> dict[str, object]:
    norms = [
        float(torch.linalg.vector_norm(value).detach().cpu().item())
        for value in gradients
    ]
    return {
        "finite": all(bool(torch.isfinite(value).all().item()) for value in gradients),
        "nonzero": all(value > 0.0 for value in norms),
        "gradient_norms": norms,
        "gradient_sha256": _tensor_sha256(gradients),
    }


def _run_production_forward(config, adapter, device, activity):
    adapter.solver.model.parameters["alpha"] = activity
    for _ in range(config.warmup_steps):
        adapter.advance(1)
    _reset_peak_memory(device)
    samples = _timed_samples(
        lambda: adapter.advance(1), config.profile_steps, device
    )
    memory = _memory(device)
    state = tuple(value.detach() for value in _production_state(adapter))
    return _sample_summary(samples), memory, {
        "finite": all(bool(torch.isfinite(value).all().item()) for value in state),
        "state_sha256": _tensor_sha256(state),
    }


def _run_functional_forward(config, runtime, device, controls):
    state = runtime.initial_state()
    with torch.no_grad():
        for step in range(config.warmup_steps):
            state = runtime.step(state, controls, step)
    counter = config.warmup_steps

    def operation():
        nonlocal state, counter
        with torch.no_grad():
            state = runtime.step(state, controls, counter)
        counter += 1

    _reset_peak_memory(device)
    samples = _timed_samples(operation, config.profile_steps, device)
    memory = _memory(device)
    return _sample_summary(samples), memory, {
        "finite": all(bool(torch.isfinite(value).all().item()) for value in state),
        "state_sha256": _tensor_sha256(state),
    }


def _run_functional_vjp(config, runtime, device, controls):
    state = runtime.initial_state()
    for _ in range(config.warmup_steps):
        _one_vjp(runtime, state, controls)
    last = None

    def operation():
        nonlocal last
        last = _one_vjp(runtime, state, controls)

    _reset_peak_memory(device)
    samples = _timed_samples(operation, config.profile_steps, device)
    memory = _memory(device)
    if last is None:  # pragma: no cover - guarded by positive profile_steps
        raise AssertionError("VJP profile produced no gradient")
    evidence = _vjp_evidence(last)
    replay = _one_vjp(runtime, state, controls)
    evidence["replay_bitwise_equal"] = all(
        torch.equal(left, right)
        for left, right in zip(last, replay, strict=True)
    )
    return _sample_summary(samples), memory, evidence


def run_profile(
    config: PeriodicFunctionalProfileConfig,
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    _validate_config(config)
    device = _allocated_device(config.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    root = (
        Path(__file__).resolve().parents[1]
        if repository_root is None
        else Path(repository_root).resolve()
    )
    initial_path = Path(config.initial_q_path).expanduser().resolve()
    values = np.load(initial_path, allow_pickle=False, mmap_mode="r")
    expected = (*config.shape, len(Q_COMPONENTS))
    if values.shape != expected or values.dtype != np.dtype(config.dtype):
        raise ValueError("initial Q shape or dtype differs from the profile contract")
    if not bool(np.isfinite(values).all()):
        raise ValueError("initial Q contains NaN or Inf")

    with tempfile.TemporaryDirectory(prefix="pssolver-p95-") as temporary:
        simulation = _simulation(
            config,
            device=device,
            output_directory=Path(temporary) / "unused-output",
        )
        functional = None
        controls = None
        correctness: dict[str, object] = {}
        if config.role == "production_forward":
            adapter = _production(simulation, device)
            activity_spec = periodic_activity_functional_request(
                simulation.specification
            ).control_fields[0].tensor
            indices = torch.arange(
                math.prod(activity_spec.shape),
                dtype=torch.float64,
                device=activity_spec.device,
            ).reshape(activity_spec.shape)
            activity = config.base_activity + 0.002 * torch.sin(indices * 0.071)
            timing, memory, result = _run_production_forward(
                config, adapter, device, activity
            )
        else:
            request = periodic_activity_functional_request(simulation.specification)
            functional = build_functional_runtime(request)
            controls = _activity(functional, config.base_activity)
            if config.trial == 1:
                correctness["r12"] = validate_periodic_production_consistency(
                    functional,
                    _production(simulation, device),
                    functional.initial_state(),
                    controls,
                ).to_metadata()
            if config.role == "functional_forward":
                timing, memory, result = _run_functional_forward(
                    config, functional, device, controls
                )
            else:
                if config.trial == 1:
                    correctness["gradient"] = validate_periodic_activity_gradients(
                        functional,
                        functional.initial_state(),
                        controls,
                    ).to_metadata()
                timing, memory, result = _run_functional_vjp(
                    config, functional, device, controls
                )

    git = _git_provenance(root)
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "git": git,
    }
    identity = (
        None if functional is None else functional.identity().to_metadata()
    )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "kind": PROFILE_KIND,
        "phase": "P9.5",
        "config": {
            **asdict(config),
            "shape": list(config.shape),
            "lengths": list(config.lengths),
            "device": str(device),
            "initial_q_path": str(initial_path),
            "initial_q_sha256": _sha256(initial_path),
        },
        "environment": environment,
        "functional_identity": identity,
        "timing": timing,
        "memory": memory,
        "result": result,
        "correctness": correctness,
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
    parser.add_argument("--role", choices=PROFILE_ROLES, required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", nargs=3, required=True)
    parser.add_argument("--lengths", nargs=3, required=True)
    parser.add_argument("--initial-q-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--base-activity", type=float, default=0.013)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = PeriodicFunctionalProfileConfig(
        role=args.role,
        trial=args.trial,
        shape=_tuple(args.shape, int),
        lengths=_tuple(args.lengths, float),
        initial_q_path=args.initial_q_path,
        device=args.device,
        dtype=args.dtype,
        dt=args.dt,
        base_activity=args.base_activity,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
    )
    _atomic_json(args.output, run_profile(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
