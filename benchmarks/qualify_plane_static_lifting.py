#!/usr/bin/env python3
"""P8.4.5 qualification helpers for static Plane Dirichlet lifting.

The helper deliberately exercises the package-owned runtime.  It does not
reimplement the Beris--Edwards timestep and it never changes a production
default.  The three commands cover the independent evidence classes needed
by the H100 closure: steady-state performance, a manufactured wall-normal
oracle, and exact checkpoint/restart.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

if __package__:
    from benchmarks.profile_beris_edwards_timestep import (
        RegionTimer,
        _counter_delta,
        _dynamo_counter_snapshot,
        _git_provenance,
        _install_transform_timers,
    )
    from benchmarks.profile_plane_runtime_timestep import (
        RuntimeProfileConfig,
        _initial_q,
        _run_spec,
        _state_sha256,
        _tensor_mapping_sha256,
    )
else:
    from profile_beris_edwards_timestep import (
        RegionTimer,
        _counter_delta,
        _dynamo_counter_snapshot,
        _git_provenance,
        _install_transform_timers,
    )
    from profile_plane_runtime_timestep import (
        RuntimeProfileConfig,
        _initial_q,
        _run_spec,
        _state_sha256,
        _tensor_mapping_sha256,
    )
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    strong_planar_q,
)
from pssolver.operators import materialize_plane_static_lifting
from pssolver.runtime.package_construction import (
    PackageRuntimeConstructionInput,
    build_package_simulation_runtime,
)
from pssolver.runtime.plane_beris_edwards import (
    PlaneRuntimeBuildRequest,
    plane_physical_component,
)
from pssolver.workflows.plane_checkpoint import (
    capture_plane_checkpoint,
    restore_plane_checkpoint,
)


PROFILE_SCHEMA_VERSION = 1
MANUFACTURED_SCHEMA_VERSION = 1
RESTART_SCHEMA_VERSION = 1
PROFILE_VARIANTS = ("homogeneous_control", "strong_planar_lifting")


@dataclass(frozen=True, slots=True)
class LiftingProfileConfig:
    """Identity of one balanced control/lifting profiler invocation."""

    variant: str = "strong_planar_lifting"
    trial: int = 1
    shape: tuple[int, int, int] = (64, 64, 32)
    lengths: tuple[float, float, float] = (100.0, 100.0, 20.0)
    device: str = "cpu"
    dtype: str = "float64"
    dt: float = 0.005
    activity_number: float = 18.0
    warmup_steps: int = 10
    profile_steps: int = 50
    pointwise_execution: str = "compile"
    seed: int = 20260926

    def __post_init__(self) -> None:
        if self.variant not in PROFILE_VARIANTS:
            raise ValueError(f"variant must be one of {PROFILE_VARIANTS!r}")
        if not isinstance(self.trial, int) or isinstance(self.trial, bool):
            raise TypeError("trial must be an integer")
        if self.trial <= 0:
            raise ValueError("trial must be positive")
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise ValueError("shape must contain three positive entries")
        if len(self.lengths) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.lengths
        ):
            raise ValueError("lengths must contain three positive finite entries")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.warmup_steps < 0 or self.profile_steps <= 0:
            raise ValueError("profile window lengths are invalid")


def _base_profile_config(config: LiftingProfileConfig) -> RuntimeProfileConfig:
    return RuntimeProfileConfig(
        runtime_path="legacy_production",
        trial=config.trial,
        shape=config.shape,
        lengths=config.lengths,
        device=config.device,
        dtype=config.dtype,
        dt=config.dt,
        activity_number=config.activity_number,
        warmup_steps=config.warmup_steps,
        profile_steps=config.profile_steps,
        pointwise_execution=config.pointwise_execution,
        seed=config.seed,
    )


def _strong_planar_simulation(run_spec):
    base = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )
    normals = {
        (2, "lower"): (0.0, 0.0, -1.0),
        (2, "upper"): (0.0, 0.0, 1.0),
    }
    q_policy = strong_planar_q(
        scalar_order=0.6,
        face_directors={
            (2, "lower"): (1.0, 0.0, 0.0),
            (2, "upper"): (0.0, 1.0, 0.0),
        },
        face_normals=normals,
    )
    boundaries = assign_boundaries(
        model=base.equation_system,
        geometry=base.geometry,
        policies={
            "Q": q_policy,
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
        name="p845_strong_planar_lifting",
    )
    return replace(base, boundaries=boundaries)


def _build_runtime(
    config: LiftingProfileConfig,
    initial_values: dict[str, torch.Tensor],
):
    profile = _base_profile_config(config)
    run_spec = _run_spec(profile)
    simulation = compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )
    if config.variant == "strong_planar_lifting":
        simulation = _strong_planar_simulation(run_spec)
    metadata = {
        "configuration": run_spec.identity_metadata(),
        "runtime_selection": run_spec.runtime_selection_metadata(),
    }
    request = PlaneRuntimeBuildRequest(
        run_spec,
        metadata,
        initial_values,
        config.device,
        simulation,
    )
    construction = PackageRuntimeConstructionInput(
        plan_package_runtime_construction(simulation),
        request,
    )
    return run_spec, simulation, build_package_simulation_runtime(construction)


def _wall_residual_report(adapter) -> dict[str, object] | None:
    lifting = getattr(adapter.solver.model, "static_lifting_runtime", None)
    if lifting is None:
        return None
    operator = lifting.operator
    plan = operator.plan
    axis = plan.wall_normal_axis
    dz = plan.domain_lengths[axis] / plan.domain_shape[axis]
    residuals: dict[str, dict[str, float]] = {}
    maximum = 0.0
    for index, name in enumerate(operator.component_order):
        component = plan.for_component(name)
        slope = (
            component.upper_value.value - component.lower_value.value
        ) / plan.domain_lengths[axis]
        lower_cell = operator.stacked_lift[index].select(axis, 0)
        upper_cell = operator.stacked_lift[index].select(
            axis,
            plan.domain_shape[axis] - 1,
        )
        lower = lower_cell - 0.5 * dz * slope
        upper = upper_cell + 0.5 * dz * slope
        lower_error = float(
            torch.max(torch.abs(lower - component.lower_value.value)).item()
        )
        upper_error = float(
            torch.max(torch.abs(upper - component.upper_value.value)).item()
        )
        maximum = max(maximum, lower_error, upper_error)
        residuals[name] = {
            "lower_linf": lower_error,
            "upper_linf": upper_error,
        }
    return {
        "definition": "analytic_half_cell_extrapolation_of_affine_lift",
        "component_residuals": residuals,
        "max_linf": maximum,
        "homogeneous_remainder_face_value": 0.0,
    }


def _physical_state_sha256(adapter) -> str:
    digest = hashlib.sha256()
    names = (*Q_COMPONENTS, "ux", "uy", "uz", "p")
    for name in names:
        value = (
            plane_physical_component(adapter, name)
            if name in Q_COMPONENTS
            else adapter.fields[name]
        )
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _step_times(adapter, *, steps: int, device: torch.device) -> list[float]:
    if device.type == "cuda":
        events = []
        for _ in range(steps):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            adapter.advance(1)
            stop.record()
            events.append((start, stop))
        torch.cuda.synchronize(device)
        return [start.elapsed_time(stop) / 1000.0 for start, stop in events]
    samples = []
    for _ in range(steps):
        started = time.perf_counter()
        adapter.advance(1)
        samples.append(time.perf_counter() - started)
    return samples


def run_profile(config: LiftingProfileConfig) -> dict[str, object]:
    """Profile one real package runtime with or without static lifting."""

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    base = _base_profile_config(config)
    initial_values = _initial_q(base)
    initial_sha256 = _tensor_mapping_sha256(initial_values)
    transform_timer = RegionTimer(
        device,
        timed_regions=frozenset(),
    )
    counters_before = _dynamo_counter_snapshot()

    with torch.no_grad():
        run_spec, simulation, adapter = _build_runtime(config, initial_values)
        # The runtime owns its evolved state after construction.  Do not keep
        # the caller-owned physical initial tensors alive in the measurement
        # scope, and release cached construction temporaries before measuring
        # the frozen steady-state CUDA footprint.
        del initial_values
        _install_transform_timers(adapter.solver, transform_timer)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
        counters_after_build = _dynamo_counter_snapshot()
        adapter.advance(config.warmup_steps)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        counters_after_warmup = _dynamo_counter_snapshot()
        transform_timer.reset()
        samples = _step_times(
            adapter,
            steps=config.profile_steps,
            device=device,
        )
        transform_summary = transform_timer.summarize()
        counters_after_profile = _dynamo_counter_snapshot()
        adapter.synchronize_for_observation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        finite = all(
            bool(torch.isfinite(adapter.fields[name]).all().item())
            for name in (*Q_COMPONENTS, "ux", "uy", "uz", "p")
        )
        internal_sha256 = _state_sha256(adapter)
        physical_sha256 = _physical_state_sha256(adapter)
        wall_residual = _wall_residual_report(adapter)
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
        memory_stats = (
            torch.cuda.memory_stats(device) if device.type == "cuda" else {}
        )
        peak_active = int(memory_stats.get("active_bytes.all.peak", 0))
        peak_requested = int(memory_stats.get("requested_bytes.all.peak", 0))

    runtime = adapter.to_metadata()
    if runtime["requested"] != runtime["effective"]:
        raise RuntimeError("runtime identity mismatch")
    if runtime["fallback_used"] is not False:
        raise RuntimeError("runtime fallback detected")
    forward = int(transform_summary.get("transform_forward", {}).get("calls", 0))
    inverse = int(transform_summary.get("transform_inverse", {}).get("calls", 0))
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    standard_deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    lifting_runtime = getattr(
        adapter.solver.model,
        "static_lifting_runtime",
        None,
    )
    pointwise_metadata = (
        adapter.solver.model.nlmodel.pointwise_kernels.metadata()
    )
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_profile",
        "config": asdict(config),
        "configuration_identity": run_spec.identity_metadata(),
        "simulation_identity": simulation.canonical_sha256(),
        "runtime_identity": runtime,
        "git": _git_provenance(),
        "pssolver_import": __import__("pssolver").__file__,
        "timing": {
            "samples_seconds": samples,
            "mean_timestep_seconds": mean,
            "median_timestep_seconds": median,
            "sample_standard_deviation_seconds": standard_deviation,
            "coefficient_of_variation": standard_deviation / mean,
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_active_bytes": peak_active,
            "peak_requested_bytes": peak_requested,
            "peak_reserved_bytes": peak_reserved,
        },
        "transform_calls": {
            "forward": forward,
            "inverse": inverse,
            "forward_per_step": forward / config.profile_steps,
            "inverse_per_step": inverse / config.profile_steps,
        },
        "pointwise_compile": {
            "execution": pointwise_metadata,
            "during_build": _counter_delta(counters_after_build, counters_before),
            "during_warmup": _counter_delta(
                counters_after_warmup,
                counters_after_build,
            ),
            "during_profile": _counter_delta(
                counters_after_profile,
                counters_after_warmup,
            ),
        },
        "lifting": (
            None
            if lifting_runtime is None
            else lifting_runtime.restart_metadata()
        ),
        "lifting_storage": (
            None
            if lifting_runtime is None
            else lifting_runtime.storage_metadata()
        ),
        "wall_residual": wall_residual,
        "initial_q_sha256": initial_sha256,
        "internal_final_state_sha256": internal_sha256,
        "physical_final_state_sha256": physical_sha256,
        "completed_steps": adapter.completed_steps,
        "finite": finite,
    }


def _manufactured_error(
    wall_points: int,
    *,
    device: str,
) -> float:
    config = LiftingProfileConfig(
        variant="strong_planar_lifting",
        shape=(4, 4, wall_points),
        device=device,
        pointwise_execution="eager",
        warmup_steps=0,
        profile_steps=1,
    )
    profile = _base_profile_config(config)
    run_spec = _run_spec(profile)
    simulation = _strong_planar_simulation(run_spec)
    from pssolver.configuration.simulation_lowering import lower_simulation_spec

    plan = lower_simulation_spec(simulation).lifting_plan
    if plan is None:
        raise RuntimeError("lifting plan was not lowered")
    operator = materialize_plane_static_lifting(
        plan,
        dtype=torch.float64,
        device=device,
    )
    length = plan.domain_lengths[plan.wall_normal_axis]
    dz = length / wall_points
    z = (torch.arange(wall_points, dtype=torch.float64, device=device) + 0.5) * dz
    one_d = torch.sin(math.pi * z / length)
    remainder = one_d.reshape(1, 1, wall_points).expand(plan.domain_shape).clone()
    physical = operator.reconstruct_physical("Qxx", remainder)
    recovered = operator.extract_homogeneous_remainder("Qxx", physical)
    extended = torch.cat((-recovered[..., :1], recovered, -recovered[..., -1:]), dim=-1)
    numerical = (
        extended[..., 2:] - 2.0 * extended[..., 1:-1] + extended[..., :-2]
    ) / (dz * dz)
    exact = -((math.pi / length) ** 2) * remainder
    return float(torch.sqrt(torch.mean((numerical - exact) ** 2)).item())


def run_manufactured(*, device: str) -> dict[str, object]:
    """Run an independent second-order homogeneous-remainder oracle."""

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    sizes = (16, 32, 64)
    errors = [_manufactured_error(size, device=device) for size in sizes]
    rates = [
        math.log(errors[index] / errors[index + 1], 2.0)
        for index in range(len(errors) - 1)
    ]
    return {
        "schema_version": MANUFACTURED_SCHEMA_VERSION,
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_manufactured_convergence",
        "device": device,
        "wall_points": list(sizes),
        "errors": errors,
        "rates": rates,
        "minimum_rate": min(rates),
        "finite": all(math.isfinite(value) for value in (*errors, *rates)),
    }


def _runtime_for_restart(
    *,
    shape: tuple[int, int, int],
    device: str,
    seed: int,
    pointwise_execution: str,
):
    config = LiftingProfileConfig(
        variant="strong_planar_lifting",
        shape=shape,
        device=device,
        pointwise_execution=pointwise_execution,
        warmup_steps=0,
        profile_steps=1,
        seed=seed,
    )
    initial = _initial_q(_base_profile_config(config))
    return _build_runtime(config, initial)


def run_restart(
    *,
    shape: tuple[int, int, int],
    device: str,
    segment_steps: int,
    final_steps: int,
    seed: int,
    pointwise_execution: str = "compile",
) -> dict[str, object]:
    """Compare continuous and restored lifted trajectories exactly."""

    if segment_steps <= 0 or final_steps <= segment_steps:
        raise ValueError("restart steps must satisfy 0 < segment < final")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    identity, _simulation, continuous = _runtime_for_restart(
        shape=shape,
        device=device,
        seed=seed,
        pointwise_execution=pointwise_execution,
    )
    _, _, segment = _runtime_for_restart(
        shape=shape,
        device=device,
        seed=seed,
        pointwise_execution=pointwise_execution,
    )
    _, _, resumed = _runtime_for_restart(
        shape=shape,
        device=device,
        seed=seed,
        pointwise_execution=pointwise_execution,
    )
    with torch.no_grad():
        continuous.advance(final_steps)
        segment.advance(segment_steps)
        checkpoint = capture_plane_checkpoint(
            segment,
            run_spec=identity,
        )
        restore_plane_checkpoint(
            resumed,
            checkpoint,
            run_spec=identity,
        )
        resumed.advance(final_steps - segment_steps)
        if device == "cuda":
            torch.cuda.synchronize()
    fields_equal = {
        name: bool(torch.equal(continuous.fields[name], resumed.fields[name]))
        for name in (*Q_COMPONENTS, "ux", "uy", "uz", "p")
    }
    physical_equal = {
        name: bool(
            torch.equal(
                plane_physical_component(continuous, name),
                plane_physical_component(resumed, name),
            )
        )
        for name in Q_COMPONENTS
    }
    return {
        "schema_version": RESTART_SCHEMA_VERSION,
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_restart",
        "shape": list(shape),
        "device": device,
        "pointwise_execution": pointwise_execution,
        "segment_steps": segment_steps,
        "final_steps": final_steps,
        "fields_byte_identical": fields_equal,
        "physical_q_byte_identical": physical_equal,
        "all_byte_identical": all((*fields_equal.values(), *physical_equal.values())),
        "continuous_physical_sha256": _physical_state_sha256(continuous),
        "resumed_physical_sha256": _physical_state_sha256(resumed),
        "lifting_restart": dict(checkpoint.lifting_restart),
        "finite": all(
            bool(torch.isfinite(resumed.fields[name]).all().item())
            for name in (*Q_COMPONENTS, "ux", "uy", "uz", "p")
        ),
    }


def _shape(value: str) -> tuple[int, int, int]:
    parsed = tuple(int(item) for item in value.split(","))
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("shape requires three comma-separated ints")
    return parsed


def _lengths(value: str) -> tuple[float, float, float]:
    parsed = tuple(float(item) for item in value.split(","))
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("lengths require three comma-separated values")
    return parsed


def _write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    print(payload, end="")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--variant", choices=PROFILE_VARIANTS, required=True)
    profile.add_argument("--trial", type=int, required=True)
    profile.add_argument("--shape", type=_shape, required=True)
    profile.add_argument("--lengths", type=_lengths, default=(100.0, 100.0, 20.0))
    profile.add_argument("--device", choices=("cpu", "cuda"), required=True)
    profile.add_argument("--warmup-steps", type=int, default=10)
    profile.add_argument("--profile-steps", type=int, default=50)
    profile.add_argument(
        "--pointwise-execution",
        choices=("eager", "compile"),
        default="compile",
    )
    profile.add_argument("--seed", type=int, default=20260926)
    profile.add_argument("--output", type=Path, required=True)

    manufactured = subparsers.add_parser("manufactured")
    manufactured.add_argument("--device", choices=("cpu", "cuda"), required=True)
    manufactured.add_argument("--output", type=Path, required=True)

    restart = subparsers.add_parser("restart")
    restart.add_argument("--shape", type=_shape, required=True)
    restart.add_argument("--device", choices=("cpu", "cuda"), required=True)
    restart.add_argument("--segment-steps", type=int, default=10)
    restart.add_argument("--final-steps", type=int, default=20)
    restart.add_argument("--seed", type=int, default=20260926)
    restart.add_argument(
        "--pointwise-execution",
        choices=("eager", "compile"),
        default="compile",
    )
    restart.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "profile":
        result = run_profile(
            LiftingProfileConfig(
                variant=args.variant,
                trial=args.trial,
                shape=args.shape,
                lengths=args.lengths,
                device=args.device,
                warmup_steps=args.warmup_steps,
                profile_steps=args.profile_steps,
                pointwise_execution=args.pointwise_execution,
                seed=args.seed,
            )
        )
    elif args.command == "manufactured":
        result = run_manufactured(device=args.device)
    else:
        result = run_restart(
            shape=args.shape,
            device=args.device,
            segment_steps=args.segment_steps,
            final_steps=args.final_steps,
            seed=args.seed,
            pointwise_execution=args.pointwise_execution,
        )
    _write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
