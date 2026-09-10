"""Profile one complete Beris--Edwards--Stokes timestep locally.

The benchmark uses the production model and transform implementations while
keeping profiling, optional snapshot I/O, and synthetic initial data outside
the solver package. Timed regions are inclusive where explicitly documented:
``transform_*``, ``nematic_force``, and ``stokes_solve`` are cross-cutting
subregions of the additive timestep regions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from pssolver import (
    BasisAwareSpectralProjector,
    DEALIAS_RULE_FRACTIONS,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    PROJECTED_TRANSFORM_EXECUTION_MODES,
    SpectralSolver,
)
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsQGradientCache,
    BerisEdwardsQNonlinearModel,
    BerisEdwardsPointwiseKernels,
    DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE,
    DEFAULT_POINTWISE_EXECUTION,
    DEFAULT_STRESS_DIVERGENCE_SUM_SPACE,
    POINTWISE_EXECUTION_MODES,
    Q_COMPONENTS,
    beris_edwards_linear_operator,
)


Q_BCS = ("periodic", "periodic", "neumann")
TANGENTIAL_BCS = ("periodic", "periodic", "neumann")
NORMAL_BCS = ("periodic", "periodic", "dirichlet")
PRESSURE_BCS = ("periodic", "periodic", "neumann")


@dataclass(frozen=True)
class ProfileConfig:
    """Inputs defining one reproducible complete-timestep profile."""

    shape: tuple[int, int, int] = (64, 64, 32)
    lengths: tuple[float, float, float] = (100.0, 100.0, 20.0)
    device: str = "cpu"
    dtype: str = "float64"
    dt: float = 0.005
    dealias_rule: str = "cubic_half"
    projected_transform_execution: str = DEFAULT_PROJECTED_TRANSFORM_EXECUTION
    warmup_steps: int = 3
    profile_steps: int = 10
    spectral_refresh_interval: int | None = None
    pressure_diagnostics: bool = False
    reuse_q_gradients: bool = True
    molecular_field_linear_space: str = DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE
    stress_divergence_sum_space: str = DEFAULT_STRESS_DIVERGENCE_SUM_SPACE
    pointwise_execution: str = DEFAULT_POINTWISE_EXECUTION
    transform_execution_order: str = DEFAULT_TRANSFORM_EXECUTION_ORDER
    snapshot_interval: int | None = None
    snapshot_directory: str | None = None
    save_hydrodynamics: bool = False
    seed: int = 20260908


class RegionTimer:
    """Collect nested CPU wall-clock or asynchronous CUDA-event timings."""

    def __init__(self, device: torch.device):
        self.device = device
        self.enabled = False
        self._cpu_seconds: dict[str, list[float]] = defaultdict(list)
        self._cuda_events: dict[
            str,
            list[tuple[torch.cuda.Event, torch.cuda.Event]],
        ] = defaultdict(list)

    @contextmanager
    def region(self, name: str, *, host: bool = False) -> Iterator[None]:
        """Time a region without synchronizing inside the profiled timestep."""
        if not self.enabled:
            yield
            return

        if self.device.type == "cuda" and not host:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                stop.record()
                self._cuda_events[name].append((start, stop))
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            self._cpu_seconds[name].append(time.perf_counter() - start_time)

    def reset(self) -> None:
        """Discard warm-up measurements and enable profiling."""
        self._cpu_seconds.clear()
        self._cuda_events.clear()
        self.enabled = True

    def summarize(self) -> dict[str, dict[str, float | int]]:
        """Synchronize once and return aggregate timings by region."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        names = sorted(set(self._cpu_seconds) | set(self._cuda_events))
        summary: dict[str, dict[str, float | int]] = {}
        for name in names:
            samples = list(self._cpu_seconds[name])
            samples.extend(
                start.elapsed_time(stop) / 1000.0
                for start, stop in self._cuda_events[name]
            )
            total = math.fsum(samples)
            summary[name] = {
                "calls": len(samples),
                "total_seconds": total,
                "mean_seconds": total / len(samples),
            }
        return summary


class ProfiledBerisEdwardsFreeSlipStokes(BerisEdwardsFreeSlipStokes):
    """Production static model with nested profiler ranges."""

    def __init__(self, *args, profile_timer: RegionTimer, **kwargs):
        self.profile_timer = profile_timer
        super().__init__(*args, **kwargs)

    def compute_nematic_force(self, fields, alpha):
        with self.profile_timer.region("nematic_force"):
            return super().compute_nematic_force(fields, alpha)

    def solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        with self.profile_timer.region("stokes_solve"):
            return super().solve_force_hats(fx_hat, fy_hat, fz_hat)


class ProfiledDealiasedIntegrator(SemiImplicitEulerIntegrator):
    """Production dealiased Euler step split into additive profiler regions."""

    def __init__(self, model, dt, qx, qy, q2, *, profile_timer: RegionTimer):
        super().__init__(model, dt, qx, qy, q2)
        self.spectral_projector = model.spectral_projector
        self.profile_timer = profile_timer

    def _refresh_dynamic_spectra(self):
        self.spectral_projector.refresh_dynamic_fields(
            self.model.fields,
            sync_spatial=True,
        )

    def step(self, pre_update_callback=None):
        with self.profile_timer.region("whole_timestep"):
            with self.profile_timer.region("static_fields"):
                if self._static_fields_are_current:
                    self._static_fields_are_current = False
                else:
                    self.model.update_static_fields()

            if pre_update_callback is not None:
                pre_update_callback()

            with self.profile_timer.region("q_nonlinear"):
                nonlinear_hats = self.model.compute_nonlinear()

            with self.profile_timer.region("imex_and_dealias"):
                dynamic_fields = self.model.fields.spectral[: self.dyn_count]
                dynamic_fields.add_(self.dt * nonlinear_hats)
                dynamic_fields.div_(self.denom)
                self.spectral_projector.project_dynamic_fields(
                    self.model.fields,
                    sync_spatial=False,
                )

            with self.profile_timer.region("dynamic_inverse"):
                for group in self.dynamic_transform_groups:
                    boundary_conditions = (
                        self.model.fields.get_boundary_conditions(group[0])
                    )
                    self.model.fields.spatial[group] = (
                        self.spectral_projector.inverse_transform(
                            self.model.fields.spectral[group],
                            boundary_conditions,
                        )
                    )

            with self.profile_timer.region("spectral_refresh"):
                self._advance_spectral_refresh_clock()


def _validate_config(config: ProfileConfig) -> None:
    if len(config.shape) != 3 or any(value <= 0 for value in config.shape):
        raise ValueError("shape must contain three positive integers")
    if len(config.lengths) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in config.lengths
    ):
        raise ValueError("lengths must contain three positive finite numbers")
    if config.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if config.transform_execution_order not in {"legacy", "real_first"}:
        raise ValueError(
            "transform_execution_order must be 'legacy' or 'real_first'"
        )
    if (
        config.projected_transform_execution
        not in PROJECTED_TRANSFORM_EXECUTION_MODES
    ):
        raise ValueError(
            "projected_transform_execution must be 'full' or 'truncated'"
        )
    if (
        config.projected_transform_execution == "truncated"
        and config.dealias_rule == "none"
    ):
        raise ValueError(
            "truncated projected transforms require enabled dealiasing"
        )
    if config.molecular_field_linear_space not in {"physical", "spectral"}:
        raise ValueError(
            "molecular_field_linear_space must be 'physical' or 'spectral'"
        )
    if config.stress_divergence_sum_space not in {"physical", "spectral"}:
        raise ValueError(
            "stress_divergence_sum_space must be 'physical' or 'spectral'"
        )
    if config.pointwise_execution not in POINTWISE_EXECUTION_MODES:
        raise ValueError(
            "pointwise_execution must be 'eager' or 'compile'"
        )
    if not math.isfinite(config.dt) or config.dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.pointwise_execution == "compile" and config.warmup_steps < 1:
        raise ValueError(
            "compiled pointwise execution requires at least one warmup step"
        )
    if config.profile_steps <= 0:
        raise ValueError("profile_steps must be positive")
    if (
        config.spectral_refresh_interval is not None
        and config.spectral_refresh_interval <= 0
    ):
        raise ValueError("spectral_refresh_interval must be positive or None")
    if config.snapshot_interval is not None and config.snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be positive or None")
    if (config.snapshot_interval is None) != (config.snapshot_directory is None):
        raise ValueError(
            "snapshot_interval and snapshot_directory must be enabled together"
        )


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _install_transform_timers(solver, timer: RegionTimer) -> None:
    backend = solver.transform_backend
    original_forward = backend.forward
    original_inverse = backend.inverse

    def timed_forward(tensor, boundary_conditions, **kwargs):
        with timer.region("transform_forward"):
            return original_forward(tensor, boundary_conditions, **kwargs)

    def timed_inverse(spectral, boundary_conditions, **kwargs):
        with timer.region("transform_inverse"):
            return original_inverse(spectral, boundary_conditions, **kwargs)

    backend.forward = timed_forward
    backend.inverse = timed_inverse


def _synthetic_initial_q(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = 0.01 * torch.randn(5, *shape, generator=generator, dtype=dtype)
    q_amplitude = 0.5
    noise[0].add_(2.0 * q_amplitude / 3.0)
    noise[3].add_(-q_amplitude / 3.0)
    return {name: noise[index] for index, name in enumerate(Q_COMPONENTS)}


def _build_solver(config: ProfileConfig, timer: RegionTimer):
    dtype = _torch_dtype(config.dtype)
    solver = SpectralSolver(
        shape=config.shape,
        L=config.lengths,
        dt=config.dt,
        device=config.device,
        batchsize=1,
        dtype=dtype,
        transform_execution_order=config.transform_execution_order,
    )
    projector = BasisAwareSpectralProjector(
        solver,
        rule=config.dealias_rule,
        transform_execution=config.projected_transform_execution,
    )
    solver.model.spectral_projector = projector
    solver.model.set_static_inverse_transform(projector.inverse_transform)
    initial_q = _synthetic_initial_q(config.shape, dtype=dtype, seed=config.seed)
    q_gradient_cache = (
        BerisEdwardsQGradientCache() if config.reuse_q_gradients else None
    )
    pointwise_kernels = BerisEdwardsPointwiseKernels(
        config.pointwise_execution
    )
    solver.pointwise_kernels = pointwise_kernels

    frank_k = 1.0 / 81.0
    ldg_l1 = 2.0 * frank_k
    gamma = 2.94
    linear_operator = beris_edwards_linear_operator(
        solver.get_q2(Q_BCS),
        ldg_a=0.0,
        ldg_l1=ldg_l1,
        rotational_viscosity=gamma,
    )
    for name in Q_COMPONENTS:
        solver.model.add_dynamic_field(
            name,
            init=initial_q[name],
            L_hat=linear_operator,
            boundary_conditions=Q_BCS,
        )
    solver.model.add_static_field("ux", boundary_conditions=TANGENTIAL_BCS)
    solver.model.add_static_field("uy", boundary_conditions=TANGENTIAL_BCS)
    solver.model.add_static_field("uz", boundary_conditions=NORMAL_BCS)
    solver.model.add_static_field("p", boundary_conditions=PRESSURE_BCS)
    solver.model.set_nonlinear_model(
        BerisEdwardsQNonlinearModel(
            projector,
            Q_BCS,
            ldg_b=-0.3,
            ldg_c=0.3,
            rotational_viscosity=gamma,
            flow_alignment=0.3,
            q_gradient_cache=q_gradient_cache,
            pointwise_kernels=pointwise_kernels,
        )
    )
    solver.model.set_static_compute_model(
        ProfiledBerisEdwardsFreeSlipStokes(
            solver,
            spectral_projector=projector,
            beta_value=-1.0,
            friction=0.0,
            viscosity=2.0 / 3.0,
            ldg_a=0.0,
            ldg_b=-0.3,
            ldg_c=0.3,
            ldg_l1=ldg_l1,
            flow_alignment=0.3,
            molecular_field_linear_space=(
                config.molecular_field_linear_space
            ),
            stress_divergence_sum_space=(
                config.stress_divergence_sum_space
            ),
            cache_force_diagnostics=False,
            cache_pressure_diagnostics=config.pressure_diagnostics,
            q_gradient_cache=q_gradient_cache,
            pointwise_kernels=pointwise_kernels,
            zero_mode_policy="zero_mean",
            profile_timer=timer,
        )
    )
    zeta = frank_k * (18.0 / config.lengths[2]) ** 2
    solver.model.parameters.new_param(
        "alpha",
        torch.tensor(zeta, device=config.device, dtype=dtype),
    )

    solver.model.build()
    solver.integrator = ProfiledDealiasedIntegrator(
        solver.model,
        solver.dt,
        solver.qx,
        solver.qy,
        solver.q2,
        profile_timer=timer,
    )
    solver.integrator.set_spectral_refresh_interval(
        config.spectral_refresh_interval
    )
    projector.project_dynamic_fields(solver.model.fields, sync_spatial=True)
    solver.integrator._static_fields_are_current = False
    _install_transform_timers(solver, timer)
    return solver


def _state_sha256(solver) -> str:
    digest = hashlib.sha256()
    for tensor in (solver.model.fields.spatial, solver.model.fields.spectral):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _prepare_snapshot_directory(config: ProfileConfig) -> Path | None:
    if config.snapshot_directory is None:
        return None
    directory = Path(config.snapshot_directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"snapshot directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save_snapshot(
    solver,
    step: int,
    directory: Path,
    timer: RegionTimer,
    *,
    save_hydrodynamics: bool,
) -> None:
    with timer.region("snapshot_cpu_transfer", host=True):
        q = torch.stack(
            [solver.model.fields[name].detach().cpu() for name in Q_COMPONENTS]
        ).permute(1, 2, 3, 4, 0)[0]
        hydro = (
            {
                name: solver.model.fields[name].detach().cpu().numpy()
                for name in ("ux", "uy", "uz", "p")
            }
            if save_hydrodynamics
            else {}
        )
    with timer.region("snapshot_disk_write", host=True):
        np.save(directory / f"Q_{step}.npy", q.numpy())
        if hydro:
            velocity = np.stack(
                [hydro[name][0] for name in ("ux", "uy", "uz")],
                axis=-1,
            )
            np.save(directory / f"u_{step}.npy", velocity)
            np.save(directory / f"p_{step}.npy", hydro["p"][0])


def _git_provenance() -> dict[str, object]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ("git", *args),
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "status": status,
    }


def _dynamo_counter_snapshot() -> dict[str, int | None]:
    """Return bounded TorchDynamo counters without making them a dependency."""
    try:
        from torch._dynamo.utils import counters
    except (AttributeError, ImportError):
        return {
            "unique_graphs": None,
            "calls_captured": None,
            "graph_breaks": None,
        }

    return {
        "unique_graphs": int(counters["stats"]["unique_graphs"]),
        "calls_captured": int(counters["stats"]["calls_captured"]),
        "graph_breaks": int(sum(counters["graph_break"].values())),
    }


def _counter_delta(after, before):
    return {
        name: (
            None
            if before[name] is None or after[name] is None
            else after[name] - before[name]
        )
        for name in before
    }


def run_profile(config: ProfileConfig) -> dict[str, object]:
    """Run one profile and return an auditable JSON-compatible result."""
    _validate_config(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    snapshot_directory = _prepare_snapshot_directory(config)
    timer = RegionTimer(device)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    counters_before = _dynamo_counter_snapshot()
    with torch.no_grad():
        build_wall_start = time.perf_counter()
        solver = _build_solver(config, timer)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        build_wall_seconds = time.perf_counter() - build_wall_start
        counters_after_build = _dynamo_counter_snapshot()

        warmup_wall_start = time.perf_counter()
        for _ in range(config.warmup_steps):
            solver.integrator.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        warmup_wall_seconds = time.perf_counter() - warmup_wall_start
        counters_after_warmup = _dynamo_counter_snapshot()

        timer.reset()
        profile_wall_start = time.perf_counter()
        for profile_index in range(1, config.profile_steps + 1):
            solver.integrator.step()
            if (
                snapshot_directory is not None
                and profile_index % config.snapshot_interval == 0
            ):
                _save_snapshot(
                    solver,
                    profile_index,
                    snapshot_directory,
                    timer,
                    save_hydrodynamics=config.save_hydrodynamics,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        profile_wall_seconds = time.perf_counter() - profile_wall_start
        timings = timer.summarize()
        state_sha256 = _state_sha256(solver)
        counters_after_profile = _dynamo_counter_snapshot()

    whole_timestep_seconds = timings["whole_timestep"]["total_seconds"]
    memory: dict[str, int | None] = {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        memory = {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    return {
        "schema_version": 1,
        "config": asdict(config),
        "model": {
            "name": "Beris--Edwards complete-nematic-force Stokes",
            "activity_number": 18.0,
            "height": config.lengths[2],
            "frank_k": 1.0 / 81.0,
            "zeta": (1.0 / 81.0) * (18.0 / config.lengths[2]) ** 2,
            "ldg_a": 0.0,
            "ldg_b": -0.3,
            "ldg_c": 0.3,
            "ldg_l1": 2.0 / 81.0,
            "gamma": 2.94,
            "flow_alignment": 0.3,
            "eta": 2.0 / 3.0,
            "zero_mode_policy": "zero_mean",
            "reuse_q_gradients": config.reuse_q_gradients,
            "molecular_field_linear_space": (
                config.molecular_field_linear_space
            ),
            "stress_divergence_sum_space": (
                config.stress_divergence_sum_space
            ),
            "pointwise_execution": config.pointwise_execution,
            "projected_transform_execution": (
                config.projected_transform_execution
            ),
            "initial_condition": "deterministic synthetic aligned Q plus noise",
        },
        "environment": {
            "git": _git_provenance(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
            "cpu_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "timing_notes": {
            "additive_regions": [
                "static_fields",
                "q_nonlinear",
                "imex_and_dealias",
                "dynamic_inverse",
                "spectral_refresh",
            ],
            "cross_cutting_nested_regions": [
                "nematic_force",
                "stokes_solve",
                "transform_forward",
                "transform_inverse",
            ],
            "snapshot_regions_are_outside_whole_timestep": True,
            "cuda_synchronization_inside_timestep": False,
        },
        "pointwise_kernels": {
            **solver.pointwise_kernels.metadata(),
            "build_wall_seconds": build_wall_seconds,
            "warmup_steps": config.warmup_steps,
            "warmup_wall_seconds": warmup_wall_seconds,
            "preprofile_wall_seconds": (
                build_wall_seconds + warmup_wall_seconds
            ),
            "dynamo_during_build": _counter_delta(
                counters_after_build,
                counters_before,
            ),
            "dynamo_during_warmup": _counter_delta(
                counters_after_warmup,
                counters_after_build,
            ),
            "dynamo_during_profile": _counter_delta(
                counters_after_profile,
                counters_after_warmup,
            ),
        },
        "projected_transforms": {
            **solver.model.spectral_projector.execution_metadata(),
            "retained_axis_counts": {
                "q": list(
                    solver.model.spectral_projector.retained_axis_counts(
                        Q_BCS
                    )
                ),
                "normal_velocity": list(
                    solver.model.spectral_projector.retained_axis_counts(
                        NORMAL_BCS
                    )
                ),
            },
            "computed_axis_sizes": {
                "q": list(
                    solver.model.spectral_projector.computed_axis_sizes(
                        Q_BCS
                    )
                ),
                "normal_velocity": list(
                    solver.model.spectral_projector.computed_axis_sizes(
                        NORMAL_BCS
                    )
                ),
            },
        },
        "timings": timings,
        "throughput": {
            "profile_wall_seconds": profile_wall_seconds,
            "whole_timestep_seconds": whole_timestep_seconds,
            "mean_timestep_seconds": whole_timestep_seconds / config.profile_steps,
            "timesteps_per_second": config.profile_steps / whole_timestep_seconds,
        },
        "memory": memory,
        "final_state_sha256": state_sha256,
    }


def _three_ints(value: str) -> tuple[int, int, int]:
    values = tuple(int(item) for item in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated integers")
    return values


def _three_floats(value: str) -> tuple[float, float, float]:
    values = tuple(float(item) for item in value.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_three_ints, default=(64, 64, 32))
    parser.add_argument("--lengths", type=_three_floats, default=(100.0, 100.0, 20.0))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument(
        "--dealias-rule",
        choices=tuple(DEALIAS_RULE_FRACTIONS),
        default="cubic_half",
    )
    parser.add_argument(
        "--projected-transform-execution",
        choices=PROJECTED_TRANSFORM_EXECUTION_MODES,
        default=DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
        help=(
            "A/B control for transforms whose output is immediately "
            "projected or whose input is already projected. full retains "
            "the qualified path; truncated skips discarded DCT/DST modes "
            "while preserving full-shape spectral storage."
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument(
        "--spectral-refresh-interval",
        type=int,
        help="Positive step interval; omit to disable refresh during timing.",
    )
    parser.add_argument("--pressure-diagnostics", action="store_true")
    parser.add_argument("--disable-q-gradient-reuse", action="store_true")
    parser.add_argument(
        "--molecular-field-linear-space",
        choices=("physical", "spectral"),
        default=DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE,
        help=(
            "A/B control for the raw molecular-field linear terms. "
            "spectral is the production default and keeps the L1 laplacian "
            "in modal space; physical retains the compatibility path."
        ),
    )
    parser.add_argument(
        "--stress-divergence-sum-space",
        choices=("physical", "spectral"),
        default=DEFAULT_STRESS_DIVERGENCE_SUM_SPACE,
        help=(
            "A/B control for divergence assembly. spectral adds compatible "
            "derivative coefficients before inverse transforms; physical "
            "retains the production path."
        ),
    )
    parser.add_argument(
        "--pointwise-execution",
        choices=POINTWISE_EXECUTION_MODES,
        default=DEFAULT_POINTWISE_EXECUTION,
        help=(
            "A/B control for Beris--Edwards pointwise algebra. compile is "
            "the H100-qualified production default and uses fixed-shape "
            "full-graph TorchInductor with no silent fallback; eager retains "
            "the validated compatibility path."
        ),
    )
    parser.add_argument(
        "--transform-execution-order",
        choices=("legacy", "real_first"),
        default=DEFAULT_TRANSFORM_EXECUTION_ORDER,
        help=(
            "Tensor-product execution plan (default: real_first). "
            "real_first applies DCT/DST axes while data are real; legacy "
            "retains the historical axis order."
        ),
    )
    parser.add_argument("--snapshot-interval", type=int)
    parser.add_argument("--snapshot-directory")
    parser.add_argument("--save-hydrodynamics", action="store_true")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"output exists: {args.output}; pass --overwrite to replace it")
    return args


def main() -> None:
    args = parse_args()
    result = run_profile(
        ProfileConfig(
            shape=args.shape,
            lengths=args.lengths,
            device=args.device,
            dtype=args.dtype,
            dt=args.dt,
            dealias_rule=args.dealias_rule,
            projected_transform_execution=(
                args.projected_transform_execution
            ),
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            spectral_refresh_interval=args.spectral_refresh_interval,
            pressure_diagnostics=args.pressure_diagnostics,
            reuse_q_gradients=not args.disable_q_gradient_reuse,
            molecular_field_linear_space=(
                args.molecular_field_linear_space
            ),
            stress_divergence_sum_space=(
                args.stress_divergence_sum_space
            ),
            pointwise_execution=args.pointwise_execution,
            transform_execution_order=args.transform_execution_order,
            snapshot_interval=args.snapshot_interval,
            snapshot_directory=args.snapshot_directory,
            save_hydrodynamics=args.save_hydrodynamics,
            seed=args.seed,
        )
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
