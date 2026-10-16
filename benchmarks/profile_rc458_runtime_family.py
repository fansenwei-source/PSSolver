#!/usr/bin/env python3
"""Profile one frozen RC4.5.8 runtime-family cell.

This module is qualification support, not a production runtime.  It gives the
eight RC4.5.8 runtime families one fail-closed result schema while continuing
to construct and step the real Provider runtimes.
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
from typing import Callable, Iterable, Mapping

import numpy as np
import torch

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.profile_beris_edwards_timestep import (
    RegionTimer,
    _counter_delta,
    _dynamo_counter_snapshot,
    _install_transform_timers,
)
from benchmarks.profile_plane_runtime_timestep import (
    RuntimeProfileConfig,
    _build_runtime as _build_plane_runtime,
    _initial_q as _plane_initial_q,
)
from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
)
from pssolver.applications.channel_beris_edwards import load_channel_initial_q
from pssolver.applications.periodic_beris_edwards import load_periodic_initial_q
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.channel import build_active_nematic_channel
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    ChannelRuntimePath,
)
from pssolver.functional import (
    build_channel_activity_functional_runtime,
    build_functional_runtime,
    channel_activity_functional_request,
    periodic_activity_functional_request,
)
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    Q_COMPONENTS,
)
from pssolver.runtime.channel_active_nematics import (
    ChannelRuntimeBuildRequest,
    build_channel_active_nematic_runtime,
)
from pssolver.runtime.channel_application_bridge import (
    build_package_compiled_channel_runtime,
)
from pssolver.runtime.channel_beris_edwards import (
    ChannelBerisEdwardsRuntimeBuildRequest,
    build_channel_beris_edwards_runtime,
)
from pssolver.runtime.periodic_beris_edwards import (
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)


SCHEMA = "pssolver.rc4_5_8.runtime_family_profile.v1"
FAMILIES = (
    "plane_legacy_production",
    "plane_compiled_v2",
    "periodic_complete_stress",
    "channel_legacy",
    "channel_compiled_v2",
    "channel_complete_stress",
    "periodic_functional",
    "channel_functional",
)


@dataclass(frozen=True, slots=True)
class RC458ProfileConfig:
    family: str
    provider_role: str
    provider_commit: str
    provider_wheel_sha256: str
    trial: int
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    device: str = "cuda"
    dtype: str = "float64"
    warmup_steps: int = 10
    profile_steps: int = 60
    post_flush_rewarm_steps: int = 0
    seed: int = 20261013


@dataclass(slots=True)
class _RuntimeCase:
    step: Callable[[], None]
    state: Callable[[], tuple[torch.Tensor, ...]]
    completed_steps: Callable[[], int]
    solver: object
    runtime_identity: dict[str, object]
    configuration_identity: dict[str, object]
    initial_state_sha256: str
    public_diagnostics: Callable[[], object] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate(config: RC458ProfileConfig) -> None:
    if config.family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if config.provider_role not in {"baseline", "candidate"}:
        raise ValueError("provider_role must be baseline or candidate")
    if len(config.provider_commit) != 40:
        raise ValueError("provider_commit must be a full commit identity")
    if len(config.provider_wheel_sha256) != 64:
        raise ValueError("provider_wheel_sha256 must be a SHA-256")
    if not isinstance(config.trial, int) or isinstance(config.trial, bool):
        raise TypeError("trial must be an integer")
    if config.trial <= 0:
        raise ValueError("trial must be positive")
    if len(config.shape) != 3 or any(value <= 1 for value in config.shape):
        raise ValueError("shape must contain three integers greater than one")
    if len(config.lengths) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in config.lengths
    ):
        raise ValueError("lengths must contain three positive finite values")
    if config.device != "cuda":
        raise ValueError("RC4.5.8 formal profiles require --device cuda")
    expected_dtype = (
        "float32"
        if config.family in {"channel_legacy", "channel_compiled_v2"}
        else "float64"
    )
    if config.dtype != expected_dtype:
        raise ValueError(
            f"{config.family} is frozen to the qualified {expected_dtype} contract"
        )
    if (
        config.warmup_steps < 0
        or config.profile_steps <= 0
        or config.post_flush_rewarm_steps < 0
    ):
        raise ValueError("warmup/profile step counts are invalid")


def _tensor_sha256(values: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        current = value.detach().contiguous().cpu()
        digest.update(str(current.dtype).encode("ascii"))
        digest.update(json.dumps(list(current.shape)).encode("ascii"))
        digest.update(current.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite(values: Iterable[torch.Tensor]) -> bool:
    return all(bool(value.detach().isfinite().all().item()) for value in values)


def _q_values(
    shape: tuple[int, int, int], *, seed: int
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.normal(scale=0.01, size=(*shape, 5)).astype(np.float64)
    values[..., 0] += 0.2
    values[..., 3] -= 0.1
    return values


def _snapshot(root: Path, config: RC458ProfileConfig) -> Path:
    directory = root / "snapshot"
    directory.mkdir(parents=True)
    path = directory / "Q_0.npy"
    np.save(path, _q_values(config.shape, seed=config.seed), allow_pickle=False)
    return directory


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


def _complete_simulation(
    root: Path, config: RC458ProfileConfig, *, channel: bool
) -> Simulation:
    model = _model()
    geometry = (
        RectangularChannel(shape=config.shape, lengths=config.lengths)
        if channel
        else PeriodicBox(shape=config.shape, lengths=config.lengths)
    )
    policies = {
        "Q": neumann_q(),
        "velocity": no_slip_velocity() if channel else free_slip_velocity(),
        "pressure": neumann_pressure_compatibility(),
    }
    runtime_path = "channel_complete_stress" if channel else "periodic_spectral"
    options = {
        "tf32": "off",
        "molecular_field_linear_space": "spectral",
        "stress_divergence_sum_space": "physical" if channel else "spectral",
        "pointwise_execution": "eager",
        "disable_q_gradient_reuse": channel is False,
    }
    discretization = (
        {
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        }
        if channel
        else {}
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies=policies,
        ),
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex" if channel else "hermitian_half",
            hermitian_axis=None if channel else 1,
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(_snapshot(root, config), step=0),
        execution=TorchSpectralExecution(
            runtime_path=runtime_path,
            device=config.device,
            options=options,
        ),
        output=Output(
            directory=root / "unused-output",
            steps=(
                config.warmup_steps
                + config.post_flush_rewarm_steps
                + config.profile_steps
            ),
            save_interval=(
                config.warmup_steps
                + config.post_flush_rewarm_steps
                + config.profile_steps
                + 1
            ),
            diagnostic_interval=(
                config.warmup_steps
                + config.post_flush_rewarm_steps
                + config.profile_steps
                + 1
            ),
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization=discretization,
    )


def _production_case(
    root: Path, config: RC458ProfileConfig, *, channel: bool
) -> _RuntimeCase:
    simulation = _complete_simulation(root, config, channel=channel)
    compiled = compile_simulation(simulation)
    spec = compiled.application_request
    if channel:
        initial, _, _ = load_channel_initial_q(spec)
        adapter = build_channel_beris_edwards_runtime(
            ChannelBerisEdwardsRuntimeBuildRequest(spec, initial, config.device)
        )
    else:
        initial, _, _ = load_periodic_initial_q(spec)
        adapter = build_periodic_beris_edwards_runtime(
            PeriodicRuntimeBuildRequest(spec, initial, config.device)
        )
    state = lambda: (adapter.fields.spatial, adapter.fields.spectral)
    initial_sha = _tensor_sha256(state())
    return _RuntimeCase(
        step=lambda: adapter.advance(1),
        state=state,
        completed_steps=lambda: int(adapter.completed_steps),
        solver=adapter.solver,
        runtime_identity=dict(adapter.to_metadata()),
        configuration_identity=dict(spec.simulation.identity_metadata()),
        initial_state_sha256=initial_sha,
        public_diagnostics=getattr(adapter, "flow_diagnostics", None),
    )


def _functional_case(
    root: Path, config: RC458ProfileConfig, *, channel: bool
) -> _RuntimeCase:
    simulation = _complete_simulation(root, config, channel=channel)
    request = (
        channel_activity_functional_request(simulation.specification)
        if channel
        else periodic_activity_functional_request(simulation.specification)
    )
    runtime = (
        build_channel_activity_functional_runtime(request)
        if channel
        else build_functional_runtime(request)
    )
    holder = {"state": runtime.initial_state(), "step": 0}
    control_spec = runtime.control_specs[0].tensor
    control_dtype = (
        getattr(torch, control_spec.dtype)
        if isinstance(control_spec.dtype, str)
        else control_spec.dtype
    )
    controls = {
        "activity": torch.full(
            control_spec.shape,
            0.01,
            dtype=control_dtype,
            device=control_spec.device,
        )
    }

    def step() -> None:
        holder["state"] = runtime.step(  # type: ignore[assignment]
            holder["state"], controls, holder["step"]  # type: ignore[arg-type]
        )
        holder["step"] = int(holder["step"]) + 1

    def state() -> tuple[torch.Tensor, ...]:
        return tuple(holder["state"])  # type: ignore[arg-type]

    kind = "channel_activity_batch_one" if channel else "periodic_activity_batch_one"
    identity = runtime.identity().to_metadata()
    return _RuntimeCase(
        step=step,
        state=state,
        completed_steps=lambda: int(holder["step"]),
        solver=runtime._solver,
        runtime_identity={
            "requested": kind,
            "effective": kind,
            "fallback_allowed": False,
            "fallback_used": False,
            "functional_identity": identity,
        },
        configuration_identity=simulation.identity_metadata(),
        initial_state_sha256=_tensor_sha256(state()),
        public_diagnostics=getattr(runtime, "diagnostics", None),
    )


def _channel_active_case(config: RC458ProfileConfig) -> _RuntimeCase:
    runtime_path = {
        "channel_legacy": ChannelRuntimePath.LEGACY_CHANNEL,
        "channel_compiled_v2": ChannelRuntimePath.COMPILED_CHANNEL_V2,
    }[config.family]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    dtype = {"float32": torch.float32, "float64": torch.float64}[config.dtype]
    values = 0.01 * torch.randn(
        5, *config.shape, generator=generator, dtype=dtype
    )
    values[0].add_(1.0 / 3.0)
    values[3].add_(-1.0 / 6.0)
    initial = {name: values[index] for index, name in enumerate(Q_COMPONENTS)}
    spec = ChannelActiveNematicRunSpec(
        shape=config.shape,
        lengths=config.lengths,
        dt=0.01,
        steps=(
            config.warmup_steps
            + config.post_flush_rewarm_steps
            + config.profile_steps
        ),
        save_interval=(
            config.warmup_steps
            + config.post_flush_rewarm_steps
            + config.profile_steps
        ),
        diagnostic_interval=(
            config.warmup_steps
            + config.post_flush_rewarm_steps
            + config.profile_steps
        ),
        activity=5.0,
        device=config.device,
        dtype=config.dtype,
        pressure_relative_tolerance=1.0e-6,
        pressure_max_iterations=80,
        runtime_path=runtime_path,
    )
    request = ChannelRuntimeBuildRequest(
        spec,
        {
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        },
        initial,
        config.device,
    )
    material = spec.components.material
    pressure = spec.components.pressure_solver

    def legacy_builder():
        solver = build_active_nematic_channel(
            spec.shape,
            spec.lengths,
            spec.dt,
            initial,
            device=config.device,
            batchsize=1,
            rho=material.rho,
            elastic_constant=material.elastic_constant,
            beta=material.beta,
            friction=material.friction,
            viscosity=material.viscosity,
            pressure_rel_tol=pressure.relative_tolerance,
            pressure_max_iter=pressure.max_iterations,
        )
        solver.parameters["alpha"] = torch.full(
            (1, *spec.shape),
            material.activity,
            dtype=solver.dtype,
            device=solver.device,
        )
        return solver

    adapter = build_channel_active_nematic_runtime(
        request,
        legacy_builder=(
            legacy_builder
            if runtime_path is ChannelRuntimePath.LEGACY_CHANNEL
            else None
        ),
        compiled_builder=(
            lambda: build_package_compiled_channel_runtime(request)
            if runtime_path is ChannelRuntimePath.COMPILED_CHANNEL_V2
            else None
        ),
    )
    state = lambda: (adapter.fields.spatial, adapter.fields.spectral)
    return _RuntimeCase(
        step=lambda: adapter.advance(1),
        state=state,
        completed_steps=lambda: int(adapter.completed_steps),
        solver=adapter.solver,
        runtime_identity=dict(adapter.to_metadata()),
        configuration_identity=dict(spec.identity_metadata()),
        initial_state_sha256=_tensor_sha256(state()),
        public_diagnostics=getattr(adapter, "flow_diagnostics", None),
    )


def _plane_case(config: RC458ProfileConfig) -> _RuntimeCase:
    runtime_path = (
        "legacy_production"
        if config.family == "plane_legacy_production"
        else "compiled_v2"
    )
    native = RuntimeProfileConfig(
        runtime_path=runtime_path,
        trial=config.trial,
        shape=config.shape,
        lengths=config.lengths,
        device=config.device,
        dtype=config.dtype,
        warmup_steps=config.warmup_steps,
        profile_steps=config.profile_steps + config.post_flush_rewarm_steps,
        seed=config.seed,
        pointwise_execution="compile",
    )
    initial = _plane_initial_q(native)
    spec, adapter = _build_plane_runtime(native, initial)
    state = lambda: (adapter.fields.spatial, adapter.fields.spectral)
    return _RuntimeCase(
        step=lambda: adapter.advance(1),
        state=state,
        completed_steps=lambda: int(adapter.completed_steps),
        solver=adapter.solver,
        runtime_identity=dict(adapter.to_metadata()),
        configuration_identity=dict(spec.identity_metadata()),
        initial_state_sha256=_tensor_sha256(state()),
        public_diagnostics=getattr(adapter, "flow_diagnostics", None),
    )


def _build_case(root: Path, config: RC458ProfileConfig) -> _RuntimeCase:
    if config.family.startswith("plane_"):
        return _plane_case(config)
    if config.family in {"channel_legacy", "channel_compiled_v2"}:
        return _channel_active_case(config)
    if config.family == "periodic_complete_stress":
        return _production_case(root, config, channel=False)
    if config.family == "channel_complete_stress":
        return _production_case(root, config, channel=True)
    if config.family == "periodic_functional":
        return _functional_case(root, config, channel=False)
    if config.family == "channel_functional":
        return _functional_case(root, config, channel=True)
    raise AssertionError("unreachable family")


def _pointwise_metadata(solver: object) -> dict[str, object]:
    model = getattr(solver, "model", None)
    owners = (
        getattr(model, "static_model", None),
        getattr(model, "nlmodel", None),
    )
    for owner in owners:
        kernels = getattr(owner, "pointwise_kernels", None)
        if kernels is not None and hasattr(kernels, "metadata"):
            return {"applicable": True, **dict(kernels.metadata())}
    return {
        "applicable": False,
        "requested": "not_exposed",
        "effective": "not_exposed",
        "fallback_allowed": None,
        "fallback_reason": None,
    }


def _timed_steps(
    operation: Callable[[], None], count: int, device: torch.device
) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    for start, stop in zip(starts, stops, strict=True):
        start.record()
        operation()
        stop.record()
    torch.cuda.synchronize(device)
    return [
        float(start.elapsed_time(stop)) / 1000.0
        for start, stop in zip(starts, stops, strict=True)
    ]


def _cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "active_bytes": int(stats.get("active_bytes.all.current", 0)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "peak_active_bytes": int(stats.get("active_bytes.all.peak", 0)),
    }


def _find_nested_value(value: object, names: tuple[str, ...]) -> tuple[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    for name in names:
        if name in value:
            return name, value[name]
    for nested in value.values():
        found = _find_nested_value(nested, names)
        if found is not None:
            return found
    return None


def _normalize_public_diagnostics(callback: Callable[[], object] | None) -> dict[str, object]:
    if callback is None or not callable(callback):
        return {
            "applicable": False,
            "reason": "runtime_does_not_expose_public_diagnostics",
        }
    try:
        raw = callback()
    except Exception as error:  # diagnostic evidence must not hide the exception
        return {
            "applicable": False,
            "reason": f"public_diagnostics_raised_{type(error).__name__}",
        }
    if not isinstance(raw, Mapping):
        return {
            "applicable": False,
            "reason": "public_diagnostics_did_not_return_a_mapping",
        }
    aliases = {
        "pressure_iteration_count": (
            "last_pressure_iterations",
            "pressure_iterations",
            "iterations",
        ),
        "pressure_termination_reason": (
            "last_pressure_termination_reason",
            "termination_reason",
        ),
        "pressure_absolute_residual": (
            "last_pressure_residual",
            "absolute_residual",
            "residual",
        ),
        "pressure_relative_residual": (
            "last_pressure_relative_residual",
            "relative_residual",
        ),
    }
    normalized: dict[str, object] = {
        "applicable": False,
        "observed_fields": [],
        "source_keys": {},
    }
    observed_fields: list[str] = []
    source_keys: dict[str, str] = {}
    for destination, names in aliases.items():
        found = _find_nested_value(raw, names)
        if found is None:
            continue
        source, observed = found
        if destination == "pressure_iteration_count":
            if isinstance(observed, bool) or not isinstance(observed, int):
                continue
            value: object = int(observed)
        elif destination == "pressure_termination_reason":
            if not isinstance(observed, str) or not observed:
                continue
            value = observed
        else:
            if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                continue
            numeric = float(observed)
            if not math.isfinite(numeric):
                normalized[f"{destination}_finite"] = False
                continue
            normalized[f"{destination}_finite"] = True
            value = numeric
        normalized[destination] = value
        observed_fields.append(destination)
        source_keys[destination] = source
    normalized["observed_fields"] = observed_fields
    normalized["source_keys"] = source_keys
    normalized["applicable"] = bool(observed_fields)
    if not observed_fields:
        normalized["reason"] = "no_supported_pressure_field_was_exposed"
    return normalized


def _nvidia_smi_telemetry() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as error:
        return {
            "applicable": False,
            "reason": f"nvidia_smi_raised_{type(error).__name__}",
            "command": command,
        }
    if result.returncode != 0:
        return {
            "applicable": False,
            "reason": "nvidia_smi_returned_nonzero",
            "returncode": result.returncode,
            "stderr": result.stderr.strip(),
            "command": command,
        }
    rows: list[dict[str, object]] = []
    try:
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            pstate, sm, memory, power, temperature = (
                item.strip() for item in line.split(",")
            )
            numeric = [float(sm), float(memory), float(power), float(temperature)]
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError("non-finite telemetry")
            rows.append(
                {
                    "pstate": pstate,
                    "sm_clock_mhz": numeric[0],
                    "memory_clock_mhz": numeric[1],
                    "power_draw_watts": numeric[2],
                    "temperature_celsius": numeric[3],
                }
            )
    except Exception as error:
        return {
            "applicable": False,
            "reason": f"nvidia_smi_output_parse_{type(error).__name__}",
            "stdout": result.stdout.strip(),
            "command": command,
        }
    return {
        "applicable": bool(rows),
        "reason": None if rows else "nvidia_smi_returned_no_rows",
        "rows": rows,
        "command": command,
    }


def _diagnostic_replay(
    root: Path,
    config: RC458ProfileConfig,
    *,
    device: torch.device,
    expected_initial_sha256: str,
    expected_final_sha256: str,
) -> dict[str, object]:
    replay = _build_case(root, config)
    _require(
        replay.initial_state_sha256 == expected_initial_sha256,
        "diagnostic replay initial state differs",
    )
    for _ in range(config.warmup_steps):
        replay.step()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(config.post_flush_rewarm_steps):
        replay.step()
    torch.cuda.synchronize(device)
    records: list[dict[str, object]] = []
    for index in range(1, config.profile_steps + 1):
        replay.step()
        torch.cuda.synchronize(device)
        records.append(
            {
                "step_index": index,
                "runtime_completed_steps": replay.completed_steps(),
                "diagnostics": _normalize_public_diagnostics(
                    replay.public_diagnostics
                ),
            }
        )
    replay_state = replay.state()
    replay_final_sha256 = _tensor_sha256(replay_state)
    _require(_finite(replay_state), "diagnostic replay state is non-finite")
    _require(
        replay_final_sha256 == expected_final_sha256,
        "diagnostic replay final state differs from formal profile",
    )
    return {
        "separate_fresh_runtime": True,
        "used_for_performance_or_memory_adjudication": False,
        "initial_state_sha256": replay.initial_state_sha256,
        "final_state_sha256": replay_final_sha256,
        "completed_steps": replay.completed_steps(),
        "step_records": records,
        "finite": True,
    }


def _collect_and_empty_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run_profile(config: RC458ProfileConfig) -> dict[str, object]:
    _validate(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda")
    allocated = torch.empty((), device=device).device
    _require(str(allocated) == "cuda:0", "profile did not bind allocated cuda:0")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    caller_before = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    counters_before = _dynamo_counter_snapshot()

    with tempfile.TemporaryDirectory(prefix="pssolver-rc458-") as temporary:
        with torch.no_grad():
            case = _build_case(Path(temporary), config)
            # The timer is installed before warm-up but remains disabled until
            # reset(), so construction and warm-up calls cannot enter the
            # formal measurement window.
            timer = RegionTimer(device, timed_regions=frozenset())
            _install_transform_timers(case.solver, timer)
            counters_after_build = _dynamo_counter_snapshot()
            for _ in range(config.warmup_steps):
                case.step()
            torch.cuda.synchronize(device)
            counters_after_warmup = _dynamo_counter_snapshot()
            steps_after_warmup = case.completed_steps()
            memory_after_warmup = _cuda_memory_snapshot(device)
            gc.collect()
            torch.cuda.empty_cache()
            memory_after_empty_cache = _cuda_memory_snapshot(device)
            for _ in range(config.post_flush_rewarm_steps):
                case.step()
            torch.cuda.synchronize(device)
            counters_after_rewarm = _dynamo_counter_snapshot()
            steps_after_rewarm = case.completed_steps()
            memory_after_rewarm = _cuda_memory_snapshot(device)
            torch.cuda.reset_peak_memory_stats(device)
            timer.reset()
            samples = _timed_steps(case.step, config.profile_steps, device)
            transform_summary = timer.summarize()
            counters_after_profile = _dynamo_counter_snapshot()
            state = case.state()
            torch.cuda.synchronize(device)
            memory_stats = torch.cuda.memory_stats(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            memory_after_profile = _cuda_memory_snapshot(device)
            final_sha = _tensor_sha256(state)
            finite = _finite(state)
            runtime_completed_steps = case.completed_steps()
            runtime_identity = case.runtime_identity
            configuration_identity = dict(case.configuration_identity)
            pointwise = _pointwise_metadata(case.solver)
            initial_sha = case.initial_state_sha256
            telemetry_after_formal_window = _nvidia_smi_telemetry()
            diagnostic_replay: dict[str, object] | None = None
            if config.post_flush_rewarm_steps:
                del state
                del case
                del timer
                _collect_and_empty_cuda_cache()
                diagnostic_replay = _diagnostic_replay(
                    Path(temporary) / "diagnostic-replay",
                    config,
                    device=device,
                    expected_initial_sha256=initial_sha,
                    expected_final_sha256=final_sha,
                )

    caller_after = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    _require(
        runtime_identity.get("requested") == runtime_identity.get("effective"),
        "runtime requested/effective identity differs",
    )
    _require(runtime_identity.get("fallback_used") is False, "runtime fallback used")
    graph = _counter_delta(counters_after_profile, counters_after_rewarm)
    graph_breaks = graph["graph_breaks"]
    _require(isinstance(graph_breaks, int), "graph-break counter unavailable")
    compile_fallback = bool(
        pointwise.get("applicable")
        and pointwise.get("requested") == "compile"
        and pointwise.get("effective") != "compile"
    ) or pointwise.get("fallback_reason") not in {None, ""}
    forward = int(transform_summary.get("transform_forward", {}).get("calls", 0))
    inverse = int(transform_summary.get("transform_inverse", {}).get("calls", 0))
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "schema": SCHEMA,
        "config": {**asdict(config), "shape": list(config.shape), "lengths": list(config.lengths)},
        "provider_identity": {
            "role": config.provider_role,
            "commit": config.provider_commit,
            "wheel_sha256": config.provider_wheel_sha256,
        },
        "configuration_identity": configuration_identity,
        "runtime_identity": runtime_identity,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "requested_device": "cuda",
            "allocated_device": str(allocated),
            "device_name": torch.cuda.get_device_name(device),
            "pssolver_import": str(Path(__import__("pssolver").__file__).resolve()),
            "tf32_caller_before": caller_before,
            "tf32_caller_after": caller_after,
        },
        "timing": {
            "samples_seconds": samples,
            "mean_timestep_seconds": mean,
            "median_timestep_seconds": statistics.median(samples),
            "sample_standard_deviation_seconds": stdev,
            "coefficient_of_variation": stdev / mean,
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_active_bytes": int(memory_stats.get("active_bytes.all.peak", 0)),
            "peak_reserved_bytes": peak_reserved,
        },
        "transform_calls": {
            "forward": forward,
            "inverse": inverse,
            "forward_per_step": forward / config.profile_steps,
            "inverse_per_step": inverse / config.profile_steps,
        },
        "compiler": {
            "pointwise": pointwise,
            "during_build": _counter_delta(counters_after_build, counters_before),
            "during_warmup": _counter_delta(counters_after_warmup, counters_after_build),
            "during_post_flush_rewarm": _counter_delta(
                counters_after_rewarm, counters_after_warmup
            ),
            "during_profile": graph,
            "graph_breaks": graph_breaks,
            "compile_fallback": compile_fallback,
        },
        "measurement_window": {
            "warmup_steps": config.warmup_steps,
            "post_flush_rewarm_steps": config.post_flush_rewarm_steps,
            "timed_steps": config.profile_steps,
            "steps_after_warmup": steps_after_warmup,
            "measurement_start_completed_steps": steps_after_rewarm,
            "rewarm_timed": False,
            "rewarm_in_formal_transform_counters": False,
            "rewarm_in_formal_compiler_counters": False,
            "rewarm_in_formal_memory_peaks": False,
            "formal_memory_semantics": "steady_state_after_allocation_recovery",
            "memory_snapshots": {
                "after_warmup": memory_after_warmup,
                "after_empty_cache": memory_after_empty_cache,
                "after_post_flush_rewarm": memory_after_rewarm,
                "after_formal_timed_window": memory_after_profile,
            },
            "cold_recovery_memory": {
                "allocated_delta_bytes": (
                    memory_after_rewarm["allocated_bytes"]
                    - memory_after_empty_cache["allocated_bytes"]
                ),
                "reserved_delta_bytes": (
                    memory_after_rewarm["reserved_bytes"]
                    - memory_after_empty_cache["reserved_bytes"]
                ),
                "active_delta_bytes": (
                    memory_after_rewarm["active_bytes"]
                    - memory_after_empty_cache["active_bytes"]
                ),
            },
            "device_telemetry_after_formal_window": telemetry_after_formal_window,
        },
        "diagnostic_replay": diagnostic_replay,
        "initial_state_sha256": initial_sha,
        "final_state_sha256": final_sha,
        "completed_profile_steps": config.profile_steps,
        "runtime_completed_steps": runtime_completed_steps,
        "finite": finite,
    }


def _triplet(value: str, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return result


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
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
            raise RuntimeError("profile JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--provider-role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--provider-commit", required=True)
    parser.add_argument("--provider-wheel-sha256", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", type=lambda value: _triplet(value, int), required=True)
    parser.add_argument("--lengths", type=lambda value: _triplet(value, float), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--profile-steps", type=int, default=60)
    parser.add_argument("--post-flush-rewarm-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_profile(
        RC458ProfileConfig(
            family=args.family,
            provider_role=args.provider_role,
            provider_commit=args.provider_commit,
            provider_wheel_sha256=args.provider_wheel_sha256,
            trial=args.trial,
            shape=args.shape,
            lengths=args.lengths,
            device=args.device,
            dtype=args.dtype,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            post_flush_rewarm_steps=args.post_flush_rewarm_steps,
            seed=args.seed,
        )
    )
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
