#!/usr/bin/env python3
"""RC4.4.7 installed-wheel Channel pressure production profiler.

The profiler is deliberately evidence-only: it records every timed sample,
pressure termination history, transform counts, compiler counters and CUDA
memory observations.  Threshold adjudication belongs to
``analyze_rc447_pressure_h100.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

import torch

if __package__:
    from benchmarks.profile_beris_edwards_timestep import (
        RegionTimer,
        _counter_delta,
        _dynamo_counter_snapshot,
        _git_provenance,
        _install_transform_timers,
    )
    from benchmarks.profile_channel_runtime_timestep import (
        ChannelRuntimeProfileConfig,
        _build,
        _hash_tensors,
        _initial_q,
        _scheduled_refreshes,
        _validate,
    )
else:
    from profile_beris_edwards_timestep import (
        RegionTimer,
        _counter_delta,
        _dynamo_counter_snapshot,
        _git_provenance,
        _install_transform_timers,
    )
    from profile_channel_runtime_timestep import (
        ChannelRuntimeProfileConfig,
        _build,
        _hash_tensors,
        _initial_q,
        _scheduled_refreshes,
        _validate,
    )


SCHEMA = "pssolver.rc4_4_7.channel_pressure_profile.v1"
ROLES = ("baseline", "candidate")


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
        if temporary.exists():
            temporary.unlink()


def _graph_break_count(value: object) -> int:
    if isinstance(value, dict):
        return sum(
            int(item)
            if key in {"graph_break", "graph_breaks"}
            and isinstance(item, int)
            and not isinstance(item, bool)
            else _graph_break_count(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_graph_break_count(item) for item in value)
    return 0


def _pointwise_metadata(adapter) -> dict[str, object]:
    model = adapter.solver.model
    owners = (
        _pressure_static_model(adapter),
        getattr(model, "nlmodel", None),
    )
    for owner in owners:
        kernels = getattr(owner, "pointwise_kernels", None)
        if kernels is not None and hasattr(kernels, "metadata"):
            return {"applicable": True, **dict(kernels.metadata())}
    return {
        "applicable": False,
        "reason": "legacy_channel_does_not_expose_pointwise_execution",
        "requested": "not_exposed",
        "effective": "not_exposed",
        "fallback_allowed": None,
        "fallback_reason": None,
    }


def _pressure_static_model(adapter):
    """Return the pressure owner used by the frozen Channel profiler layer."""

    try:
        return adapter.solver.model.static_model
    except AttributeError as exc:
        raise RuntimeError(
            "Channel profile adapter does not expose solver.model.static_model"
        ) from exc


def _pressure_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _pressure_residual(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise RuntimeError(f"{name} must be a finite non-negative number")
    return result


def _pressure_observation(adapter, *, role: str) -> dict[str, object]:
    """Read the exact diagnostics schema available to each frozen Provider role.

    Both installed Provider versions are driven through
    ``LegacyChannelRuntimeAdapter``.  That adapter does not define
    ``flow_diagnostics()``; the established Channel benchmark surface owns the
    pressure diagnostics on ``solver.model.static_model``.  Role selection is
    explicit so a malformed candidate cannot silently fall back to the legacy
    evidence schema.
    """

    if role not in ROLES:
        raise ValueError(f"unsupported role: {role!r}")
    pressure = _pressure_static_model(adapter)
    if role == "baseline":
        iterations = _pressure_count(
            getattr(pressure, "last_pressure_iterations", None),
            "legacy last_pressure_iterations",
        )
        residual = _pressure_residual(
            getattr(pressure, "last_pressure_relative_residual", None),
            "legacy last_pressure_relative_residual",
        )
        return {
            "schema": "legacy_public_flow_diagnostics.v1",
            "achieved_iteration_count": iterations,
            "termination_reason": "not_exposed_by_legacy_schema",
            "termination_reason_observed": False,
            "acceptable": None,
            "acceptable_observed": False,
            "achieved_normalized_residual": residual,
        }

    diagnostics = getattr(pressure, "last_pressure_diagnostics", None)
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError(
            "candidate static pressure model does not expose "
            "last_pressure_diagnostics"
        )
    if diagnostics.get("schema_version") != 2:
        raise RuntimeError("candidate pressure diagnostics schema_version must be 2")
    iterations = _pressure_count(
        diagnostics.get("achieved_iteration_count"),
        "candidate achieved_iteration_count",
    )
    termination_reason = diagnostics.get("termination_reason")
    if not isinstance(termination_reason, str) or not termination_reason:
        raise RuntimeError("candidate termination_reason must be a non-empty string")
    acceptable = diagnostics.get("acceptable")
    if not isinstance(acceptable, bool):
        raise RuntimeError("candidate acceptable must be a boolean")
    residual = _pressure_residual(
        diagnostics.get("achieved_normalized_residual"),
        "candidate achieved_normalized_residual",
    )
    return {
        "schema": "channel_pressure_diagnostics.v2",
        "achieved_iteration_count": iterations,
        "termination_reason": termination_reason,
        "termination_reason_observed": True,
        "acceptable": acceptable,
        "acceptable_observed": True,
        "achieved_normalized_residual": residual,
    }


def run_profile(
    config: ChannelRuntimeProfileConfig,
    *,
    role: str,
    provider_commit: str,
    provider_wheel_sha256: str,
) -> dict[str, object]:
    """Run one fresh 10+50-step profile using the allocated CUDA device."""

    _validate(config)
    if role not in ROLES:
        raise ValueError(f"unsupported role: {role!r}")
    if len(provider_commit) != 40 or len(provider_wheel_sha256) != 64:
        raise ValueError("full provider commit and wheel SHA-256 are required")
    device = torch.device(config.device)
    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RC4.4.7 profiles require an allocated CUDA device")
    before_tf32 = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    after_tf32 = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    initial_q = _initial_q(config)
    initial_sha256 = _hash_tensors(
        (name, initial_q[name]) for name in sorted(initial_q)
    )
    timer = RegionTimer(device, timed_regions=frozenset())
    counters_before = _dynamo_counter_snapshot()

    with torch.no_grad():
        spec, adapter = _build(config, initial_q)
        del initial_q
        _install_transform_timers(adapter.solver, timer)
        torch.cuda.synchronize(device)
        counters_after_build = _dynamo_counter_snapshot()
        adapter.advance(config.warmup_steps)
        torch.cuda.synchronize(device)
        counters_after_warmup = _dynamo_counter_snapshot()
        integrator = adapter.solver.integrator
        refresh_interval = integrator.spectral_refresh_interval
        refresh_step_before = int(integrator.step_count)
        refresh_count_before = int(integrator.refresh_count)
        transform_group_count = len(integrator.dynamic_transform_groups)
        expected_refreshes = _scheduled_refreshes(
            start_step_count=refresh_step_before,
            interval=refresh_interval,
            steps=config.profile_steps,
        )
        torch.cuda.reset_peak_memory_stats(device)
        timer.reset()
        samples: list[float] = []
        pressure_history: list[dict[str, object]] = []
        for _ in range(config.profile_steps):
            started = time.perf_counter()
            adapter.advance(1)
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
            pressure_history.append(_pressure_observation(adapter, role=role))
        transform_summary = timer.summarize()
        counters_after_profile = _dynamo_counter_snapshot()
        adapter.synchronize_for_observation()
        torch.cuda.synchronize(device)
        final_sha256 = _hash_tensors(
            (("spatial", adapter.fields.spatial), ("spectral", adapter.fields.spectral))
        )
        memory_stats = torch.cuda.memory_stats(device)
        runtime = adapter.to_metadata()
        pointwise = _pointwise_metadata(adapter)
        finite = bool(
            torch.isfinite(adapter.fields.spatial).all().item()
            and torch.isfinite(adapter.fields.spectral).all().item()
        )
        refresh_step_after = int(integrator.step_count)
        refresh_count_after = int(integrator.refresh_count)
        completed_steps = int(adapter.completed_steps)

    if runtime["requested"] != runtime["effective"] or runtime["fallback_used"]:
        raise RuntimeError("runtime fallback or identity mismatch")
    observed_refreshes = refresh_count_after - refresh_count_before
    if observed_refreshes != expected_refreshes:
        raise RuntimeError("spectral-refresh count differs from integration clock")
    forward = int(transform_summary.get("transform_forward", {}).get("calls", 0))
    inverse = int(transform_summary.get("transform_inverse", {}).get("calls", 0))
    refresh_forward = expected_refreshes * transform_group_count
    base_forward = forward - refresh_forward
    if base_forward < 0:
        raise RuntimeError("scheduled refreshes exceed observed forward transforms")
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    profile_delta = _counter_delta(counters_after_profile, counters_after_warmup)
    return {
        "schema": SCHEMA,
        "role": role,
        "provider_identity": {
            "commit": provider_commit,
            "wheel_sha256": provider_wheel_sha256,
        },
        "config": asdict(config),
        "configuration_identity": spec.identity_metadata(),
        "runtime_identity": runtime,
        "environment": {
            "git_diagnostic_only": _git_provenance(),
            "pssolver_import": str(Path(__import__("pssolver").__file__).resolve()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "requested_device": "cuda",
            "allocated_device": str(torch.empty((), device=device).device),
            "device_name": torch.cuda.get_device_name(device),
            "tf32_before_policy": before_tf32,
            "tf32_after_policy": after_tf32,
        },
        "timing": {
            "samples_seconds": samples,
            "mean_timestep_seconds": mean,
            "median_timestep_seconds": median,
            "sample_standard_deviation_seconds": stdev,
            "coefficient_of_variation": stdev / mean,
        },
        "memory": {
            "peak_allocated_bytes": int(memory_stats["allocated_bytes.all.peak"]),
            "peak_active_bytes": int(memory_stats["active_bytes.all.peak"]),
            "peak_reserved_bytes": int(memory_stats["reserved_bytes.all.peak"]),
        },
        "transform_calls": {
            "forward": forward,
            "inverse": inverse,
            "forward_per_step": forward / config.profile_steps,
            "inverse_per_step": inverse / config.profile_steps,
            "base_forward_per_step": base_forward / config.profile_steps,
            "scheduled_refresh_forward_calls": refresh_forward,
        },
        "spectral_refresh": {
            "interval": refresh_interval,
            "step_count_before": refresh_step_before,
            "step_count_after": refresh_step_after,
            "count_before": refresh_count_before,
            "count_after": refresh_count_after,
            "expected_in_window": expected_refreshes,
            "observed_in_window": observed_refreshes,
            "dynamic_transform_group_count": transform_group_count,
        },
        "compiler": {
            "pointwise": pointwise,
            "during_build": _counter_delta(counters_after_build, counters_before),
            "during_warmup": _counter_delta(counters_after_warmup, counters_after_build),
            "during_profile": profile_delta,
            "graph_breaks_during_profile": _graph_break_count(profile_delta),
        },
        "pressure_history": pressure_history,
        "initial_q_sha256": initial_sha256,
        "final_state_sha256": final_sha256,
        "completed_steps": completed_steps,
        "finite": finite,
    }


def _triplet(value: str, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--provider-commit", required=True)
    parser.add_argument("--provider-wheel-sha256", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", type=lambda value: _triplet(value, int), required=True)
    parser.add_argument("--lengths", type=lambda value: _triplet(value, float), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = ChannelRuntimeProfileConfig(
        runtime_path="legacy_channel",
        trial=args.trial,
        shape=args.shape,
        lengths=args.lengths,
        device="cuda",
        warmup_steps=10,
        profile_steps=50,
    )
    _atomic_json(
        args.output,
        run_profile(
            config,
            role=args.role,
            provider_commit=args.provider_commit,
            provider_wheel_sha256=args.provider_wheel_sha256,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
