#!/usr/bin/env python3
"""Run one analysis-only rc5 G-1 timing-instrumentation profile.

The asynchronous window resembles the v10 launch topology.  A second fresh
runtime is synchronized after every step so host wall time and public pressure
diagnostics have an unambiguous step association.  Neither window is formal
performance qualification evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import TextIO

import torch

from benchmarks.profile_rc458_runtime_family import (
    RC458ProfileConfig,
    _build_case,
    _finite,
    _normalize_public_diagnostics,
    _tensor_sha256,
    _validate,
)


SCHEMA = "pssolver.rc5.g1.timing_instrumentation_profile.v1"
TELEMETRY_FIELDS = (
    "timestamp",
    "index",
    "uuid",
    "pci.bus_id",
    "pstate",
    "clocks.current.sm",
    "clocks.current.memory",
    "power.draw",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite_number(value: str, name: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"telemetry {name} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"telemetry {name} is not finite")
    return result


def parse_telemetry_line(line: str) -> dict[str, object]:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != len(TELEMETRY_FIELDS):
        raise ValueError("telemetry row has the wrong field count")
    (
        timestamp,
        index,
        uuid,
        bus,
        pstate,
        sm,
        memory,
        power,
        temperature,
        gpu_utilization,
        memory_utilization,
    ) = parts
    if not timestamp or not uuid or not bus or not pstate:
        raise ValueError("telemetry identity field is empty")
    try:
        index_value = int(index)
    except ValueError as error:
        raise ValueError("telemetry GPU index is not an integer") from error
    return {
        "timestamp": timestamp,
        "index": index_value,
        "uuid": uuid,
        "pci_bus_id": bus,
        "pstate": pstate,
        "sm_clock_mhz": _finite_number(sm, "SM clock"),
        "memory_clock_mhz": _finite_number(memory, "memory clock"),
        "power_draw_watts": _finite_number(power, "power"),
        "temperature_celsius": _finite_number(temperature, "temperature"),
        "gpu_utilization_percent": _finite_number(
            gpu_utilization, "GPU utilization"
        ),
        "memory_utilization_percent": _finite_number(
            memory_utilization, "memory utilization"
        ),
    }


def resolve_visible_device_binding(
    cuda_visible_devices: str | None,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    if cuda_visible_devices is None or not cuda_visible_devices.strip():
        return {
            "resolved": False,
            "reason": "CUDA_VISIBLE_DEVICES_is_missing",
            "token": cuda_visible_devices,
        }
    tokens = [token.strip() for token in cuda_visible_devices.split(",")]
    if len(tokens) != 1 or not tokens[0]:
        return {
            "resolved": False,
            "reason": "CUDA_VISIBLE_DEVICES_does_not_name_exactly_one_device",
            "token": cuda_visible_devices,
        }
    token = tokens[0]
    identities = {(int(row["index"]), str(row["uuid"])) for row in rows}
    matches: set[tuple[int, str]] = set()
    if token.isdigit():
        matches = {identity for identity in identities if identity[0] == int(token)}
    elif token.startswith("GPU-"):
        matches = {identity for identity in identities if identity[1] == token}
    if len(matches) != 1:
        return {
            "resolved": False,
            "reason": "visible_device_token_does_not_uniquely_match_telemetry",
            "token": token,
        }
    index, uuid = next(iter(matches))
    return {
        "resolved": True,
        "reason": None,
        "token": token,
        "physical_index": index,
        "uuid": uuid,
    }


@dataclass(slots=True)
class GCStepRecorder:
    """Associate Python GC callbacks with a diagnostic step when possible."""

    active: bool = False
    current_step: int | None = None
    current_window: str | None = None
    events: list[dict[str, object]] = field(default_factory=list)
    enabled_before: bool | None = None
    thresholds: tuple[int, int, int] | None = None
    counts_at_start: tuple[int, int, int] | None = None
    counts_at_end: tuple[int, int, int] | None = None
    registered_callback: Callable[[str, Mapping[str, int]], None] | None = None

    def _callback(self, phase: str, info: Mapping[str, int]) -> None:
        if not self.active:
            return
        self.events.append(
            {
                "callback_phase": phase,
                "generation": int(info.get("generation", -1)),
                "collected": int(info.get("collected", 0)),
                "uncollectable": int(info.get("uncollectable", 0)),
                "window": self.current_window,
                "step_index": self.current_step,
                "location": (
                    "within_step" if self.current_step is not None else "between_steps"
                ),
                "monotonic_ns": time.monotonic_ns(),
            }
        )

    def start(self, window: str) -> None:
        if self.active:
            raise RuntimeError("GC recorder is already active")
        self.active = True
        self.current_window = window
        self.enabled_before = gc.isenabled()
        self.thresholds = tuple(int(value) for value in gc.get_threshold())
        self.counts_at_start = tuple(int(value) for value in gc.get_count())
        self.registered_callback = self._callback
        gc.callbacks.append(self.registered_callback)

    def begin_step(self, index: int) -> tuple[int, int, int]:
        if not self.active or self.current_step is not None:
            raise RuntimeError("GC recorder step lifecycle is invalid")
        self.current_step = index
        return tuple(int(value) for value in gc.get_count())

    def end_step(self) -> tuple[int, int, int]:
        if not self.active or self.current_step is None:
            raise RuntimeError("GC recorder step lifecycle is invalid")
        counts = tuple(int(value) for value in gc.get_count())
        self.current_step = None
        return counts

    def stop(self) -> None:
        if not self.active:
            return
        self.current_step = None
        self.counts_at_end = tuple(int(value) for value in gc.get_count())
        try:
            if self.registered_callback is not None:
                gc.callbacks.remove(self.registered_callback)
        finally:
            self.registered_callback = None
            self.active = False

    def to_metadata(self) -> dict[str, object]:
        if self.active:
            raise RuntimeError("GC recorder must be stopped before serialization")
        return {
            "enabled_before": self.enabled_before,
            "thresholds": list(self.thresholds or ()),
            "counts_at_start": list(self.counts_at_start or ()),
            "counts_at_end": list(self.counts_at_end or ()),
            "events": self.events,
            "event_count": len(self.events),
            "within_step_event_count": sum(
                event["location"] == "within_step" for event in self.events
            ),
        }


class BackgroundTelemetrySampler:
    """Run nvidia-smi as an external low-frequency background process."""

    def __init__(self, root: Path, *, interval_ms: int = 100) -> None:
        if not isinstance(interval_ms, int) or isinstance(interval_ms, bool):
            raise TypeError("telemetry interval must be an integer")
        if interval_ms < 50:
            raise ValueError("telemetry interval must be at least 50 ms")
        self.root = root
        self.interval_ms = interval_ms
        self.stdout_path = root / "nvidia_smi.stdout"
        self.stderr_path = root / "nvidia_smi.stderr"
        self.command = [
            "nvidia-smi",
            "--query-gpu=" + ",".join(TELEMETRY_FIELDS),
            "--format=csv,noheader,nounits",
            f"--loop-ms={interval_ms}",
        ]
        self.process: subprocess.Popen[str] | None = None
        self.stdout_stream: TextIO | None = None
        self.stderr_stream: TextIO | None = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("telemetry sampler is already started")
        self.root.mkdir(parents=True, exist_ok=False)
        self.stdout_stream = self.stdout_path.open("x", encoding="utf-8")
        self.stderr_stream = self.stderr_path.open("x", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=self.stdout_stream,
                stderr=self.stderr_stream,
                text=True,
            )
        except Exception:
            self.stdout_stream.close()
            self.stderr_stream.close()
            self.stdout_stream = None
            self.stderr_stream = None
            raise

    def stop(self) -> dict[str, object]:
        if self.process is None:
            raise RuntimeError("telemetry sampler was not started")
        process = self.process
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            assert self.stdout_stream is not None
            assert self.stderr_stream is not None
            self.stdout_stream.close()
            self.stderr_stream.close()
            self.process = None
        parsed: list[dict[str, object]] = []
        parse_errors: list[dict[str, object]] = []
        for line_number, line in enumerate(
            self.stdout_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                parsed.append(parse_telemetry_line(line))
            except ValueError as error:
                parse_errors.append(
                    {"line_number": line_number, "line": line, "error": str(error)}
                )
        binding = resolve_visible_device_binding(
            os.environ.get("CUDA_VISIBLE_DEVICES"), parsed
        )
        return {
            "command": self.command,
            "interval_ms": self.interval_ms,
            "returncode_after_termination": process.returncode,
            "rows": parsed,
            "row_count": len(parsed),
            "parse_errors": parse_errors,
            "binding": binding,
            "stderr": self.stderr_path.read_text(encoding="utf-8"),
            "all_physical_rows_preserved": True,
        }


def scheduled_refresh_due(
    family: str, completed_before: int, completed_after: int
) -> dict[str, object]:
    _require(
        completed_after == completed_before + 1,
        "step progress did not advance once",
    )
    applicable = family == "channel_compiled_v2"
    return {
        "applicable": applicable,
        "due": bool(applicable and completed_after % 20 == 0),
        "interval": 20 if applicable else None,
        "source": "frozen_rc5_g1_transform_call_contract",
    }


def _prepare(
    case, config: RC458ProfileConfig, device: torch.device
) -> dict[str, object]:
    for _ in range(config.warmup_steps):
        case.step()
    torch.cuda.synchronize(device)
    steps_after_warmup = case.completed_steps()
    gc.collect()
    torch.cuda.empty_cache()
    for _ in range(config.post_flush_rewarm_steps):
        case.step()
    torch.cuda.synchronize(device)
    return {
        "steps_after_warmup": steps_after_warmup,
        "measurement_start_completed_steps": case.completed_steps(),
    }


def _run_window(
    *,
    name: str,
    case,
    config: RC458ProfileConfig,
    device: torch.device,
    telemetry_root: Path,
    synchronize_each_step: bool,
    capture_public_diagnostics: bool,
) -> dict[str, object]:
    lifecycle = _prepare(case, config, device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(config.profile_steps)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(config.profile_steps)]
    gc_recorder = GCStepRecorder()
    telemetry = BackgroundTelemetrySampler(telemetry_root)
    records: list[dict[str, object]] = []
    telemetry.start()
    try:
        gc_recorder.start(name)
        window_start = time.perf_counter_ns()
        window_start_unix_ns = time.time_ns()
        try:
            for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
                completed_before = int(case.completed_steps())
                counts_before = gc_recorder.begin_step(index)
                host_start = time.perf_counter_ns()
                host_start_unix_ns = time.time_ns()
                start.record()
                case.step()
                stop.record()
                if synchronize_each_step:
                    stop.synchronize()
                host_stop = time.perf_counter_ns()
                host_stop_unix_ns = time.time_ns()
                counts_after = gc_recorder.end_step()
                completed_after = int(case.completed_steps())
                diagnostics = (
                    _normalize_public_diagnostics(case.public_diagnostics)
                    if capture_public_diagnostics
                    else {
                        "applicable": False,
                        "reason": (
                            "not_sampled_in_asynchronous_compatibility_window"
                        ),
                    }
                )
                records.append(
                    {
                        "step_index_zero_based": index,
                        "step_index_one_based": index + 1,
                        "runtime_completed_steps_before": completed_before,
                        "runtime_completed_steps_after": completed_after,
                        "host_duration_seconds": (
                            host_stop - host_start
                        ) / 1.0e9,
                        "host_start_unix_ns": host_start_unix_ns,
                        "host_stop_unix_ns": host_stop_unix_ns,
                        "host_duration_semantics": (
                            "synchronized_end_to_end"
                            if synchronize_each_step
                            else "asynchronous_submit_only"
                        ),
                        "gc_counts_before": list(counts_before),
                        "gc_counts_after": list(counts_after),
                        "scheduled_refresh": scheduled_refresh_due(
                            config.family, completed_before, completed_after
                        ),
                        "public_diagnostics": diagnostics,
                    }
                )
            torch.cuda.synchronize(device)
            window_stop = time.perf_counter_ns()
            window_stop_unix_ns = time.time_ns()
        finally:
            gc_recorder.stop()
    finally:
        telemetry_result = telemetry.stop()
    cuda_seconds = [
        float(start.elapsed_time(stop)) / 1000.0
        for start, stop in zip(starts, stops, strict=True)
    ]
    for record, seconds in zip(records, cuda_seconds, strict=True):
        record["cuda_event_seconds"] = seconds
        step_index = int(record["step_index_zero_based"])
        record["gc_event_indices"] = [
            event_index
            for event_index, event in enumerate(gc_recorder.events)
            if event["step_index"] == step_index
        ]
    state = case.state()
    return {
        "name": name,
        "fresh_runtime": True,
        "synchronize_each_step": synchronize_each_step,
        "used_for_formal_performance_adjudication": False,
        "lifecycle": lifecycle,
        "total_host_window_seconds": (window_stop - window_start) / 1.0e9,
        "window_start_unix_ns": window_start_unix_ns,
        "window_stop_unix_ns": window_stop_unix_ns,
        "records": records,
        "gc": gc_recorder.to_metadata(),
        "background_gpu_telemetry": telemetry_result,
        "initial_state_sha256": case.initial_state_sha256,
        "final_state_sha256": _tensor_sha256(state),
        "finite": _finite(state),
        "completed_steps": int(case.completed_steps()),
        "runtime_identity": case.runtime_identity,
        "configuration_identity": case.configuration_identity,
    }


def run_profile(config: RC458ProfileConfig) -> dict[str, object]:
    _validate(config)
    _require(
        config.post_flush_rewarm_steps == 3,
        "instrumentation requires 3 rewarm steps",
    )
    _require(
        config.profile_steps == 60,
        "instrumentation requires all 60 samples",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(
        torch.cuda.device_count() == 1,
        "exactly one visible CUDA device is required",
    )
    device = torch.device("cuda")
    allocated = torch.empty((), device=device).device
    _require(str(allocated) == "cuda:0", "allocated device is not cuda:0")
    _require("H100" in torch.cuda.get_device_name(device), "H100 is required")
    _require(not torch.backends.cuda.matmul.allow_tf32, "matmul TF32 must be disabled")
    _require(not torch.backends.cudnn.allow_tf32, "cuDNN TF32 must be disabled")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    with tempfile.TemporaryDirectory(prefix="pssolver-rc5-g1-instrument-") as temporary:
        root = Path(temporary)
        with torch.no_grad():
            asynchronous_case = _build_case(root / "asynchronous", config)
            asynchronous = _run_window(
                name="asynchronous_compatibility",
                case=asynchronous_case,
                config=config,
                device=device,
                telemetry_root=root / "telemetry-asynchronous",
                synchronize_each_step=False,
                capture_public_diagnostics=False,
            )
            del asynchronous_case
            gc.collect()
            torch.cuda.empty_cache()
            synchronized_case = _build_case(root / "synchronized", config)
            synchronized = _run_window(
                name="synchronized_attribution_replay",
                case=synchronized_case,
                config=config,
                device=device,
                telemetry_root=root / "telemetry-synchronized",
                synchronize_each_step=True,
                capture_public_diagnostics=True,
            )
    _require(asynchronous["finite"] is True, "asynchronous state is non-finite")
    _require(synchronized["finite"] is True, "synchronized state is non-finite")
    _require(
        asynchronous["initial_state_sha256"] == synchronized["initial_state_sha256"],
        "fresh-runtime initial states differ",
    )
    _require(
        asynchronous["final_state_sha256"] == synchronized["final_state_sha256"],
        "instrumentation windows changed the final state",
    )
    return {
        "schema": SCHEMA,
        "classification": "ANALYSIS_ONLY_TIMING_INSTRUMENTATION_PROFILE",
        "analysis_only": True,
        "used_for_formal_performance_adjudication": False,
        "config": {
            **asdict(config),
            "shape": list(config.shape),
            "lengths": list(config.lengths),
        },
        "environment": {
            "allocated_device": str(allocated),
            "device_name": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "windows": {
            "asynchronous_compatibility": asynchronous,
            "synchronized_attribution_replay": synchronized,
        },
        "identity": {
            "initial_state_sha256_equal": True,
            "final_state_sha256_equal": True,
        },
        "formal_v10_reclassified": False,
        "thresholds_changed": False,
        "raw_sample_deletion_allowed": False,
        "production_default_changed": False,
    }


def _tf32_state() -> tuple[str, bool, bool]:
    return (
        torch.get_float32_matmul_precision(),
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
    )


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def _triple(value: str, cast: Callable[[str], object]) -> tuple[object, object, object]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return tuple(cast(part) for part in parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--provider-role", required=True)
    parser.add_argument("--provider-commit", required=True)
    parser.add_argument("--provider-wheel-sha256", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument(
        "--shape", type=lambda value: _triple(value, int), required=True
    )
    parser.add_argument(
        "--lengths", type=lambda value: _triple(value, float), required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--post-flush-rewarm-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = RC458ProfileConfig(
        family=args.family,
        provider_role=args.provider_role,
        provider_commit=args.provider_commit,
        provider_wheel_sha256=args.provider_wheel_sha256,
        trial=args.trial,
        shape=tuple(args.shape),
        lengths=tuple(args.lengths),
        device=args.device,
        dtype=args.dtype,
        warmup_steps=args.warmup_steps,
        post_flush_rewarm_steps=args.post_flush_rewarm_steps,
        profile_steps=args.profile_steps,
        seed=args.seed,
    )
    before = _tf32_state()
    enforced: tuple[str, bool, bool]
    result: dict[str, object]
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        enforced = _tf32_state()
        if enforced != ("highest", False, False):
            raise RuntimeError("cannot enforce the frozen TF32-off profile policy")
        result = run_profile(config)
    finally:
        torch.set_float32_matmul_precision(before[0])
        torch.backends.cuda.matmul.allow_tf32 = before[1]
        torch.backends.cudnn.allow_tf32 = before[2]
    restored = _tf32_state()
    if restored != before:
        raise RuntimeError("TF32 child policy was not restored")
    result["tf32_child_policy"] = {
        "before": {
            "float32_matmul_precision": before[0],
            "cuda_matmul_allow_tf32": before[1],
            "cudnn_allow_tf32": before[2],
        },
        "enforced": {
            "float32_matmul_precision": enforced[0],
            "cuda_matmul_allow_tf32": enforced[1],
            "cudnn_allow_tf32": enforced[2],
        },
        "restored_before_atomic_write": {
            "float32_matmul_precision": restored[0],
            "cuda_matmul_allow_tf32": restored[1],
            "cudnn_allow_tf32": restored[2],
        },
    }
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
