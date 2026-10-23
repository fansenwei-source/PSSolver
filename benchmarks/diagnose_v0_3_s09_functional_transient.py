#!/usr/bin/env python3
"""One-shot S-09 CPU telemetry for the Channel functional tolerance path.

This is an analysis-only diagnostic.  It does not replace or relax either
threshold in the frozen H100 qualification contract.  Every timestep and
every telemetry observation is retained.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.benchmark_v0_3_s09_channel_cpu import (  # noqa: E402
    Cell,
    _atomic_json,
    _build_cell,
    _diagnostic_fields,
    _install_provider_root,
    _validate_cell,
)


SCHEMA = "pssolver.v0_3.s09.functional_transient_telemetry.v1"
TRIALS = 6
WARMUP_STEPS = 3
MEASURED_STEPS = 60
BLOCK_SIZE = 10
CPU = 8
FAMILY = "channel_functional"
MODE = "relative_tolerance"
MEDIAN_LIMIT = 1.03
PAIRED_LIMIT = 1.05
FREQUENCY_RATIO_EVENT_MAX = 0.95
UNATTRIBUTED_EXPLANATION_FRACTION = 0.50


def _read_frequency_khz(cpu: int) -> int | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
    try:
        return int(path.read_text(encoding="ascii").strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def _current_cpu() -> int | None:
    try:
        value = int(ctypes.CDLL(None, use_errno=True).sched_getcpu())
    except (AttributeError, OSError):
        return None
    return value if value >= 0 else None


def _loadavg() -> list[float]:
    return [float(value) for value in os.getloadavg()]


def _proc_stat() -> dict[str, int]:
    result: dict[str, int] = {}
    for line in Path("/proc/stat").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields and fields[0] in {"ctxt", "processes", "procs_running", "procs_blocked"}:
            result[fields[0]] = int(fields[1])
    return result


def _rusage() -> resource.struct_rusage:
    return resource.getrusage(resource.RUSAGE_SELF)


def _rusage_delta(before: resource.struct_rusage, after: resource.struct_rusage) -> dict[str, float | int]:
    user = float(after.ru_utime - before.ru_utime)
    system = float(after.ru_stime - before.ru_stime)
    return {
        "user_seconds": user,
        "system_seconds": system,
        "process_cpu_seconds": user + system,
        "voluntary_context_switches": int(after.ru_nvcsw - before.ru_nvcsw),
        "involuntary_context_switches": int(after.ru_nivcsw - before.ru_nivcsw),
        "minor_faults": int(after.ru_minflt - before.ru_minflt),
        "major_faults": int(after.ru_majflt - before.ru_majflt),
    }


class _SystemSampler:
    def __init__(self, cpu: int, interval_seconds: float = 0.25) -> None:
        self.cpu = cpu
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_SystemSampler":
        def sample() -> None:
            started = time.monotonic_ns()
            while not self._stop.is_set():
                self.samples.append(
                    {
                        "elapsed_seconds": (time.monotonic_ns() - started) * 1.0e-9,
                        "loadavg": _loadavg(),
                        "frequency_khz": _read_frequency_khz(self.cpu),
                        "proc_stat": _proc_stat(),
                    }
                )
                self._stop.wait(self.interval_seconds)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()


def _block_means(values: list[float]) -> list[float]:
    if len(values) != MEASURED_STEPS:
        raise ValueError(f"expected {MEASURED_STEPS} values")
    return [
        statistics.fmean(values[offset : offset + BLOCK_SIZE])
        for offset in range(0, MEASURED_STEPS, BLOCK_SIZE)
    ]


def run_cell(cell: Cell, cpu: int) -> dict[str, object]:
    _validate_cell(cell)
    os.sched_setaffinity(0, {cpu})
    _install_provider_root(cell.provider_root)
    import numpy as np
    import torch
    import pssolver

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    imported = Path(pssolver.__file__).resolve()
    if not imported.is_relative_to(cell.provider_root.resolve()):
        raise RuntimeError(f"source shadow import: {imported}")
    with tempfile.TemporaryDirectory(prefix="pssolver-s09-telemetry-") as temporary:
        step, diagnostics = _build_cell(cell, Path(temporary))
        with torch.no_grad():
            for _ in range(cell.warmup_steps):
                step()
            samples: list[dict[str, object]] = []
            for index in range(cell.measured_steps):
                frequency_before = _read_frequency_khz(cpu)
                load_before = _loadavg()
                usage_before = _rusage()
                started = time.perf_counter_ns()
                step()
                wall = (time.perf_counter_ns() - started) * 1.0e-9
                usage_after = _rusage()
                values: dict[str, object] = {
                    "index": index,
                    "wall_seconds": wall,
                    **_rusage_delta(usage_before, usage_after),
                    "frequency_before_khz": frequency_before,
                    "frequency_after_khz": _read_frequency_khz(cpu),
                    "loadavg_before": load_before,
                    "loadavg_after": _loadavg(),
                    "cpu_after": _current_cpu(),
                }
                samples.append(values)
            observed = _diagnostic_fields(diagnostics())
    wall_values = [float(value["wall_seconds"]) for value in samples]
    cpu_values = [float(value["process_cpu_seconds"]) for value in samples]
    if not all(math.isfinite(value) and value > 0.0 for value in wall_values):
        raise RuntimeError("invalid wall-clock measurements")
    return {
        "schema": SCHEMA,
        "kind": "cell",
        "config": {
            "family": cell.family,
            "mode": cell.mode,
            "role": cell.role,
            "trial": cell.trial,
            "provider_root": str(cell.provider_root.resolve()),
            "provider_commit": cell.provider_commit,
            "warmup_steps": cell.warmup_steps,
            "measured_steps": cell.measured_steps,
            "shape": list(cell.shape),
            "lengths": list(cell.lengths),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pssolver_import": str(imported),
            "sys_path": list(sys.path),
            "current_cpu_after_window": _current_cpu(),
        },
        "samples": samples,
        "summary": {
            "wall_block_means_seconds": _block_means(wall_values),
            "cpu_block_means_seconds": _block_means(cpu_values),
            "wall_center_seconds": statistics.median(_block_means(wall_values)),
            "cpu_center_seconds": statistics.median(_block_means(cpu_values)),
            "voluntary_context_switches": sum(int(value["voluntary_context_switches"]) for value in samples),
            "involuntary_context_switches": sum(int(value["involuntary_context_switches"]) for value in samples),
            "minor_faults": sum(int(value["minor_faults"]) for value in samples),
            "major_faults": sum(int(value["major_faults"]) for value in samples),
        },
        "pressure_diagnostics": observed,
    }


def _ratio(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError("ratio denominator must be positive")
    return candidate / baseline


def _median_frequency(profile: dict[str, object]) -> float | None:
    values = [
        float(item[key])
        for item in profile["samples"]
        for key in ("frequency_before_khz", "frequency_after_khz")
        if item[key] is not None
    ]
    return statistics.median(values) if values else None


def analyze(cells: list[dict[str, object]], frozen_median: float) -> dict[str, object]:
    pairs: list[dict[str, object]] = []
    for trial in range(1, TRIALS + 1):
        selected = [item for item in cells if int(item["config"]["trial"]) == trial]
        by_role = {str(item["config"]["role"]): item for item in selected}
        if set(by_role) != {"baseline", "candidate"}:
            raise ValueError(f"missing trial {trial} pair")
        baseline, candidate = by_role["baseline"], by_role["candidate"]
        bs, cs = baseline["summary"], candidate["summary"]
        wall_ratio = _ratio(float(cs["wall_center_seconds"]), float(bs["wall_center_seconds"]))
        cpu_ratio = _ratio(float(cs["cpu_center_seconds"]), float(bs["cpu_center_seconds"]))
        baseline_unattributed = max(0.0, float(bs["wall_center_seconds"]) - float(bs["cpu_center_seconds"]))
        candidate_unattributed = max(0.0, float(cs["wall_center_seconds"]) - float(cs["cpu_center_seconds"]))
        wall_excess = max(0.0, float(cs["wall_center_seconds"]) - float(bs["wall_center_seconds"]))
        freq_base, freq_candidate = _median_frequency(baseline), _median_frequency(candidate)
        freq_ratio = None if freq_base is None or freq_candidate is None else _ratio(freq_candidate, freq_base)
        inv_excess = int(cs["involuntary_context_switches"]) > int(bs["involuntary_context_switches"])
        major_excess = int(cs["major_faults"]) > int(bs["major_faults"])
        deschedule_explains = (
            candidate_unattributed - baseline_unattributed
            >= UNATTRIBUTED_EXPLANATION_FRACTION * wall_excess
            and wall_excess > 0.0
        )
        frequency_event = freq_ratio is not None and freq_ratio < FREQUENCY_RATIO_EVENT_MAX
        over_limit = wall_ratio > PAIRED_LIMIT
        environmental = over_limit and (
            cpu_ratio <= PAIRED_LIMIT
            or major_excess
            or (inv_excess and deschedule_explains)
            or frequency_event
        )
        block_ratios = [
            _ratio(float(c), float(b))
            for b, c in zip(bs["wall_block_means_seconds"], cs["wall_block_means_seconds"])
        ]
        candidate_warmup = (
            over_limit
            and not environmental
            and statistics.fmean(block_ratios[:3]) > PAIRED_LIMIT
            and statistics.fmean(block_ratios[3:]) <= MEDIAN_LIMIT
        )
        pairs.append(
            {
                "trial": trial,
                "wall_center_ratio": wall_ratio,
                "process_cpu_center_ratio": cpu_ratio,
                "frequency_ratio": freq_ratio,
                "wall_block_ratios": block_ratios,
                "events": {
                    "candidate_involuntary_context_switch_excess": inv_excess,
                    "candidate_major_fault_excess": major_excess,
                    "descheduled_time_explains_at_least_half_wall_excess": deschedule_explains,
                    "frequency_drop_below_0_95": frequency_event,
                },
                "paired_limit_exceeded": over_limit,
                "environmental_interference_supported": environmental,
                "candidate_warmup_supported": candidate_warmup,
            }
        )
    new_median = statistics.median(float(item["wall_center_ratio"]) for item in pairs)
    exceeded = [item for item in pairs if item["paired_limit_exceeded"]]
    if not exceeded:
        classification = "NO_PAIRED_LIMIT_EXCEEDANCE"
    elif all(item["environmental_interference_supported"] for item in exceeded):
        classification = "ENVIRONMENTAL_INTERFERENCE_SUPPORTED"
    elif any(item["candidate_warmup_supported"] for item in exceeded):
        classification = "CANDIDATE_WARMUP_OR_FIRST_CALL_OVERHEAD_SUPPORTED"
    else:
        classification = "CAUSE_UNRESOLVED"
    reference_pass = (
        frozen_median <= MEDIAN_LIMIT
        and new_median <= MEDIAN_LIMIT
        and classification in {"NO_PAIRED_LIMIT_EXCEEDANCE", "ENVIRONMENTAL_INTERFERENCE_SUPPORTED"}
    )
    return {
        "classification": classification,
        "frozen_six_by_sixty_median_ratio": frozen_median,
        "focused_median_ratio": new_median,
        "pairs": pairs,
        "cpu_preflight_reference_estimate_passed": reference_pass,
        "candidate_fix_required_before_h100": classification in {
            "CANDIDATE_WARMUP_OR_FIRST_CALL_OVERHEAD_SUPPORTED",
            "CAUSE_UNRESOLVED",
        },
        "formal_h100_contract_unchanged": True,
    }


def _cell_command(args: argparse.Namespace, role: str, trial: int, output: Path) -> list[str]:
    root = args.baseline_root if role == "baseline" else args.candidate_root
    commit = args.baseline_commit if role == "baseline" else args.candidate_commit
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell",
        "--role", role,
        "--trial", str(trial),
        "--provider-root", str(root),
        "--provider-commit", commit,
        "--cpu", str(args.cpu),
        "--output", str(output),
    ]


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    cell_root = output.parent / f"{output.stem}_cells"
    cell_root.mkdir(parents=True)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    cells: list[dict[str, object]] = []
    sequence: list[dict[str, object]] = []
    with _SystemSampler(args.cpu) as sampler:
        for trial in range(1, TRIALS + 1):
            roles = ("baseline", "candidate") if trial % 2 else ("candidate", "baseline")
            for role in roles:
                cell_output = cell_root / f"{role}_trial{trial}.json"
                command = _cell_command(args, role, trial, cell_output)
                completed = subprocess.run(
                    command,
                    env=env,
                    cwd=cell_root,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                sequence.append({
                    "role": role,
                    "trial": trial,
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                })
                if completed.returncode != 0:
                    raise RuntimeError(f"telemetry cell failed: {role}/trial{trial}: {completed.stderr}")
                cells.append(json.loads(cell_output.read_text(encoding="utf-8")))
    result = {
        "schema": SCHEMA,
        "kind": "matrix",
        "contract": {
            "analysis_only": True,
            "family": FAMILY,
            "mode": MODE,
            "trials": TRIALS,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "block_size": BLOCK_SIZE,
            "cpu": args.cpu,
            "alternating_pair_order": True,
            "all_samples_retained": True,
            "rerun_allowed": False,
            "median_limit": MEDIAN_LIMIT,
            "paired_limit": PAIRED_LIMIT,
        },
        "baseline": {"root": str(args.baseline_root.resolve()), "commit": args.baseline_commit},
        "candidate": {"root": str(args.candidate_root.resolve()), "commit": args.candidate_commit},
        "sequence": sequence,
        "raw_cells": cells,
        "system_samples": sampler.samples,
        "analysis": analyze(cells, args.frozen_median),
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--matrix", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--trial", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--provider-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--provider-commit", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-commit")
    parser.add_argument("--cpu", type=int, default=CPU)
    parser.add_argument("--frozen-median", type=float, default=1.0168360838379518)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cell:
        required = (args.role, args.trial, args.provider_root, args.provider_commit)
        if any(value is None for value in required):
            raise ValueError("cell invocation is incomplete")
        report = run_cell(
            Cell(
                family=FAMILY,
                mode=MODE,
                role=args.role,
                trial=args.trial,
                provider_root=args.provider_root,
                provider_commit=args.provider_commit,
                warmup_steps=WARMUP_STEPS,
                measured_steps=MEASURED_STEPS,
            ),
            args.cpu,
        )
        _atomic_json(args.output, report)
        return 0
    required = (args.baseline_root, args.baseline_commit, args.candidate_root, args.candidate_commit)
    if any(value is None for value in required):
        raise ValueError("matrix invocation is incomplete")
    report = run_matrix(args)
    print(json.dumps(report["analysis"], allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
