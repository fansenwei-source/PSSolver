#!/usr/bin/env python3
"""Measure the S-06 Channel pressure cost before S-09 may use H100.

The matrix runner alternates isolated baseline and candidate child processes.
Only steady timesteps are timed.  Peak resident memory is sampled from
``/proc/self/statm`` during the same measurement window, so Python import and
runtime construction are outside the timing window while the complete process
footprint remains visible to the memory gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Callable


SCHEMA = "pssolver.v0_3.s09.channel_cpu_preflight.v1"
FAMILIES = (
    "channel_legacy",
    "channel_compiled_v2",
    "channel_complete_stress",
    "channel_functional",
)
MODES = ("relative_tolerance", "fixed_work")
TIME_CENTER_LIMIT = 1.03
TIME_PAIRED_LIMIT = 1.05
MEMORY_LIMIT = 1.03


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    expected = json.loads(payload)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("atomic JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _rss_bytes() -> int:
    fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
    return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))


class _RssSampler:
    def __init__(self, interval_seconds: float = 0.0005) -> None:
        self.interval_seconds = interval_seconds
        self.baseline = 0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_RssSampler":
        self.baseline = _rss_bytes()
        self.peak = self.baseline

        def sample() -> None:
            while not self._stop.is_set():
                self.peak = max(self.peak, _rss_bytes())
                self._stop.wait(self.interval_seconds)
            self.peak = max(self.peak, _rss_bytes())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()


@dataclass(frozen=True, slots=True)
class Cell:
    family: str
    mode: str
    role: str
    trial: int
    provider_root: Path
    provider_commit: str
    warmup_steps: int
    measured_steps: int
    shape: tuple[int, int, int] = (128, 20, 20)
    lengths: tuple[float, float, float] = (40.0, 6.0, 6.0)
    seed: int = 20261023


def _validate_cell(cell: Cell) -> None:
    if cell.family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    if cell.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if cell.role not in {"baseline", "candidate"}:
        raise ValueError("role must be baseline or candidate")
    if cell.trial <= 0 or cell.warmup_steps < 0 or cell.measured_steps <= 0:
        raise ValueError("invalid trial or step count")
    if len(cell.provider_commit) != 40:
        raise ValueError("provider commit must be complete")
    if not (cell.provider_root / "pssolver" / "__init__.py").is_file():
        raise FileNotFoundError("provider root does not contain pssolver")


def _install_provider_root(root: Path) -> None:
    resolved = str(root.resolve())
    sys.path[:] = [item for item in sys.path if Path(item or ".").resolve() != Path(__file__).resolve().parents[1]]
    sys.path.insert(0, resolved)


def _solver_pressure_diagnostics(solver: object) -> dict[str, object]:
    """Read only stable diagnostic values from old and new solver layouts."""
    candidates = [solver]
    model = getattr(solver, "model", None)
    if model is not None:
        candidates.extend(
            item
            for item in (
                model,
                getattr(model, "static_model", None),
                getattr(model, "flow_model", None),
            )
            if item is not None
        )
    aliases = {
        "iterations": ("last_pressure_iterations", "pressure_iterations"),
        "termination_reason": (
            "last_pressure_termination_reason",
            "pressure_termination_reason",
        ),
        "relative_residual": (
            "last_pressure_relative_residual",
            "pressure_relative_residual",
        ),
    }
    result: dict[str, object] = {}
    for destination, names in aliases.items():
        for candidate in candidates:
            for name in names:
                if hasattr(candidate, name):
                    value = getattr(candidate, name)
                    if value is not None:
                        result[destination] = value
                        break
            if destination in result:
                break
    return result


def _build_cell(cell: Cell, root: Path):
    import numpy as np
    import torch

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
    from pssolver.boundaries import (
        assign_boundaries,
        neumann_pressure_compatibility,
        neumann_q,
        no_slip_velocity,
    )
    from pssolver.channel import build_active_nematic_channel
    from pssolver.configuration.channel_active_nematics import ChannelActiveNematicRunSpec
    from pssolver.configuration.channel_active_nematics_declarations import ChannelRuntimePath
    from pssolver.functional import (
        build_channel_activity_functional_runtime,
        channel_activity_functional_request,
    )
    from pssolver.geometries import RectangularChannel
    from pssolver.models.active_nematics import CompleteStressBerisEdwards, Q_COMPONENTS
    from pssolver.runtime.channel_active_nematics import (
        ChannelRuntimeBuildRequest,
        build_channel_active_nematic_runtime,
    )
    from pssolver.runtime.channel_application_bridge import build_package_compiled_channel_runtime
    from pssolver.runtime.channel_beris_edwards import (
        ChannelBerisEdwardsRuntimeBuildRequest,
        build_channel_beris_edwards_runtime,
    )

    fixed = 12 if cell.mode == "fixed_work" else None
    if cell.family in {"channel_legacy", "channel_compiled_v2"}:
        generator = torch.Generator(device="cpu").manual_seed(cell.seed)
        values = 0.01 * torch.randn(5, *cell.shape, generator=generator, dtype=torch.float32)
        values[0].add_(1.0 / 3.0)
        values[3].add_(-1.0 / 6.0)
        initial = {name: values[index] for index, name in enumerate(Q_COMPONENTS)}
        runtime_path = (
            ChannelRuntimePath.LEGACY_CHANNEL
            if cell.family == "channel_legacy"
            else ChannelRuntimePath.COMPILED_CHANNEL_V2
        )
        spec = ChannelActiveNematicRunSpec(
            shape=cell.shape,
            lengths=cell.lengths,
            dt=0.01,
            steps=cell.warmup_steps + cell.measured_steps,
            save_interval=cell.warmup_steps + cell.measured_steps,
            diagnostic_interval=cell.warmup_steps + cell.measured_steps,
            activity=5.0,
            device="cpu",
            dtype="float32",
            pressure_relative_tolerance=1.0e-5,
            pressure_max_iterations=80,
            pressure_fixed_iterations=fixed,
            runtime_path=runtime_path,
        )
        request = ChannelRuntimeBuildRequest(
            spec,
            {"configuration": spec.identity_metadata(), "runtime_selection": spec.runtime_selection_metadata()},
            initial,
            "cpu",
        )
        material = spec.components.material
        pressure = spec.components.pressure_solver

        def legacy_builder():
            solver = build_active_nematic_channel(
                spec.shape,
                spec.lengths,
                spec.dt,
                initial,
                device="cpu",
                batchsize=1,
                rho=material.rho,
                elastic_constant=material.elastic_constant,
                beta=material.beta,
                friction=material.friction,
                viscosity=material.viscosity,
                pressure_rel_tol=pressure.relative_tolerance,
                pressure_max_iter=pressure.max_iterations,
                pressure_fixed_iterations=pressure.fixed_iterations,
            )
            solver.parameters["alpha"] = torch.full(
                (1, *spec.shape), material.activity, dtype=solver.dtype, device=solver.device
            )
            return solver

        adapter = build_channel_active_nematic_runtime(
            request,
            legacy_builder=legacy_builder if runtime_path is ChannelRuntimePath.LEGACY_CHANNEL else None,
            compiled_builder=(lambda: build_package_compiled_channel_runtime(request))
            if runtime_path is ChannelRuntimePath.COMPILED_CHANNEL_V2
            else None,
        )
        return (
            (lambda: adapter.advance(1)),
            (lambda: _solver_pressure_diagnostics(adapter.solver)),
        )

    snapshot = root / "snapshot"
    snapshot.mkdir()
    rng = np.random.default_rng(cell.seed)
    values = rng.normal(scale=0.01, size=(*cell.shape, 5)).astype(np.float64)
    values[..., 0] += 0.2
    values[..., 3] -= 0.1
    np.save(snapshot / "Q_0.npy", values, allow_pickle=False)
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
    )
    geometry = RectangularChannel(shape=cell.shape, lengths=cell.lengths)
    simulation = Simulation(
        model=model,
        geometry=geometry,
        boundaries=assign_boundaries(
            model=model,
            geometry=geometry,
            policies={"Q": neumann_q(), "velocity": no_slip_velocity(), "pressure": neumann_pressure_compatibility()},
        ),
        numerics=SpectralNumerics(dtype="float64", dealias_rule="cubic_half", spectral_storage="full_complex"),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=root / "unused-output",
            steps=cell.warmup_steps + cell.measured_steps,
            save_interval=cell.warmup_steps + cell.measured_steps + 1,
            diagnostic_interval=cell.warmup_steps + cell.measured_steps + 1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 80,
                "fixed_iterations": fixed,
                "warm_start": True,
            }
        },
    )
    if cell.family == "channel_complete_stress":
        compiled = compile_simulation(simulation)
        initial, _, _ = load_channel_initial_q(compiled.application_request)
        adapter = build_channel_beris_edwards_runtime(
            ChannelBerisEdwardsRuntimeBuildRequest(compiled.application_request, initial, "cpu")
        )
        public_diagnostics = getattr(adapter, "flow_diagnostics", None)
        return (
            (lambda: adapter.advance(1)),
            public_diagnostics
            if callable(public_diagnostics)
            else (lambda: _solver_pressure_diagnostics(adapter.solver)),
        )
    request = channel_activity_functional_request(simulation.specification)
    runtime = build_channel_activity_functional_runtime(request)
    holder = {"state": runtime.initial_state(), "step": 0}
    control_spec = runtime.control_specs[0].tensor
    controls = {
        "activity": torch.full(
            control_spec.shape,
            0.01,
            dtype=getattr(torch, control_spec.dtype) if isinstance(control_spec.dtype, str) else control_spec.dtype,
            device=control_spec.device,
        )
    }

    def functional_step() -> None:
        holder["state"] = runtime.step(holder["state"], controls, holder["step"])
        holder["step"] += 1

    return functional_step, (lambda: runtime.diagnostics())


def _diagnostic_fields(raw: object) -> dict[str, object]:
    if hasattr(raw, "to_metadata"):
        raw = raw.to_metadata()
    if not isinstance(raw, dict):
        return {"available": False}
    payload = json.loads(json.dumps(raw, allow_nan=False))
    return {"available": True, "payload": payload}


def run_cell(cell: Cell) -> dict[str, object]:
    _validate_cell(cell)
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
    with tempfile.TemporaryDirectory(prefix="pssolver-s09-cpu-cell-") as temporary:
        step, diagnostics = _build_cell(cell, Path(temporary))
        with torch.no_grad():
            for _ in range(cell.warmup_steps):
                step()
            gc.collect()
            samples: list[float] = []
            with _RssSampler() as rss:
                for _ in range(cell.measured_steps):
                    started = time.perf_counter_ns()
                    step()
                    samples.append((time.perf_counter_ns() - started) * 1.0e-9)
            observed = _diagnostic_fields(diagnostics())
    if not samples or not all(math.isfinite(value) and value > 0.0 for value in samples):
        raise RuntimeError("invalid timestep measurements")
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
            "seed": cell.seed,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pssolver_import": str(imported),
            "sys_path": list(sys.path),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "timing": {
            "samples_seconds": samples,
            "mean_step_seconds": statistics.fmean(samples),
            "median_step_seconds": statistics.median(samples),
        },
        "memory": {
            "rss_before_window_bytes": rss.baseline,
            "peak_rss_during_window_bytes": rss.peak,
            "measurement_window_rss_delta_bytes": max(0, rss.peak - rss.baseline),
        },
        "pressure_diagnostics": observed,
    }


def analyze(cells: list[dict[str, object]]) -> dict[str, object]:
    analyses: list[dict[str, object]] = []
    passed = True
    for family in FAMILIES:
        for mode in MODES:
            subset = [
                value for value in cells
                if value["config"]["family"] == family and value["config"]["mode"] == mode
            ]
            trials = sorted({int(value["config"]["trial"]) for value in subset})
            paired_time: list[float] = []
            paired_memory: list[float] = []
            for trial in trials:
                by_role = {
                    str(value["config"]["role"]): value
                    for value in subset
                    if int(value["config"]["trial"]) == trial
                }
                if set(by_role) != {"baseline", "candidate"}:
                    raise ValueError(f"missing paired role for {family}/{mode}/trial{trial}")
                baseline = by_role["baseline"]
                candidate = by_role["candidate"]
                paired_time.append(
                    float(candidate["timing"]["mean_step_seconds"])
                    / float(baseline["timing"]["mean_step_seconds"])
                )
                paired_memory.append(
                    float(candidate["memory"]["peak_rss_during_window_bytes"])
                    / float(baseline["memory"]["peak_rss_during_window_bytes"])
                )
            center = statistics.median(paired_time)
            memory_center = statistics.median(paired_memory)
            gate = (
                center <= TIME_CENTER_LIMIT
                and max(paired_time) <= TIME_PAIRED_LIMIT
                and memory_center <= MEMORY_LIMIT
                and max(paired_memory) <= MEMORY_LIMIT
            )
            passed = passed and gate
            analyses.append(
                {
                    "family": family,
                    "mode": mode,
                    "paired_time_ratios": paired_time,
                    "median_time_ratio": center,
                    "maximum_time_ratio": max(paired_time),
                    "paired_peak_rss_ratios": paired_memory,
                    "median_peak_rss_ratio": memory_center,
                    "maximum_peak_rss_ratio": max(paired_memory),
                    "passed": gate,
                }
            )
    return {
        "thresholds": {
            "median_time_ratio_max": TIME_CENTER_LIMIT,
            "paired_time_ratio_max": TIME_PAIRED_LIMIT,
            "median_peak_rss_ratio_max": MEMORY_LIMIT,
            "paired_peak_rss_ratio_max": MEMORY_LIMIT,
            "relaxation_allowed": False,
        },
        "cells": analyses,
        "passed": passed,
        "eligible_for_h100_preparation": passed,
    }


def _cell_command(args: argparse.Namespace, *, family: str, mode: str, role: str, trial: int, output: Path) -> list[str]:
    root = args.baseline_root if role == "baseline" else args.candidate_root
    commit = args.baseline_commit if role == "baseline" else args.candidate_commit
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell",
        "--family", family,
        "--mode", mode,
        "--role", role,
        "--trial", str(trial),
        "--provider-root", str(root),
        "--provider-commit", commit,
        "--warmup-steps", str(args.warmup_steps),
        "--measured-steps", str(args.measured_steps),
        "--shape", *(str(value) for value in args.shape),
        "--lengths", *(str(value) for value in args.lengths),
        "--output", str(output),
    ]


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    cell_root = output.parent / f"{output.stem}_cells"
    if cell_root.exists() or cell_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {cell_root}")
    cell_root.mkdir(parents=True)
    cells: list[dict[str, object]] = []
    sequence: list[dict[str, object]] = []
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    for family in FAMILIES:
        for mode in MODES:
            for trial in range(1, args.trials + 1):
                roles = ("baseline", "candidate") if trial % 2 else ("candidate", "baseline")
                for role in roles:
                    cell_output = cell_root / f"{family}_{mode}_{role}_trial{trial}.json"
                    command = _cell_command(args, family=family, mode=mode, role=role, trial=trial, output=cell_output)
                    completed = subprocess.run(command, env=env, cwd=cell_root, text=True, capture_output=True, check=False)
                    sequence.append({"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
                    if completed.returncode != 0:
                        raise RuntimeError(f"cell failed: {family}/{mode}/{role}/trial{trial}: {completed.stderr}")
                    cells.append(json.loads(cell_output.read_text(encoding="utf-8")))
    result = {
        "schema": SCHEMA,
        "kind": "matrix",
        "baseline": {"root": str(args.baseline_root.resolve()), "commit": args.baseline_commit},
        "candidate": {"root": str(args.candidate_root.resolve()), "commit": args.candidate_commit},
        "contract": {"families": list(FAMILIES), "modes": list(MODES), "trials": args.trials, "warmup_steps": args.warmup_steps, "measured_steps": args.measured_steps, "shape": list(args.shape), "lengths": list(args.lengths), "alternating_pair_order": True},
        "sequence": sequence,
        "raw_cells": cells,
        "analysis": analyze(cells),
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cell", action="store_true")
    modes.add_argument("--matrix", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--role", choices=("baseline", "candidate"))
    parser.add_argument("--trial", type=int)
    parser.add_argument("--provider-root", type=Path)
    parser.add_argument("--provider-commit")
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-commit")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--shape", type=int, nargs=3, default=(128, 20, 20))
    parser.add_argument("--lengths", type=float, nargs=3, default=(40.0, 6.0, 6.0))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cell:
        required = (args.family, args.mode, args.role, args.trial, args.provider_root, args.provider_commit)
        if any(value is None for value in required):
            raise ValueError("cell mode requires family, mode, role, trial, provider root and commit")
        report = run_cell(
            Cell(
                family=args.family,
                mode=args.mode,
                role=args.role,
                trial=args.trial,
                provider_root=args.provider_root,
                provider_commit=args.provider_commit,
                warmup_steps=args.warmup_steps,
                measured_steps=args.measured_steps,
                shape=tuple(args.shape),
                lengths=tuple(args.lengths),
            )
        )
        _atomic_json(args.output, report)
        return 0
    required = (args.baseline_root, args.baseline_commit, args.candidate_root, args.candidate_commit)
    if any(value is None for value in required):
        raise ValueError("matrix mode requires both provider roots and commits")
    report = run_matrix(args)
    print(json.dumps(report["analysis"], allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["analysis"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
