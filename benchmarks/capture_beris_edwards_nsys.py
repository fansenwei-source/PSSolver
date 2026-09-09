"""Run a bounded Beris--Edwards timestep capture under Nsight Systems.

This module does not invoke Nsight itself. It exposes nested NVTX ranges and
uses the CUDA profiler API to delimit only the requested timesteps. A typical
capture is:

    nsys profile --trace=cuda,nvtx,osrt \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --output=/tmp/pssolver_nsys \
      python -m benchmarks.capture_beris_edwards_nsys
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch

from pssolver import DEFAULT_TRANSFORM_EXECUTION_ORDER
from pssolver.models.active_nematics import (
    DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE,
)

from benchmarks.profile_beris_edwards_timestep import (
    ProfileConfig,
    _build_solver,
    _git_provenance,
    _state_sha256,
    _three_floats,
    _three_ints,
    _validate_config,
)


@dataclass(frozen=True)
class NsysCaptureConfig:
    """Inputs defining one bounded Nsight Systems workload."""

    shape: tuple[int, int, int] = (64, 64, 32)
    lengths: tuple[float, float, float] = (100.0, 100.0, 20.0)
    dtype: str = "float64"
    dt: float = 0.005
    dealias_rule: str = "cubic_half"
    warmup_steps: int = 5
    capture_steps: int = 3
    spectral_refresh_interval: int | None = None
    pressure_diagnostics: bool = False
    reuse_q_gradients: bool = True
    molecular_field_linear_space: str = DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE
    transform_execution_order: str = DEFAULT_TRANSFORM_EXECUTION_ORDER
    seed: int = 20260908

    def profile_config(self) -> ProfileConfig:
        """Return the matching production-path profiler configuration."""
        return ProfileConfig(
            shape=self.shape,
            lengths=self.lengths,
            device="cuda",
            dtype=self.dtype,
            dt=self.dt,
            dealias_rule=self.dealias_rule,
            warmup_steps=self.warmup_steps,
            profile_steps=self.capture_steps,
            spectral_refresh_interval=self.spectral_refresh_interval,
            pressure_diagnostics=self.pressure_diagnostics,
            reuse_q_gradients=self.reuse_q_gradients,
            molecular_field_linear_space=(
                self.molecular_field_linear_space
            ),
            transform_execution_order=self.transform_execution_order,
            seed=self.seed,
        )


class NvtxRegionTimer:
    """Expose existing benchmark regions as nested NVTX push/pop ranges."""

    def __init__(self):
        self.enabled = False

    @contextmanager
    def region(self, name: str, *, host: bool = False) -> Iterator[None]:
        del host
        if not self.enabled:
            yield
            return

        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


def _validate_capture_config(config: NsysCaptureConfig) -> None:
    if config.capture_steps <= 0:
        raise ValueError("capture_steps must be positive")
    _validate_config(config.profile_config())


def run_capture(config: NsysCaptureConfig) -> dict[str, object]:
    """Execute warmup plus a CUDA-profiler-delimited timestep capture."""
    _validate_capture_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Nsight capture requires an available CUDA device")

    device = torch.device("cuda")
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    timer = NvtxRegionTimer()

    with torch.no_grad():
        solver = _build_solver(config.profile_config(), timer)
        for _ in range(config.warmup_steps):
            solver.integrator.step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.cudart().cudaProfilerStart()
        timer.enabled = True
        start.record()
        try:
            for _ in range(config.capture_steps):
                solver.integrator.step()
        finally:
            stop.record()
            stop.synchronize()
            timer.enabled = False
            torch.cuda.cudart().cudaProfilerStop()

        elapsed_seconds = start.elapsed_time(stop) / 1000.0
        final_state_sha256 = _state_sha256(solver)

    return {
        "schema_version": 1,
        "capture": asdict(config),
        "environment": {
            "git": _git_provenance(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        },
        "elapsed_seconds": elapsed_seconds,
        "mean_timestep_seconds": elapsed_seconds / config.capture_steps,
        "memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "final_state_sha256": final_state_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_three_ints, default=(64, 64, 32))
    parser.add_argument(
        "--lengths",
        type=_three_floats,
        default=(100.0, 100.0, 20.0),
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument(
        "--dealias-rule",
        choices=("none", "quadratic_two_thirds", "cubic_half"),
        default="cubic_half",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--capture-steps", type=int, default=3)
    parser.add_argument("--spectral-refresh-interval", type=int)
    parser.add_argument("--pressure-diagnostics", action="store_true")
    parser.add_argument("--disable-q-gradient-reuse", action="store_true")
    parser.add_argument(
        "--molecular-field-linear-space",
        choices=("physical", "spectral"),
        default=DEFAULT_MOLECULAR_FIELD_LINEAR_SPACE,
    )
    parser.add_argument(
        "--transform-execution-order",
        choices=("legacy", "real_first"),
        default=DEFAULT_TRANSFORM_EXECUTION_ORDER,
    )
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"output exists: {args.output}; pass --overwrite to replace it")
    return args


def main() -> None:
    args = parse_args()
    result = run_capture(
        NsysCaptureConfig(
            shape=args.shape,
            lengths=args.lengths,
            dtype=args.dtype,
            dt=args.dt,
            dealias_rule=args.dealias_rule,
            warmup_steps=args.warmup_steps,
            capture_steps=args.capture_steps,
            spectral_refresh_interval=args.spectral_refresh_interval,
            pressure_diagnostics=args.pressure_diagnostics,
            reuse_q_gradients=not args.disable_q_gradient_reuse,
            molecular_field_linear_space=(
                args.molecular_field_linear_space
            ),
            transform_execution_order=args.transform_execution_order,
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
