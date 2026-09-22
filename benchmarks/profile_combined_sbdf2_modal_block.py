"""Profile the Phase 4.5 combined SBDF2/modal-block reference canary.

The two measured roles execute the same numerical contract.  Role A advances
continuously; role B rebinds an equivalent stepper at a complete-history
boundary before the timed interval.  The comparison therefore measures
restart/rebind non-regression without comparing different time integrators.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import time

import torch

from pssolver.core.integrators import IntegratorSpec
from pssolver.core.modal_blocks import TwoComponentModalOperatorSpec
from pssolver.experimental.modal_block_reference import (
    TwoComponentReferenceState,
    TwoComponentSBDF2CanaryModel,
    TwoComponentSBDF2ReferenceStepper,
)
from pssolver.operators.modal_block_reference import (
    TwoComponentPeriodicReactionDiffusionReference,
)


CONTINUOUS = "A_continuous"
REBOUND = "B_rebound"
VARIANTS = (CONTINUOUS, REBOUND)


@dataclass(frozen=True, slots=True)
class CombinedCanaryProfileConfig:
    point_counts: tuple[int, ...] = (131_072, 1_048_576)
    device: str = "cuda"
    dtype: str = "float64"
    dt: float = 0.005
    warmup_steps: int = 5
    measured_steps: int = 30
    trials: int = 3
    maximum_timing_ratio: float = 1.10
    maximum_memory_ratio: float = 1.10


def _validate_config(config: CombinedCanaryProfileConfig) -> None:
    if (
        not config.point_counts
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 8
            or value % 2
            for value in config.point_counts
        )
        or tuple(sorted(set(config.point_counts))) != config.point_counts
    ):
        raise ValueError("point_counts must be unique increasing even integers")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if config.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")
    if not math.isfinite(config.dt) or config.dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if config.warmup_steps < 1 or config.measured_steps < 1 or config.trials < 1:
        raise ValueError("warmup, measured steps, and trials must be positive")
    for value, name in (
        (config.maximum_timing_ratio, "maximum_timing_ratio"),
        (config.maximum_memory_ratio, "maximum_memory_ratio"),
    ):
        if not math.isfinite(value) or value < 1.0:
            raise ValueError(f"{name} must be finite and at least one")


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _dtype(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.float64


def _model(point_count: int) -> TwoComponentSBDF2CanaryModel:
    declaration = TwoComponentModalOperatorSpec(
        component_order=("u", "v"),
        diffusion=(0.12, 0.2),
        coupling=((-0.3, 0.4), (-0.2, -0.1)),
    )
    implicit = TwoComponentPeriodicReactionDiffusionReference(
        operator_spec=declaration,
        point_count=point_count,
        initial_mode=2,
        initial_amplitudes=(0.8, -0.35),
    )
    return TwoComponentSBDF2CanaryModel(
        implicit_model=implicit,
        explicit_growth_rate=0.07,
    )


def _stepper(
    *,
    point_count: int,
    dt: float,
    dtype: torch.dtype,
    device: torch.device,
) -> TwoComponentSBDF2ReferenceStepper:
    return TwoComponentSBDF2ReferenceStepper(
        model=_model(point_count),
        spec=IntegratorSpec.sbdf2(dt=dt),
        dtype=dtype,
        device=device,
        implementation="closed_form_2x2",
    )


def _workspace_pointers(
    stepper: TwoComponentSBDF2ReferenceStepper,
) -> tuple[int, ...]:
    workspace = stepper.workspace
    return tuple(
        value.data_ptr()
        for value in (
            workspace.current_explicit_rhs,
            workspace.assembled_implicit_rhs,
            workspace.modal_solve.output,
            workspace.modal_solve.diagonal_0,
            workspace.modal_solve.diagonal_1,
            workspace.modal_solve.determinant,
            workspace.modal_solve.temporary,
        )
    )


def _prepare_role(
    config: CombinedCanaryProfileConfig,
    *,
    point_count: int,
    device: torch.device,
    variant: str,
) -> tuple[TwoComponentSBDF2ReferenceStepper, TwoComponentReferenceState]:
    dtype = _dtype(config.dtype)
    stepper = _stepper(
        point_count=point_count,
        dt=config.dt,
        dtype=dtype,
        device=device,
    )
    state = stepper.run(steps=config.warmup_steps)
    if variant == REBOUND:
        stepper = _stepper(
            point_count=point_count,
            dt=config.dt,
            dtype=dtype,
            device=device,
        )
        stepper.validate_state(state)
    return stepper, state


def _advance_timed(
    stepper: TwoComponentSBDF2ReferenceStepper,
    state: TwoComponentReferenceState,
    *,
    steps: int,
    device: torch.device,
) -> tuple[TwoComponentReferenceState, float, dict[str, int] | None, bool]:
    pointers = _workspace_pointers(stepper)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        # Keep the two roles on the same allocator footing.  In particular,
        # cached blocks left by a previously measured role are not part of the
        # current role's live tensor set and must not bias peak-reserved memory.
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        starting_allocated = int(torch.cuda.memory_allocated(device))
        starting_reserved = int(torch.cuda.memory_reserved(device))
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        current = stepper.run(steps=steps, state=state)
        stop.record()
        torch.cuda.synchronize(device)
        elapsed_ms = float(start.elapsed_time(stop))
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        memory = {
            "starting_allocated_bytes": starting_allocated,
            "starting_reserved_bytes": starting_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": (
                peak_allocated - starting_allocated
            ),
            "incremental_peak_reserved_bytes": peak_reserved - starting_reserved,
        }
    else:
        start_time = time.perf_counter()
        current = stepper.run(steps=steps, state=state)
        elapsed_ms = 1_000.0 * (time.perf_counter() - start_time)
        memory = None
    stable = pointers == _workspace_pointers(stepper)
    return current, elapsed_ms / steps, memory, stable


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((actual - expected).reshape(-1))
    denominator = torch.linalg.vector_norm(expected.reshape(-1))
    tiny = torch.finfo(expected.dtype).tiny
    return float((numerator / denominator.clamp_min(tiny)).item())


def _aggregate(records: list[dict[str, object]]) -> dict[str, float | int]:
    values = [float(record["milliseconds_per_step"]) for record in records]
    return {
        "trials": len(values),
        "mean_milliseconds_per_step": statistics.fmean(values),
        "median_milliseconds_per_step": statistics.median(values),
        "sample_standard_deviation_milliseconds_per_step": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
    }


def _run_case(
    config: CombinedCanaryProfileConfig,
    *,
    point_count: int,
    device: torch.device,
) -> dict[str, object]:
    trial_records: dict[str, list[dict[str, object]]] = {
        name: [] for name in VARIANTS
    }
    comparison_states: dict[
        int,
        dict[str, tuple[torch.Tensor, torch.Tensor]],
    ] = {}
    for trial in range(config.trials):
        order = VARIANTS if trial % 2 == 0 else tuple(reversed(VARIANTS))
        comparison_states[trial] = {}
        for position, variant in enumerate(order, start=1):
            stepper, state = _prepare_role(
                config,
                point_count=point_count,
                device=device,
                variant=variant,
            )
            final, milliseconds, memory, stable = _advance_timed(
                stepper,
                state,
                steps=config.measured_steps,
                device=device,
            )
            # Comparison artifacts live on CPU so that the second role is not
            # measured while the first role's complete GPU state is retained.
            comparison_states[trial][variant] = (
                final.physical.detach().cpu().clone(),
                final.native_spectrum.detach().cpu().clone(),
            )
            trial_records[variant].append(
                {
                    "trial": trial + 1,
                    "order": position,
                    "milliseconds_per_step": milliseconds,
                    "memory": memory,
                    "workspace_pointer_stable": stable,
                    "finite": bool(torch.isfinite(final.physical).all()),
                    "workspace_allocated_tensor_count": (
                        stepper.workspace.allocated_tensor_count
                    ),
                    "implementation": stepper.operator.implementation,
                }
            )
            del final, state, stepper
            if device.type == "cuda":
                torch.cuda.empty_cache()

    comparisons = []
    for trial in range(config.trials):
        continuous_physical, continuous_spectrum = comparison_states[trial][
            CONTINUOUS
        ]
        rebound_physical, rebound_spectrum = comparison_states[trial][REBOUND]
        comparisons.append(
            {
                "trial": trial + 1,
                "physical_byte_identical": torch.equal(
                    continuous_physical,
                    rebound_physical,
                ),
                "spectrum_byte_identical": torch.equal(
                    continuous_spectrum,
                    rebound_spectrum,
                ),
                "physical_relative_l2": _relative_l2(
                    rebound_physical,
                    continuous_physical,
                ),
                "physical_linf": float(
                    torch.max(
                        torch.abs(rebound_physical - continuous_physical)
                    )
                ),
            }
        )
    aggregate = {
        variant: _aggregate(trial_records[variant]) for variant in VARIANTS
    }
    continuous_median = float(
        aggregate[CONTINUOUS]["median_milliseconds_per_step"]
    )
    rebound_median = float(
        aggregate[REBOUND]["median_milliseconds_per_step"]
    )
    paired_ratios = [
        float(rebound["milliseconds_per_step"])
        / float(continuous["milliseconds_per_step"])
        for continuous, rebound in zip(
            trial_records[CONTINUOUS],
            trial_records[REBOUND],
            strict=True,
        )
    ]
    return {
        "point_count": point_count,
        "trial_records": trial_records,
        "comparisons": comparisons,
        "aggregate": aggregate,
        "rebound_over_continuous_median_ratio": (
            rebound_median / continuous_median
        ),
        "paired_rebound_over_continuous_ratios": paired_ratios,
    }


def _convergence(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    final_time = 0.4
    model = _model(32)
    exact = model.exact_physical(final_time, dtype=dtype, device=device)
    samples = []
    for dt in (0.04, 0.02, 0.01, 0.005):
        stepper = _stepper(
            point_count=32,
            dt=dt,
            dtype=dtype,
            device=device,
        )
        state = stepper.run(steps=round(final_time / dt))
        difference = state.physical - exact
        samples.append(
            {
                "dt": dt,
                "l2_error": float(
                    torch.linalg.vector_norm(difference)
                    / math.sqrt(difference.numel())
                ),
                "linf_error": float(torch.max(torch.abs(difference))),
                "finite": bool(torch.isfinite(state.physical).all()),
            }
        )
    orders = [
        math.log(
            float(coarse["l2_error"]) / float(fine["l2_error"]),
            2.0,
        )
        for coarse, fine in zip(samples[:-1], samples[1:], strict=True)
    ]
    return {
        "final_time": final_time,
        "samples": samples,
        "pairwise_l2_orders": orders,
        "minimum_observed_l2_order": min(orders),
        "minimum_required_l2_order": 1.8,
        "passed": min(orders) >= 1.8
        and all(bool(sample["finite"]) for sample in samples),
    }


def run_profile(config: CombinedCanaryProfileConfig) -> dict[str, object]:
    _validate_config(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        rows = [
            _run_case(
                config,
                point_count=point_count,
                device=device,
            )
            for point_count in config.point_counts
        ]
        convergence = _convergence(device=device, dtype=_dtype(config.dtype))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn
    concrete_device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if device.type == "cuda"
        else device
    )
    return {
        "schema_version": 2,
        "identity": "p4_6_combined_sbdf2_modal_block_profile",
        "scope": "qualification_only_no_production_selection",
        "memory_measurement": {
            "role_live_sets_isolated": True,
            "comparison_snapshots_device": "cpu",
            "unused_allocator_cache_cleared_before_timing": True,
            "peak_scope": "current_role_only",
        },
        "config": asdict(config),
        "environment": {
            "git_head": _git_head(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "requested_device": str(device),
            "concrete_device": str(concrete_device),
            "device_name": (
                torch.cuda.get_device_name(concrete_device)
                if device.type == "cuda"
                else "CPU"
            ),
            "tf32_matmul_effective": False,
            "tf32_cudnn_effective": False,
        },
        "variants": {
            CONTINUOUS: "continuous complete-history SBDF2 execution",
            REBOUND: "equivalent stepper rebound before timed execution",
        },
        "rows": rows,
        "convergence": convergence,
        "eligibility": {
            "profile_complete": True,
            "production_default_changed": False,
            "phase_5_authorized": False,
        },
    }


def _point_counts(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("point counts must be integers") from exc
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--point-counts",
        type=_point_counts,
        default=CombinedCanaryProfileConfig().point_counts,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--maximum-timing-ratio", type=float, default=1.10)
    parser.add_argument("--maximum-memory-ratio", type=float, default=1.10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite")
    return args


def main() -> None:
    args = parse_args()
    result = run_profile(
        CombinedCanaryProfileConfig(
            point_counts=args.point_counts,
            device=args.device,
            dtype=args.dtype,
            dt=args.dt,
            warmup_steps=args.warmup_steps,
            measured_steps=args.measured_steps,
            trials=args.trials,
            maximum_timing_ratio=args.maximum_timing_ratio,
            maximum_memory_ratio=args.maximum_memory_ratio,
        )
    )
    serialized = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
