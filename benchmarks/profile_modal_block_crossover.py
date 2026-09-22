"""Map the H100 crossover of the two qualified 2-by-2 modal solvers.

This is an evidence-only microbenchmark.  It does not select a production
implementation or change the P4.4 eligibility state.  The reference and
closed-form paths share input, output-copy, and workspace validation.  Matrix
rejection remains implementation-specific: ``torch.linalg.solve`` performs
its own factorization checks, while the analytic path explicitly checks its
determinant.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch

from pssolver.core.modal_blocks import TwoComponentModalOperatorSpec
from pssolver.operators.modal_block import (
    BoundTwoComponentModalOperator,
    TwoComponentModalSolveWorkspace,
    bind_two_component_modal_operator,
)


REFERENCE = "A_torch_linalg"
CANDIDATE = "B_closed_form_2x2"
VARIANTS = (REFERENCE, CANDIDATE)


@dataclass(frozen=True, slots=True)
class ModalBlockCrossoverConfig:
    """Inputs defining one reproducible modal-block crossover scan."""

    mode_counts: tuple[int, ...] = (
        131_072,
        262_144,
        524_288,
        1_048_576,
        2_097_152,
        4_194_304,
    )
    device: str = "cuda"
    dtype: str = "complex128"
    alpha: float = 4.0
    warmup: int = 5
    repeats: int = 30
    trials: int = 3
    seed: int = 20260922
    maximum_acceptable_median_ratio: float = 1.05


def _validate_config(config: ModalBlockCrossoverConfig) -> None:
    if (
        not config.mode_counts
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in config.mode_counts
        )
        or tuple(sorted(set(config.mode_counts))) != config.mode_counts
    ):
        raise ValueError("mode_counts must be unique increasing positive integers")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if config.dtype not in {"complex64", "complex128"}:
        raise ValueError("dtype must be 'complex64' or 'complex128'")
    if not math.isfinite(config.alpha) or config.alpha <= 0.0:
        raise ValueError("alpha must be positive and finite")
    if config.warmup < 0 or config.repeats <= 0 or config.trials <= 0:
        raise ValueError("warmup must be non-negative; repeats and trials positive")
    if (
        not math.isfinite(config.maximum_acceptable_median_ratio)
        or config.maximum_acceptable_median_ratio <= 0.0
    ):
        raise ValueError("maximum acceptable median ratio must be positive")


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


def _dtypes(name: str) -> tuple[torch.dtype, torch.dtype]:
    if name == "complex64":
        return torch.float32, torch.complex64
    return torch.float64, torch.complex128


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm((actual - reference).reshape(-1))
    scale = torch.linalg.vector_norm(reference.reshape(-1))
    tiny = torch.finfo(reference.real.dtype).tiny
    return float((difference / scale.clamp_min(tiny)).item())


def _normalized_l2(value: torch.Tensor, scale: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(value.reshape(-1))
    denominator = torch.linalg.vector_norm(scale.reshape(-1))
    tiny = torch.finfo(scale.real.dtype).tiny
    return float((numerator / denominator.clamp_min(tiny)).item())


def _reference_solve_into(
    operator: BoundTwoComponentModalOperator,
    rhs: torch.Tensor,
    *,
    alpha: float,
    workspace: TwoComponentModalSolveWorkspace,
) -> torch.Tensor:
    """Execute the qualification reference with the common public gates."""

    operator._validate_rhs(rhs)
    operator._validate_workspace(rhs, workspace)
    identity = torch.eye(2, dtype=operator.dtype, device=operator.device)
    matrices = alpha * identity - operator.coefficients
    solution = torch.linalg.solve(matrices, rhs.unsqueeze(-1)).squeeze(-1)
    workspace.output.copy_(solution)
    return workspace.output


def _time_calls(
    call: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        events = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            call()
            stop.record()
            events.append((start, stop))
        torch.cuda.synchronize(device)
        return math.fsum(start.elapsed_time(stop) for start, stop in events) / repeats

    start_time = time.perf_counter()
    for _ in range(repeats):
        call()
    return 1_000.0 * (time.perf_counter() - start_time) / repeats


def _memory_probe(
    call: Callable[[], torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    starting_allocated = int(torch.cuda.memory_allocated(device))
    starting_reserved = int(torch.cuda.memory_reserved(device))
    call()
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    return {
        "starting_allocated_bytes": starting_allocated,
        "starting_reserved_bytes": starting_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_allocated_bytes": peak_allocated - starting_allocated,
        "incremental_peak_reserved_bytes": peak_reserved - starting_reserved,
    }


def _aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "trials": len(values),
        "mean_milliseconds": statistics.fmean(values),
        "median_milliseconds": statistics.median(values),
        "sample_standard_deviation_milliseconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
    }


def select_stable_suffix_threshold(
    rows: list[dict[str, object]],
    *,
    maximum_ratio: float,
) -> int | None:
    """Return the first count whose complete measured suffix meets the gate."""

    for index, row in enumerate(rows):
        suffix = rows[index:]
        if all(
            float(item["candidate_over_reference_median_ratio"])
            <= maximum_ratio
            for item in suffix
        ):
            return int(row["mode_count"])
    return None


def _run_case(
    config: ModalBlockCrossoverConfig,
    *,
    mode_count: int,
    device: torch.device,
) -> dict[str, object]:
    real_dtype, spectral_dtype = _dtypes(config.dtype)
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + mode_count
    )
    eigenvalues = -torch.linspace(
        0.0,
        64.0,
        mode_count,
        dtype=real_dtype,
        device=device,
    )
    spec = TwoComponentModalOperatorSpec(
        component_order=("u", "v"),
        diffusion=(0.12, 0.2),
        coupling=((-0.3, 0.4), (-0.2, -0.1)),
    )
    operator = bind_two_component_modal_operator(
        spec,
        eigenvalues,
        component_order=spec.component_order,
        geometry_identity=spec.geometry_identity,
        basis_signature=spec.basis_signature,
        spectral_dtype=spectral_dtype,
        implementation="closed_form_2x2",
    )
    real = torch.randn(mode_count, 2, generator=generator, dtype=real_dtype)
    imaginary = torch.randn(mode_count, 2, generator=generator, dtype=real_dtype)
    rhs = torch.complex(real, imaginary).to(device=device)
    reference_workspace = operator.allocate_workspace()
    candidate_workspace = operator.allocate_workspace()

    calls = {
        REFERENCE: lambda: _reference_solve_into(
            operator,
            rhs,
            alpha=config.alpha,
            workspace=reference_workspace,
        ),
        CANDIDATE: lambda: operator.solve_into(
            rhs,
            alpha=config.alpha,
            workspace=candidate_workspace,
        ),
    }

    reference = calls[REFERENCE]().clone()
    candidate = calls[CANDIDATE]().clone()
    identity = torch.eye(2, dtype=spectral_dtype, device=device)
    residual = torch.matmul(
        config.alpha * identity - operator.coefficients,
        candidate.unsqueeze(-1),
    ).squeeze(-1) - rhs
    correctness = {
        "relative_l2": _relative_l2(candidate, reference),
        "linf": float(torch.max(torch.abs(candidate - reference)).item()),
        "residual_relative_l2": _normalized_l2(residual, rhs),
        "finite": bool(torch.isfinite(candidate).all()),
        "reference_workspace_pointer_stable": True,
        "candidate_workspace_pointer_stable": True,
    }
    tolerance = 2.0e-6 if spectral_dtype is torch.complex64 else 1.0e-12
    if (
        not correctness["finite"]
        or correctness["relative_l2"] > tolerance
        or correctness["linf"] > tolerance
        or correctness["residual_relative_l2"] > tolerance
    ):
        raise RuntimeError(
            f"modal-block correctness gate failed at mode_count={mode_count}: "
            f"{correctness}"
        )

    pointer_snapshots = {
        name: tuple(
            tensor.data_ptr()
            for tensor in (
                workspace.output,
                workspace.diagonal_0,
                workspace.diagonal_1,
                workspace.determinant,
                workspace.temporary,
            )
        )
        for name, workspace in (
            (REFERENCE, reference_workspace),
            (CANDIDATE, candidate_workspace),
        )
    }
    trial_records = {name: [] for name in VARIANTS}
    for trial in range(config.trials):
        order = VARIANTS if trial % 2 == 0 else tuple(reversed(VARIANTS))
        for position, variant in enumerate(order, start=1):
            milliseconds = _time_calls(
                calls[variant],
                device=device,
                warmup=config.warmup,
                repeats=config.repeats,
            )
            trial_records[variant].append(
                {
                    "trial": trial + 1,
                    "order": position,
                    "mean_milliseconds": milliseconds,
                }
            )

    for name, workspace in (
        (REFERENCE, reference_workspace),
        (CANDIDATE, candidate_workspace),
    ):
        current = tuple(
            tensor.data_ptr()
            for tensor in (
                workspace.output,
                workspace.diagonal_0,
                workspace.diagonal_1,
                workspace.determinant,
                workspace.temporary,
            )
        )
        correctness[f"{name}_workspace_pointer_stable"] = (
            current == pointer_snapshots[name]
        )
        if current != pointer_snapshots[name]:
            raise RuntimeError(f"{name} workspace identity changed")

    aggregate = {
        name: _aggregate(
            [float(record["mean_milliseconds"]) for record in records]
        )
        for name, records in trial_records.items()
    }
    reference_median = float(aggregate[REFERENCE]["median_milliseconds"])
    candidate_median = float(aggregate[CANDIDATE]["median_milliseconds"])
    paired_ratios = [
        float(candidate_record["mean_milliseconds"])
        / float(reference_record["mean_milliseconds"])
        for reference_record, candidate_record in zip(
            trial_records[REFERENCE],
            trial_records[CANDIDATE],
            strict=True,
        )
    ]
    return {
        "mode_count": mode_count,
        "correctness": correctness,
        "trial_records": trial_records,
        "aggregate": aggregate,
        "candidate_over_reference_median_ratio": (
            candidate_median / reference_median
        ),
        "paired_candidate_over_reference_ratios": paired_ratios,
        "memory": {
            name: _memory_probe(calls[name], device=device)
            for name in VARIANTS
        },
    }


def run_crossover_profile(
    config: ModalBlockCrossoverConfig,
) -> dict[str, object]:
    """Execute a balanced scan and return a non-promoting policy proposal."""

    _validate_config(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        rows = [
            _run_case(
                config,
                mode_count=mode_count,
                device=device,
            )
            for mode_count in config.mode_counts
        ]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32

    threshold = select_stable_suffix_threshold(
        rows,
        maximum_ratio=config.maximum_acceptable_median_ratio,
    )
    return {
        "schema_version": 1,
        "identity": "p4_4_modal_block_crossover_profile",
        "scope": "evidence_only_no_production_selection",
        "config": asdict(config),
        "environment": {
            "git_head": _git_head(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
            "tf32_matmul_effective": False,
            "tf32_cudnn_effective": False,
        },
        "variants": {
            REFERENCE: "torch.linalg.solve with shared input/workspace gates",
            CANDIDATE: "closed_form_2x2 with fixed P4.4 workspace",
        },
        "rows": rows,
        "crossover": {
            "gate": "stable measured suffix",
            "maximum_candidate_over_reference_median_ratio": (
                config.maximum_acceptable_median_ratio
            ),
            "recommended_switch_at_mode_count": threshold,
            "eligible_for_auto_policy_implementation": threshold is not None,
            "production_default_changed": False,
            "p4_5_authorized": False,
        },
    }


def _mode_counts(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("mode counts must be integers") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("mode counts must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode-counts",
        type=_mode_counts,
        default=ModalBlockCrossoverConfig().mode_counts,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("complex64", "complex128"),
        default="complex128",
    )
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--maximum-median-ratio", type=float, default=1.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite")
    return args


def main() -> None:
    args = parse_args()
    result = run_crossover_profile(
        ModalBlockCrossoverConfig(
            mode_counts=args.mode_counts,
            device=args.device,
            dtype=args.dtype,
            alpha=args.alpha,
            warmup=args.warmup,
            repeats=args.repeats,
            trials=args.trials,
            seed=args.seed,
            maximum_acceptable_median_ratio=args.maximum_median_ratio,
        )
    )
    serialized = json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
