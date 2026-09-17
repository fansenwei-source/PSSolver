"""Compare dense and opt-in FFT bounded-axis DCT/DST transforms.

The default dense-only configuration preserves the R2R-A attribution baseline.
Selecting both algorithms adds the R2R-B candidate, checks it against the
frozen orthonormal dense reference, and reports the size-dependent crossover.
No benchmark selection changes the production default.
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

from pssolver import load_bounded_transform_policy
from pssolver.transforms import TensorProductTransformBackend


TRANSFORM_KINDS = ("dct", "dst")
EXECUTION_MODES = ("full", "truncated")
VALUE_TYPES = ("real", "complex")
ALGORITHMS = ("dense", "fft", "auto")
DEFAULT_SIZES = (20, 40, 80, 120, 160, 240, 320)


@dataclass(frozen=True)
class BoundedAxisBenchmarkConfig:
    """Inputs defining one reproducible bounded-axis transform sweep."""

    sizes: tuple[int, ...] = DEFAULT_SIZES
    kinds: tuple[str, ...] = TRANSFORM_KINDS
    execution_modes: tuple[str, ...] = EXECUTION_MODES
    value_types: tuple[str, ...] = VALUE_TYPES
    algorithms: tuple[str, ...] = ("dense",)
    geometry: str = "bounded_axis_microbenchmark"
    policy_path: str | None = None
    retained_fraction: float = 0.5
    line_count: int = 256
    device: str = "cpu"
    dtype: str = "float64"
    warmup: int = 3
    repeats: int = 10
    seed: int = 20260917


def _validate_config(config: BoundedAxisBenchmarkConfig) -> None:
    if not config.sizes or any(size <= 0 for size in config.sizes):
        raise ValueError("sizes must contain positive integers")
    if not config.kinds or any(kind not in TRANSFORM_KINDS for kind in config.kinds):
        raise ValueError(f"kinds must be drawn from {TRANSFORM_KINDS}")
    if not config.execution_modes or any(
        mode not in EXECUTION_MODES for mode in config.execution_modes
    ):
        raise ValueError(
            f"execution_modes must be drawn from {EXECUTION_MODES}"
        )
    if not config.value_types or any(
        value_type not in VALUE_TYPES for value_type in config.value_types
    ):
        raise ValueError(f"value_types must be drawn from {VALUE_TYPES}")
    if not config.algorithms or any(
        algorithm not in ALGORITHMS for algorithm in config.algorithms
    ):
        raise ValueError(f"algorithms must be drawn from {ALGORITHMS}")
    if not config.geometry:
        raise ValueError("geometry must be non-empty")
    if "auto" in config.algorithms:
        if config.policy_path is None:
            raise ValueError("auto requires a bounded-transform policy")
        if not Path(config.policy_path).is_file():
            raise FileNotFoundError(
                f"bounded-transform policy is missing: {config.policy_path}"
            )
    elif config.policy_path is not None:
        raise ValueError("policy_path is only valid when auto is requested")
    if not math.isfinite(config.retained_fraction) or not (
        0.0 < config.retained_fraction <= 1.0
    ):
        raise ValueError("retained_fraction must be in (0, 1]")
    if config.line_count <= 0:
        raise ValueError("line_count must be positive")
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if config.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be 'float32' or 'float64'")
    if config.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if config.repeats <= 0:
        raise ValueError("repeats must be positive")


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timing_samples(
    operation: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        _synchronize(device)

        if device.type == "cuda":
            events = []
            for _ in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                operation()
                stop.record()
                events.append((start, stop))
            _synchronize(device)
            return [
                start.elapsed_time(stop) / 1000.0
                for start, stop in events
            ]

        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            operation()
            samples.append(time.perf_counter() - start)
        return samples


def _sample_summary(samples: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(samples),
        "mean_seconds": statistics.fmean(samples),
        "median_seconds": statistics.median(samples),
        "sample_std_seconds": (
            statistics.stdev(samples) if len(samples) > 1 else 0.0
        ),
        "minimum_seconds": min(samples),
        "maximum_seconds": max(samples),
    }


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm((actual - reference).reshape(-1))
    scale = torch.linalg.vector_norm(reference.reshape(-1))
    tiny = torch.finfo(reference.real.dtype).tiny
    return float((difference / scale.clamp_min(tiny)).item())


def _git_provenance() -> dict[str, str | bool | None]:
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


def _make_values(
    line_count: int,
    size: int,
    *,
    value_type: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = torch.randn(
        line_count,
        size,
        device=device,
        dtype=dtype,
    )
    if value_type == "complex":
        imaginary = torch.randn(
            line_count,
            size,
            device=device,
            dtype=dtype,
        )
        values = torch.complex(values, imaginary).to(_complex_dtype(dtype))
    return values


def _matrix_cache_bytes(backend: TensorProductTransformBackend) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in backend._matrix_cache.values()
    )


def _fft_r2r_cache_bytes(backend: TensorProductTransformBackend) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for data in backend._fft_r2r_cache.values()
        for tensor in data.values()
    )


def _run_case(
    config: BoundedAxisBenchmarkConfig,
    *,
    size: int,
    kind: str,
    execution_mode: str,
    value_type: str,
    algorithm: str,
    bounded_policy,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    backend = TensorProductTransformBackend(
        shape=(size,),
        lengths=(1.0,),
        device=device,
        dtype=dtype,
        bounded_transform_algorithm=algorithm,
        bounded_transform_geometry=config.geometry,
        bounded_transform_policy=(
            bounded_policy if algorithm == "auto" else None
        ),
    )
    values = _make_values(
        config.line_count,
        size,
        value_type=value_type,
        device=device,
        dtype=dtype,
    )
    retained_count = (
        size
        if execution_mode == "full"
        else max(1, min(size, math.floor(size * config.retained_fraction)))
    )

    def forward() -> torch.Tensor:
        if execution_mode == "full":
            return backend._apply_axis_transform(
                values,
                kind,
                -1,
                inverse=False,
            )
        return backend._apply_retained_real_axis_transform(
            values,
            kind,
            0,
            retained_count,
            inverse=False,
        )

    with torch.inference_mode():
        coefficients = forward()

    def inverse() -> torch.Tensor:
        if execution_mode == "full":
            return backend._apply_axis_transform(
                coefficients,
                kind,
                -1,
                inverse=True,
            )
        return backend._apply_retained_real_axis_transform(
            coefficients,
            kind,
            0,
            retained_count,
            inverse=True,
        )

    with torch.inference_mode():
        restored = inverse()
        recovered_coefficients = (
            backend._apply_axis_transform(
                restored,
                kind,
                -1,
                inverse=False,
            )
            if execution_mode == "full"
            else backend._apply_retained_real_axis_transform(
                restored,
                kind,
                0,
                retained_count,
                inverse=False,
            )
        )

    if algorithm == "dense":
        dense_forward_relative_l2 = 0.0
        dense_inverse_relative_l2 = 0.0
    else:
        dense_backend = TensorProductTransformBackend(
            shape=(size,),
            lengths=(1.0,),
            device=device,
            dtype=dtype,
            bounded_transform_algorithm="dense",
        )
        with torch.inference_mode():
            if execution_mode == "full":
                dense_coefficients = dense_backend._apply_axis_transform(
                    values, kind, -1, inverse=False
                )
                dense_restored = dense_backend._apply_axis_transform(
                    coefficients, kind, -1, inverse=True
                )
            else:
                dense_coefficients = (
                    dense_backend._apply_retained_real_axis_transform(
                        values, kind, 0, retained_count, inverse=False
                    )
                )
                dense_restored = (
                    dense_backend._apply_retained_real_axis_transform(
                        coefficients, kind, 0, retained_count, inverse=True
                    )
                )
        dense_forward_relative_l2 = _relative_l2(
            coefficients,
            dense_coefficients,
        )
        dense_inverse_relative_l2 = _relative_l2(
            restored,
            dense_restored,
        )

    tolerance = 5.0e-6 if dtype == torch.float32 else 5.0e-13
    coefficient_relative_l2 = _relative_l2(
        recovered_coefficients,
        coefficients,
    )
    roundtrip_relative_l2 = (
        _relative_l2(restored, values)
        if execution_mode == "full"
        else None
    )
    if coefficient_relative_l2 > tolerance or (
        roundtrip_relative_l2 is not None
        and roundtrip_relative_l2 > tolerance
    ):
        raise RuntimeError(
            "bounded-axis transform failed its consistency gate: "
            f"kind={kind}, mode={execution_mode}, size={size}, "
            f"value_type={value_type}, "
            f"coefficient_relative_l2={coefficient_relative_l2:.6e}, "
            f"roundtrip_relative_l2={roundtrip_relative_l2}, "
            f"tolerance={tolerance:.6e}"
        )
    if (
        dense_forward_relative_l2 > tolerance
        or dense_inverse_relative_l2 > tolerance
    ):
        raise RuntimeError(
            "bounded-axis candidate failed its dense-reference gate: "
            f"algorithm={algorithm}, kind={kind}, mode={execution_mode}, "
            f"size={size}, value_type={value_type}, "
            f"forward_relative_l2={dense_forward_relative_l2:.6e}, "
            f"inverse_relative_l2={dense_inverse_relative_l2:.6e}, "
            f"tolerance={tolerance:.6e}"
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
    else:
        baseline_allocated = None

    forward_samples = _timing_samples(
        forward,
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )
    inverse_samples = _timing_samples(
        inverse,
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None
    )
    element_size = torch.empty((), dtype=dtype).element_size()
    scalar_products = config.line_count * size * retained_count

    return {
        "case_id": (
            f"{algorithm}_{kind}_{execution_mode}_{value_type}_"
            f"n{size}_r{retained_count}"
        ),
        "algorithm": algorithm,
        "bounded_transform_selection": (
            backend.bounded_transform_selection_metadata()
        ),
        "size": size,
        "kind": kind,
        "execution_mode": execution_mode,
        "value_type": value_type,
        "retained_count": retained_count,
        "retained_fraction_effective": retained_count / size,
        "input_shape": list(values.shape),
        "coefficient_shape": list(coefficients.shape),
        "correctness": {
            "tolerance_relative_l2": tolerance,
            "all_finite": bool(
                torch.isfinite(coefficients).all().item()
                and torch.isfinite(restored).all().item()
            ),
            "coefficient_recovery_relative_l2": coefficient_relative_l2,
            "full_roundtrip_relative_l2": roundtrip_relative_l2,
            "dense_reference_forward_relative_l2": (
                dense_forward_relative_l2
            ),
            "dense_reference_inverse_relative_l2": (
                dense_inverse_relative_l2
            ),
        },
        "dense_work_model": {
            "reference_scalar_products_per_direction": scalar_products,
            "active_asymptotic_cost": (
                "O(line_count * size * log(size))"
                if algorithm == "fft"
                else "O(line_count * size * retained_count)"
            ),
            "cached_matrix_bytes_actual": _matrix_cache_bytes(backend),
            "cached_fft_r2r_bytes_actual": _fft_r2r_cache_bytes(backend),
            "one_full_real_matrix_bytes": size * size * element_size,
            "plane_dct_plus_dst_full_cache_bytes": (
                2 * size * size * element_size
            ),
        },
        "timing": {
            "forward": _sample_summary(forward_samples),
            "inverse": _sample_summary(inverse_samples),
        },
        "cuda_memory": {
            "baseline_allocated_bytes": baseline_allocated,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_delta_bytes": (
                None
                if peak_allocated is None or baseline_allocated is None
                else peak_allocated - baseline_allocated
            ),
        },
    }


def run_benchmark(
    config: BoundedAxisBenchmarkConfig,
) -> dict[str, object]:
    """Run the configured DCT/DST sweep and return JSON-safe results."""
    _validate_config(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    dtype = _torch_dtype(config.dtype)
    bounded_policy = (
        None
        if config.policy_path is None
        else load_bounded_transform_policy(config.policy_path)
    )
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    cases = [
        _run_case(
            config,
            size=size,
            kind=kind,
            execution_mode=execution_mode,
            value_type=value_type,
            algorithm=algorithm,
            bounded_policy=bounded_policy,
            device=device,
            dtype=dtype,
        )
        for size in config.sizes
        for kind in config.kinds
        for execution_mode in config.execution_modes
        for value_type in config.value_types
        for algorithm in config.algorithms
    ]
    candidate_present = any(
        algorithm != "dense" for algorithm in config.algorithms
    )
    policy_present = "auto" in config.algorithms
    return {
        "schema_version": 1,
        "benchmark": (
            "r2r_c_qualified_policy_bounded_axis"
            if policy_present
            else (
                "r2r_b_dense_vs_fft_bounded_axis"
                if candidate_present
                else "r2r_a_dense_bounded_axis_baseline"
            )
        ),
        "scope": {
            "candidate_algorithm_present": candidate_present,
            "qualified_policy_present": policy_present,
            "production_default_changed": False,
            "basis_or_normalization_changed": False,
        },
        "config": asdict(config),
        "environment": {
            "git": _git_provenance(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
        },
        "cases": cases,
    }


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        )
    return values


def _comma_separated_choices(
    value: str,
    *,
    choices: tuple[str, ...],
    option: str,
) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in value.split(","))
    if not values or any(item not in choices for item in values):
        raise argparse.ArgumentTypeError(
            f"{option} must contain comma-separated values from {choices}"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_comma_separated_ints,
        default=DEFAULT_SIZES,
    )
    parser.add_argument(
        "--kinds",
        type=lambda value: _comma_separated_choices(
            value,
            choices=TRANSFORM_KINDS,
            option="--kinds",
        ),
        default=TRANSFORM_KINDS,
    )
    parser.add_argument(
        "--execution-modes",
        type=lambda value: _comma_separated_choices(
            value,
            choices=EXECUTION_MODES,
            option="--execution-modes",
        ),
        default=EXECUTION_MODES,
    )
    parser.add_argument(
        "--value-types",
        type=lambda value: _comma_separated_choices(
            value,
            choices=VALUE_TYPES,
            option="--value-types",
        ),
        default=VALUE_TYPES,
    )
    parser.add_argument(
        "--algorithms",
        type=lambda value: _comma_separated_choices(
            value,
            choices=ALGORITHMS,
            option="--algorithms",
        ),
        default=("dense",),
    )
    parser.add_argument(
        "--geometry",
        default="bounded_axis_microbenchmark",
    )
    parser.add_argument(
        "--policy",
        dest="policy_path",
        help="R2R-C qualification artifact required by auto.",
    )
    parser.add_argument("--retained-fraction", type=float, default=0.5)
    parser.add_argument("--line-count", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        _validate_config(
            BoundedAxisBenchmarkConfig(
                sizes=args.sizes,
                kinds=args.kinds,
                execution_modes=args.execution_modes,
                value_types=args.value_types,
                algorithms=args.algorithms,
                geometry=args.geometry,
                policy_path=args.policy_path,
                retained_fraction=args.retained_fraction,
                line_count=args.line_count,
                device=args.device,
                dtype=args.dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                seed=args.seed,
            )
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    return args


def main() -> None:
    args = parse_args()
    result = run_benchmark(
        BoundedAxisBenchmarkConfig(
            sizes=args.sizes,
            kinds=args.kinds,
            execution_modes=args.execution_modes,
            value_types=args.value_types,
            algorithms=args.algorithms,
            geometry=args.geometry,
            policy_path=args.policy_path,
            retained_fraction=args.retained_fraction,
            line_count=args.line_count,
            device=args.device,
            dtype=args.dtype,
            warmup=args.warmup,
            repeats=args.repeats,
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
