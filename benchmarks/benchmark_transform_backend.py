"""Benchmark tensor-product transforms against the legacy axis-wise path."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from pssolver.transforms import TensorProductTransformBackend


SUPPORTED_BOUNDARY_CONDITIONS = ("periodic", "dirichlet", "neumann")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs that define one reproducible transform benchmark."""

    shape: tuple[int, ...]
    lengths: tuple[float, ...]
    boundary_conditions: tuple[str, ...]
    batch_size: int
    device: str
    dtype: str
    warmup: int
    repeats: int
    seed: int


def _comma_separated_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("shape entries must be positive integers")
    return values


def _comma_separated_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(","))
    if not values or any(not math.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("length entries must be positive finite numbers")
    return values


def _boundary_conditions(value: str) -> tuple[str, ...]:
    values = tuple(item.strip().lower() for item in value.split(","))
    invalid = tuple(item for item in values if item not in SUPPORTED_BOUNDARY_CONDITIONS)
    if invalid:
        raise argparse.ArgumentTypeError(
            "boundary conditions must be periodic, dirichlet, or neumann; "
            f"got {invalid}"
        )
    return values


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _legacy_forward(
    backend: TensorProductTransformBackend,
    tensor: torch.Tensor,
    boundary_conditions: Sequence[str],
) -> torch.Tensor:
    """Apply the pre-optimization forward transform one axis at a time."""
    output = tensor
    metadata = backend.get_metadata(boundary_conditions)
    for local_axis, kind in enumerate(metadata.transform_kinds):
        axis = output.ndim - backend.dim + local_axis
        output = backend._apply_axis_transform(output, kind, axis, inverse=False)
    return output.to(backend.spectral_dtype)


def _legacy_inverse(
    backend: TensorProductTransformBackend,
    spectral: torch.Tensor,
    boundary_conditions: Sequence[str],
) -> torch.Tensor:
    """Apply the pre-optimization inverse transform one axis at a time."""
    output = spectral
    metadata = backend.get_metadata(boundary_conditions)
    for local_axis, kind in reversed(tuple(enumerate(metadata.transform_kinds))):
        axis = output.ndim - backend.dim + local_axis
        output = backend._apply_axis_transform(output, kind, axis, inverse=True)
    return output.real


def _measure_seconds(
    operation: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> float:
    """Return mean synchronized wall time for one operation."""
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        _synchronize(device)

        start = time.perf_counter()
        for _ in range(repeats):
            operation()
        _synchronize(device)
        elapsed = time.perf_counter() - start
    return elapsed / repeats


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm((actual - reference).reshape(-1))
    scale = torch.linalg.vector_norm(reference.reshape(-1))
    tiny = torch.finfo(reference.real.dtype).tiny
    return float((difference / scale.clamp_min(tiny)).item())


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


def run_benchmark(config: BenchmarkConfig) -> dict[str, object]:
    """Validate and time active and legacy transform implementations."""
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    dtype = _torch_dtype(config.dtype)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    backend = TensorProductTransformBackend(
        shape=config.shape,
        lengths=config.lengths,
        device=device,
        dtype=dtype,
    )
    values = torch.randn(
        config.batch_size,
        *config.shape,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        active_spectral = backend.forward(values, config.boundary_conditions)
        legacy_spectral = _legacy_forward(backend, values, config.boundary_conditions)
        active_roundtrip = backend.inverse(active_spectral, config.boundary_conditions)
        legacy_roundtrip = _legacy_inverse(backend, legacy_spectral, config.boundary_conditions)

    forward_max_abs = float((active_spectral - legacy_spectral).abs().max().item())
    forward_relative_l2 = _relative_l2(active_spectral, legacy_spectral)
    roundtrip_max_abs = float((active_roundtrip - legacy_roundtrip).abs().max().item())
    roundtrip_relative_l2 = _relative_l2(active_roundtrip, legacy_roundtrip)

    tolerance = 5.0e-6 if dtype == torch.float32 else 5.0e-13
    if forward_relative_l2 > tolerance or roundtrip_relative_l2 > tolerance:
        raise RuntimeError(
            "active transform does not agree with the legacy reference: "
            f"forward_relative_l2={forward_relative_l2:.6e}, "
            f"roundtrip_relative_l2={roundtrip_relative_l2:.6e}, "
            f"tolerance={tolerance:.6e}"
        )

    def active_pair() -> torch.Tensor:
        spectral = backend.forward(values, config.boundary_conditions)
        return backend.inverse(spectral, config.boundary_conditions)

    def legacy_pair() -> torch.Tensor:
        spectral = _legacy_forward(backend, values, config.boundary_conditions)
        return _legacy_inverse(backend, spectral, config.boundary_conditions)

    legacy_seconds = _measure_seconds(
        legacy_pair,
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )
    active_seconds = _measure_seconds(
        active_pair,
        device=device,
        warmup=config.warmup,
        repeats=config.repeats,
    )

    elements = config.batch_size * math.prod(config.shape)
    result: dict[str, object] = {
        "schema_version": 1,
        "config": asdict(config),
        "environment": {
            "git_head": _git_head(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
            ),
        },
        "correctness": {
            "tolerance_relative_l2": tolerance,
            "forward_max_abs": forward_max_abs,
            "forward_relative_l2": forward_relative_l2,
            "roundtrip_max_abs": roundtrip_max_abs,
            "roundtrip_relative_l2": roundtrip_relative_l2,
        },
        "timing": {
            "legacy_pair_seconds": legacy_seconds,
            "active_pair_seconds": active_seconds,
            "speedup_legacy_over_active": legacy_seconds / active_seconds,
            "active_megagridpoints_per_second": elements / active_seconds / 1.0e6,
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=_comma_separated_ints, default=(64, 64, 32))
    parser.add_argument("--lengths", type=_comma_separated_floats)
    parser.add_argument(
        "--boundary-conditions",
        type=_boundary_conditions,
        default=("periodic", "periodic", "neumann"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.lengths is None:
        args.lengths = tuple(float(item) for item in args.shape)
    if len(args.lengths) != len(args.shape):
        parser.error("--lengths and --shape must have the same number of entries")
    if len(args.boundary_conditions) != len(args.shape):
        parser.error("--boundary-conditions and --shape must have the same number of entries")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.output is not None and args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; pass --overwrite to replace it")
    return args


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig(
        shape=args.shape,
        lengths=args.lengths,
        boundary_conditions=args.boundary_conditions,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    result = run_benchmark(config)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(serialized, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
