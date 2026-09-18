"""Map bounded DCT/DST execution cost without changing production numerics.

The benchmark compares the qualified dense execution plan with a benchmark-
local full-FFT reference.  The latter deliberately computes a complete DCT or
DST before retaining a coefficient prefix; it is not a pruned transform, is
not imported by the solver, and cannot change a production default.  Its only
purpose is to make the size, retained-fraction, line-count, dtype and value-
type crossover visible before a genuine pruned implementation is attempted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, ClassVar, Iterable

import torch
import torch.nn.functional as functional

from pssolver.backends.bounded import (
    BoundedAxisPlanKey,
    DenseBoundedAxisExecutionPlan,
    build_dense_orthonormal_matrix,
)


ALGORITHMS = ("dense", "full_fft_reference")
TRANSFORM_KINDS = ("dct", "dst")
DIRECTIONS = ("forward", "inverse")
VALUE_TYPES = ("real", "complex")
DTYPES = ("float32", "float64")
DEFAULT_SIZES = (32, 40, 64, 80, 128, 160, 256, 320, 512, 1024)
DEFAULT_RETAINED_FRACTIONS = (1.0, 2.0 / 3.0, 0.5, 0.25, 0.125, 0.0625)
CSV_FIELDS = (
    "algorithm_case_id",
    "case_id",
    "algorithm",
    "physical_size",
    "retained_count",
    "retained_fraction_requested",
    "retained_fraction_effective",
    "kind",
    "direction",
    "dtype",
    "value_type",
    "line_count",
    "cold_wall_mean_seconds",
    "cold_wall_median_seconds",
    "cold_wall_sample_std_seconds",
    "cold_device_mean_seconds",
    "steady_mean_seconds",
    "steady_median_seconds",
    "steady_sample_std_seconds",
    "steady_minimum_seconds",
    "steady_maximum_seconds",
    "transforms_per_second",
    "logical_scalar_products_per_second",
    "persistent_cache_unique_bytes_max",
    "peak_allocated_delta_bytes_max",
    "peak_reserved_bytes_max",
    "dense_reference_bitwise_equal",
    "dense_reference_relative_l2",
    "dense_reference_linf",
    "coefficient_identity_relative_l2",
    "full_roundtrip_relative_l2",
    "full_fft_speedup_over_dense_median",
    "paired_trial_speedup_mean",
    "paired_trial_speedup_sample_std",
    "paired_trial_speedup_minimum",
    "paired_trial_speedup_maximum",
    "full_fft_faster_in_all_trials",
    "full_fft_at_least_15_percent_faster_in_all_trials",
)


@dataclass(frozen=True, slots=True)
class ApplicabilityConfig:
    """Inputs defining one reproducible bounded-axis applicability sweep."""

    sizes: tuple[int, ...] = DEFAULT_SIZES
    retained_fractions: tuple[float, ...] = DEFAULT_RETAINED_FRACTIONS
    line_counts: tuple[int, ...] = (256,)
    kinds: tuple[str, ...] = TRANSFORM_KINDS
    directions: tuple[str, ...] = DIRECTIONS
    dtypes: tuple[str, ...] = ("float64",)
    value_types: tuple[str, ...] = VALUE_TYPES
    device: str = "cpu"
    warmup: int = 3
    repeats: int = 10
    trials: int = 3
    seed: int = 20260917
    tf32: str = "off"
    require_device_name: str | None = None
    require_clean_git: bool = False


@dataclass(slots=True)
class FullFftReferencePlan:
    """Benchmark-only full FFT DCT-II/DST-II reference.

    Forward execution always forms all ``N`` coefficients before slicing to
    ``R``.  Inverse execution pads the retained prefix to ``N`` before a full
    inverse transform.  This explicitly is not a pruned algorithm.
    """

    key: BoundedAxisPlanKey
    data: dict[str, torch.Tensor]
    algorithm: ClassVar[str] = "full_fft_reference"

    @classmethod
    def build(cls, key: BoundedAxisPlanKey) -> "FullFftReferencePlan":
        size = key.physical_size
        modes = torch.arange(
            size,
            device=key.device,
            dtype=key.real_dtype,
        )
        scale = torch.full(
            (size,),
            math.sqrt(2.0 / size),
            device=key.device,
            dtype=key.real_dtype,
        )
        scale[0] = math.sqrt(1.0 / size)
        phase_inverse = torch.polar(
            torch.ones_like(modes),
            math.pi * modes / (2.0 * size),
        )
        modulation = torch.where(
            modes.remainder(2) == 0,
            torch.ones_like(modes),
            -torch.ones_like(modes),
        )
        indices = torch.arange(size, device=key.device)
        return cls(
            key=key,
            data={
                "scale": scale,
                "phase_forward": phase_inverse.conj(),
                "phase_inverse": phase_inverse,
                "modulation": modulation,
                "forward_permutation": torch.cat(
                    (indices[::2], indices[1::2].flip(0))
                ),
                "inverse_permutation": torch.where(
                    indices.remainder(2) == 0,
                    indices // 2,
                    size - 1 - indices // 2,
                ),
            },
        )

    def _dct(self, tensor: torch.Tensor, *, inverse: bool) -> torch.Tensor:
        size = self.key.physical_size
        data = self.data
        if not inverse:
            if tensor.is_complex():
                mirrored = torch.cat((tensor, tensor.flip(-1)), dim=-1)
                spectrum = torch.fft.fft(mirrored, dim=-1)[..., :size]
                return (
                    0.5
                    * spectrum
                    * data["phase_forward"]
                    * data["scale"]
                )

            reordered = tensor.index_select(
                -1,
                data["forward_permutation"],
            )
            half_spectrum = torch.fft.rfft(reordered, dim=-1)
            half_size = half_spectrum.shape[-1]
            phase = data["phase_inverse"]
            transformed = (
                half_spectrum.real * phase.real[:half_size]
                + half_spectrum.imag * phase.imag[:half_size]
            )
            tail_size = size - half_size
            if tail_size:
                reflected = (
                    half_spectrum[..., 1 : 1 + tail_size]
                    .flip(-1)
                    .conj()
                )
                transformed = torch.cat(
                    (
                        transformed,
                        reflected.real * phase.real[half_size:]
                        + reflected.imag * phase.imag[half_size:],
                    ),
                    dim=-1,
                )
            return transformed * data["scale"]

        coefficients = functional.pad(
            tensor,
            (0, size - tensor.shape[-1]),
        )
        unscaled = coefficients / data["scale"]
        if tensor.is_complex():
            positive = 2.0 * unscaled * data["phase_inverse"]
            nyquist = torch.zeros_like(positive[..., :1])
            negative = (
                2.0
                * unscaled[..., 1:]
                * data["phase_forward"][1:]
            ).flip(-1)
            spectrum = torch.cat((positive, nyquist, negative), dim=-1)
            return torch.fft.ifft(spectrum, dim=-1)[..., :size]

        half_size = size // 2 + 1
        imaginary = torch.cat(
            (
                torch.zeros_like(unscaled[..., :1]),
                -unscaled.flip(-1)[..., : half_size - 1],
            ),
            dim=-1,
        )
        phase = data["phase_inverse"][:half_size]
        half_spectrum = torch.complex(
            unscaled[..., :half_size] * phase.real
            - imaginary * phase.imag,
            unscaled[..., :half_size] * phase.imag
            + imaginary * phase.real,
        )
        reordered = torch.fft.irfft(
            half_spectrum,
            n=size,
            dim=-1,
        )
        return reordered.index_select(-1, data["inverse_permutation"])

    def forward_last_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.key.kind == "dct":
            transformed = self._dct(tensor, inverse=False)
        else:
            transformed = self._dct(
                tensor * self.data["modulation"],
                inverse=False,
            ).flip(-1)
        return transformed[..., : self.key.retained_count]

    def inverse_last_axis(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.key.kind == "dct":
            return self._dct(tensor, inverse=True)
        size = self.key.physical_size
        padded = functional.pad(
            tensor.flip(-1),
            (size - tensor.shape[-1], 0),
        )
        return self._dct(padded, inverse=True) * self.data["modulation"]


def _validate_unique(name: str, values: tuple[object, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")


def _validate_config(config: ApplicabilityConfig) -> None:
    if not config.sizes or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in config.sizes
    ):
        raise ValueError("sizes must contain positive integers")
    if not config.retained_fractions or any(
        not math.isfinite(value) or not 0.0 < value <= 1.0
        for value in config.retained_fractions
    ):
        raise ValueError("retained_fractions must be finite and in (0, 1]")
    if not config.line_counts or any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in config.line_counts
    ):
        raise ValueError("line_counts must contain positive integers")
    for name, values, allowed in (
        ("kinds", config.kinds, TRANSFORM_KINDS),
        ("directions", config.directions, DIRECTIONS),
        ("dtypes", config.dtypes, DTYPES),
        ("value_types", config.value_types, VALUE_TYPES),
    ):
        if not values or any(value not in allowed for value in values):
            raise ValueError(f"{name} must contain values from {allowed}")
        _validate_unique(name, values)
    _validate_unique("sizes", config.sizes)
    _validate_unique("retained_fractions", config.retained_fractions)
    _validate_unique("line_counts", config.line_counts)
    for size in config.sizes:
        counts = tuple(
            _retained_count(size, fraction)
            for fraction in config.retained_fractions
        )
        if len(counts) != len(set(counts)):
            raise ValueError(
                "retained_fractions map to duplicate retained counts for "
                f"physical size {size}: {counts}"
            )
    if config.device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if config.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if config.repeats <= 0:
        raise ValueError("repeats must be positive")
    if config.trials <= 0:
        raise ValueError("trials must be positive")
    if config.tf32 not in {"off", "on"}:
        raise ValueError("tf32 must be 'off' or 'on'")
    if config.require_device_name is not None and not (
        config.require_device_name.strip()
    ):
        raise ValueError("require_device_name must be non-empty when provided")


def _retained_count(size: int, fraction: float) -> int:
    return max(1, min(size, math.floor(size * fraction)))


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _case_seed(config: ApplicabilityConfig, descriptor: str) -> int:
    digest = hashlib.sha256(
        f"{config.seed}:{descriptor}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def _values(
    line_count: int,
    width: int,
    *,
    dtype: torch.dtype,
    value_type: str,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(
        line_count,
        width,
        generator=generator,
        dtype=dtype,
    )
    if value_type == "complex":
        imaginary = torch.randn(
            line_count,
            width,
            generator=generator,
            dtype=dtype,
        )
        real = torch.complex(real, imaginary).to(_complex_dtype(dtype))
    return real.to(device)


def _dense_plan(key: BoundedAxisPlanKey) -> DenseBoundedAxisExecutionPlan:
    matrix = build_dense_orthonormal_matrix(
        key.kind,
        key.physical_size,
        device=key.device,
        dtype=key.real_dtype,
    )[: key.retained_count]
    return DenseBoundedAxisExecutionPlan(key=key, matrix=matrix)


def _plan_factory(
    algorithm: str,
) -> Callable[[BoundedAxisPlanKey], object]:
    if algorithm == "dense":
        return _dense_plan
    if algorithm == "full_fft_reference":
        return FullFftReferencePlan.build
    raise ValueError(f"unsupported benchmark algorithm {algorithm!r}")


def _execute(plan, tensor: torch.Tensor, direction: str) -> torch.Tensor:
    if direction == "forward":
        return plan.forward_last_axis(tensor)
    return plan.inverse_last_axis(tensor)


def _timing_samples(
    operation: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    repeats: int,
) -> list[float]:
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
        return [start.elapsed_time(stop) / 1000.0 for start, stop in events]

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return samples


def _summary(samples: Iterable[float]) -> dict[str, float | int]:
    values = list(samples)
    if not values:
        raise ValueError("timing samples must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise RuntimeError("timing samples must be positive and finite")
    return {
        "samples": len(values),
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "sample_std_seconds": (
            statistics.stdev(values) if len(values) > 1 else 0.0
        ),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def _scalar_summary(values: Iterable[float]) -> dict[str, float | int]:
    samples = list(values)
    if not samples:
        raise ValueError("scalar samples must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in samples):
        raise RuntimeError("scalar samples must be positive and finite")
    return {
        "samples": len(samples),
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "sample_std": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "minimum": min(samples),
        "maximum": max(samples),
    }


def _relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm((actual - reference).reshape(-1))
    scale = torch.linalg.vector_norm(reference.reshape(-1))
    tiny = torch.finfo(reference.real.dtype).tiny
    return float((difference / scale.clamp_min(tiny)).item())


def _linf(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return float((actual - reference).abs().max().item())


def _correctness_tolerance(dtype: torch.dtype, size: int) -> tuple[float, str]:
    if dtype == torch.float32:
        tolerance = max(
            5.0e-5,
            24.0 * torch.finfo(dtype).eps * math.sqrt(size),
        )
        return tolerance, "max(5e-5, 24*eps*sqrt(N))"
    return 8.0e-12, "8e-12"


def _require_finite_nonnegative(name: str, value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value < 0.0:
        raise RuntimeError(f"{name} must be finite and non-negative")


def _tensor_sha256(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _unique_storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    storages: dict[tuple[str, int], int] = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        storages[key] = storage.nbytes()
    return sum(storages.values())


def _plan_cache_bytes(plan) -> int:
    if isinstance(plan, DenseBoundedAxisExecutionPlan):
        return _unique_storage_bytes((plan.matrix,))
    if isinstance(plan, FullFftReferencePlan):
        return _unique_storage_bytes(plan.data.values())
    raise TypeError(f"unknown benchmark plan {type(plan)!r}")


def _measure_trial(
    key: BoundedAxisPlanKey,
    values: torch.Tensor,
    *,
    algorithm: str,
    direction: str,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, object]:
    factory = _plan_factory(algorithm)
    if device.type == "cuda":
        _synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        cold_baseline_allocated = torch.cuda.memory_allocated(device)
        cold_start_event = torch.cuda.Event(enable_timing=True)
        cold_stop_event = torch.cuda.Event(enable_timing=True)
        cold_start_event.record()
    else:
        cold_baseline_allocated = None
        cold_start_event = None
        cold_stop_event = None

    _synchronize(device)
    cold_wall_start = time.perf_counter()
    cold_plan = factory(key)
    cold_output = _execute(cold_plan, values, direction)
    if cold_stop_event is not None:
        cold_stop_event.record()
    _synchronize(device)
    cold_wall_seconds = time.perf_counter() - cold_wall_start
    cold_device_seconds = (
        cold_start_event.elapsed_time(cold_stop_event) / 1000.0
        if cold_start_event is not None and cold_stop_event is not None
        else None
    )
    cold_peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    del cold_output, cold_plan

    plan = factory(key)

    def operation() -> torch.Tensor:
        return _execute(plan, values, direction)

    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        _synchronize(device)
        if device.type == "cuda":
            steady_baseline_allocated = torch.cuda.memory_allocated(device)
            steady_baseline_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
        else:
            steady_baseline_allocated = None
            steady_baseline_reserved = None
        steady_samples = _timing_samples(
            operation,
            device=device,
            repeats=repeats,
        )
        _synchronize(device)

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
    return {
        "algorithm": algorithm,
        "cold": {
            "wall_seconds": cold_wall_seconds,
            "device_seconds": cold_device_seconds,
            "baseline_allocated_bytes": cold_baseline_allocated,
            "peak_allocated_bytes": cold_peak_allocated,
            "peak_allocated_delta_bytes": (
                None
                if cold_peak_allocated is None
                or cold_baseline_allocated is None
                else cold_peak_allocated - cold_baseline_allocated
            ),
        },
        "steady_samples_seconds": steady_samples,
        "memory": {
            "baseline_allocated_bytes": steady_baseline_allocated,
            "baseline_reserved_bytes": steady_baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_delta_bytes": (
                None
                if peak_allocated is None
                or steady_baseline_allocated is None
                else peak_allocated - steady_baseline_allocated
            ),
            "persistent_cache_unique_bytes": _plan_cache_bytes(plan),
        },
    }


def _correctness(
    key: BoundedAxisPlanKey,
    values: torch.Tensor,
    *,
    direction: str,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, dict[str, object]]:
    plans = {name: _plan_factory(name)(key) for name in ALGORITHMS}
    with torch.inference_mode():
        outputs = {
            name: _execute(plan, values, direction)
            for name, plan in plans.items()
        }
    dense = outputs["dense"]
    tolerance, tolerance_model = _correctness_tolerance(
        dtype,
        key.physical_size,
    )
    coefficient_values = _values(
        values.shape[0],
        key.retained_count,
        dtype=dtype,
        value_type=key.value_type,
        device=key.device,
        seed=seed + 1,
    )
    physical_values = _values(
        values.shape[0],
        key.physical_size,
        dtype=dtype,
        value_type=key.value_type,
        device=key.device,
        seed=seed + 2,
    )
    result = {}
    with torch.inference_mode():
        for name, plan in plans.items():
            output = outputs[name]
            restored_coefficients = plan.forward_last_axis(
                plan.inverse_last_axis(coefficient_values)
            )
            full_roundtrip = (
                plan.inverse_last_axis(plan.forward_last_axis(physical_values))
                if key.retained_count == key.physical_size
                else None
            )
            record = {
                "tolerance_relative_l2": tolerance,
                "tolerance_model": tolerance_model,
                "all_finite": bool(
                    torch.isfinite(output).all().item()
                    and torch.isfinite(restored_coefficients).all().item()
                    and (
                        full_roundtrip is None
                        or torch.isfinite(full_roundtrip).all().item()
                    )
                ),
                "dense_reference_bitwise_equal": bool(
                    torch.equal(output, dense)
                ),
                "dense_reference_relative_l2": _relative_l2(output, dense),
                "dense_reference_linf": _linf(output, dense),
                "coefficient_identity_relative_l2": _relative_l2(
                    restored_coefficients,
                    coefficient_values,
                ),
                "full_roundtrip_relative_l2": (
                    None
                    if full_roundtrip is None
                    else _relative_l2(full_roundtrip, physical_values)
                ),
                "output_sha256": _tensor_sha256(output),
            }
            if not record["all_finite"]:
                raise RuntimeError(
                    f"{name} produced NaN or Inf for {key}"
                )
            for metric in (
                "dense_reference_relative_l2",
                "dense_reference_linf",
                "coefficient_identity_relative_l2",
                "full_roundtrip_relative_l2",
            ):
                _require_finite_nonnegative(metric, record[metric])
            if record["dense_reference_relative_l2"] > tolerance:
                raise RuntimeError(
                    f"{name} failed the dense-reference gate for {key}: "
                    f"{record['dense_reference_relative_l2']:.6e} > "
                    f"{tolerance:.6e}"
                )
            if record["coefficient_identity_relative_l2"] > tolerance:
                raise RuntimeError(
                    f"{name} failed coefficient identity for {key}: "
                    f"{record['coefficient_identity_relative_l2']:.6e} > "
                    f"{tolerance:.6e}"
                )
            if (
                record["full_roundtrip_relative_l2"] is not None
                and record["full_roundtrip_relative_l2"] > tolerance
            ):
                raise RuntimeError(
                    f"{name} failed full roundtrip for {key}: "
                    f"{record['full_roundtrip_relative_l2']:.6e} > "
                    f"{tolerance:.6e}"
                )
            result[name] = record
    if not result["dense"]["dense_reference_bitwise_equal"]:
        raise RuntimeError("dense execution is not bitwise self-consistent")
    return result


def _aggregate_algorithm(
    records: list[dict[str, object]],
) -> dict[str, object]:
    steady = [
        sample
        for record in records
        for sample in record["steady_samples_seconds"]
    ]
    cold_wall = [record["cold"]["wall_seconds"] for record in records]
    cold_device = [
        record["cold"]["device_seconds"]
        for record in records
        if record["cold"]["device_seconds"] is not None
    ]
    memory_records = [record["memory"] for record in records]
    for record in records:
        for name in (
            "wall_seconds",
            "device_seconds",
            "baseline_allocated_bytes",
            "peak_allocated_bytes",
            "peak_allocated_delta_bytes",
        ):
            value = record["cold"][name]
            if name.endswith("seconds"):
                if value is not None and (
                    not math.isfinite(value) or value <= 0.0
                ):
                    raise RuntimeError(
                        f"cold {name} must be positive and finite"
                    )
            else:
                _require_finite_nonnegative(f"cold {name}", value)
    for record in memory_records:
        for name in (
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "peak_allocated_delta_bytes",
            "persistent_cache_unique_bytes",
        ):
            _require_finite_nonnegative(name, record[name])
    return {
        "trials": len(records),
        "cold_wall": _summary(cold_wall),
        "cold_device": _summary(cold_device) if cold_device else None,
        "steady": _summary(steady),
        "memory": {
            "persistent_cache_unique_bytes_max": max(
                record["persistent_cache_unique_bytes"]
                for record in memory_records
            ),
            "peak_allocated_delta_bytes_max": (
                max(
                    record["peak_allocated_delta_bytes"]
                    for record in memory_records
                )
                if memory_records[0]["peak_allocated_delta_bytes"] is not None
                else None
            ),
            "peak_reserved_bytes_max": (
                max(record["peak_reserved_bytes"] for record in memory_records)
                if memory_records[0]["peak_reserved_bytes"] is not None
                else None
            ),
        },
    }


def _run_case(
    config: ApplicabilityConfig,
    *,
    size: int,
    retained_count: int,
    retained_fraction: float,
    line_count: int,
    kind: str,
    direction: str,
    dtype_name: str,
    value_type: str,
    device: torch.device,
) -> dict[str, object]:
    dtype = _torch_dtype(dtype_name)
    descriptor = (
        f"n{size}:r{retained_count}:l{line_count}:{kind}:{direction}:"
        f"{dtype_name}:{value_type}"
    )
    seed = _case_seed(config, descriptor)
    width = size if direction == "forward" else retained_count
    values = _values(
        line_count,
        width,
        dtype=dtype,
        value_type=value_type,
        device=device,
        seed=seed,
    )
    key = BoundedAxisPlanKey(
        kind=kind,
        physical_size=size,
        retained_count=retained_count,
        device=values.device,
        real_dtype=dtype,
        value_type=value_type,
    )
    correctness = _correctness(
        key,
        values,
        direction=direction,
        dtype=dtype,
        seed=seed,
    )
    records = {name: [] for name in ALGORITHMS}
    for trial in range(config.trials):
        order = ALGORITHMS[trial % len(ALGORITHMS) :] + ALGORITHMS[
            : trial % len(ALGORITHMS)
        ]
        for order_index, algorithm in enumerate(order, start=1):
            record = _measure_trial(
                key,
                values,
                algorithm=algorithm,
                direction=direction,
                warmup=config.warmup,
                repeats=config.repeats,
                device=device,
            )
            record.update(
                {
                    "trial": trial + 1,
                    "order": order_index,
                }
            )
            records[algorithm].append(record)

    aggregate = {
        name: _aggregate_algorithm(algorithm_records)
        for name, algorithm_records in records.items()
    }
    paired_speedups = []
    for trial in range(config.trials):
        dense_median = statistics.median(
            records["dense"][trial]["steady_samples_seconds"]
        )
        fft_median = statistics.median(
            records["full_fft_reference"][trial]["steady_samples_seconds"]
        )
        speedup = dense_median / fft_median
        if not math.isfinite(speedup) or speedup <= 0.0:
            raise RuntimeError("paired trial speedup must be positive and finite")
        paired_speedups.append(speedup)
    speedup_summary = _scalar_summary(paired_speedups)
    speedup = speedup_summary["median"]
    logical_scalar_products = line_count * size * retained_count
    case_id = descriptor.replace(":", "_")
    return {
        "case_id": case_id,
        "physical_size": size,
        "retained_count": retained_count,
        "retained_fraction_requested": retained_fraction,
        "retained_fraction_effective": retained_count / size,
        "line_count": line_count,
        "kind": kind,
        "direction": direction,
        "dtype": dtype_name,
        "value_type": value_type,
        "input_shape": list(values.shape),
        "output_shape": [
            line_count,
            retained_count if direction == "forward" else size,
        ],
        "dense_work_model": {
            "logical_scalar_products_per_call": logical_scalar_products,
            "asymptotic_cost": "O(line_count * N * R)",
        },
        "full_fft_reference_semantics": {
            "forward": "complete transform then retain prefix",
            "inverse": "zero-pad retained prefix then complete inverse",
            "is_pruned": False,
        },
        "correctness": correctness,
        "records": records,
        "aggregate": aggregate,
        "comparison": {
            "full_fft_speedup_over_dense_median": speedup,
            "paired_trial_speedups": paired_speedups,
            "paired_trial_speedup_summary": speedup_summary,
            "full_fft_faster": speedup > 1.0,
            "full_fft_at_least_15_percent_faster": speedup >= 1.15,
            "full_fft_faster_in_all_trials": all(
                value > 1.0 for value in paired_speedups
            ),
            "full_fft_at_least_15_percent_faster_in_all_trials": all(
                value >= 1.15 for value in paired_speedups
            ),
        },
    }


def run_benchmark(config: ApplicabilityConfig) -> dict[str, object]:
    """Run a dense/full-FFT research sweep and return JSON-safe evidence."""
    _validate_config(config)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = "CPU"
    if (
        config.require_device_name is not None
        and device_name != config.require_device_name
    ):
        raise RuntimeError(
            f"device name mismatch: expected {config.require_device_name!r}, "
            f"got {device_name!r}"
        )
    git = _git_provenance()
    if config.require_clean_git and git["dirty"] is not False:
        raise RuntimeError("a clean Git worktree is required")

    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_matmul_precision = torch.get_float32_matmul_precision()
    requested_tf32 = config.tf32 == "on"
    torch.backends.cuda.matmul.allow_tf32 = requested_tf32
    torch.set_float32_matmul_precision("high" if requested_tf32 else "highest")
    try:
        if device.type == "cuda":
            with torch.inference_mode():
                preflight = torch.ones(1, device=device)
                (preflight + 1.0).sum().item()
            _synchronize(device)

        precision = {
            "tf32_requested": config.tf32,
            "tf32_effective": bool(
                device.type == "cuda"
                and torch.backends.cuda.matmul.allow_tf32
            ),
            "cuda_matmul_allow_tf32": (
                torch.backends.cuda.matmul.allow_tf32
                if device.type == "cuda"
                else None
            ),
            "float32_matmul_precision": (
                torch.get_float32_matmul_precision()
            ),
        }
        cases = []
        for size in config.sizes:
            for fraction in config.retained_fractions:
                retained_count = _retained_count(size, fraction)
                for line_count in config.line_counts:
                    for kind in config.kinds:
                        for direction in config.directions:
                            for dtype_name in config.dtypes:
                                for value_type in config.value_types:
                                    cases.append(
                                        _run_case(
                                            config,
                                            size=size,
                                            retained_count=retained_count,
                                            retained_fraction=fraction,
                                            line_count=line_count,
                                            kind=kind,
                                            direction=direction,
                                            dtype_name=dtype_name,
                                            value_type=value_type,
                                            device=device,
                                        )
                                    )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_matmul_precision)

    speedups = [
        case["comparison"]["full_fft_speedup_over_dense_median"]
        for case in cases
    ]
    maximum_error = max(
        record["dense_reference_relative_l2"]
        for case in cases
        for record in case["correctness"].values()
    )
    low_fraction = [
        case
        for case in cases
        if case["retained_fraction_effective"] <= 0.125
    ]
    properties = (
        torch.cuda.get_device_properties(device)
        if device.type == "cuda"
        else None
    )
    return {
        "schema_version": 1,
        "benchmark": "bounded_axis_dense_full_fft_applicability_map",
        "scope": {
            "microbenchmark_only": True,
            "production_solver_imports_full_fft_reference": False,
            "production_default_changed": False,
            "production_cli_changed": False,
            "basis_or_normalization_changed": False,
            "pruned_algorithm_present": False,
            "full_fft_reference_previously_rejected_for_production": True,
        },
        "config": asdict(config),
        "environment": {
            "git": git,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": device_name,
            "device_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
            "device_total_memory_bytes": (
                properties.total_memory if properties is not None else None
            ),
            "precision": precision,
        },
        "summary": {
            "logical_case_count": len(cases),
            "algorithm_case_count": len(cases) * len(ALGORITHMS),
            "maximum_dense_reference_relative_l2": maximum_error,
            "full_fft_faster_case_count": sum(value > 1.0 for value in speedups),
            "full_fft_at_least_15_percent_faster_case_count": sum(
                value >= 1.15 for value in speedups
            ),
            "low_retained_fraction_case_count": len(low_fraction),
            "low_retained_fraction_full_fft_at_least_15_percent_faster_count": (
                sum(
                    case["comparison"][
                        "full_fft_at_least_15_percent_faster"
                    ]
                    for case in low_fraction
                )
            ),
            "full_fft_faster_in_all_trials_case_count": sum(
                case["comparison"]["full_fft_faster_in_all_trials"]
                for case in cases
            ),
            "full_fft_at_least_15_percent_faster_in_all_trials_case_count": (
                sum(
                    case["comparison"][
                        "full_fft_at_least_15_percent_faster_in_all_trials"
                    ]
                    for case in cases
                )
            ),
        },
        "timing_semantics": {
            "cold_label": "fresh_python_plan_after_runtime_preflight",
            "cold_runtime_state": (
                "runtime libraries may already cache kernels or FFT plans; "
                "only the Python execution plan and its explicit tensors "
                "are guaranteed fresh"
            ),
            "steady_samples": "per-call synchronized device or wall time",
            "paired_speedup": (
                "per-trial dense median divided by full-FFT median"
            ),
        },
        "cases": cases,
    }


def csv_rows(result: dict[str, object]) -> list[dict[str, object]]:
    """Flatten aggregate results into stable, one-algorithm-per-row CSV."""
    rows = []
    for case in result["cases"]:
        logical_products = case["dense_work_model"][
            "logical_scalar_products_per_call"
        ]
        comparison = case["comparison"]
        for algorithm in ALGORITHMS:
            aggregate = case["aggregate"][algorithm]
            steady = aggregate["steady"]
            cold_wall = aggregate["cold_wall"]
            cold_device = aggregate["cold_device"]
            memory = aggregate["memory"]
            correctness = case["correctness"][algorithm]
            rows.append(
                {
                    "algorithm_case_id": f"{case['case_id']}_{algorithm}",
                    "case_id": case["case_id"],
                    "algorithm": algorithm,
                    "physical_size": case["physical_size"],
                    "retained_count": case["retained_count"],
                    "retained_fraction_requested": case[
                        "retained_fraction_requested"
                    ],
                    "retained_fraction_effective": case[
                        "retained_fraction_effective"
                    ],
                    "kind": case["kind"],
                    "direction": case["direction"],
                    "dtype": case["dtype"],
                    "value_type": case["value_type"],
                    "line_count": case["line_count"],
                    "cold_wall_mean_seconds": cold_wall["mean_seconds"],
                    "cold_wall_median_seconds": cold_wall["median_seconds"],
                    "cold_wall_sample_std_seconds": cold_wall[
                        "sample_std_seconds"
                    ],
                    "cold_device_mean_seconds": (
                        None if cold_device is None else cold_device["mean_seconds"]
                    ),
                    "steady_mean_seconds": steady["mean_seconds"],
                    "steady_median_seconds": steady["median_seconds"],
                    "steady_sample_std_seconds": steady["sample_std_seconds"],
                    "steady_minimum_seconds": steady["minimum_seconds"],
                    "steady_maximum_seconds": steady["maximum_seconds"],
                    "transforms_per_second": 1.0 / steady["mean_seconds"],
                    "logical_scalar_products_per_second": (
                        logical_products / steady["mean_seconds"]
                    ),
                    "persistent_cache_unique_bytes_max": memory[
                        "persistent_cache_unique_bytes_max"
                    ],
                    "peak_allocated_delta_bytes_max": memory[
                        "peak_allocated_delta_bytes_max"
                    ],
                    "peak_reserved_bytes_max": memory[
                        "peak_reserved_bytes_max"
                    ],
                    "dense_reference_bitwise_equal": correctness[
                        "dense_reference_bitwise_equal"
                    ],
                    "dense_reference_relative_l2": correctness[
                        "dense_reference_relative_l2"
                    ],
                    "dense_reference_linf": correctness[
                        "dense_reference_linf"
                    ],
                    "coefficient_identity_relative_l2": correctness[
                        "coefficient_identity_relative_l2"
                    ],
                    "full_roundtrip_relative_l2": correctness[
                        "full_roundtrip_relative_l2"
                    ],
                    "full_fft_speedup_over_dense_median": comparison[
                        "full_fft_speedup_over_dense_median"
                    ],
                    "paired_trial_speedup_mean": comparison[
                        "paired_trial_speedup_summary"
                    ]["mean"],
                    "paired_trial_speedup_sample_std": comparison[
                        "paired_trial_speedup_summary"
                    ]["sample_std"],
                    "paired_trial_speedup_minimum": comparison[
                        "paired_trial_speedup_summary"
                    ]["minimum"],
                    "paired_trial_speedup_maximum": comparison[
                        "paired_trial_speedup_summary"
                    ]["maximum"],
                    "full_fft_faster_in_all_trials": comparison[
                        "full_fft_faster_in_all_trials"
                    ],
                    "full_fft_at_least_15_percent_faster_in_all_trials": (
                        comparison[
                            "full_fft_at_least_15_percent_faster_in_all_trials"
                        ]
                    ),
                }
            )
    return rows


def write_artifacts(
    result: dict[str, object],
    *,
    json_output: Path,
    csv_output: Path,
    overwrite: bool,
) -> None:
    """Write JSON and aggregate CSV after all scientific gates have passed."""
    if json_output.resolve() == csv_output.resolve():
        raise ValueError("JSON and CSV outputs must be different files")
    existing = [path for path in (json_output, csv_output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing output: "
            + ", ".join(str(path) for path in existing)
        )
    json_output.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    rows = csv_rows(result)
    temporary_csv = csv_output.with_name(
        f".{csv_output.name}.tmp.{os.getpid()}"
    )
    temporary_json = json_output.with_name(
        f".{json_output.name}.tmp.{os.getpid()}"
    )
    backup_json = json_output.with_name(
        f".{json_output.name}.backup.{os.getpid()}"
    )
    backup_csv = csv_output.with_name(
        f".{csv_output.name}.backup.{os.getpid()}"
    )
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        csv_sha256 = hashlib.sha256(temporary_csv.read_bytes()).hexdigest()
        json_payload = {
            **result,
            "artifacts": {
                "authoritative_completion_artifact": "json",
                "csv": {
                    "path": str(csv_output.resolve()),
                    "sha256": csv_sha256,
                    "row_count": len(rows),
                },
            },
        }
        json_text = json.dumps(
            json_payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
        temporary_json.write_text(json_text, encoding="utf-8")

        for target, backup in (
            (csv_output, backup_csv),
            (json_output, backup_json),
        ):
            if target.exists():
                os.replace(target, backup)
                backups.append((target, backup))

        os.replace(temporary_csv, csv_output)
        published.append(csv_output)
        # JSON is published last and binds the CSV hash and row count.  Its
        # presence therefore acts as the two-artifact completion marker.
        os.replace(temporary_json, json_output)
        published.append(json_output)
    except Exception as publish_error:
        recovery_errors: list[tuple[Path, Path, OSError]] = []
        for path in reversed(published):
            if path.exists():
                try:
                    path.unlink()
                except OSError as recovery_error:
                    recovery_errors.append((path, path, recovery_error))
        for target, backup in reversed(backups):
            if backup.exists():
                try:
                    os.replace(backup, target)
                except OSError as recovery_error:
                    recovery_errors.append(
                        (target, backup, recovery_error)
                    )
        if recovery_errors:
            details = "; ".join(
                f"target={target.resolve()}, preserved_backup="
                f"{backup.resolve()}: {error}"
                for target, backup, error in recovery_errors
            )
            raise RuntimeError(
                "artifact publication failed and rollback was incomplete; "
                f"preserved recovery paths: {details}"
            ) from publish_error
        raise
    else:
        for _, backup in backups:
            if backup.exists():
                backup.unlink()
    finally:
        # Never remove backups here: if rollback itself fails they are the
        # only remaining copy of the caller's pre-existing artifacts.
        for path in (temporary_csv, temporary_json):
            if path.exists():
                path.unlink()


def _comma_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _comma_floats(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("expected at least one number")
    return values


def _comma_choices(
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=_comma_ints, default=DEFAULT_SIZES)
    parser.add_argument(
        "--retained-fractions",
        type=_comma_floats,
        default=DEFAULT_RETAINED_FRACTIONS,
    )
    parser.add_argument("--line-counts", type=_comma_ints, default=(256,))
    parser.add_argument(
        "--kinds",
        type=lambda value: _comma_choices(
            value,
            choices=TRANSFORM_KINDS,
            option="--kinds",
        ),
        default=TRANSFORM_KINDS,
    )
    parser.add_argument(
        "--directions",
        type=lambda value: _comma_choices(
            value,
            choices=DIRECTIONS,
            option="--directions",
        ),
        default=DIRECTIONS,
    )
    parser.add_argument(
        "--dtypes",
        type=lambda value: _comma_choices(
            value,
            choices=DTYPES,
            option="--dtypes",
        ),
        default=("float64",),
    )
    parser.add_argument(
        "--value-types",
        type=lambda value: _comma_choices(
            value,
            choices=VALUE_TYPES,
            option="--value-types",
        ),
        default=VALUE_TYPES,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--tf32", choices=("off", "on"), default="off")
    parser.add_argument("--require-device-name")
    parser.add_argument("--require-clean-git", action="store_true")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        _validate_config(
            ApplicabilityConfig(
                sizes=args.sizes,
                retained_fractions=args.retained_fractions,
                line_counts=args.line_counts,
                kinds=args.kinds,
                directions=args.directions,
                dtypes=args.dtypes,
                value_types=args.value_types,
                device=args.device,
                warmup=args.warmup,
                repeats=args.repeats,
                trials=args.trials,
                seed=args.seed,
                tf32=args.tf32,
                require_device_name=args.require_device_name,
                require_clean_git=args.require_clean_git,
            )
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.json_output.resolve() == args.csv_output.resolve():
        parser.error("--json-output and --csv-output must differ")
    existing = [
        path
        for path in (args.json_output, args.csv_output)
        if path.exists()
    ]
    if existing and not args.overwrite:
        parser.error(
            "output already exists; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ApplicabilityConfig(
        sizes=args.sizes,
        retained_fractions=args.retained_fractions,
        line_counts=args.line_counts,
        kinds=args.kinds,
        directions=args.directions,
        dtypes=args.dtypes,
        value_types=args.value_types,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        trials=args.trials,
        seed=args.seed,
        tf32=args.tf32,
        require_device_name=args.require_device_name,
        require_clean_git=args.require_clean_git,
    )
    result = run_benchmark(config)
    write_artifacts(
        result,
        json_output=args.json_output,
        csv_output=args.csv_output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "benchmark": result["benchmark"],
                "summary": result["summary"],
                "json_output": str(args.json_output.resolve()),
                "csv_output": str(args.csv_output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
