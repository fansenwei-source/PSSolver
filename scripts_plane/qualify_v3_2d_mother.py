#!/usr/bin/env python3
"""Conservatively qualify a completed target-parameter V3 Q2D mother run."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


FRAME_PATTERN = re.compile(r"^Q2D_(\d+)\.npy$")
OBSERVABLES = (
    "defect_density",
    "u_rms",
    "vorticity_rms",
    "mean_S",
    "ldg_free_energy_density",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def effective_sample_size(values: np.ndarray) -> float:
    """Estimate ESS with the initial-positive autocorrelation sequence."""
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count < 2:
        return float(count)
    centered = values - values.mean()
    variance = float(np.dot(centered, centered) / count)
    if variance <= np.finfo(np.float64).tiny:
        return float(count)
    autocorrelation = np.correlate(centered, centered, mode="full")[count - 1 :]
    autocorrelation /= autocorrelation[0]
    integrated_time = 1.0
    for lag in range(1, count):
        rho = float(autocorrelation[lag])
        if not math.isfinite(rho) or rho <= 0.0:
            break
        integrated_time += 2.0 * rho
    return max(1.0, min(float(count), count / integrated_time))


def time_series_statistics(times: np.ndarray, values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if len(values) != len(times) or len(values) < 3:
        raise ValueError("time-series statistics require at least three paired values")
    if not np.all(np.isfinite(values)):
        raise ValueError("time series contains NaN or Inf")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    slope = float(np.polyfit(times - times[0], values, 1)[0])
    drift = abs(slope) * float(times[-1] - times[0])
    scale = max(std, 1.0e-12 * max(1.0, abs(mean)))
    midpoint = len(values) // 2
    first_mean = float(values[:midpoint].mean())
    second_mean = float(values[midpoint:].mean())
    half_difference = abs(second_mean - first_mean)
    return {
        "mean": mean,
        "std": std,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "slope_per_time": slope,
        "full_window_drift_over_std": drift / scale,
        "first_half_mean": first_mean,
        "second_half_mean": second_mean,
        "half_mean_difference_over_std": half_difference / scale,
        "effective_sample_size": effective_sample_size(values),
    }


def count_nematic_defects(q2d: np.ndarray) -> tuple[int, int, int]:
    """Count periodic planar +/-1/2 plaquette windings from compact Q."""
    if q2d.ndim != 3 or q2d.shape[-1] != 5:
        raise ValueError(f"expected Q2D shape (Nx,Ny,5), got {q2d.shape}")
    if not np.all(np.isfinite(q2d)):
        raise ValueError("Q2D frame contains NaN or Inf")
    doubled_angle = np.arctan2(2.0 * q2d[..., 1], q2d[..., 0] - q2d[..., 3])

    def wrapped_difference(end: np.ndarray, start: np.ndarray) -> np.ndarray:
        difference = end - start
        return np.angle(np.exp(1j * difference))

    right = np.roll(doubled_angle, -1, axis=0)
    up = np.roll(doubled_angle, -1, axis=1)
    right_up = np.roll(right, -1, axis=1)
    winding_phase = (
        wrapped_difference(right, doubled_angle)
        + wrapped_difference(right_up, right)
        + wrapped_difference(up, right_up)
        + wrapped_difference(doubled_angle, up)
    )
    winding = np.rint(winding_phase / (2.0 * math.pi)).astype(np.int8)
    positive = int(np.count_nonzero(winding == 1))
    negative = int(np.count_nonzero(winding == -1))
    return positive + negative, positive, negative


def _numeric_frames(run_dir: Path) -> dict[int, Path]:
    frames = {}
    for path in run_dir.glob("Q2D_*.npy"):
        match = FRAME_PATTERN.match(path.name)
        if match is None:
            continue
        step = int(match.group(1))
        if step in frames:
            raise ValueError(f"duplicate V3 Q2D step {step}")
        frames[step] = path.resolve()
    if not frames:
        raise ValueError(f"no numeric Q2D frames found in {run_dir}")
    return frames


def _load_diagnostics(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "step",
            "time",
            "u_rms",
            "vorticity_rms",
            "mean_S",
            "ldg_free_energy_density",
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(f"mother diagnostics is missing {sorted(required)}")
        for raw in reader:
            raw_step = finite_float(raw["step"], label="diagnostic step")
            step = int(round(raw_step))
            if not math.isclose(raw_step, step, rel_tol=0.0, abs_tol=1.0e-9):
                raise ValueError(f"diagnostic step must be integral, got {raw_step}")
            if step in rows:
                raise ValueError(f"duplicate mother diagnostic step {step}")
            rows[step] = {
                name: finite_float(raw[name], label=f"diagnostic {name}")
                for name in required - {"step"}
            }
    return rows


def qualify(
    *,
    run_dir: Path | Sequence[Path],
    start_time: float,
    end_time: float,
    minimum_frames: int,
    minimum_ess: float,
    maximum_drift_over_std: float,
    maximum_half_difference_over_std: float,
    maximum_birth_death_relative_imbalance: float,
) -> tuple[dict, list[dict]]:
    run_dirs = (
        (Path(run_dir),)
        if isinstance(run_dir, (str, Path))
        else tuple(Path(path) for path in run_dir)
    )
    if not run_dirs:
        raise ValueError("at least one V3 mother run directory is required")
    run_dirs = tuple(path.expanduser().resolve() for path in run_dirs)
    frames: dict[int, Path] = {}
    all_frame_paths: dict[int, list[Path]] = {}
    diagnostics: dict[int, dict[str, float]] = {}
    control_paths: list[Path] = []
    metadata_records: list[dict] = []
    overlap_checks: list[dict] = []
    compatibility_reference = None
    for current_run_dir in run_dirs:
        metadata_path = current_run_dir / "metadata.json"
        complete_path = current_run_dir / "COMPLETE"
        diagnostics_path = current_run_dir / "mother_online_diagnostics.csv"
        for path in (metadata_path, complete_path, diagnostics_path):
            if not path.is_file():
                raise FileNotFoundError(f"required V3 mother input is missing: {path}")
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("metadata root must be a JSON object")
        if metadata.get("status") != "complete":
            raise ValueError("V3 mother metadata status must be complete")
        if metadata.get("initialization_protocol") != "v3-mother":
            raise ValueError("run is not a v3-mother initialization protocol")
        if metadata.get("solver", {}).get("save_layout") != "q2d":
            raise ValueError("V3 mother run must use q2d save layout")
        if metadata.get("q2d_z_invariance", {}).get(
            "all_saved_frames_passed"
        ) is not True:
            raise ValueError("V3 mother run did not pass its z-invariance gate")
        compatibility = {
            "dt": metadata.get("dt"),
            "domain": metadata.get("domain"),
            "shape": metadata.get("shape"),
            "dtype": metadata.get("dtype"),
            "parameterization": metadata.get("parameterization"),
            "parameters": metadata.get("model", {}).get("parameters"),
            "q_boundary_conditions": metadata.get("q_boundary_conditions"),
            "zero_mode_policy": metadata.get("zero_mode_policy"),
            "dealias_rule": metadata.get("dealias_rule"),
            "spectral_refresh_mode": metadata.get("numerics", {})
            .get("spectral_refresh", {})
            .get("mode"),
        }
        if compatibility_reference is None:
            compatibility_reference = compatibility
        elif compatibility != compatibility_reference:
            raise ValueError("V3 mother continuation metadata is incompatible")
        metadata_records.append(metadata)
        control_paths.extend((metadata_path, complete_path, diagnostics_path))

        for step, path in _numeric_frames(current_run_dir).items():
            all_frame_paths.setdefault(step, []).append(path)
            if step in frames:
                first = np.load(frames[step], allow_pickle=False)
                second = np.load(path, allow_pickle=False)
                if first.shape != second.shape or first.dtype != second.dtype:
                    raise ValueError(
                        f"overlapping V3 mother frame {step} changed shape or dtype"
                    )
                difference = np.asarray(second - first, dtype=np.float64)
                difference_rms = float(np.sqrt(np.mean(difference * difference)))
                reference_rms = float(
                    np.sqrt(np.mean(np.asarray(first, dtype=np.float64) ** 2))
                )
                relative_rms = difference_rms / max(
                    reference_rms,
                    np.finfo(np.float64).tiny,
                )
                maximum_absolute = float(np.max(np.abs(difference)))
                tolerance = 1.0e-12 if first.dtype == np.float64 else 2.0e-6
                overlap_checks.append({
                    "step": step,
                    "first_path": str(frames[step]),
                    "second_path": str(path),
                    "bitwise_equal": bool(np.array_equal(first, second)),
                    "relative_rms": relative_rms,
                    "maximum_absolute": maximum_absolute,
                    "relative_tolerance": tolerance,
                })
                if relative_rms > tolerance:
                    raise ValueError(
                        f"overlapping V3 mother frame {step} relative RMS "
                        f"{relative_rms:.6e} exceeds {tolerance:.6e}"
                    )
            else:
                frames[step] = path
        for step, row in _load_diagnostics(diagnostics_path).items():
            if step in diagnostics:
                if any(
                    not math.isclose(
                        diagnostics[step][name],
                        row[name],
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-14,
                    )
                    for name in row
                ):
                    raise ValueError(
                        f"overlapping V3 mother diagnostic {step} is inconsistent"
                    )
            else:
                diagnostics[step] = row

    metadata = metadata_records[0]
    dt = finite_float(metadata["dt"], label="metadata dt")
    lx, ly, _ = (float(value) for value in metadata["domain"])
    area = lx * ly
    selected_steps = [
        step
        for step in sorted(frames)
        if start_time - 1.0e-12 <= step * dt <= end_time + 1.0e-12
    ]
    if len(selected_steps) < minimum_frames:
        raise ValueError(
            f"analysis window has {len(selected_steps)} frames; requires {minimum_frames}"
        )
    missing_diagnostics = [step for step in selected_steps if step not in diagnostics]
    if missing_diagnostics:
        raise ValueError(
            f"saved Q2D frames lack online diagnostics at steps {missing_diagnostics[:8]}"
        )

    fixed_paths = control_paths + [
        path
        for step in selected_steps
        for path in all_frame_paths[step]
    ]
    identities_before = {str(path): sha256_file(path) for path in fixed_paths}
    rows = []
    previous_count = None
    births = 0
    deaths = 0
    for step in selected_steps:
        q2d = np.load(frames[step], allow_pickle=False)
        count, positive, negative = count_nematic_defects(q2d)
        if previous_count is not None:
            change = count - previous_count
            births += max(change, 0)
            deaths += max(-change, 0)
        previous_count = count
        diagnostic = diagnostics[step]
        rows.append({
            "step": step,
            "time": step * dt,
            "defect_count": count,
            "positive_half_count": positive,
            "negative_half_count": negative,
            "defect_density": count / area,
            "u_rms": diagnostic["u_rms"],
            "vorticity_rms": diagnostic["vorticity_rms"],
            "mean_S": diagnostic["mean_S"],
            "ldg_free_energy_density": diagnostic["ldg_free_energy_density"],
        })

    times = np.asarray([row["time"] for row in rows])
    statistics = {
        name: time_series_statistics(
            times,
            np.asarray([row[name] for row in rows]),
        )
        for name in OBSERVABLES
    }
    gates = {
        "minimum_frame_count": len(rows) >= minimum_frames,
        "nonzero_dynamic_defect_population": (
            statistics["defect_density"]["mean"] > 0 and births > 0 and deaths > 0
        ),
        "periodic_charge_neutrality": all(
            row["positive_half_count"] == row["negative_half_count"]
            for row in rows
        ),
        "all_observable_ess_at_least_minimum": all(
            statistics[name]["effective_sample_size"] >= minimum_ess
            for name in OBSERVABLES
        ),
        "all_full_window_drifts_within_tolerance": all(
            statistics[name]["full_window_drift_over_std"]
            <= maximum_drift_over_std
            for name in OBSERVABLES
        ),
        "all_half_window_means_compatible": all(
            statistics[name]["half_mean_difference_over_std"]
            <= maximum_half_difference_over_std
            for name in OBSERVABLES
        ),
        "birth_death_rates_compatible": (
            abs(births - deaths) / max(births + deaths, 1)
            <= maximum_birth_death_relative_imbalance
        ),
        "z_invariance": True,
    }
    qualified = all(gates.values())

    standardized = np.column_stack([
        np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in OBSERVABLES
    ])
    center = np.mean(standardized, axis=0)
    scale = np.std(standardized, axis=0)
    scale[scale <= np.finfo(float).tiny] = 1.0
    medoid_index = int(np.argmin(np.sum(((standardized - center) / scale) ** 2, axis=1)))
    checkpoint_path = frames[rows[medoid_index]["step"]]
    identities_after = {str(path): sha256_file(path) for path in fixed_paths}
    if identities_after != identities_before:
        raise RuntimeError("a V3 mother input changed during qualification")

    parameters = metadata["model"]["parameters"]
    report = {
        "schema_version": 1,
        "protocol": "V3",
        "analysis": "target_parameter_2d_mother_stationarity_screen",
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "candidate_2d_statistical_steady_state": qualified,
        "classification": "qualified" if qualified else "inconclusive_or_nonstationary",
        "parameters": {
            "frank_k": float(parameters["frank_K"]),
            "zeta": float(parameters["zeta"]),
            "activity_number": float(parameters["activity_number"]),
        },
        "analysis_window": {
            "start_time": float(times[0]),
            "end_time": float(times[-1]),
            "frame_count": len(rows),
        },
        "source_run_directories": [str(path) for path in run_dirs],
        "continuation_overlap_checks": overlap_checks,
        "thresholds": {
            "minimum_frames": minimum_frames,
            "minimum_ess": minimum_ess,
            "maximum_drift_over_std": maximum_drift_over_std,
            "maximum_half_difference_over_std": maximum_half_difference_over_std,
            "maximum_birth_death_relative_imbalance": (
                maximum_birth_death_relative_imbalance
            ),
        },
        "defect_count_change_proxy": {
            "births": births,
            "deaths": deaths,
            "definition": "positive/negative changes between saved-frame counts",
        },
        "statistics": statistics,
        "gates": gates,
        "recommended_checkpoint": (
            {
                "path": str(checkpoint_path),
                "sha256": identities_before[str(checkpoint_path)],
                "step": rows[medoid_index]["step"],
                "time": rows[medoid_index]["time"],
                "selection": "standardized observable medoid of qualified window",
            }
            if qualified
            else None
        ),
        "input_identities": identities_before,
        "limitations": [
            "A finite time window cannot prove mathematical ergodicity.",
            "Defect birth/death values are count-change proxies, not tracked identities.",
            "Independent mother seeds remain necessary for ensemble uncertainty.",
        ],
    }
    return report, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--minimum-frames", type=int, default=41)
    parser.add_argument("--minimum-ess", type=float, default=3.0)
    parser.add_argument("--maximum-drift-over-std", type=float, default=1.0)
    parser.add_argument("--maximum-half-difference-over-std", type=float, default=1.0)
    parser.add_argument(
        "--maximum-birth-death-relative-imbalance",
        type=float,
        default=0.25,
    )
    args = parser.parse_args()
    if not math.isfinite(args.start_time) or not math.isfinite(args.end_time):
        parser.error("analysis times must be finite")
    if args.start_time >= args.end_time:
        parser.error("--start-time must be smaller than --end-time")
    if args.minimum_frames < 3:
        parser.error("--minimum-frames must be at least three")
    if args.minimum_ess < 1:
        parser.error("--minimum-ess must be at least one")
    for name in (
        "maximum_drift_over_std",
        "maximum_half_difference_over_std",
        "maximum_birth_death_relative_imbalance",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative and finite")
    return args


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    report, rows = qualify(
        run_dir=args.run_dir,
        start_time=args.start_time,
        end_time=args.end_time,
        minimum_frames=args.minimum_frames,
        minimum_ess=args.minimum_ess,
        maximum_drift_over_std=args.maximum_drift_over_std,
        maximum_half_difference_over_std=args.maximum_half_difference_over_std,
        maximum_birth_death_relative_imbalance=(
            args.maximum_birth_death_relative_imbalance
        ),
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "v3_mother_qualification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with (output_dir / "v3_mother_time_series.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["candidate_2d_statistical_steady_state"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
