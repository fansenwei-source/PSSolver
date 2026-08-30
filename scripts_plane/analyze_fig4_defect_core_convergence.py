#!/usr/bin/env python3
"""Defect-core resolution precheck for short Fig. 4 benchmark runs.

The analysis deliberately separates three objects which are easy to confuse:

* ``Q2D_initial.npy`` is the unprojected, discrete two-dimensional generator.
* ``Q_0.npy`` is the projected three-dimensional state that actually enters
  the evolution.
* ``Q_200.npy`` (or the completed step recorded in metadata) is the evolved
  state.

The three-dimensional snapshots are memory mapped and evaluated at common
``z/H`` values with the solver's cell-centred, orthonormal DCT-II basis.  All
sub-grid xy samples use a fifth-order periodic spline whose accuracy is tested
against analytic periodic Fourier modes.  Low order parameter only proposes a
core candidate: a candidate receives a charge only after two independent
director/Q winding checks agree.  Hungarian matching is then performed within
each reliable charge class.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any

import numpy as np
from scipy import fft as scipy_fft
from scipy import ndimage
from scipy.optimize import brentq, linear_sum_assignment, minimize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pssolver.models.active_nematics import Q_convention_metadata, S_from_Q
from scripts_plane.analyze_fig4_convergence import (
    RunArtifact,
    load_run,
    validate_comparability,
    validate_run_identity,
)


Q_COMPONENT_COUNT = 5
DEFAULT_Z_FRACTIONS = (0.1, 0.5, 0.9)
DEFAULT_RADIAL_MAX = 4.0
DEFAULT_RADIAL_STEP = 0.05
DEFAULT_ANGULAR_SAMPLES = 128
DEFAULT_MATCH_RADIUS = 4.0
DEFAULT_INTERPOLATION_ORDER = 5
BIAXIALITY_RELATIVE_TRQ2_CUTOFF = 1.0e-12
S_FAR_MINIMUM_VALID_FRACTION = 0.50
S_FAR_LOW_ORDER_CUTOFF_FRACTION = 0.10
PROFILE_MINIMUM_ANGULAR_FRACTION = 0.50
PROFILE_NORM_FLOOR = 1.0e-12
LOCAL_REFERENCE_MINIMUM_SPAN_FRACTION = 1.0e-3
WINDING_RING_NEIGHBOR_CLEARANCE_MULTIPLIER = 2.0
RADIUS_METRICS = (
    "r50_absolute",
    "r90_absolute",
    "r50_local",
    "r90_local",
)
CORE_METRICS = (
    "S_min",
    *RADIUS_METRICS,
    "S_far",
    "S_far_mad",
    "S_far_valid_fraction",
    "r50_contour_anisotropy",
    "r50_contour_valid_fraction",
    "core_deficit_absolute",
    "core_deficit_local",
    "low_s_area_absolute",
    "low_s_area_local",
    "center_displacement",
    "biaxiality_mean",
    "biaxiality_median",
    "biaxiality_max",
    "biaxiality_valid_fraction",
    "biaxiality_mean_cutoff_lo",
    "biaxiality_mean_cutoff_hi",
)
NEAR_ZERO_METRICS = frozenset(
    {
        "S_min",
        "center_displacement",
        "biaxiality_mean",
        "biaxiality_median",
        "biaxiality_max",
        "biaxiality_mean_cutoff_lo",
        "biaxiality_mean_cutoff_hi",
    }
)
TIME_GATE_ABSOLUTE_TOLERANCES = {
    "S_min": 1.0e-6,
    "center_displacement": 1.0e-4,
    "biaxiality_mean": 1.0e-6,
    "biaxiality_median": 1.0e-6,
    "biaxiality_max": 1.0e-6,
    "default": 1.0e-4,
    "profile_weighted_relative_l2": 1.0e-4,
}
REFERENCE_METRICS = (
    "S_min",
    *RADIUS_METRICS,
    "core_deficit_absolute",
    "core_deficit_local",
    "low_s_area_absolute",
    "low_s_area_local",
)
RAW_STATE = "raw_q2d"
PROJECTED_STATE = "projected_q0"
FINAL_STATE = "final"
LEGACY_BENCHMARK_SCRIPT = "Plane_fig4_benchmark.py"
BERIS_EDWARDS_BENCHMARK_SCRIPT = "Plane_beris_edwards_stokes.py"
BERIS_EDWARDS_MODEL_VARIANT = "beris_edwards_complete_nematic_stress_stokes"
REQUIRED_BERIS_EDWARDS_IMPLEMENTATION_FILES = frozenset(
    {
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/Field.py",
        "pssolver/PDEmodel.py",
        "pssolver/integrator.py",
        "pssolver/transforms.py",
        "pssolver/__init__.py",
        "pssolver/models/active_nematics/__init__.py",
        "pssolver/models/active_nematics/fields.py",
        "pssolver/models/active_nematics/q_tensor.py",
        "pssolver/models/active_nematics/beris_edwards.py",
        "pssolver/models/active_nematics/initial_conditions.py",
    }
)


@dataclass(frozen=True)
class CoreRun:
    artifact: RunArtifact
    q0_path: Path
    q2d_path: Path
    defects_path: Path
    defect_positions: np.ndarray
    defect_charges: np.ndarray

    @property
    def label(self) -> str:
        return self.artifact.label

    @property
    def metadata(self) -> dict[str, Any]:
        return self.artifact.metadata

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.artifact.shape

    @property
    def lengths(self) -> tuple[float, float, float]:
        return self.artifact.lengths


@dataclass(frozen=True)
class CoreCandidate:
    x: float
    y: float
    S_min: float
    charge: float | None
    charge_reason: str
    q_windings: tuple[float, ...]
    director_windings: tuple[float, ...]
    center_fit_status: str = "optimized"
    center_fit_initial_S: float = math.nan
    center_fit_objective_improvement: float = math.nan
    center_fit_boundary_margin_fraction: float = math.nan
    nearest_other_core_distance: float = math.nan
    winding_ring_required_clearance: float = math.nan
    winding_ring_isolation_status: str = "not_evaluated"


@dataclass(frozen=True)
class RadiusCrossing:
    value: float
    reason: str
    crossing_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze projected-initial and final defect cores from completed "
            "Plane_fig4_benchmark.py resolution runs."
        )
    )
    parser.add_argument("--mode", choices=("space", "time"), default="space")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z-fractions", type=float, nargs="+", default=DEFAULT_Z_FRACTIONS)
    parser.add_argument("--r-max", type=float, default=DEFAULT_RADIAL_MAX)
    parser.add_argument("--dr", type=float, default=DEFAULT_RADIAL_STEP)
    parser.add_argument("--angular-samples", type=int, default=DEFAULT_ANGULAR_SAMPLES)
    parser.add_argument("--match-radius", type=float, default=DEFAULT_MATCH_RADIUS)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument(
        "--space-core-summary",
        type=Path,
        default=None,
        help=(
            "Space-mode defect_core_summary.json used to apply the prescribed "
            "20%% time-vs-space gate in --mode time."
        ),
    )
    parser.add_argument(
        "--interpolation-order",
        type=int,
        choices=(5,),
        default=DEFAULT_INTERPOLATION_ORDER,
    )
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        parser.error("at least two completed run directories are required")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in args.z_fractions):
        parser.error("--z-fractions must lie in [0, 1]")
    if len(set(args.z_fractions)) != len(args.z_fractions):
        parser.error("--z-fractions must be unique")
    for required in DEFAULT_Z_FRACTIONS:
        if not any(math.isclose(value, required, rel_tol=0.0, abs_tol=1.0e-12) for value in args.z_fractions):
            parser.error("--z-fractions must include 0.1, 0.5, and 0.9")
    if not math.isfinite(args.r_max) or not math.isclose(
        args.r_max, 4.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        parser.error("--r-max must be exactly 4 for the common physical integration domain")
    if not math.isfinite(args.dr) or args.dr <= 0.0 or args.dr > 0.05:
        parser.error("--dr must be positive, finite, and no larger than 0.05")
    if args.angular_samples < 128:
        parser.error("--angular-samples must be at least 128")
    if not math.isfinite(args.match_radius) or args.match_radius <= 0.0:
        parser.error("--match-radius must be positive and finite")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.space_core_summary is not None and args.mode != "time":
        parser.error("--space-core-summary is valid only with --mode time")
    return args


def minimum_image_delta(delta: np.ndarray, periods: Sequence[float]) -> np.ndarray:
    values = np.asarray(delta, dtype=np.float64)
    period_values = np.asarray(periods, dtype=np.float64)
    return values - period_values * np.round(values / period_values)


def periodic_distances(
    first: np.ndarray,
    second: np.ndarray,
    periods: Sequence[float],
) -> np.ndarray:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    delta = a[..., None, :] - b[None, ...]
    return np.linalg.norm(minimum_image_delta(delta, periods), axis=-1)


def load_defects_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing initial defect table: {path}")
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if table.dtype.names != ("x", "y", "charge"):
        raise ValueError(f"{path} must have columns x,y,charge")
    table = np.atleast_1d(table)
    positions = np.column_stack((table["x"], table["y"]))
    charges = np.asarray(table["charge"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0:
        raise ValueError(f"{path} contains no valid defect coordinates")
    if not np.isfinite(positions).all() or not np.isfinite(charges).all():
        raise ValueError(f"{path} contains NaN or Inf")
    if not np.all(np.isin(charges, (-0.5, 0.5))):
        raise ValueError(f"{path} charges must be exactly +/-0.5")
    if not math.isclose(float(np.sum(charges)), 0.0, abs_tol=1.0e-14):
        raise ValueError(f"{path} is not charge neutral")
    return positions, charges


def _mmap_float_array(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required array: {path}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != expected_shape:
        raise ValueError(f"{path} has shape {values.shape}, expected {expected_shape}")
    if np.dtype(values.dtype) != np.dtype(np.float64):
        raise ValueError(f"{path} must contain float64 values, got {values.dtype}")
    return values


def projected_snapshot_sha256(
    path: Path,
    shape: tuple[int, int, int],
    *,
    chunk_size: int = 8,
) -> str:
    """Reproduce ``tensor_sha256`` for a component-last saved Q snapshot.

    The benchmark hashes five separate tensors of shape ``(1,Nx,Ny,Nz)``.
    Iterating over x chunks preserves each component's C-order byte stream
    without allocating a complete component or a complete 3-D Q copy.
    """
    q = _mmap_float_array(path, (*shape, Q_COMPONENT_COUNT))
    digest = hashlib.sha256()
    component_shape = np.asarray((1, *shape), dtype=np.int64)
    for component in range(Q_COMPONENT_COUNT):
        digest.update(str(q.dtype).encode("ascii"))
        digest.update(component_shape.tobytes())
        for start in range(0, shape[0], chunk_size):
            stop = min(start + chunk_size, shape[0])
            values = np.ascontiguousarray(q[start:stop, :, :, component])
            if not np.isfinite(values).all():
                raise ValueError(f"{path} contains NaN or Inf in component {component}")
            digest.update(values.tobytes())
    return digest.hexdigest()


def load_core_run(directory: str | Path) -> CoreRun:
    artifact = load_run(directory)
    q0_path = artifact.directory / "Q_0.npy"
    q2d_path = artifact.directory / "Q2D_initial.npy"
    defects_path = artifact.directory / "Q2D_defects.csv"
    _mmap_float_array(q0_path, (*artifact.shape, Q_COMPONENT_COUNT))
    _mmap_float_array(q2d_path, (*artifact.shape[:2], Q_COMPONENT_COUNT))
    positions, charges = load_defects_csv(defects_path)
    save_start = artifact.metadata.get("save_start_step")
    if save_start != 0:
        raise ValueError(
            f"{artifact.directory} must declare save_start_step=0 so Q_0 is the "
            "actual projected initial state"
        )
    expected_hash = artifact.metadata.get("initial_condition", {}).get(
        "projected_q_sha256"
    )
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError(
            f"{artifact.directory} is missing a valid projected_q_sha256"
        )
    observed_hash = projected_snapshot_sha256(q0_path, artifact.shape)
    if observed_hash != expected_hash:
        raise ValueError(
            f"{q0_path} does not match metadata projected_q_sha256: "
            f"observed {observed_hash}, expected {expected_hash}"
        )
    return CoreRun(
        artifact=artifact,
        q0_path=q0_path,
        q2d_path=q2d_path,
        defects_path=defects_path,
        defect_positions=positions,
        defect_charges=charges,
    )


def validate_core_runs(runs: Sequence[CoreRun], mode: str) -> None:
    validate_comparability([run.artifact for run in runs], mode)
    reference = runs[0]
    for run in runs[1:]:
        if run.defect_positions.shape != reference.defect_positions.shape:
            raise ValueError("Q2D_defects.csv defect counts differ between runs")
        if not np.allclose(
            run.defect_positions,
            reference.defect_positions,
            rtol=0.0,
            atol=2.0e-13,
        ):
            raise ValueError("Q2D_defects.csv continuous coordinates differ between runs")
        if not np.array_equal(run.defect_charges, reference.defect_charges):
            raise ValueError("Q2D_defects.csv charge order/values differ between runs")

    expected = Q_convention_metadata()
    for run in runs:
        if run.metadata["model"].get("Q_convention") != expected:
            raise ValueError(f"{run.artifact.directory} has a noncanonical Q convention")
        parameters = run.metadata["model"].get("parameters", {})
        if run.metadata.get("zero_mode_policy") == "zero_mean":
            if float(run.metadata.get("friction", math.nan)) != 0.0:
                raise ValueError("zero_mean benchmark metadata must have friction=0.0")
            if float(parameters.get("fric", math.nan)) != 0.0:
                raise ValueError("zero_mean model parameters must have fric=0.0")


def _metadata_close(value: Any, expected: float) -> bool:
    try:
        observed = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(observed) and math.isclose(
        observed, expected, rel_tol=2.0e-14, abs_tol=2.0e-14
    )


def _resolved_core_physics(metadata: Mapping[str, Any]) -> dict[str, float]:
    """Resolve raw physical coefficients without conflating K and L1/gamma."""
    script = metadata.get("script")
    ldg = metadata.get("ldg_coefficients")
    if not isinstance(ldg, Mapping):
        raise ValueError("metadata is missing ldg_coefficients")
    if script == LEGACY_BENCHMARK_SCRIPT:
        ldg_l1 = metadata.get("landau_l1")
    elif script == BERIS_EDWARDS_BENCHMARK_SCRIPT:
        ldg_l1 = metadata.get("ldg_l1")
    else:
        raise ValueError(f"unsupported core-analysis script {script!r}")
    fields = {
        "ldg_A": ldg.get("A"),
        "ldg_B": ldg.get("B"),
        "ldg_C": ldg.get("C"),
        "ldg_L1": ldg_l1,
        "S_bulk": metadata.get("S_bulk"),
        "frank_K": metadata.get("frank_k"),
        "zeta": metadata.get("zeta"),
        "gamma": metadata.get("gamma"),
    }
    resolved: dict[str, float] = {}
    for name, value in fields.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"metadata has invalid {name}={value!r}") from error
        if not math.isfinite(number):
            raise ValueError(f"metadata has invalid {name}={value!r}")
        resolved[name] = number
    return resolved


def _check_scalar_mapping(
    mapping: Any,
    expected: Mapping[str, float],
    *,
    label: str,
    prefix: str,
    errors: list[str],
) -> None:
    if not isinstance(mapping, Mapping):
        errors.append(f"{prefix}: {label} must be a mapping")
        return
    for field, expected_value in expected.items():
        if not _metadata_close(mapping.get(field), expected_value):
            errors.append(
                f"{prefix}: {label}.{field} must be {expected_value!r}"
            )


def validate_task_contract(runs: Sequence[CoreRun], mode: str) -> None:
    """Reject complete-but-wrong inputs before labeling this fixed A=18 study."""
    if mode not in {"space", "time"}:
        raise ValueError(f"unsupported defect-core validation mode {mode!r}")
    expected_space_shapes = {(256, 256, 64), (320, 320, 80), (512, 512, 128)}
    observed_shapes = {run.shape for run in runs}
    errors: list[str] = []
    if mode == "space" and (len(runs) != 3 or observed_shapes != expected_space_shapes):
        errors.append(
            f"space run shapes {sorted(observed_shapes)} do not equal {sorted(expected_space_shapes)}"
        )
    if mode == "time" and (len(runs) != 2 or observed_shapes != {(512, 512, 128)}):
        errors.append("time-control inputs must all use shape (512,512,128)")
    if mode == "time" and sorted(run.artifact.dt for run in runs) != [0.0025, 0.005]:
        errors.append("time control must contain exactly dt=0.005 and dt=0.0025")

    expected_scalar_fields = {
        "activity_number": 18.0,
        "height": 20.0,
        "zeta": 0.01,
        "frank_k": 0.012345679012345678,
        "gamma": 2.94,
        "flow_alignment": 0.3,
        "eta": 0.6666666666666666,
        "active_stress_beta": -1.0,
        "S_initial": 1.0 / 3.0,
        "S_bulk": 1.0 / 3.0,
        "friction": 0.0,
    }
    expected_legacy_model_fields = {
        "aQ": 0.0,
        "bQ": -0.10204081632653061,
        "cQ": 0.10204081632653061,
        "KQ": 0.008398421096833794,
        "alpha": 0.01,
        "beta": -1.0,
        "S_initial": 1.0 / 3.0,
        "S_bulk": 1.0 / 3.0,
        "fric": 0.0,
        "eta": 0.6666666666666666,
        "flow_alignment": 0.3,
    }
    expected_beris_model_fields = {
        "activity_number": 18.0,
        "zeta": 0.01,
        "frank_K": 0.012345679012345678,
        "ldg_A": 0.0,
        "ldg_B": -0.3,
        "ldg_C": 0.3,
        "ldg_L1": 0.024691358024691357,
        "rotational_viscosity_gamma": 2.94,
        "flow_alignment_lambda": 0.3,
        "alpha": 0.01,
        "beta": -1.0,
        "S_initial": 1.0 / 3.0,
        "S_bulk": 1.0 / 3.0,
        "fric": 0.0,
        "eta": 0.6666666666666666,
    }
    expected_beris_raw_fields = {
        "A": 0.0,
        "B": -0.3,
        "C": 0.3,
        "L1": 0.024691358024691357,
        "rotational_viscosity_gamma": 2.94,
        "flow_alignment_lambda": 0.3,
    }
    expected_beris_evolution_fields = {
        "A_over_gamma": 0.0,
        "B_over_gamma": -0.10204081632653061,
        "C_over_gamma": 0.10204081632653061,
        "L1_over_gamma": 0.008398421096833794,
    }
    for run in runs:
        metadata = run.metadata
        prefix = str(run.artifact.directory)
        try:
            validate_run_identity(
                metadata, path=run.artifact.directory / "metadata.json"
            )
        except ValueError as error:
            errors.append(str(error))
        script = metadata.get("script")
        if tuple(run.lengths) != (100.0, 100.0, 20.0):
            errors.append(f"{prefix}: physical box must be 100x100x20")
        if not math.isclose(run.artifact.final_time, 1.0, rel_tol=0.0, abs_tol=2.0e-14):
            errors.append(f"{prefix}: final time must be T=1")
        if mode == "space" and not (
            _metadata_close(run.artifact.dt, 0.005) and run.artifact.steps == 200
        ):
            errors.append(f"{prefix}: spatial run must use dt=0.005 and steps=200")
        if metadata.get("parameterization") != "paper-window":
            errors.append(f"{prefix}: parameterization must be paper-window")
        if metadata.get("zero_mode_policy") != "zero_mean":
            errors.append(f"{prefix}: zero_mode_policy must be zero_mean")
        if metadata.get("dealias_rule") != "cubic_half":
            errors.append(f"{prefix}: dealias_rule must be cubic_half")
        if not _metadata_close(metadata.get("dealias_fraction"), 0.5):
            errors.append(f"{prefix}: dealias_fraction must be 0.5")
        requested_device = metadata.get("device")
        if script == BERIS_EDWARDS_BENCHMARK_SCRIPT:
            if not isinstance(requested_device, str) or not requested_device.startswith("cuda"):
                errors.append(f"{prefix}: requested device must be cuda or cuda:N")
            runtime = metadata.get("runtime_environment")
            if not isinstance(runtime, Mapping):
                errors.append(f"{prefix}: runtime_environment must be a mapping")
            else:
                resolved_device = runtime.get("resolved_device")
                if runtime.get("device_type") != "cuda":
                    errors.append(f"{prefix}: runtime device_type must be cuda")
                if not isinstance(resolved_device, str) or not resolved_device.startswith("cuda"):
                    errors.append(f"{prefix}: runtime resolved_device must be cuda or cuda:N")
                if runtime.get("cuda_available") is not True:
                    errors.append(f"{prefix}: runtime must report cuda_available=true")
                if not isinstance(runtime.get("cuda_device_name"), str) or not runtime.get("cuda_device_name"):
                    errors.append(f"{prefix}: runtime must record a CUDA device name")
                if type(runtime.get("cuda_total_memory_bytes")) is not int or runtime.get("cuda_total_memory_bytes", 0) <= 0:
                    errors.append(f"{prefix}: runtime must record positive CUDA memory")
        elif script == LEGACY_BENCHMARK_SCRIPT and requested_device != "cuda":
            errors.append(f"{prefix}: device must be cuda")
        if metadata.get("dtype") != "float64" or metadata.get("tf32") != "off":
            errors.append(f"{prefix}: top-level dtype/tf32 must be float64/off")
        if metadata.get("seed") != 24:
            errors.append(f"{prefix}: seed must be 24")
        if metadata.get("save_start_step") != 0:
            errors.append(f"{prefix}: save_start_step must be 0")
        if metadata.get("save_interval") != run.artifact.steps:
            errors.append(f"{prefix}: save_interval must equal completed steps")
        expected_diagnostic_interval = (
            min(100, run.artifact.steps)
            if script == BERIS_EDWARDS_BENCHMARK_SCRIPT
            else run.artifact.steps
        )
        if metadata.get("diagnostic_interval") != expected_diagnostic_interval:
            errors.append(
                f"{prefix}: diagnostic_interval must be "
                f"{expected_diagnostic_interval}"
            )
        if metadata.get("save_hydrodynamics") is not True:
            errors.append(f"{prefix}: save_hydrodynamics must be true")
        expected_boundaries = {
            "Q": ["periodic", "periodic", "neumann"],
            "velocity_tangential": ["periodic", "periodic", "neumann"],
            "velocity_normal": ["periodic", "periodic", "dirichlet"],
            "pressure": ["periodic", "periodic", "neumann"],
        }
        if script == BERIS_EDWARDS_BENCHMARK_SCRIPT:
            expected_boundaries = {
                **expected_boundaries,
                "velocity_wall_model_note": (
                    "homogeneous kinematic free-slip; not zero total nematic traction"
                ),
            }
        if metadata.get("boundary_conditions") != expected_boundaries:
            errors.append(
                f"{prefix}: boundary_conditions do not match the fixed free-slip study"
            )
        for field, expected in expected_scalar_fields.items():
            if not _metadata_close(metadata.get(field), expected):
                errors.append(f"{prefix}: {field} must resolve to {expected!r}")
        expected_l1_field = (
            "ldg_l1"
            if script == BERIS_EDWARDS_BENCHMARK_SCRIPT
            else "landau_l1"
        )
        if not _metadata_close(
            metadata.get(expected_l1_field), 0.024691358024691357
        ):
            errors.append(
                f"{prefix}: {expected_l1_field} must resolve to "
                "0.024691358024691357"
            )
        if metadata.get("ldg_coefficients") != {"A": 0.0, "B": -0.3, "C": 0.3}:
            errors.append(
                f"{prefix}: raw LdG coefficients must be A=0,B=-0.3,C=0.3"
            )

        model = metadata.get("model")
        if not isinstance(model, Mapping):
            errors.append(f"{prefix}: model must be a mapping")
            model = {}
        parameters = model.get("parameters", {})
        if script == LEGACY_BENCHMARK_SCRIPT:
            _check_scalar_mapping(
                parameters,
                expected_legacy_model_fields,
                label="model.parameters",
                prefix=prefix,
                errors=errors,
            )
        elif script == BERIS_EDWARDS_BENCHMARK_SCRIPT:
            _check_scalar_mapping(
                parameters,
                expected_beris_model_fields,
                label="model.parameters",
                prefix=prefix,
                errors=errors,
            )
            if model.get("stage") != "reusable_beris_edwards_complete_stress_stokes":
                errors.append(f"{prefix}: unexpected Beris--Edwards model stage")
            q_dynamics = model.get("q_dynamics")
            if not isinstance(q_dynamics, Mapping):
                errors.append(f"{prefix}: model.q_dynamics must be a mapping")
                q_dynamics = {}
            if q_dynamics.get("implementation") != (
                "pssolver.models.active_nematics.BerisEdwardsQNonlinearModel"
            ):
                errors.append(f"{prefix}: unexpected Beris--Edwards Q implementation")
            if q_dynamics.get("flow_alignment_form") != "full_beris_edwards":
                errors.append(f"{prefix}: Q flow alignment must be full_beris_edwards")
            _check_scalar_mapping(
                q_dynamics.get("raw_coefficients"),
                expected_beris_raw_fields,
                label="model.q_dynamics.raw_coefficients",
                prefix=prefix,
                errors=errors,
            )
            _check_scalar_mapping(
                q_dynamics.get("evolution_coefficients"),
                expected_beris_evolution_fields,
                label="model.q_dynamics.evolution_coefficients",
                prefix=prefix,
                errors=errors,
            )
            if not _metadata_close(
                metadata.get("q_elastic_relaxation"),
                expected_beris_evolution_fields["L1_over_gamma"],
            ):
                errors.append(
                    f"{prefix}: q_elastic_relaxation must equal L1/gamma, not Frank K"
                )
            flow_dynamics = model.get("flow_dynamics")
            if not isinstance(flow_dynamics, Mapping) or flow_dynamics.get(
                "nematic_stress"
            ) != "complete_one_constant_beris_edwards_up_to_isotropic_pressure":
                errors.append(f"{prefix}: complete one-constant nematic stress is required")
            provenance = metadata.get("implementation_provenance")
            files = (
                provenance.get("files")
                if isinstance(provenance, Mapping)
                else None
            )
            if not isinstance(provenance, Mapping) or provenance.get("schema_version") != 1:
                errors.append(f"{prefix}: implementation provenance schema must be 1")
            if not isinstance(files, Mapping) or set(files) != (
                REQUIRED_BERIS_EDWARDS_IMPLEMENTATION_FILES
            ):
                errors.append(
                    f"{prefix}: implementation provenance must contain the fixed 12-file set"
                )
        precision = metadata.get("numerics", {}).get("precision", {})
        expected_precision = {
            "real_dtype": "float64",
            "spectral_dtype": "complex128",
            "tf32_requested": "off",
            "tf32_effective": False,
            "float32_matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        for field, expected in expected_precision.items():
            if precision.get(field) != expected:
                errors.append(f"{prefix}: precision.{field} must be {expected!r}")
        solver = metadata.get("solver", {})
        if solver.get("real_dtype") != "float64" or solver.get("spectral_dtype") != "complex128":
            errors.append(f"{prefix}: solver dtypes must be float64/complex128")
        refresh = metadata.get("numerics", {}).get("spectral_refresh", {})
        if refresh.get("mode") != "disabled" or refresh.get("actual_count") != 0:
            errors.append(f"{prefix}: spectral refresh must be disabled with actual_count=0")
        if any(
            refresh.get(field) is not None
            for field in (
                "requested_interval_time",
                "requested_interval_steps",
                "effective_interval_steps",
                "effective_interval_time",
            )
        ):
            errors.append(f"{prefix}: disabled spectral refresh intervals must be null")

        generator = metadata.get("initial_defect_gas", {})
        expected_generator = {
            "num_pairs": 6,
            "minimum_separation": 10.0,
            "core_radius": 1.5,
            "S_initial": 1.0 / 3.0,
            "background_angle": 0.0,
        }
        for field, expected in expected_generator.items():
            if field == "num_pairs":
                valid = generator.get(field) == expected
            else:
                valid = _metadata_close(generator.get(field), float(expected))
            if not valid:
                errors.append(f"{prefix}: initial_defect_gas.{field} must be {expected!r}")
        twist = metadata.get("initial_neumann_twist", {})
        if not _metadata_close(twist.get("rms_amplitude_radians"), 0.01) or twist.get(
            "dct_modes"
        ) != [1, 2, 3]:
            errors.append(f"{prefix}: twist must be amplitude 0.01 with modes [1,2,3]")

        nx, ny, nz = run.shape
        expected_q_modes = [nx // 2 - 1, ny // 2 - 1, nz // 2]
        expected_normal_modes = [nx // 2 - 1, ny // 2 - 1, nz // 2 - 1]
        if metadata.get("retained_q_modes") != expected_q_modes:
            errors.append(f"{prefix}: retained_q_modes must be {expected_q_modes}")
        if metadata.get("retained_normal_velocity_modes") != expected_normal_modes:
            errors.append(
                f"{prefix}: retained_normal_velocity_modes must be {expected_normal_modes}"
            )
        initial = metadata.get("initial_condition", {})
        if initial.get("name") != "extruded_analytic_periodic_defect_gas_2d":
            errors.append(f"{prefix}: unexpected initial_condition.name")
        if initial.get("seed") != 24:
            errors.append(f"{prefix}: initial_condition.seed must be 24")
        if not _metadata_close(initial.get("S_initial"), 1.0 / 3.0):
            errors.append(f"{prefix}: initial_condition.S_initial must be 1/3")
        if not _metadata_close(initial.get("twist_amplitude"), 0.01) or initial.get(
            "twist_modes"
        ) != [1, 2, 3]:
            errors.append(f"{prefix}: initial_condition twist must be 0.01/[1,2,3]")
        for hash_name in ("raw_q_sha256", "projected_q_sha256"):
            value = initial.get(hash_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                errors.append(
                    f"{prefix}: initial_condition.{hash_name} must be a canonical SHA-256"
                )
        if len(run.defect_positions) != 12:
            errors.append(f"{prefix}: Q2D_defects.csv must contain 12 defects")
        if np.count_nonzero(run.defect_charges == 0.5) != 6 or np.count_nonzero(
            run.defect_charges == -0.5
        ) != 6:
            errors.append(f"{prefix}: Q2D_defects.csv must contain six defects of each charge")

        required_arrays = {
            run.q0_path: (*run.shape, Q_COMPONENT_COUNT),
            run.artifact.q_path: (*run.shape, Q_COMPONENT_COUNT),
            run.artifact.u_path: (*run.shape, 3),
            run.artifact.p_path: run.shape,
            run.q2d_path: (*run.shape[:2], Q_COMPONENT_COUNT),
        }
        for path, shape in required_arrays.items():
            try:
                _mmap_float_array(path, shape)
            except (FileNotFoundError, ValueError) as error:
                errors.append(str(error))
        for step in (0, run.artifact.steps):
            for prefix_name in ("Q", "u", "p"):
                if not (run.artifact.directory / f"{prefix_name}_{step}.npy").is_file():
                    errors.append(f"{prefix}: missing {prefix_name}_{step}.npy")
        for required_name in ("diagnostics.npy", "diagnostics.csv", "COMPLETE"):
            if not (run.artifact.directory / required_name).is_file():
                errors.append(f"{prefix}: missing {required_name}")
    if errors:
        raise ValueError("Fixed A=18/T=1 benchmark contract failed:\n- " + "\n- ".join(errors))


def physical_scale_summary(runs: Sequence[CoreRun]) -> dict[str, Any]:
    metadata = runs[0].metadata
    physics = _resolved_core_physics(metadata)
    S_bulk = physics["S_bulk"]
    mu_S = (
        physics["ldg_A"]
        + physics["ldg_B"] * S_bulk
        + 4.5 * physics["ldg_C"] * S_bulk**2
    )
    landau_l1 = physics["ldg_L1"]
    frank_k = physics["frank_K"]
    zeta = physics["zeta"]
    gamma = physics["gamma"]
    xi_S = math.sqrt(landau_l1 / mu_S)
    active_length = math.sqrt(frank_k / zeta)
    tau_S = gamma / mu_S
    isolated_r50 = 1.5 * float(np.arctanh(0.5))
    grids = []
    for run in sorted(runs, key=lambda item: item.artifact.resolution_h, reverse=True):
        dx, dy, dz = run.artifact.cell_sizes
        grids.append(
            {
                "resolution": run.label,
                "shape": run.shape,
                "dx": dx,
                "dy": dy,
                "dz": dz,
                "h": run.artifact.resolution_h,
                "xi_S_over_dx": xi_S / dx,
                "xi_S_over_dz": xi_S / dz,
                "active_length_over_dx": active_length / dx,
                "active_length_over_dz": active_length / dz,
                "isolated_r50_diameter_over_dx": 2.0 * isolated_r50 / dx,
                "retained_q_modes": run.metadata["retained_q_modes"],
                "retained_normal_velocity_modes": run.metadata[
                    "retained_normal_velocity_modes"
                ],
            }
        )
    return {
        "S_bulk": S_bulk,
        "mu_S": mu_S,
        "tau_S": tau_S,
        "T_over_tau_S": 1.0 / tau_S,
        "xi_S": xi_S,
        "active_length": active_length,
        "isolated_r50": isolated_r50,
        "grids": grids,
    }


def dct_modal_basis(nz: int, z_fractions: Sequence[float]) -> np.ndarray:
    fractions = np.asarray(z_fractions, dtype=np.float64)
    if nz <= 0 or fractions.ndim != 1:
        raise ValueError("nz must be positive and z_fractions must be one-dimensional")
    if not np.isfinite(fractions).all() or np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("z_fractions must be finite and lie in [0, 1]")
    modes = np.arange(nz, dtype=np.float64)
    basis = np.cos(np.pi * fractions[:, None] * modes[None, :])
    basis[:, 0] *= math.sqrt(1.0 / nz)
    if nz > 1:
        basis[:, 1:] *= math.sqrt(2.0 / nz)
    return basis


def dct_modal_evaluate_chunk(
    values: np.ndarray,
    z_fractions: Sequence[float],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1:
        raise ValueError("DCT input must have at least one dimension")
    nz = array.shape[-1]
    coefficients = scipy_fft.dct(array, type=2, norm="ortho", axis=-1)
    basis = dct_modal_basis(nz, z_fractions)
    return np.tensordot(coefficients, basis, axes=([-1], [1]))


def evaluate_q_dct_planes(
    path: Path,
    shape: tuple[int, int, int],
    z_fractions: Sequence[float],
    *,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate Q at common z/H values without materializing a full 3-D copy."""
    q = _mmap_float_array(path, (*shape, Q_COMPONENT_COUNT))
    nx, ny, nz = shape
    fractions = tuple(float(value) for value in z_fractions)
    basis = dct_modal_basis(nz, fractions)
    planes = np.empty((nx, ny, len(fractions), Q_COMPONENT_COUNT), dtype=np.float64)
    for start in range(0, nx, chunk_size):
        stop = min(start + chunk_size, nx)
        chunk = np.asarray(q[start:stop, :, :, :], dtype=np.float64)
        if not np.isfinite(chunk).all():
            raise ValueError(f"{path} contains NaN or Inf in x slice {start}:{stop}")
        coefficients = scipy_fft.dct(chunk, type=2, norm="ortho", axis=2)
        planes[start:stop] = np.einsum(
            "xykc,tk->xytc", coefficients, basis, optimize=True
        )
    return planes


class PeriodicQSampler:
    """High-order periodic interpolation of a compact-Q xy plane."""

    def __init__(
        self,
        q_plane: np.ndarray,
        lengths: Sequence[float],
        *,
        order: int = DEFAULT_INTERPOLATION_ORDER,
    ) -> None:
        q = np.asarray(q_plane, dtype=np.float64)
        if q.ndim != 3 or q.shape[-1] != Q_COMPONENT_COUNT:
            raise ValueError(f"q_plane must have shape (Nx,Ny,5), got {q.shape}")
        if not np.isfinite(q).all():
            raise ValueError("q_plane contains NaN or Inf")
        lengths_array = np.asarray(lengths, dtype=np.float64)
        if lengths_array.shape != (2,) or np.any(lengths_array <= 0.0):
            raise ValueError("lengths must contain two positive values")
        if order not in (3, 5):
            raise ValueError("periodic interpolation order must be 3 or 5")
        self.shape = q.shape[:2]
        self.lengths = lengths_array
        self.spacing = lengths_array / np.asarray(self.shape, dtype=np.float64)
        self.order = order
        # A two-dimensional filtered plane is small even for R512 (~2 MiB per
        # float64 component).  This never duplicates a full 3-D snapshot.
        self._coefficients = tuple(
            ndimage.spline_filter(q[..., component], order=order, mode="grid-wrap")
            for component in range(Q_COMPONENT_COUNT)
        )

    def _index_coordinates(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_array, y_array = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        return np.vstack(
            (
                ((x_array.ravel() % self.lengths[0]) / self.spacing[0]) - 0.5,
                ((y_array.ravel() % self.lengths[1]) / self.spacing[1]) - 0.5,
            )
        )

    def sample_q(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_array, y_array = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        coordinates = self._index_coordinates(x_array, y_array)
        values = np.stack(
            [
                ndimage.map_coordinates(
                    coefficients,
                    coordinates,
                    order=self.order,
                    mode="grid-wrap",
                    prefilter=False,
                )
                for coefficients in self._coefficients
            ],
            axis=-1,
        )
        return values.reshape((*x_array.shape, Q_COMPONENT_COUNT))

    def sample_S(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.asarray(S_from_Q(self.sample_q(x, y)), dtype=np.float64)


def continuous_multidefect_S(
    x: np.ndarray,
    y: np.ndarray,
    *,
    positions: np.ndarray,
    lengths: Sequence[float],
    core_radius: float,
    S_initial: float,
) -> np.ndarray:
    x_values, y_values = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    result = np.full(x_values.shape, float(S_initial), dtype=np.float64)
    periods = np.asarray(lengths, dtype=np.float64)
    for position in np.asarray(positions, dtype=np.float64):
        dx = minimum_image_delta(x_values - position[0], (periods[0],))
        dy = minimum_image_delta(y_values - position[1], (periods[1],))
        result *= np.tanh(np.hypot(dx, dy) / float(core_radius))
    return result


def compact_q_matrix(q: np.ndarray) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64)
    if values.shape[-1] != Q_COMPONENT_COUNT:
        raise ValueError("compact Q must have final dimension 5")
    matrix = np.empty(values.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = values[..., 0]
    matrix[..., 0, 1] = matrix[..., 1, 0] = values[..., 1]
    matrix[..., 0, 2] = matrix[..., 2, 0] = values[..., 2]
    matrix[..., 1, 1] = values[..., 3]
    matrix[..., 1, 2] = matrix[..., 2, 1] = values[..., 4]
    matrix[..., 2, 2] = -(values[..., 0] + values[..., 3])
    return matrix


def biaxiality_from_Q(
    q: np.ndarray,
    *,
    S_scale: float,
    relative_trq2_cutoff: float = BIAXIALITY_RELATIVE_TRQ2_CUTOFF,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = compact_q_matrix(q)
    tr_q2 = np.einsum("...ij,...ji->...", matrix, matrix, optimize=True)
    tr_q3 = np.einsum("...ij,...jk,...ki->...", matrix, matrix, matrix, optimize=True)
    cutoff = float(relative_trq2_cutoff) * float(S_scale) ** 2
    valid = np.isfinite(tr_q2) & np.isfinite(tr_q3) & (tr_q2 > cutoff)
    result = np.full(tr_q2.shape, np.nan, dtype=np.float64)
    result[valid] = 1.0 - 6.0 * tr_q3[valid] ** 2 / tr_q2[valid] ** 3
    result[valid] = np.clip(result[valid], 0.0, 1.0)
    return result, valid


def wrapped_phase_winding(phase_values: np.ndarray) -> float:
    phase = np.asarray(phase_values, dtype=np.float64).reshape(-1)
    if len(phase) < 8 or not np.isfinite(phase).all():
        return math.nan
    increments = np.angle(np.exp(1j * (np.roll(phase, -1) - phase)))
    return float(np.sum(increments, dtype=np.float64) / (2.0 * np.pi))


def classify_candidate_charge(
    sampler: PeriodicQSampler,
    center: Sequence[float],
    *,
    S_bulk: float,
    radii: Sequence[float] = (1.5, 2.0),
    angular_samples: int = DEFAULT_ANGULAR_SAMPLES,
    winding_tolerance: float = 0.2,
    minimum_projection: float = 0.15,
    minimum_ring_S_fraction: float = 0.05,
    minimum_q_amplitude_fraction: float = 0.05,
    other_centers: np.ndarray | None = None,
    neighbor_clearance_multiplier: float = WINDING_RING_NEIGHBOR_CLEARANCE_MULTIPLIER,
) -> tuple[float | None, str, tuple[float, ...], tuple[float, ...]]:
    center_array = np.asarray(center, dtype=np.float64)
    radii_array = np.asarray(radii, dtype=np.float64)
    if center_array.shape != (2,) or not np.isfinite(center_array).all():
        raise ValueError("winding center must contain two finite coordinates")
    if (
        radii_array.ndim != 1
        or radii_array.size == 0
        or not np.isfinite(radii_array).all()
        or np.any(radii_array <= 0.0)
    ):
        raise ValueError("winding radii must be a nonempty sequence of positive finite values")
    if not math.isfinite(neighbor_clearance_multiplier) or neighbor_clearance_multiplier < 2.0:
        raise ValueError("neighbor clearance multiplier must be finite and at least two")
    if other_centers is not None:
        other_array = np.asarray(other_centers, dtype=np.float64)
        if other_array.size == 0:
            other_array = np.empty((0, 2), dtype=np.float64)
        if other_array.ndim != 2 or other_array.shape[1] != 2:
            raise ValueError("other winding centers must have shape (N,2)")
        if not np.isfinite(other_array).all():
            raise ValueError("other winding centers must be finite")
        if len(other_array):
            nearest_distance = float(
                np.min(
                    periodic_distances(
                        center_array[None, :], other_array, sampler.lengths
                    )
                )
            )
            required_clearance = (
                neighbor_clearance_multiplier * float(np.max(radii_array))
                + float(np.max(sampler.spacing))
            )
            if nearest_distance <= required_clearance:
                return (
                    None,
                    "insufficient_winding_ring_clearance_from_other_core",
                    (),
                    (),
                )
    theta = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    q_windings: list[float] = []
    director_windings: list[float] = []
    integers: list[int] = []
    for radius in radii_array:
        x = float(center[0]) + float(radius) * np.cos(theta)
        y = float(center[1]) + float(radius) * np.sin(theta)
        q = sampler.sample_q(x, y)
        ring_S = np.asarray(S_from_Q(q), dtype=np.float64)
        if np.min(ring_S) < minimum_ring_S_fraction * S_bulk:
            return None, "ring_S_below_reliable_winding_threshold", tuple(q_windings), tuple(director_windings)
        q_complex = q[..., 0] - q[..., 3] + 2j * q[..., 1]
        if np.min(np.abs(q_complex)) < minimum_q_amplitude_fraction * S_bulk:
            return None, "in_plane_q_amplitude_below_reliable_winding_threshold", tuple(q_windings), tuple(director_windings)
        q_winding = wrapped_phase_winding(np.angle(q_complex))

        _, eigenvectors = np.linalg.eigh(compact_q_matrix(q))
        director = eigenvectors[..., :, -1]
        projection = director[..., 0] + 1j * director[..., 1]
        projection_norm = np.abs(projection)
        if np.min(projection_norm) < minimum_projection:
            return None, "director_projection_too_small", tuple(q_windings), tuple(director_windings)
        director_phase = np.angle((projection / projection_norm) ** 2)
        director_winding = wrapped_phase_winding(director_phase)
        q_windings.append(q_winding)
        director_windings.append(director_winding)
        nearest_q = int(np.rint(q_winding))
        nearest_director = int(np.rint(director_winding))
        if (
            abs(q_winding - nearest_q) > winding_tolerance
            or abs(director_winding - nearest_director) > winding_tolerance
            or nearest_q != nearest_director
            or nearest_q not in (-1, 1)
        ):
            return None, "inconsistent_or_non_half_winding", tuple(q_windings), tuple(director_windings)
        integers.append(nearest_q)
    if len(set(integers)) != 1:
        return None, "winding_changes_with_ring_radius", tuple(q_windings), tuple(director_windings)
    return 0.5 * integers[0], "reliable_double_winding", tuple(q_windings), tuple(director_windings)


def refine_core_center(
    sampler: PeriodicQSampler,
    initial_xy: Sequence[float],
    *,
    search_half_width: float,
    return_status: bool = False,
) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, str]:
    start = np.asarray(initial_xy, dtype=np.float64)

    def objective(unwrapped: np.ndarray) -> float:
        return float(sampler.sample_S(unwrapped[0], unwrapped[1]))

    bounds = [
        (start[0] - search_half_width, start[0] + search_half_width),
        (start[1] - search_half_width, start[1] + search_half_width),
    ]
    result = minimize(
        objective,
        start,
        method="Powell",
        bounds=bounds,
        options={"xtol": 1.0e-7, "ftol": 1.0e-12, "maxiter": 120},
    )
    center = np.mod(result.x if result.success else start, sampler.lengths)
    status = "optimized" if result.success else "optimizer_failed_using_seed"
    basic = (center, objective(center))
    return (*basic, status) if return_status else basic


def detect_core_candidates(
    q_plane: np.ndarray,
    lengths: Sequence[float],
    *,
    S_bulk: float,
    expected_count: int,
    interpolation_order: int = DEFAULT_INTERPOLATION_ORDER,
    angular_samples: int = DEFAULT_ANGULAR_SAMPLES,
) -> tuple[list[CoreCandidate], PeriodicQSampler]:
    q = np.asarray(q_plane, dtype=np.float64)
    sampler = PeriodicQSampler(q, lengths, order=interpolation_order)
    S_grid = np.asarray(S_from_Q(q), dtype=np.float64)
    nx, ny = S_grid.shape
    spacing = np.asarray(lengths, dtype=np.float64) / np.asarray((nx, ny), dtype=np.float64)
    local_minimum = S_grid <= ndimage.minimum_filter(S_grid, size=3, mode="wrap") + 1.0e-14
    threshold = min(0.8 * float(S_bulk), float(np.quantile(S_grid, 0.05)))
    indices = np.argwhere(local_minimum & (S_grid <= threshold))
    if len(indices) < expected_count:
        indices = np.argwhere(local_minimum)
    indices = sorted(indices, key=lambda ij: float(S_grid[tuple(ij)]))

    centers: list[np.ndarray] = []
    provisional: list[dict[str, Any]] = []
    minimum_separation = max(0.5, 1.5 * float(np.max(spacing)))
    max_candidates = max(expected_count * 4, expected_count + 8)
    for i, j in indices:
        seed = np.asarray(((i + 0.5) * spacing[0], (j + 0.5) * spacing[1]))
        if centers:
            distances = periodic_distances(seed[None, :], np.asarray(centers), lengths)[0]
            if float(np.min(distances)) < minimum_separation:
                continue
        search_half_width = max(1.5 * float(np.max(spacing)), 0.35)
        seed_S = float(sampler.sample_S(seed[0], seed[1]))
        center, S_min, fit_status = refine_core_center(
            sampler,
            seed,
            search_half_width=search_half_width,
            return_status=True,
        )
        fit_delta = minimum_image_delta(center - seed, sampler.lengths)
        boundary_margin_fraction = max(
            0.0,
            1.0 - float(np.max(np.abs(fit_delta))) / search_half_width,
        )
        if fit_status == "optimized" and boundary_margin_fraction < 1.0e-3:
            fit_status = "optimized_boundary_hit"
        if centers:
            distances = periodic_distances(center[None, :], np.asarray(centers), lengths)[0]
            if float(np.min(distances)) < minimum_separation:
                continue
        centers.append(center)
        provisional.append(
            {
                "center": center,
                "S_min": float(S_min),
                "center_fit_status": fit_status,
                "center_fit_initial_S": seed_S,
                "center_fit_objective_improvement": seed_S - float(S_min),
                "center_fit_boundary_margin_fraction": boundary_margin_fraction,
            }
        )
        if len(provisional) >= max_candidates:
            break

    candidates: list[CoreCandidate] = []
    all_centers = np.asarray(centers, dtype=np.float64)
    winding_radii = (1.5, 2.0)
    required_clearance = (
        WINDING_RING_NEIGHBOR_CLEARANCE_MULTIPLIER * max(winding_radii)
        + float(np.max(sampler.spacing))
    )
    for index, record in enumerate(provisional):
        center = np.asarray(record["center"], dtype=np.float64)
        other_centers = np.delete(all_centers, index, axis=0)
        nearest_other_distance = (
            float(
                np.min(
                    periodic_distances(
                        center[None, :], other_centers, sampler.lengths
                    )
                )
            )
            if len(other_centers)
            else math.nan
        )
        charge, reason, q_winding, director_winding = classify_candidate_charge(
            sampler,
            center,
            S_bulk=S_bulk,
            radii=winding_radii,
            angular_samples=angular_samples,
            other_centers=other_centers,
        )
        candidates.append(
            CoreCandidate(
                x=float(center[0]),
                y=float(center[1]),
                S_min=float(record["S_min"]),
                charge=charge,
                charge_reason=reason,
                q_windings=q_winding,
                director_windings=director_winding,
                center_fit_status=str(record["center_fit_status"]),
                center_fit_initial_S=float(record["center_fit_initial_S"]),
                center_fit_objective_improvement=float(
                    record["center_fit_objective_improvement"]
                ),
                center_fit_boundary_margin_fraction=float(
                    record["center_fit_boundary_margin_fraction"]
                ),
                nearest_other_core_distance=nearest_other_distance,
                winding_ring_required_clearance=required_clearance,
                winding_ring_isolation_status=(
                    "isolated"
                    if math.isfinite(nearest_other_distance)
                    and nearest_other_distance > required_clearance
                    else (
                        "not_isolated"
                        if math.isfinite(nearest_other_distance)
                        else "no_other_candidate"
                    )
                ),
            )
        )
    return candidates, sampler


def charge_aware_hungarian(
    candidates: Sequence[CoreCandidate],
    reference_positions: np.ndarray,
    reference_charges: np.ndarray,
    lengths: Sequence[float],
    *,
    maximum_distance: float,
) -> tuple[dict[int, int], set[int]]:
    """Return reference->candidate matches and all unmatched candidate indices."""
    matches: dict[int, int] = {}
    unmatched = set(range(len(candidates)))
    for charge in (-0.5, 0.5):
        candidate_indices = [
            index for index, candidate in enumerate(candidates) if candidate.charge == charge
        ]
        reference_indices = np.flatnonzero(np.asarray(reference_charges) == charge).tolist()
        if not candidate_indices or not reference_indices:
            continue
        detected = np.asarray([[candidates[index].x, candidates[index].y] for index in candidate_indices])
        expected = np.asarray(reference_positions, dtype=np.float64)[reference_indices]
        real_cost = periodic_distances(detected, expected, lengths)
        n_detected, n_reference = real_cost.shape
        huge = maximum_distance * 1.0e6
        cost = np.full((n_detected, n_reference + n_detected), huge, dtype=np.float64)
        cost[:, :n_reference] = np.where(real_cost <= maximum_distance, real_cost, huge)
        for local_index in range(n_detected):
            cost[local_index, n_reference + local_index] = maximum_distance + 1.0e-10
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            if column < n_reference and real_cost[row, column] <= maximum_distance:
                reference_index = reference_indices[column]
                candidate_index = candidate_indices[row]
                matches[reference_index] = candidate_index
                unmatched.discard(candidate_index)
    return matches, unmatched


def first_outward_crossing(
    radii: np.ndarray,
    profile: np.ndarray,
    threshold: float,
) -> RadiusCrossing:
    r = np.asarray(radii, dtype=np.float64)
    values = np.asarray(profile, dtype=np.float64)
    if r.ndim != 1 or values.ndim != 1 or r.shape != values.shape:
        return RadiusCrossing(math.nan, "invalid_profile_shape", 0)
    valid = np.isfinite(r) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return RadiusCrossing(math.nan, "insufficient_profile_samples", 0)
    if not math.isfinite(threshold):
        return RadiusCrossing(math.nan, "invalid_threshold", 0)
    if not valid[0]:
        return RadiusCrossing(math.nan, "invalid_center_profile_sample", 0)
    if np.any(np.diff(r[np.isfinite(r)]) <= 0.0):
        return RadiusCrossing(math.nan, "radii_not_strictly_increasing", 0)
    if values[0] >= threshold:
        return RadiusCrossing(math.nan, "center_at_or_above_threshold", 0)
    adjacent_valid = valid[:-1] & valid[1:]
    crossing_indices = np.flatnonzero(
        adjacent_valid & (values[:-1] < threshold) & (values[1:] >= threshold)
    )
    valid_indices = np.flatnonzero(valid)
    gap_brackets_crossing = any(
        right > left + 1
        and values[left] < threshold
        and values[right] >= threshold
        for left, right in zip(valid_indices, valid_indices[1:])
    )
    if gap_brackets_crossing:
        return RadiusCrossing(
            math.nan,
            "threshold_crossing_bracket_interrupted_by_invalid_profile_gap",
            int(len(crossing_indices)),
        )
    if len(crossing_indices) and np.any(~valid[: int(crossing_indices[0]) + 1]):
        return RadiusCrossing(
            math.nan,
            "profile_gap_before_threshold_crossing",
            int(len(crossing_indices)),
        )
    if len(crossing_indices) == 0:
        if np.any(~valid[1:]):
            return RadiusCrossing(
                math.nan,
                "insufficient_contiguous_profile_samples",
                0,
            )
        return RadiusCrossing(math.nan, "threshold_not_reached_by_rmax", 0)
    if len(crossing_indices) > 1:
        return RadiusCrossing(
            math.nan,
            "multiple_outward_crossings_unstable_branch",
            int(len(crossing_indices)),
        )
    index = int(crossing_indices[0])
    denominator = values[index + 1] - values[index]
    fraction = 0.0 if denominator == 0.0 else (threshold - values[index]) / denominator
    value = r[index] + fraction * (r[index + 1] - r[index])
    return RadiusCrossing(float(value), "ok", int(len(crossing_indices)))


def local_reference_quality(
    S_min: float,
    S_far: float,
    S_bulk: float,
) -> tuple[bool, float, float, str]:
    """Validate the span used to normalize local core-radius metrics."""
    if not all(math.isfinite(value) for value in (S_min, S_far, S_bulk)) or S_bulk <= 0.0:
        return False, math.nan, math.nan, "invalid_local_reference_values"
    scale = max(abs(float(S_bulk)), abs(float(S_far)), abs(float(S_min)), 1.0)
    minimum_span = max(
        LOCAL_REFERENCE_MINIMUM_SPAN_FRACTION * abs(float(S_bulk)),
        256.0 * np.finfo(np.float64).eps * scale,
    )
    span = float(S_far - S_min)
    if span <= 0.0:
        return False, span, minimum_span, "invalid_S_far_not_above_S_min"
    if span < minimum_span:
        return (
            False,
            span,
            minimum_span,
            "local_reference_span_below_minimum_fraction_of_S_bulk",
        )
    return True, span, minimum_span, "ok"


def _owned_mask(
    x: np.ndarray,
    y: np.ndarray,
    centers: np.ndarray,
    target_index: int,
    lengths: Sequence[float],
) -> np.ndarray:
    points = np.column_stack((np.asarray(x).ravel(), np.asarray(y).ravel()))
    ownership = np.argmin(periodic_distances(points, centers, lengths), axis=1)
    return (ownership == target_index).reshape(np.broadcast_shapes(np.shape(x), np.shape(y)))


def measure_core_profile(
    sampler: PeriodicQSampler,
    center: Sequence[float],
    *,
    all_centers: np.ndarray,
    center_index: int,
    S_bulk: float,
    r_max: float,
    dr: float,
    angular_samples: int,
    use_local_reference: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    radii = np.arange(0.0, r_max + 0.5 * dr, dr, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    x = float(center[0]) + radii[:, None] * np.cos(angles)[None, :]
    y = float(center[1]) + radii[:, None] * np.sin(angles)[None, :]
    q = sampler.sample_q(x, y)
    S = np.asarray(S_from_Q(q), dtype=np.float64)
    owned = _owned_mask(x, y, all_centers, center_index, sampler.lengths)
    profile_median = np.full(len(radii), np.nan)
    profile_mean = np.full(len(radii), np.nan)
    profile_std = np.full(len(radii), np.nan)
    profile_count = np.sum(owned, axis=1, dtype=np.int64)
    minimum_profile_samples = math.ceil(
        PROFILE_MINIMUM_ANGULAR_FRACTION * angular_samples
    )
    for index in range(len(radii)):
        values = S[index, owned[index]]
        if values.size >= minimum_profile_samples:
            profile_median[index] = float(np.median(values))
            profile_mean[index] = float(np.mean(values, dtype=np.float64))
            profile_std[index] = float(np.std(values, dtype=np.float64))

    S_min = float(sampler.sample_S(float(center[0]), float(center[1])))
    far_ring = (radii[:, None] >= 3.0) & (radii[:, None] <= 4.0)
    far_mask = owned & far_ring
    far_candidates = S[far_mask]
    far_values = far_candidates[
        np.isfinite(far_candidates)
        & (far_candidates >= S_FAR_LOW_ORDER_CUTOFF_FRACTION * S_bulk)
    ]
    far_possible = int(np.count_nonzero(far_ring) * angular_samples)
    S_far_valid_fraction = float(far_values.size / far_possible) if far_possible else 0.0
    if far_values.size == 0:
        S_far = S_far_mad = math.nan
        S_far_reason = "no_owned_finite_far_ring_samples_above_low_order_cutoff"
    elif S_far_valid_fraction < S_FAR_MINIMUM_VALID_FRACTION:
        S_far = S_far_mad = math.nan
        S_far_reason = "insufficient_far_ring_valid_fraction"
    else:
        S_far = float(np.median(far_values))
        S_far_mad = float(np.median(np.abs(far_values - S_far)))
        S_far_reason = "ok"
    absolute_50 = first_outward_crossing(radii, profile_median, 0.5 * S_bulk)
    absolute_90 = first_outward_crossing(radii, profile_median, 0.9 * S_bulk)
    (
        local_reference_well_conditioned,
        local_reference_span,
        local_reference_minimum_span,
        local_reference_quality_reason,
    ) = local_reference_quality(S_min, S_far, S_bulk)
    if not local_reference_well_conditioned:
        local_50 = RadiusCrossing(math.nan, local_reference_quality_reason, 0)
        local_90 = RadiusCrossing(math.nan, local_reference_quality_reason, 0)
    else:
        local_50 = first_outward_crossing(
            radii, profile_median, S_min + 0.5 * (S_far - S_min)
        )
        local_90 = first_outward_crossing(
            radii, profile_median, S_min + 0.9 * (S_far - S_min)
        )

    midpoint_radii = np.arange(0.5 * dr, r_max, dr, dtype=np.float64)
    xm = float(center[0]) + midpoint_radii[:, None] * np.cos(angles)[None, :]
    ym = float(center[1]) + midpoint_radii[:, None] * np.sin(angles)[None, :]
    qm = sampler.sample_q(xm, ym)
    Sm = np.asarray(S_from_Q(qm), dtype=np.float64)
    owned_mid = _owned_mask(xm, ym, all_centers, center_index, sampler.lengths)
    weights = midpoint_radii[:, None] * dr * (2.0 * np.pi / angular_samples)
    weights = np.broadcast_to(weights, Sm.shape)
    absolute_deficit = float(
        np.sum(np.where(owned_mid, np.maximum(1.0 - Sm / S_bulk, 0.0) * weights, 0.0))
    )
    low_absolute = float(np.sum(np.where(owned_mid & (Sm < 0.5 * S_bulk), weights, 0.0)))
    if local_reference_well_conditioned:
        local_deficit = float(
            np.sum(np.where(owned_mid, np.maximum(1.0 - Sm / S_far, 0.0) * weights, 0.0))
        )
        low_local = float(np.sum(np.where(owned_mid & (Sm < 0.5 * S_far), weights, 0.0)))
        local_integral_reason = "ok"
    else:
        local_deficit = math.nan
        low_local = math.nan
        local_integral_reason = local_reference_quality_reason

    biaxiality_scale = S_far if use_local_reference else S_bulk
    biaxiality_scale_valid = math.isfinite(biaxiality_scale) and biaxiality_scale > 0.0
    if biaxiality_scale_valid:
        biaxiality, biaxiality_valid = biaxiality_from_Q(
            qm, S_scale=biaxiality_scale
        )
    else:
        biaxiality = np.full(Sm.shape, np.nan, dtype=np.float64)
        biaxiality_valid = np.zeros(Sm.shape, dtype=bool)
    biaxiality_owned = biaxiality[owned_mid & biaxiality_valid]
    owned_count = int(np.count_nonzero(owned_mid))
    biaxiality_valid_fraction = (
        float(biaxiality_owned.size / owned_count) if owned_count else 0.0
    )
    sensitivity: dict[str, float] = {}
    for label, cutoff in (("lo", 0.1e-12), ("hi", 10.0e-12)):
        if biaxiality_scale_valid:
            values, valid = biaxiality_from_Q(
                qm,
                S_scale=biaxiality_scale,
                relative_trq2_cutoff=cutoff,
            )
        else:
            values = np.full(Sm.shape, np.nan, dtype=np.float64)
            valid = np.zeros(Sm.shape, dtype=bool)
        selected = values[owned_mid & valid]
        sensitivity[f"biaxiality_mean_cutoff_{label}"] = (
            float(np.mean(selected)) if selected.size else math.nan
        )
        sensitivity[f"biaxiality_valid_fraction_cutoff_{label}"] = (
            float(selected.size / owned_count) if owned_count else 0.0
        )

    contour_r50 = []
    for angle_index in range(angular_samples):
        angular_profile = np.where(owned[:, angle_index], S[:, angle_index], np.nan)
        contour_r50.append(
            first_outward_crossing(radii, angular_profile, 0.5 * S_bulk).value
        )
    contour_r50 = np.asarray(contour_r50, dtype=np.float64)
    valid_contour = contour_r50[np.isfinite(contour_r50)]
    contour_median = float(np.median(valid_contour)) if valid_contour.size else math.nan
    contour_p10 = float(np.quantile(valid_contour, 0.1)) if valid_contour.size else math.nan
    contour_p90 = float(np.quantile(valid_contour, 0.9)) if valid_contour.size else math.nan
    contour_anisotropy = (
        (contour_p90 - contour_p10) / (2.0 * contour_median)
        if valid_contour.size and contour_median > 0.0
        else math.nan
    )
    finite_profile = profile_median[np.isfinite(profile_median)]
    monotonicity_tolerance = 1.0e-4 * S_bulk
    nonmonotone_steps = int(
        np.count_nonzero(np.diff(finite_profile) < -monotonicity_tolerance)
    )
    metrics: dict[str, Any] = {
        "S_min": S_min,
        "S_far": S_far,
        "S_far_mad": S_far_mad,
        "S_far_valid_fraction": S_far_valid_fraction,
        "S_far_reason": S_far_reason,
        "S_far_candidate_samples": int(far_candidates.size),
        "S_far_valid_samples": int(far_values.size),
        "S_far_low_order_rejected_samples": int(
            far_candidates.size - far_values.size
        ),
        "local_reference_well_conditioned": local_reference_well_conditioned,
        "local_reference_span": local_reference_span,
        "local_reference_minimum_span": local_reference_minimum_span,
        "local_reference_span_fraction_of_S_bulk": (
            local_reference_span / S_bulk
            if math.isfinite(local_reference_span) and S_bulk > 0.0
            else math.nan
        ),
        "local_reference_quality_reason": local_reference_quality_reason,
        "r50_absolute": absolute_50.value,
        "r50_absolute_reason": absolute_50.reason,
        "r50_absolute_crossing_count": absolute_50.crossing_count,
        "r90_absolute": absolute_90.value,
        "r90_absolute_reason": absolute_90.reason,
        "r90_absolute_crossing_count": absolute_90.crossing_count,
        "r50_local": local_50.value,
        "r50_local_reason": local_50.reason,
        "r50_local_crossing_count": local_50.crossing_count,
        "r90_local": local_90.value,
        "r90_local_reason": local_90.reason,
        "r90_local_crossing_count": local_90.crossing_count,
        "core_deficit_absolute": absolute_deficit,
        "core_deficit_local": local_deficit,
        "core_deficit_local_reason": local_integral_reason,
        "low_s_area_absolute": low_absolute,
        "low_s_area_local": low_local,
        "low_s_area_local_reason": local_integral_reason,
        "profile_effective_samples": int(np.sum(profile_count)),
        "profile_min_angular_samples": int(np.min(profile_count)),
        "profile_minimum_required_angular_samples": minimum_profile_samples,
        "profile_valid_radial_fraction": float(
            np.count_nonzero(np.isfinite(profile_median)) / len(profile_median)
        ),
        "profile_nonmonotone_step_count": nonmonotone_steps,
        "profile_monotonicity_tolerance": monotonicity_tolerance,
        "profile_monotonicity_reason": (
            "monotone_within_tolerance"
            if nonmonotone_steps == 0
            else "nonmonotone_steps_present"
        ),
        "r50_contour_valid_fraction": float(valid_contour.size / angular_samples),
        "r50_contour_median": contour_median,
        "r50_contour_p10": contour_p10,
        "r50_contour_p90": contour_p90,
        "r50_contour_anisotropy": contour_anisotropy,
        "biaxiality_valid_samples": int(biaxiality_owned.size),
        "biaxiality_valid_fraction": biaxiality_valid_fraction,
        "biaxiality_S_reference": biaxiality_scale,
        "biaxiality_S_reference_reason": (
            "S_far_for_final_state"
            if use_local_reference and biaxiality_scale_valid
            else (
                "S_bulk_for_initial_state"
                if not use_local_reference
                else "invalid_S_far_biaxiality_masked"
            )
        ),
        "biaxiality_mean": float(np.mean(biaxiality_owned)) if biaxiality_owned.size else math.nan,
        "biaxiality_median": float(np.median(biaxiality_owned)) if biaxiality_owned.size else math.nan,
        "biaxiality_max": float(np.max(biaxiality_owned)) if biaxiality_owned.size else math.nan,
        **sensitivity,
    }
    profile = {
        "radii": radii,
        "median": profile_median,
        "mean": profile_mean,
        "std": profile_std,
        "count": profile_count,
    }
    return metrics, profile


def continuous_reference_profile(
    defect_index: int,
    positions: np.ndarray,
    lengths: Sequence[float],
    *,
    core_radius: float,
    S_initial: float,
    S_bulk: float,
    r_max: float,
    dr: float,
    angular_samples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    radii = np.arange(0.0, r_max + 0.5 * dr, dr, dtype=np.float64)
    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    center = positions[defect_index]
    x = center[0] + radii[:, None] * np.cos(angles)[None, :]
    y = center[1] + radii[:, None] * np.sin(angles)[None, :]
    S = continuous_multidefect_S(
        x,
        y,
        positions=positions,
        lengths=lengths,
        core_radius=core_radius,
        S_initial=S_initial,
    )
    owned = _owned_mask(x, y, positions, defect_index, lengths)
    median = np.asarray(
        [
            np.median(S[index, owned[index]])
            if np.count_nonzero(owned[index])
            >= math.ceil(PROFILE_MINIMUM_ANGULAR_FRACTION * angular_samples)
            else np.nan
            for index in range(len(radii))
        ]
    )
    mean = np.asarray(
        [
            np.mean(S[index, owned[index]])
            if np.count_nonzero(owned[index])
            >= math.ceil(PROFILE_MINIMUM_ANGULAR_FRACTION * angular_samples)
            else np.nan
            for index in range(len(radii))
        ]
    )
    std = np.asarray(
        [
            np.std(S[index, owned[index]])
            if np.count_nonzero(owned[index])
            >= math.ceil(PROFILE_MINIMUM_ANGULAR_FRACTION * angular_samples)
            else np.nan
            for index in range(len(radii))
        ]
    )
    count = np.sum(owned, axis=1, dtype=np.int64)
    far_ring = (radii[:, None] >= 3.0) & (radii[:, None] <= 4.0)
    far_candidates = S[owned & far_ring]
    far = far_candidates[
        np.isfinite(far_candidates)
        & (far_candidates >= S_FAR_LOW_ORDER_CUTOFF_FRACTION * S_bulk)
    ]
    far_possible = int(np.count_nonzero(far_ring) * angular_samples)
    far_valid_fraction = float(far.size / far_possible) if far_possible else 0.0
    if far.size == 0:
        S_far = S_far_mad = math.nan
        S_far_reason = "no_owned_finite_far_ring_samples_above_low_order_cutoff"
    elif far_valid_fraction < S_FAR_MINIMUM_VALID_FRACTION:
        S_far = S_far_mad = math.nan
        S_far_reason = "insufficient_far_ring_valid_fraction"
    else:
        S_far = float(np.median(far))
        S_far_mad = float(np.median(np.abs(far - S_far)))
        S_far_reason = "ok"
    (
        local_reference_well_conditioned,
        local_reference_span,
        local_reference_minimum_span,
        local_reference_quality_reason,
    ) = local_reference_quality(0.0, S_far, S_bulk)
    crossings = {
        "r50_absolute": first_outward_crossing(radii, median, 0.5 * S_bulk),
        "r90_absolute": first_outward_crossing(radii, median, 0.9 * S_bulk),
        "r50_local": (
            first_outward_crossing(radii, median, 0.5 * S_far)
            if local_reference_well_conditioned
            else RadiusCrossing(math.nan, local_reference_quality_reason, 0)
        ),
        "r90_local": (
            first_outward_crossing(radii, median, 0.9 * S_far)
            if local_reference_well_conditioned
            else RadiusCrossing(math.nan, local_reference_quality_reason, 0)
        ),
    }
    midpoint_radii = np.arange(0.5 * dr, r_max, dr, dtype=np.float64)
    xm = center[0] + midpoint_radii[:, None] * np.cos(angles)[None, :]
    ym = center[1] + midpoint_radii[:, None] * np.sin(angles)[None, :]
    Sm = continuous_multidefect_S(
        xm,
        ym,
        positions=positions,
        lengths=lengths,
        core_radius=core_radius,
        S_initial=S_initial,
    )
    owned_mid = _owned_mask(xm, ym, positions, defect_index, lengths)
    weights = np.broadcast_to(
        midpoint_radii[:, None] * dr * (2.0 * np.pi / angular_samples),
        Sm.shape,
    )
    metrics: dict[str, Any] = {
        "S_min": 0.0,
        "S_far": S_far,
        "S_far_mad": S_far_mad,
        "S_far_valid_fraction": far_valid_fraction,
        "S_far_reason": S_far_reason,
        "S_far_candidate_samples": int(far_candidates.size),
        "S_far_valid_samples": int(far.size),
        "S_far_low_order_rejected_samples": int(far_candidates.size - far.size),
        "local_reference_well_conditioned": local_reference_well_conditioned,
        "local_reference_span": local_reference_span,
        "local_reference_minimum_span": local_reference_minimum_span,
        "local_reference_span_fraction_of_S_bulk": (
            local_reference_span / S_bulk
            if math.isfinite(local_reference_span) and S_bulk > 0.0
            else math.nan
        ),
        "local_reference_quality_reason": local_reference_quality_reason,
        "core_deficit_absolute": float(
            np.sum(np.where(owned_mid, np.maximum(1.0 - Sm / S_bulk, 0.0) * weights, 0.0))
        ),
        "core_deficit_local": (
            float(np.sum(np.where(owned_mid, np.maximum(1.0 - Sm / S_far, 0.0) * weights, 0.0)))
            if local_reference_well_conditioned
            else math.nan
        ),
        "core_deficit_local_reason": (
            "ok" if local_reference_well_conditioned else local_reference_quality_reason
        ),
        "low_s_area_absolute": float(
            np.sum(np.where(owned_mid & (Sm < 0.5 * S_bulk), weights, 0.0))
        ),
        "low_s_area_local": (
            float(np.sum(np.where(owned_mid & (Sm < 0.5 * S_far), weights, 0.0)))
            if local_reference_well_conditioned
            else math.nan
        ),
        "low_s_area_local_reason": (
            "ok" if local_reference_well_conditioned else local_reference_quality_reason
        ),
        "center_displacement": 0.0,
        "biaxiality_mean": 0.0,
        "biaxiality_valid_fraction": 1.0,
    }
    for name, crossing in crossings.items():
        metrics[name] = crossing.value
        metrics[f"{name}_reason"] = crossing.reason
        metrics[f"{name}_crossing_count"] = crossing.crossing_count
    return metrics, {"radii": radii, "median": median, "mean": mean, "std": std, "count": count}


def _candidate_centers(candidates: Sequence[CoreCandidate]) -> np.ndarray:
    if not candidates:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([[candidate.x, candidate.y] for candidate in candidates], dtype=np.float64)


def analyze_q_plane(
    q_plane: np.ndarray,
    run: CoreRun,
    *,
    state: str,
    step: int,
    z_fraction: float | None,
    S_bulk: float,
    r_max: float,
    dr: float,
    angular_samples: int,
    match_radius: float,
    interpolation_order: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[CoreCandidate]]:
    candidates, sampler = detect_core_candidates(
        q_plane,
        run.lengths[:2],
        S_bulk=S_bulk,
        expected_count=len(run.defect_positions),
        interpolation_order=interpolation_order,
        angular_samples=angular_samples,
    )
    matches, unmatched = charge_aware_hungarian(
        candidates,
        run.defect_positions,
        run.defect_charges,
        run.lengths[:2],
        maximum_distance=match_radius,
    )
    centers = _candidate_centers(candidates)
    matched_candidate_indices = sorted(set(matches.values()))
    ownership_centers = (
        centers[matched_candidate_indices]
        if matched_candidate_indices
        else np.empty((0, 2), dtype=np.float64)
    )
    ownership_index = {
        candidate_index: local_index
        for local_index, candidate_index in enumerate(matched_candidate_indices)
    }
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    common = {
        "state": state,
        "resolution": run.label,
        "run_dir": str(run.artifact.directory),
        "step": step,
        "z_fraction": z_fraction,
        "nx": run.shape[0],
        "ny": run.shape[1],
        "nz": run.shape[2],
        "dt": run.artifact.dt,
    }
    for defect_id, (reference_position, initial_charge) in enumerate(
        zip(run.defect_positions, run.defect_charges)
    ):
        row: dict[str, Any] = {
            **common,
            "defect_id": defect_id,
            "initial_charge": float(initial_charge),
            "detected_charge": math.nan,
            "match_status": "missing",
            "match_reason": "no_reliable_same_charge_candidate_within_radius",
            "initial_x": float(reference_position[0]),
            "initial_y": float(reference_position[1]),
            "center_x": math.nan,
            "center_y": math.nan,
            "center_displacement": math.nan,
            "center_fit_status": "missing",
        }
        if defect_id in matches:
            candidate_index = matches[defect_id]
            candidate = candidates[candidate_index]
            delta = minimum_image_delta(
                np.asarray((candidate.x, candidate.y)) - reference_position,
                run.lengths[:2],
            )
            metrics, profile = measure_core_profile(
                sampler,
                (candidate.x, candidate.y),
                all_centers=ownership_centers,
                center_index=ownership_index[candidate_index],
                S_bulk=S_bulk,
                r_max=r_max,
                dr=dr,
                angular_samples=angular_samples,
                use_local_reference=state == FINAL_STATE,
            )
            row.update(
                detected_charge=float(candidate.charge),
                match_status="matched",
                match_reason="same_charge_hungarian_within_radius",
                center_x=candidate.x,
                center_y=candidate.y,
                center_displacement=float(np.linalg.norm(delta)),
                center_fit_status=candidate.center_fit_status,
                center_fit_initial_S=candidate.center_fit_initial_S,
                center_fit_objective_improvement=candidate.center_fit_objective_improvement,
                center_fit_boundary_margin_fraction=candidate.center_fit_boundary_margin_fraction,
                nearest_other_core_distance=candidate.nearest_other_core_distance,
                winding_ring_required_clearance=candidate.winding_ring_required_clearance,
                winding_ring_isolation_status=candidate.winding_ring_isolation_status,
                charge_reason=candidate.charge_reason,
                q_windings=";".join(f"{value:.8g}" for value in candidate.q_windings),
                director_windings=";".join(f"{value:.8g}" for value in candidate.director_windings),
                **metrics,
            )
            profiles.append(
                {
                    **common,
                    "defect_id": defect_id,
                    "charge": float(initial_charge),
                    **profile,
                }
            )
        rows.append(row)

    for candidate_index in sorted(unmatched):
        candidate = candidates[candidate_index]
        rows.append(
            {
                **common,
                "defect_id": "",
                "initial_charge": math.nan,
                "detected_charge": math.nan if candidate.charge is None else candidate.charge,
                "match_status": "ambiguous" if candidate.charge is None else "unmatched",
                "match_reason": candidate.charge_reason,
                "initial_x": math.nan,
                "initial_y": math.nan,
                "center_x": candidate.x,
                "center_y": candidate.y,
                "center_displacement": math.nan,
                "center_fit_status": candidate.center_fit_status,
                "center_fit_initial_S": candidate.center_fit_initial_S,
                "center_fit_objective_improvement": candidate.center_fit_objective_improvement,
                "center_fit_boundary_margin_fraction": candidate.center_fit_boundary_margin_fraction,
                "nearest_other_core_distance": candidate.nearest_other_core_distance,
                "winding_ring_required_clearance": candidate.winding_ring_required_clearance,
                "winding_ring_isolation_status": candidate.winding_ring_isolation_status,
                "S_min": candidate.S_min,
                "q_windings": ";".join(f"{value:.8g}" for value in candidate.q_windings),
                "director_windings": ";".join(f"{value:.8g}" for value in candidate.director_windings),
            }
        )
    return rows, profiles, candidates


def q2d_interpolation_calibration(
    sampler: PeriodicQSampler,
    positions: np.ndarray,
    lengths: Sequence[float],
    *,
    core_radius: float,
    S_initial: float,
    r_max: float,
    dr: float,
    angular_samples: int,
) -> dict[str, float]:
    radii = np.arange(0.0, r_max + 0.5 * dr, dr)
    angles = np.linspace(0.0, 2.0 * np.pi, angular_samples, endpoint=False)
    errors = []
    for center in positions:
        x = center[0] + radii[:, None] * np.cos(angles)[None, :]
        y = center[1] + radii[:, None] * np.sin(angles)[None, :]
        observed = sampler.sample_S(x, y)
        exact = continuous_multidefect_S(
            x,
            y,
            positions=positions,
            lengths=lengths,
            core_radius=core_radius,
            S_initial=S_initial,
        )
        errors.append((observed - exact).ravel())
    error = np.concatenate(errors)
    return {
        "sample_count": int(error.size),
        "S_error_rms": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
        "S_error_linf": float(np.max(np.abs(error))),
    }


def periodic_interpolation_synthetic_calibration(
    shape: tuple[int, int],
    lengths: Sequence[float],
    *,
    order: int,
) -> dict[str, Any]:
    """Measure periodic spline error on analytic Fourier modes, including seams."""
    nx, ny = shape
    Lx, Ly = (float(lengths[0]), float(lengths[1]))
    x = (np.arange(nx, dtype=np.float64) + 0.5) * Lx / nx
    y = (np.arange(ny, dtype=np.float64) + 0.5) * Ly / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    S = 0.25 + 0.03 * np.cos(4.0 * np.pi * xx / Lx) + 0.02 * np.sin(
        6.0 * np.pi * yy / Ly
    )
    q = np.zeros((nx, ny, Q_COMPONENT_COUNT), dtype=np.float64)
    q[..., 0] = S
    q[..., 3] = -0.5 * S
    sampler = PeriodicQSampler(q, (Lx, Ly), order=order)
    parameter = np.linspace(0.0, 1.0, 4096, endpoint=False)
    query_x = Lx * (1.37 * parameter - 0.185)
    query_y = Ly * (0.83 * parameter + 0.271)
    query_x[:4] = (-1.0e-10, Lx - 1.0e-10, Lx + 1.0e-10, 2.0 * Lx + 1.0e-10)
    query_y[:4] = (0.37 * Ly,) * 4
    expected = 0.25 + 0.03 * np.cos(4.0 * np.pi * query_x / Lx) + 0.02 * np.sin(
        6.0 * np.pi * query_y / Ly
    )
    error = sampler.sample_S(query_x, query_y) - expected
    return {
        "shape": shape,
        "sample_count": int(error.size),
        "S_error_rms": float(np.sqrt(np.mean(error * error, dtype=np.float64))),
        "S_error_linf": float(np.max(np.abs(error))),
        "includes_periodic_seam_queries": True,
        "analytic_modes": "cos(4*pi*x/Lx) and sin(6*pi*y/Ly)",
    }


def interpolation_error_vs_profile_changes(
    calibrations: Sequence[Mapping[str, Any]],
    profile_changes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {str(item["resolution"]): item for item in calibrations}
    output: list[dict[str, Any]] = []
    for row in profile_changes:
        if row.get("aggregation") != "individual":
            continue
        coarse = lookup.get(str(row["coarse"]))
        fine = lookup.get(str(row["fine"]))
        if coarse is None or fine is None:
            continue
        calibration_rms = max(
            float(coarse["S_error_rms"]), float(fine["S_error_rms"])
        )
        profile_rms = float(row.get("weighted_difference_rms", math.nan))
        resolvable = (
            math.isfinite(profile_rms)
            and math.isfinite(calibration_rms)
            and calibration_rms < profile_rms
        )
        output.append(
            {
                "state": row["state"],
                "z_fraction": row["z_fraction"],
                "charge": row["charge"],
                "defect_id": row["defect_id"],
                "coarse": row["coarse"],
                "fine": row["fine"],
                "maximum_synthetic_interpolation_S_error_rms": calibration_rms,
                "paired_profile_weighted_difference_rms": profile_rms,
                "synthetic_interpolation_error_below_compared_profile_difference": resolvable,
                "calibration_scope": (
                    "analytic periodic low Fourier modes and seam; this is an interpolation "
                    "verification, not a bound on discrete field representation error"
                ),
            }
        )
    return output


def summarize_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("match_status") != "matched":
            continue
        key = (
            row.get("state"),
            row.get("resolution"),
            row.get("step"),
            row.get("z_fraction"),
            row.get("initial_charge"),
        )
        groups[key].append(row)
        groups[(*key[:-1], "all")].append(row)
    summaries: list[dict[str, Any]] = []
    for key, members in groups.items():
        summary = {
            "state": key[0],
            "resolution": key[1],
            "step": key[2],
            "z_fraction": key[3],
            "charge": key[4],
            "count": len(members),
            "matched_defect_ids": sorted(
                int(member["defect_id"])
                for member in members
                if isinstance(member.get("defect_id"), (int, np.integer))
            ),
        }
        for metric in CORE_METRICS:
            member_values = [
                (member.get("defect_id"), float(member.get(metric, math.nan)))
                for member in members
            ]
            finite_pairs = [pair for pair in member_values if math.isfinite(pair[1])]
            values = np.asarray([pair[1] for pair in finite_pairs], dtype=float)
            summary[f"{metric}_count"] = int(values.size)
            summary[f"{metric}_defect_ids"] = [
                int(defect_id)
                for defect_id, _ in finite_pairs
                if isinstance(defect_id, (int, np.integer))
            ]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values.size else math.nan
            summary[f"{metric}_std"] = float(np.std(values)) if values.size else math.nan
            summary[f"{metric}_median"] = float(np.median(values)) if values.size else math.nan
            median = summary[f"{metric}_median"]
            mad = (
                float(np.median(np.abs(values - median))) if values.size else math.nan
            )
            summary[f"{metric}_mad"] = mad
            if values.size:
                robust_threshold = max(
                    3.0 * 1.4826 * mad,
                    0.10 * abs(median),
                    32.0 * np.finfo(np.float64).eps,
                )
                summary[f"{metric}_outlier_defect_ids"] = [
                    defect_id
                    for defect_id, value in finite_pairs
                    if abs(value - median) > robust_threshold
                ]
            else:
                summary[f"{metric}_outlier_defect_ids"] = []
        summaries.append(summary)
    return summaries


def summarize_status_counts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Count initial-core outcomes separately from extra candidate outcomes."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("state") == "continuous_multidefect_reference":
            continue
        base = (
            row.get("state"),
            row.get("resolution"),
            row.get("step"),
            row.get("z_fraction"),
        )
        groups[(*base, "all")].append(row)
        initial_charge = row.get("initial_charge")
        detected_charge = row.get("detected_charge")
        charge: float | None = None
        try:
            if math.isfinite(float(initial_charge)):
                charge = float(initial_charge)
            elif math.isfinite(float(detected_charge)):
                charge = float(detected_charge)
        except (TypeError, ValueError):
            charge = None
        if charge in (-0.5, 0.5):
            groups[(*base, charge)].append(row)

    output: list[dict[str, Any]] = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        initial_rows = [
            row
            for row in members
            if isinstance(row.get("defect_id"), (int, np.integer))
        ]
        candidate_rows = [
            row
            for row in members
            if not isinstance(row.get("defect_id"), (int, np.integer))
        ]
        matched = [row for row in initial_rows if row.get("match_status") == "matched"]
        missing = [row for row in initial_rows if row.get("match_status") == "missing"]
        ambiguous = [
            row for row in candidate_rows if row.get("match_status") == "ambiguous"
        ]
        unmatched = [
            row for row in candidate_rows if row.get("match_status") == "unmatched"
        ]
        output.append(
            {
                "state": key[0],
                "resolution": key[1],
                "step": key[2],
                "z_fraction": key[3],
                "charge": key[4],
                "expected_initial_count": len(initial_rows),
                "matched_initial_count": len(matched),
                "missing_initial_count": len(missing),
                "missing_initial_ids": [int(row["defect_id"]) for row in missing],
                "ambiguous_candidate_count": len(ambiguous),
                "unmatched_reliable_candidate_count": len(unmatched),
                "extra_candidate_count": len(candidate_rows),
                "extra_candidate_centers": [
                    [float(row["center_x"]), float(row["center_y"])]
                    for row in candidate_rows
                ],
                "all_expected_initial_reliably_matched": bool(
                    initial_rows and len(matched) == len(initial_rows)
                ),
            }
        )
    return output


def relative_change(coarse: float, fine: float) -> float:
    if not math.isfinite(coarse) or not math.isfinite(fine):
        return math.nan
    denominator = abs(fine)
    if denominator > 0.0:
        return abs(fine - coarse) / denominator
    return 0.0 if coarse == fine else math.nan


def _trapezoid_integral(values: Any, coordinates: Any) -> Any:
    """Integrate on NumPy versions before and after the trapz rename/removal."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = getattr(np, "trapz", None)
    if integrate is None:
        raise RuntimeError("NumPy provides neither trapezoid nor trapz")
    return integrate(values, coordinates)


def weighted_profile_difference(
    coarse_member: Mapping[str, Any],
    fine_member: Mapping[str, Any],
    *,
    S_scale: float,
) -> dict[str, float]:
    """Compare two profiles on identical physical radii with polar weighting."""
    coarse_radii = np.asarray(coarse_member["radii"], dtype=np.float64)
    fine_radii = np.asarray(fine_member["radii"], dtype=np.float64)
    if not np.array_equal(coarse_radii, fine_radii):
        raise ValueError("profile comparisons require identical physical radii")
    coarse_profile = np.asarray(coarse_member["median"], dtype=np.float64)
    fine_profile = np.asarray(fine_member["median"], dtype=np.float64)
    if coarse_profile.shape != coarse_radii.shape or fine_profile.shape != fine_radii.shape:
        raise ValueError("profile median and radii shapes must agree")
    valid = (
        np.isfinite(coarse_radii)
        & np.isfinite(coarse_profile)
        & np.isfinite(fine_profile)
    )
    radii = coarse_radii[valid]
    if len(radii) < 2:
        return {
            "weighted_difference_rms": math.nan,
            "weighted_relative_l2": math.nan,
            "difference_linf": math.nan,
            "valid_radial_samples": int(len(radii)),
            "comparison_reason": "insufficient_common_profile_samples",
        }
    difference = coarse_profile[valid] - fine_profile[valid]
    numerator = max(
        float(_trapezoid_integral(difference * difference * radii, radii)), 0.0
    )
    fine_norm_squared = max(
        float(
            _trapezoid_integral(
                fine_profile[valid] * fine_profile[valid] * radii,
                radii,
            )
        ),
        0.0,
    )
    area_weight = float(_trapezoid_integral(radii, radii))
    if area_weight <= 0.0:
        return {
            "weighted_difference_rms": math.nan,
            "weighted_relative_l2": math.nan,
            "difference_linf": math.nan,
            "valid_radial_samples": int(len(radii)),
            "comparison_reason": "nonpositive_radial_weight",
        }
    numerator_norm = math.sqrt(numerator)
    denominator_norm = max(
        math.sqrt(fine_norm_squared),
        abs(float(S_scale)) * math.sqrt(area_weight),
        PROFILE_NORM_FLOOR,
    )
    return {
        "weighted_difference_rms": math.sqrt(numerator / area_weight),
        "weighted_relative_l2": numerator_norm / denominator_norm,
        "difference_linf": float(np.max(np.abs(difference))),
        "valid_radial_samples": int(len(radii)),
        "comparison_reason": "ok",
    }


def resample_profile(
    profile: Mapping[str, Any],
    target_radii: np.ndarray,
) -> dict[str, np.ndarray]:
    """Resample an already high-accuracy radial statistic onto common radii."""
    source_radii = np.asarray(profile["radii"], dtype=np.float64)
    targets = np.asarray(target_radii, dtype=np.float64)
    if (
        source_radii.ndim != 1
        or targets.ndim != 1
        or source_radii.size < 2
        or targets[0] < source_radii[0]
        or targets[-1] > source_radii[-1]
    ):
        raise ValueError("invalid radial grids for profile resampling")
    output = {"radii": targets.copy()}
    for field in ("median", "mean", "std"):
        values = np.asarray(profile[field], dtype=np.float64)
        valid = np.isfinite(source_radii) & np.isfinite(values)
        output[field] = (
            np.interp(targets, source_radii[valid], values[valid])
            if np.count_nonzero(valid) >= 2
            else np.full(targets.shape, np.nan, dtype=np.float64)
        )
    counts = np.asarray(profile["count"], dtype=np.float64)
    output["count"] = np.rint(np.interp(targets, source_radii, counts)).astype(np.int64)
    return output


def continuous_reference_errors(
    rows: Sequence[dict[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    continuous_metrics: Mapping[int, Mapping[str, Any]],
    continuous_profiles: Mapping[int, Mapping[str, Any]],
    *,
    S_scale: float,
) -> list[dict[str, Any]]:
    """Attach raw/projected scalar errors and return paired profile errors."""
    profile_errors: list[dict[str, Any]] = []
    profile_lookup: dict[tuple[Any, ...], dict[str, float]] = {}
    for profile in profiles:
        if profile.get("state") not in (RAW_STATE, PROJECTED_STATE):
            continue
        defect_id = int(profile["defect_id"])
        metrics = weighted_profile_difference(
            profile,
            continuous_profiles[defect_id],
            S_scale=S_scale,
        )
        result = {
            "state": profile["state"],
            "resolution": profile["resolution"],
            "step": profile["step"],
            "z_fraction": profile["z_fraction"],
            "defect_id": defect_id,
            "charge": float(profile["charge"]),
            **metrics,
        }
        profile_errors.append(result)
        profile_lookup[
            (
                profile["state"],
                profile["resolution"],
                profile["z_fraction"],
                defect_id,
            )
        ] = metrics

    for row in rows:
        if row.get("state") not in (RAW_STATE, PROJECTED_STATE):
            continue
        defect_id = row.get("defect_id")
        if row.get("match_status") != "matched" or not isinstance(
            defect_id, (int, np.integer)
        ):
            continue
        reference = continuous_metrics[int(defect_id)]
        for metric in REFERENCE_METRICS:
            observed = float(row.get(metric, math.nan))
            expected = float(reference.get(metric, math.nan))
            signed_error = observed - expected
            absolute_error = abs(signed_error)
            row[f"continuous_reference_{metric}"] = expected
            row[f"{metric}_continuous_reference_signed_error"] = signed_error
            row[f"{metric}_continuous_reference_absolute_error"] = absolute_error
            stable_relative = (
                metric not in NEAR_ZERO_METRICS
                and math.isfinite(expected)
                and abs(expected) > max(1.0e-12, 1.0e-10 * abs(S_scale))
            )
            row[f"{metric}_continuous_reference_relative_error"] = (
                absolute_error / abs(expected)
                if stable_relative and math.isfinite(absolute_error)
                else math.nan
            )
            row[f"{metric}_continuous_reference_relative_error_reason"] = (
                "ok" if stable_relative else "reference_near_zero_or_metric_near_zero"
            )
        row["center_error_to_continuous"] = float(
            row.get("center_displacement", math.nan)
        )
        profile_key = (
            row["state"],
            row["resolution"],
            row["z_fraction"],
            int(defect_id),
        )
        if profile_key in profile_lookup:
            for metric, value in profile_lookup[profile_key].items():
                row[f"profile_continuous_reference_{metric}"] = value
    return profile_errors


def continuous_reference_error_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("state") not in (RAW_STATE, PROJECTED_STATE):
            continue
        if row.get("match_status") != "matched":
            continue
        base = (
            row["state"],
            row["resolution"],
            row["step"],
            row["z_fraction"],
        )
        groups[(*base, float(row["initial_charge"]))].append(row)
        groups[(*base, "all")].append(row)
    output: list[dict[str, Any]] = []
    fields = [
        *(f"{metric}_continuous_reference_absolute_error" for metric in REFERENCE_METRICS),
        "profile_continuous_reference_weighted_difference_rms",
        "profile_continuous_reference_weighted_relative_l2",
        "profile_continuous_reference_difference_linf",
        "center_error_to_continuous",
    ]
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        summary: dict[str, Any] = {
            "state": key[0],
            "resolution": key[1],
            "step": key[2],
            "z_fraction": key[3],
            "charge": key[4],
            "paired_defect_count": len(members),
            "defect_ids": [int(member["defect_id"]) for member in members],
        }
        for field in fields:
            finite_pairs = [
                (int(member["defect_id"]), float(member.get(field, math.nan)))
                for member in members
                if math.isfinite(float(member.get(field, math.nan)))
            ]
            values = np.asarray([value for _, value in finite_pairs], dtype=np.float64)
            median = float(np.median(values)) if values.size else math.nan
            mad = float(np.median(np.abs(values - median))) if values.size else math.nan
            summary[f"{field}_count"] = int(values.size)
            summary[f"{field}_mean"] = float(np.mean(values)) if values.size else math.nan
            summary[f"{field}_std"] = float(np.std(values)) if values.size else math.nan
            summary[f"{field}_median"] = median
            summary[f"{field}_mad"] = mad
            threshold = (
                max(3.0 * 1.4826 * mad, 0.10 * abs(median), 32.0 * np.finfo(float).eps)
                if values.size
                else math.nan
            )
            summary[f"{field}_outlier_defect_ids"] = (
                [
                    defect_id
                    for defect_id, value in finite_pairs
                    if abs(value - median) > threshold
                ]
                if values.size
                else []
            )
        output.append(summary)
    return output


def continuous_reference_error_trends(
    rows: Sequence[Mapping[str, Any]],
    runs: Sequence[CoreRun],
) -> list[dict[str, Any]]:
    ordered = sorted(runs, key=lambda run: run.artifact.resolution_h, reverse=True)
    labels = [run.label for run in ordered]
    lookup = {
        (
            row.get("state"),
            row.get("resolution"),
            row.get("z_fraction"),
            int(row["defect_id"]),
        ): row
        for row in rows
        if row.get("state") in (RAW_STATE, PROJECTED_STATE)
        and row.get("match_status") == "matched"
        and isinstance(row.get("defect_id"), (int, np.integer))
    }
    combinations = sorted(
        {(key[0], key[2], key[3]) for key in lookup},
        key=lambda item: tuple(map(str, item)),
    )
    fields = [
        *(f"{metric}_continuous_reference_absolute_error" for metric in REFERENCE_METRICS),
        "profile_continuous_reference_weighted_difference_rms",
        "profile_continuous_reference_weighted_relative_l2",
        "profile_continuous_reference_difference_linf",
        "center_error_to_continuous",
    ]
    output: list[dict[str, Any]] = []
    for state, z_fraction, defect_id in combinations:
        level_rows = [lookup.get((state, label, z_fraction, defect_id)) for label in labels]
        for field in fields:
            values = [
                float(row.get(field, math.nan)) if row is not None else math.nan
                for row in level_rows
            ]
            finite_all = all(math.isfinite(value) for value in values)
            nonincreasing = finite_all and all(
                fine <= coarse + max(1.0e-14, 1.0e-12 * abs(coarse))
                for coarse, fine in zip(values, values[1:])
            )
            output.append(
                {
                    "state": state,
                    "z_fraction": z_fraction,
                    "defect_id": defect_id,
                    "charge": (
                        float(level_rows[0]["initial_charge"])
                        if level_rows[0] is not None
                        else math.nan
                    ),
                    "error_metric": field,
                    "resolution_labels_coarse_to_fine": labels,
                    "values_coarse_to_fine": values,
                    "all_levels_finite": finite_all,
                    "nonincreasing_with_refinement": nonincreasing,
                    "interpretation": (
                        "working sampling/representation diagnostic; not a formal proof"
                    ),
                }
            )
    return output


def pairwise_metric_changes(
    summaries: Sequence[Mapping[str, Any]],
    runs: Sequence[CoreRun],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "space":
        ordered = sorted(runs, key=lambda run: run.artifact.resolution_h, reverse=True)
    else:
        ordered = sorted(runs, key=lambda run: run.artifact.dt, reverse=True)
    lookup = {
        (
            row["state"], row["resolution"], row["step"], row["z_fraction"], row["charge"]
        ): row
        for row in summaries
    }
    changes: list[dict[str, Any]] = []
    states = sorted({row["state"] for row in summaries})
    z_values = sorted({row["z_fraction"] for row in summaries if row["z_fraction"] is not None})
    charges: tuple[float | str, ...] = (-0.5, 0.5, "all")
    for coarse, fine in zip(ordered, ordered[1:]):
        for state in states:
            state_z = [None] if state == RAW_STATE else z_values
            for z_fraction in state_z:
                for charge in charges:
                    coarse_key = (state, coarse.label, 0 if state != FINAL_STATE else coarse.artifact.steps, z_fraction, charge)
                    fine_key = (state, fine.label, 0 if state != FINAL_STATE else fine.artifact.steps, z_fraction, charge)
                    if coarse_key not in lookup or fine_key not in lookup:
                        continue
                    coarse_row = lookup[coarse_key]
                    fine_row = lookup[fine_key]
                    for metric in CORE_METRICS:
                        coarse_value = float(coarse_row[f"{metric}_median"])
                        fine_value = float(fine_row[f"{metric}_median"])
                        signed_difference = fine_value - coarse_value
                        absolute_difference = abs(signed_difference)
                        relative_is_stable = metric not in NEAR_ZERO_METRICS
                        change = (
                            relative_change(coarse_value, fine_value)
                            if relative_is_stable
                            else math.nan
                        )
                        row = {
                            "mode": mode,
                            "state": state,
                            "z_fraction": z_fraction,
                            "charge": charge,
                            "metric": metric,
                            "coarse": coarse.label,
                            "fine": fine.label,
                            "coarse_value": coarse_value,
                            "fine_value": fine_value,
                            "signed_difference_fine_minus_coarse": signed_difference,
                            "absolute_difference": absolute_difference,
                            "relative_change": change,
                            "relative_change_reason": (
                                "ok"
                                if relative_is_stable
                                else "not_reported_for_near_zero_metric"
                            ),
                            "within_5_percent": bool(math.isfinite(change) and change <= 0.05),
                            "within_10_percent": bool(math.isfinite(change) and change <= 0.10),
                        }
                        if mode == "time":
                            ratio = coarse.artifact.dt / fine.artifact.dt
                            valid_time_ratio = math.isclose(
                                ratio, 2.0, rel_tol=0.0, abs_tol=1.0e-12
                            )
                            row["time_step_ratio"] = ratio
                            row["richardson_model_valid"] = valid_time_ratio
                            row["richardson_model_reason"] = (
                                "verified_first_order_with_dt_ratio_2"
                                if valid_time_ratio
                                else "requires_dt_ratio_2_for_prescribed_time_gate"
                            )
                            row["richardson_extrapolated_metric"] = (
                                2.0 * fine_value - coarse_value
                                if valid_time_ratio
                                and math.isfinite(coarse_value)
                                and math.isfinite(fine_value)
                                else math.nan
                            )
                            row["first_order_richardson_time_error"] = (
                                2.0 * absolute_difference
                                if valid_time_ratio and math.isfinite(absolute_difference)
                                else math.nan
                            )
                            row["fine_step_remaining_time_error"] = (
                                absolute_difference
                                if valid_time_ratio and math.isfinite(absolute_difference)
                                else math.nan
                            )
                        changes.append(row)
    return changes


def paired_individual_metric_changes(
    rows: Sequence[Mapping[str, Any]],
    runs: Sequence[CoreRun],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    """Compare only the same reliably matched defect across adjacent runs."""
    if mode == "space":
        ordered = sorted(runs, key=lambda run: run.artifact.resolution_h, reverse=True)
    else:
        ordered = sorted(runs, key=lambda run: run.artifact.dt, reverse=True)
    run_labels = {run.label for run in runs}
    lookup: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("resolution") not in run_labels or row.get("match_status") != "matched":
            continue
        defect_id = row.get("defect_id")
        if not isinstance(defect_id, (int, np.integer)):
            continue
        key = (
            row.get("state"),
            row.get("resolution"),
            row.get("z_fraction"),
            int(defect_id),
        )
        if key in lookup:
            raise ValueError(f"duplicate matched metric row for {key}")
        lookup[key] = row

    individual: list[dict[str, Any]] = []
    for coarse, fine in zip(ordered, ordered[1:]):
        coarse_keys = sorted(
            (key for key in lookup if key[1] == coarse.label),
            key=lambda key: (str(key[0]), str(key[2]), int(key[3])),
        )
        for coarse_key in coarse_keys:
            state, _, z_fraction, defect_id = coarse_key
            fine_key = (state, fine.label, z_fraction, defect_id)
            if fine_key not in lookup:
                continue
            coarse_row = lookup[coarse_key]
            fine_row = lookup[fine_key]
            coarse_charge = float(coarse_row.get("detected_charge", math.nan))
            fine_charge = float(fine_row.get("detected_charge", math.nan))
            initial_charge = float(coarse_row.get("initial_charge", math.nan))
            if not (
                math.isfinite(coarse_charge)
                and coarse_charge == fine_charge == initial_charge
                and float(fine_row.get("initial_charge", math.nan)) == initial_charge
            ):
                continue
            time_ratio = coarse.artifact.dt / fine.artifact.dt
            valid_time_ratio = mode != "time" or math.isclose(
                time_ratio, 2.0, rel_tol=0.0, abs_tol=1.0e-12
            )
            for metric in CORE_METRICS:
                coarse_value = float(coarse_row.get(metric, math.nan))
                fine_value = float(fine_row.get(metric, math.nan))
                if not (math.isfinite(coarse_value) and math.isfinite(fine_value)):
                    continue
                signed_difference = fine_value - coarse_value
                absolute_difference = abs(signed_difference)
                relative_is_stable = metric not in NEAR_ZERO_METRICS
                branch_consistent = True
                branch_reason = "not_a_crossing_metric"
                if metric in RADIUS_METRICS:
                    coarse_reason = coarse_row.get(f"{metric}_reason")
                    fine_reason = fine_row.get(f"{metric}_reason")
                    coarse_count = coarse_row.get(f"{metric}_crossing_count")
                    fine_count = fine_row.get(f"{metric}_crossing_count")
                    branch_consistent = (
                        coarse_reason == fine_reason == "ok"
                        and coarse_count == fine_count == 1
                    )
                    branch_reason = (
                        "same_single_outward_crossing"
                        if branch_consistent
                        else "crossing_status_or_branch_changed"
                    )
                richardson_valid = valid_time_ratio and branch_consistent
                comparison = {
                    "mode": mode,
                    "aggregation": "individual",
                    "state": state,
                    "z_fraction": z_fraction,
                    "charge": initial_charge,
                    "defect_id": defect_id,
                    "metric": metric,
                    "coarse": coarse.label,
                    "fine": fine.label,
                    "coarse_value": coarse_value,
                    "fine_value": fine_value,
                    "signed_difference_fine_minus_coarse": signed_difference,
                    "absolute_difference": absolute_difference,
                    "relative_change": (
                        relative_change(coarse_value, fine_value)
                        if relative_is_stable
                        else math.nan
                    ),
                    "relative_change_reason": (
                        "ok"
                        if relative_is_stable
                        else "not_reported_for_near_zero_metric"
                    ),
                    "crossing_branch_consistent": branch_consistent,
                    "crossing_branch_reason": branch_reason,
                }
                change = float(comparison["relative_change"])
                comparison["within_5_percent"] = bool(
                    math.isfinite(change) and change <= 0.05
                )
                comparison["within_10_percent"] = bool(
                    math.isfinite(change) and change <= 0.10
                )
                if mode == "time":
                    comparison.update(
                        time_step_ratio=time_ratio,
                        richardson_model_valid=richardson_valid,
                        richardson_model_reason=(
                            "verified_first_order_same_smooth_metric_branch"
                            if richardson_valid
                            else (
                                "requires_dt_ratio_2"
                                if not valid_time_ratio
                                else "topology_matching_or_crossing_branch_not_smooth"
                            )
                        ),
                        richardson_extrapolated_metric=(
                            2.0 * fine_value - coarse_value
                            if richardson_valid
                            else math.nan
                        ),
                        first_order_richardson_time_error=(
                            2.0 * absolute_difference
                            if richardson_valid
                            else math.nan
                        ),
                        fine_step_remaining_time_error=(
                            absolute_difference if richardson_valid else math.nan
                        ),
                    )
                individual.append(comparison)

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in individual:
        base = (
            row["mode"],
            row["state"],
            row["z_fraction"],
            row["metric"],
            row["coarse"],
            row["fine"],
        )
        grouped[(*base, row["charge"])].append(row)
        grouped[(*base, "all")].append(row)
    summaries: list[dict[str, Any]] = []
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        mode_value, state, z_fraction, metric, coarse, fine, charge = key
        summary: dict[str, Any] = {
            "mode": mode_value,
            "aggregation": "paired_distribution",
            "state": state,
            "z_fraction": z_fraction,
            "metric": metric,
            "coarse": coarse,
            "fine": fine,
            "charge": charge,
            "individual_count": len(members),
            "defect_ids": [int(member["defect_id"]) for member in members],
        }
        for field in ("absolute_difference", "relative_change"):
            finite_pairs = [
                (int(member["defect_id"]), float(member[field]))
                for member in members
                if math.isfinite(float(member[field]))
            ]
            values = np.asarray([value for _, value in finite_pairs], dtype=np.float64)
            median = float(np.median(values)) if values.size else math.nan
            mad = float(np.median(np.abs(values - median))) if values.size else math.nan
            summary[field] = median
            summary[f"{field}_count"] = int(values.size)
            summary[f"{field}_mean"] = float(np.mean(values)) if values.size else math.nan
            summary[f"{field}_std"] = float(np.std(values)) if values.size else math.nan
            summary[f"{field}_median"] = median
            summary[f"{field}_mad"] = mad
            summary[f"{field}_p90"] = float(np.quantile(values, 0.9)) if values.size else math.nan
            summary[f"{field}_max"] = float(np.max(values)) if values.size else math.nan
            threshold = (
                max(3.0 * 1.4826 * mad, 0.10 * abs(median), 32.0 * np.finfo(float).eps)
                if values.size
                else math.nan
            )
            summary[f"{field}_outlier_defect_ids"] = (
                [
                    defect_id
                    for defect_id, value in finite_pairs
                    if abs(value - median) > threshold
                ]
                if values.size
                else []
            )
        change = float(summary["relative_change"])
        summary["within_5_percent"] = bool(math.isfinite(change) and change <= 0.05)
        summary["within_10_percent"] = bool(math.isfinite(change) and change <= 0.10)
        if mode == "time":
            valid_members = [
                member for member in members if bool(member.get("richardson_model_valid"))
            ]
            errors = np.asarray(
                [member["first_order_richardson_time_error"] for member in valid_members],
                dtype=np.float64,
            )
            summary["richardson_valid_count"] = len(valid_members)
            summary["richardson_invalid_count"] = len(members) - len(valid_members)
            summary["first_order_richardson_time_error_median"] = (
                float(np.median(errors)) if errors.size else math.nan
            )
        summaries.append(summary)
    return [*individual, *summaries]


def _positive_order_from_ratio(
    h_descending: np.ndarray,
    values_descending: np.ndarray,
) -> tuple[float, float, float]:
    """Return p, observed difference ratio, and its p->0 positive-order limit."""
    h0, h1, h2 = h_descending
    d01 = values_descending[0] - values_descending[1]
    d12 = values_descending[1] - values_descending[2]
    ratio = float(d01 / d12)
    a = math.log(float(h0 / h1))
    b = math.log(float(h1 / h2))
    ratio_limit = a / b
    if not math.isfinite(ratio) or ratio <= ratio_limit:
        raise ValueError("no_positive_order_root")

    def log_expm1_positive(value: float) -> float:
        if value > 40.0:
            return value + math.log1p(-math.exp(-value))
        return math.log(math.expm1(value))

    def residual(order: float) -> float:
        log_numerator = log_expm1_positive(order * a)
        log_denominator = math.log(-math.expm1(-order * b))
        return log_numerator - log_denominator - math.log(ratio)

    lower = 1.0e-8
    upper = 64.0
    if residual(lower) >= 0.0 or residual(upper) <= 0.0:
        raise ValueError("positive_order_outside_stable_search_range")
    return float(brentq(residual, lower, upper, xtol=1.0e-12, rtol=1.0e-12)), ratio, ratio_limit


def _fit_with_fixed_order(
    h_descending: np.ndarray,
    values_descending: np.ndarray,
    order: float,
) -> dict[str, Any]:
    h_fine = float(h_descending[-1])
    t = np.power(h_descending / h_fine, order)
    design = np.column_stack((np.ones(3, dtype=np.float64), t))
    coefficients, _, _, _ = np.linalg.lstsq(design, values_descending, rcond=None)
    limit = float(coefficients[0])
    fine_amplitude = float(coefficients[1])
    reconstructed = design @ coefficients
    residual = reconstructed - values_descending
    scale = max(float(np.max(np.abs(values_descending))), np.finfo(np.float64).tiny)
    span = float(np.ptp(values_descending))
    coefficient_C = fine_amplitude / (h_fine**order)
    jacobian = np.column_stack(
        (
            np.ones(3, dtype=np.float64),
            t,
            (fine_amplitude / scale) * t * np.log(h_descending / h_fine),
        )
    )
    return {
        "m_infinity": limit,
        "coefficient_C": float(coefficient_C),
        "fine_grid_correction": float(limit - values_descending[-1]),
        "reconstructed_values": reconstructed,
        "residual_linf": float(np.max(np.abs(residual))),
        "residual_relative_l2": float(
            np.linalg.norm(residual) / max(np.linalg.norm(values_descending), np.finfo(float).tiny)
        ),
        "residual_over_span": float(
            np.max(np.abs(residual)) / max(span, np.finfo(float).tiny)
        ),
        "scaled_jacobian_condition": float(np.linalg.cond(jacobian)),
        "fine_extrapolation_ratio": float(
            abs(limit - values_descending[-1])
            / max(abs(values_descending[1] - values_descending[2]), np.finfo(float).tiny)
        ),
    }


def unequal_grid_three_level_fit(
    h: Sequence[float],
    values: Sequence[float],
    *,
    relative_perturbation: float = 1.0e-6,
    max_scaled_jacobian_condition: float = 1.0e8,
    max_order_sensitivity: float = 0.25,
    max_limit_sensitivity: float = 0.01,
) -> dict[str, Any]:
    """Auxiliary fit of M(h)=M_inf+C*h**p with explicit stability gates."""
    h_array = np.asarray(h, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    base: dict[str, Any] = {
        "accepted": False,
        "status": "invalid_input",
        "rejection_reasons": [],
        "candidate_order": math.nan,
        "observed_order": math.nan,
    }
    if h_array.shape != (3,) or value_array.shape != (3,):
        base["rejection_reasons"] = ["requires_exactly_three_levels"]
        return base
    if not np.isfinite(h_array).all() or np.any(h_array <= 0.0):
        base["rejection_reasons"] = ["h_must_be_positive_and_finite"]
        return base
    if not np.isfinite(value_array).all():
        base["rejection_reasons"] = ["metric_values_must_be_finite"]
        return base
    order_indices = np.argsort(h_array)[::-1]
    h_array = h_array[order_indices]
    value_array = value_array[order_indices]
    if not (h_array[0] > h_array[1] > h_array[2]):
        base["rejection_reasons"] = ["h_levels_must_be_distinct"]
        return base
    differences = np.diff(value_array)
    scale = max(float(np.max(np.abs(value_array))), np.finfo(np.float64).tiny)
    numerical_floor = 64.0 * np.finfo(np.float64).eps * scale
    if np.any(np.abs(differences) <= numerical_floor):
        base.update(
            status="difference_at_roundoff",
            rejection_reasons=["difference_at_roundoff"],
            difference_roundoff_floor=numerical_floor,
        )
        return base
    if differences[0] * differences[1] <= 0.0:
        base.update(status="non_monotone", rejection_reasons=["non_monotone_or_sign_change"])
        return base
    try:
        candidate_order, difference_ratio, ratio_limit = _positive_order_from_ratio(
            h_array, value_array
        )
    except ValueError as error:
        reason = str(error)
        base.update(status=reason, rejection_reasons=[reason])
        return base

    fit = _fit_with_fixed_order(h_array, value_array, candidate_order)
    sensitivity_orders: list[float] = []
    sensitivity_limits: list[float] = []
    perturbation = relative_perturbation * scale
    for index in range(3):
        for sign in (-1.0, 1.0):
            perturbed = value_array.copy()
            perturbed[index] += sign * perturbation
            perturbed_differences = np.diff(perturbed)
            if perturbed_differences[0] * perturbed_differences[1] <= 0.0:
                continue
            try:
                perturbed_order, _, _ = _positive_order_from_ratio(h_array, perturbed)
            except ValueError:
                continue
            perturbed_fit = _fit_with_fixed_order(h_array, perturbed, perturbed_order)
            sensitivity_orders.append(perturbed_order)
            sensitivity_limits.append(float(perturbed_fit["m_infinity"]))
    successful_trials = len(sensitivity_orders)
    order_sensitivity_abs = (
        max(abs(value - candidate_order) for value in sensitivity_orders)
        if sensitivity_orders
        else math.inf
    )
    order_sensitivity_relative = order_sensitivity_abs / max(candidate_order, 1.0)
    limit_sensitivity = (
        max(abs(value - float(fit["m_infinity"])) for value in sensitivity_limits) / scale
        if sensitivity_limits
        else math.inf
    )
    rejection_reasons: list[str] = []
    if not math.isfinite(float(fit["scaled_jacobian_condition"])) or float(
        fit["scaled_jacobian_condition"]
    ) > max_scaled_jacobian_condition:
        rejection_reasons.append("scaled_jacobian_ill_conditioned")
    if float(fit["residual_over_span"]) > 1.0e-8:
        rejection_reasons.append("fit_residual_too_large")
    if successful_trials != 6:
        rejection_reasons.append("perturbation_destroyed_positive_order_fit")
    if order_sensitivity_relative > max_order_sensitivity:
        rejection_reasons.append("observed_order_too_sensitive")
    if limit_sensitivity > max_limit_sensitivity:
        rejection_reasons.append("extrapolated_limit_too_sensitive")
    accepted = not rejection_reasons
    return {
        "accepted": accepted,
        "status": "accepted_auxiliary_diagnostic" if accepted else "unstable_fit_rejected",
        "rejection_reasons": rejection_reasons,
        "candidate_order": candidate_order,
        "observed_order": candidate_order if accepted else math.nan,
        "difference_ratio": difference_ratio,
        "positive_order_ratio_limit": ratio_limit,
        "difference_roundoff_floor": numerical_floor,
        "relative_input_perturbation": relative_perturbation,
        "absolute_input_perturbation": perturbation,
        "sensitivity_successful_trials": successful_trials,
        "order_sensitivity_max_abs": order_sensitivity_abs,
        "order_sensitivity_relative": order_sensitivity_relative,
        "m_infinity_sensitivity_over_scale": limit_sensitivity,
        **fit,
    }


def generalized_spatial_fits(
    summaries: Sequence[Mapping[str, Any]],
    runs: Sequence[CoreRun],
) -> list[dict[str, Any]]:
    """Apply the unequal-grid diagnostic to charge-median scalar metrics."""
    ordered = sorted(runs, key=lambda run: run.artifact.resolution_h, reverse=True)
    if len(ordered) != 3:
        return [
            {
                "accepted": False,
                "status": "requires_exactly_three_spatial_levels",
                "diagnostic_only": True,
            }
        ]
    labels = [run.label for run in ordered]
    h_values = [run.artifact.resolution_h for run in ordered]
    lookup = {
        (
            row["state"],
            row["resolution"],
            row["z_fraction"],
            row["charge"],
        ): row
        for row in summaries
        if row.get("resolution") in labels
    }
    combinations = sorted(
        {
            (row["state"], row["z_fraction"], row["charge"])
            for row in summaries
            if row.get("resolution") in labels
        },
        key=lambda item: tuple(map(str, item)),
    )
    output: list[dict[str, Any]] = []
    for state, z_fraction, charge in combinations:
        level_rows = [lookup.get((state, label, z_fraction, charge)) for label in labels]
        for metric in CORE_METRICS:
            common = {
                "diagnostic_only": True,
                "state": state,
                "z_fraction": z_fraction,
                "charge": charge,
                "metric": metric,
                "coarse": labels[0],
                "medium": labels[1],
                "fine": labels[2],
                "h_values": h_values,
            }
            if any(row is None for row in level_rows):
                output.append(
                    {
                        **common,
                        "accepted": False,
                        "status": "missing_resolution_summary",
                        "rejection_reasons": ["missing_resolution_summary"],
                        "observed_order": math.nan,
                    }
                )
                continue
            counts = [int(row[f"{metric}_count"]) for row in level_rows if row is not None]
            if len(set(counts)) != 1 or counts[0] <= 0:
                output.append(
                    {
                        **common,
                        "accepted": False,
                        "status": "matched_count_mismatch",
                        "rejection_reasons": ["matched_count_mismatch"],
                        "finite_counts": counts,
                        "observed_order": math.nan,
                    }
                )
                continue
            matched_id_sets = [
                tuple(int(value) for value in row.get(f"{metric}_defect_ids", []))
                for row in level_rows
                if row is not None
            ]
            if len(set(matched_id_sets)) != 1:
                output.append(
                    {
                        **common,
                        "accepted": False,
                        "status": "paired_defect_id_mismatch",
                        "rejection_reasons": ["paired_defect_id_mismatch"],
                        "matched_defect_ids_by_level": matched_id_sets,
                        "observed_order": math.nan,
                    }
                )
                continue
            metric_values = [
                float(row[f"{metric}_median"]) for row in level_rows if row is not None
            ]
            if metric in NEAR_ZERO_METRICS:
                output.append(
                    {
                        **common,
                        "accepted": False,
                        "status": "near_zero_metric_not_fit",
                        "rejection_reasons": ["near_zero_metric_not_fit"],
                        "values": metric_values,
                        "observed_order": math.nan,
                    }
                )
                continue
            output.append(
                {
                    **common,
                    "values": metric_values,
                    "finite_counts": counts,
                    **unequal_grid_three_level_fit(h_values, metric_values),
                }
            )
    return output


def raw_to_projected_changes(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Quantify twist+cubic-half representation changes at each resolution."""
    raw_lookup = {
        (row["resolution"], row["charge"]): row
        for row in summaries
        if row["state"] == RAW_STATE
    }
    output: list[dict[str, Any]] = []
    for projected in summaries:
        if projected["state"] != PROJECTED_STATE:
            continue
        raw = raw_lookup.get((projected["resolution"], projected["charge"]))
        if raw is None:
            continue
        for metric in CORE_METRICS:
            raw_value = float(raw[f"{metric}_median"])
            projected_value = float(projected[f"{metric}_median"])
            signed = projected_value - raw_value
            absolute = abs(signed)
            stable_relative = metric not in NEAR_ZERO_METRICS
            change = (
                relative_change(raw_value, projected_value)
                if stable_relative
                else math.nan
            )
            output.append(
                {
                    "transition": f"{RAW_STATE}_to_{PROJECTED_STATE}",
                    "resolution": projected["resolution"],
                    "z_fraction": projected["z_fraction"],
                    "charge": projected["charge"],
                    "metric": metric,
                    "raw_value": raw_value,
                    "projected_value": projected_value,
                    "signed_difference_projected_minus_raw": signed,
                    "absolute_difference": absolute,
                    "relative_change": change,
                    "relative_change_reason": (
                        "ok"
                        if stable_relative
                        else "not_reported_for_near_zero_metric"
                    ),
                }
            )
    return output


def z_fraction_variations(
    summaries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize explicit variation across the common z/H=0.1,0.5,0.9 planes."""
    grouped: dict[tuple[Any, ...], dict[float, Mapping[str, Any]]] = defaultdict(dict)
    for row in summaries:
        if row["state"] not in (PROJECTED_STATE, FINAL_STATE):
            continue
        z_fraction = row.get("z_fraction")
        if z_fraction is None:
            continue
        grouped[(row["state"], row["resolution"], row["charge"])][
            float(z_fraction)
        ] = row
    output: list[dict[str, Any]] = []
    required = (0.1, 0.5, 0.9)
    for key, by_z in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        if not all(any(math.isclose(z, target, abs_tol=1.0e-12) for z in by_z) for target in required):
            continue
        selected = [
            next(row for z, row in by_z.items() if math.isclose(z, target, abs_tol=1.0e-12))
            for target in required
        ]
        for metric in CORE_METRICS:
            values = np.asarray(
                [float(row[f"{metric}_median"]) for row in selected], dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            absolute_range = float(np.ptp(finite)) if finite.size == 3 else math.nan
            mid_value = float(values[1])
            stable_relative = metric not in NEAR_ZERO_METRICS
            relative_range = (
                absolute_range / abs(mid_value)
                if stable_relative
                and math.isfinite(absolute_range)
                and math.isfinite(mid_value)
                and mid_value != 0.0
                else math.nan
            )
            output.append(
                {
                    "state": key[0],
                    "resolution": key[1],
                    "charge": key[2],
                    "metric": metric,
                    "z_fractions": required,
                    "values": values,
                    "absolute_range": absolute_range,
                    "relative_range_over_midplane": relative_range,
                    "relative_range_reason": (
                        "ok"
                        if stable_relative
                        else "not_reported_for_near_zero_metric"
                    ),
                }
            )
    return output


def time_vs_space_error_gate(
    space_summary_path: Path,
    time_metric_changes: Sequence[Mapping[str, Any]],
    time_profile_changes: Sequence[Mapping[str, Any]],
    time_runs: Sequence[CoreRun],
    time_status_counts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare the dt=0.005 Richardson estimate with the R320->R512 change."""
    path = space_summary_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing space core summary for time gate: {path}")
    space_report = json.loads(path.read_text(encoding="utf-8"))
    if space_report.get("mode") != "space":
        raise ValueError(f"{path} is not a space-mode core summary")
    grids = space_report.get("physical_scales_and_retained_modes", {}).get("grids", [])
    label_by_nx = {
        int(item["shape"][0]): str(item["resolution"])
        for item in grids
        if isinstance(item.get("shape"), list) and len(item["shape"]) == 3
    }
    if 320 not in label_by_nx or 512 not in label_by_nx:
        raise ValueError(f"{path} does not identify both R320 and R512 grids")
    medium_label = label_by_nx[320]
    fine_label = label_by_nx[512]
    space_inputs = space_report.get("input_run_manifest", [])
    space_fine_input = next(
        (item for item in space_inputs if item.get("resolution") == fine_label), None
    )
    time_coarse_run = max(time_runs, key=lambda run: run.artifact.dt)
    if space_fine_input is None:
        raise ValueError(f"{path} is missing the R512 input-run identity manifest")
    time_coarse_identity = input_run_manifest((time_coarse_run,))[0]
    identity_fields = (
        "shape",
        "lengths",
        "dt",
        "steps",
        "script",
        "model_variant",
        "model_stage",
        "Q_convention",
        "validation_config_sha256",
        "implementation_provenance",
        "runtime_environment",
        "metadata_sha256",
        "raw_q_sha256",
        "projected_q_sha256",
        "q2d_initial_sha256",
        "defect_table_sha256",
    )
    identity_mismatches = [
        field
        for field in identity_fields
        if space_fine_input.get(field)
        != json_safe(time_coarse_identity.get(field))
    ]
    if identity_mismatches:
        raise ValueError(
            "Time-control dt=0.005 R512 does not match the space-study R512 "
            f"for fields {identity_mismatches}"
        )

    def topology_status(
        counts: Sequence[Mapping[str, Any]], labels: set[str]
    ) -> tuple[bool, list[str]]:
        problems: list[str] = []
        for label in sorted(labels):
            for z_fraction in DEFAULT_Z_FRACTIONS:
                row = next(
                    (
                        item
                        for item in counts
                        if item.get("state") == FINAL_STATE
                        and item.get("resolution") == label
                        and item.get("charge") == "all"
                        and math.isclose(
                            float(item.get("z_fraction", math.nan)),
                            z_fraction,
                            rel_tol=0.0,
                            abs_tol=1.0e-12,
                        )
                    ),
                    None,
                )
                if row is None:
                    problems.append(f"{label}@z/H={z_fraction}:missing_status_row")
                    continue
                if (
                    row.get("expected_initial_count") != 12
                    or row.get("matched_initial_count") != 12
                    or row.get("missing_initial_count") != 0
                    or row.get("ambiguous_candidate_count") != 0
                    or row.get("unmatched_reliable_candidate_count") != 0
                ):
                    problems.append(f"{label}@z/H={z_fraction}:topology_or_matching_incomplete")
        return not problems, problems

    space_topology_ok, space_topology_problems = topology_status(
        space_report.get("status_counts", []), {medium_label, fine_label}
    )
    time_topology_ok, time_topology_problems = topology_status(
        time_status_counts, {run.label for run in time_runs}
    )
    topology_matching_complete = space_topology_ok and time_topology_ok
    space_metric_rows = space_report.get(
        "paired_individual_metric_changes_and_distributions", []
    )
    space_metric_lookup = {
        (
            row.get("state"),
            row.get("z_fraction"),
            row.get("defect_id"),
            row.get("metric"),
        ): row
        for row in space_metric_rows
        if row.get("aggregation") == "individual"
        and row.get("coarse") == medium_label
        and row.get("fine") == fine_label
    }
    scalar_gates = []
    for time_row in time_metric_changes:
        if time_row.get("aggregation") != "individual" or time_row.get("state") != FINAL_STATE:
            continue
        key = (
            time_row.get("state"),
            time_row.get("z_fraction"),
            time_row.get("defect_id"),
            time_row.get("metric"),
        )
        space_row = space_metric_lookup.get(key)
        if space_row is None:
            continue
        metric = str(time_row["metric"])
        spatial_error = float(space_row.get("absolute_difference", math.nan))
        time_error = float(
            time_row.get("first_order_richardson_time_error", math.nan)
        )
        tolerance = TIME_GATE_ABSOLUTE_TOLERANCES.get(
            metric, TIME_GATE_ABSOLUTE_TOLERANCES["default"]
        )
        smooth = bool(time_row.get("richardson_model_valid"))
        absolute_tolerance_satisfied = bool(
            math.isfinite(time_error) and time_error <= tolerance
        )
        if not topology_matching_complete:
            passed: bool | None = False
            reason = "topology_or_matching_incomplete"
        elif not smooth or not (math.isfinite(spatial_error) and math.isfinite(time_error)):
            passed = False
            reason = "invalid_smooth_richardson_or_missing_metric"
        elif spatial_error <= tolerance:
            passed = None
            reason = "indeterminate_near_zero_space_difference_check_absolute_tolerance"
        else:
            passed = time_error <= 0.2 * spatial_error
            reason = "twenty_percent_of_spatial_difference"
        scalar_gates.append(
            {
                "state": time_row["state"],
                "z_fraction": time_row["z_fraction"],
                "charge": time_row["charge"],
                "defect_id": time_row["defect_id"],
                "metric": metric,
                "space_pair": [medium_label, fine_label],
                "spatial_error": spatial_error,
                "coarse_time_richardson_error": time_error,
                "absolute_physical_tolerance": tolerance,
                "absolute_time_tolerance_satisfied": absolute_tolerance_satisfied,
                "gate_reason": reason,
                "passed": passed,
            }
        )

    space_profile_rows = space_report.get(
        "charge_conditioned_profile_pairwise_changes", []
    )
    space_profile_lookup = {
        (row.get("state"), row.get("z_fraction"), row.get("defect_id")): row
        for row in space_profile_rows
        if row.get("aggregation") == "individual"
        and row.get("coarse") == medium_label
        and row.get("fine") == fine_label
    }
    profile_gates = []
    profile_tolerance = TIME_GATE_ABSOLUTE_TOLERANCES[
        "profile_weighted_relative_l2"
    ]
    for time_row in time_profile_changes:
        if time_row.get("aggregation") != "individual" or time_row.get("state") != FINAL_STATE:
            continue
        key = (
            time_row.get("state"),
            time_row.get("z_fraction"),
            time_row.get("defect_id"),
        )
        space_row = space_profile_lookup.get(key)
        if space_row is None:
            continue
        spatial_error = float(space_row.get("weighted_relative_l2", math.nan))
        time_error = float(
            time_row.get("first_order_richardson_weighted_relative_l2", math.nan)
        )
        smooth = bool(time_row.get("richardson_model_valid"))
        absolute_tolerance_satisfied = bool(
            math.isfinite(time_error) and time_error <= profile_tolerance
        )
        if not topology_matching_complete:
            passed = False
            reason = "topology_or_matching_incomplete"
        elif not smooth or not (math.isfinite(spatial_error) and math.isfinite(time_error)):
            passed = False
            reason = "invalid_smooth_richardson_or_missing_profile"
        elif spatial_error <= profile_tolerance:
            passed = None
            reason = "indeterminate_near_zero_space_difference_check_absolute_profile_tolerance"
        else:
            passed = time_error <= 0.2 * spatial_error
            reason = "twenty_percent_of_spatial_profile_difference"
        profile_gates.append(
            {
                "state": time_row["state"],
                "z_fraction": time_row["z_fraction"],
                "charge": time_row["charge"],
                "defect_id": time_row["defect_id"],
                "space_pair": [medium_label, fine_label],
                "spatial_weighted_relative_l2": spatial_error,
                "coarse_time_richardson_weighted_relative_l2": time_error,
                "absolute_profile_tolerance": profile_tolerance,
                "absolute_time_tolerance_satisfied": absolute_tolerance_satisfied,
                "gate_reason": reason,
                "passed": passed,
            }
        )
    return {
        "applied": True,
        "space_summary": str(path),
        "space_r512_time_coarse_identity_verified": True,
        "space_r512_directory": space_fine_input.get("directory"),
        "time_coarse_r512_directory": time_coarse_identity["directory"],
        "topology_matching_complete": topology_matching_complete,
        "space_topology_problems": space_topology_problems,
        "time_topology_problems": time_topology_problems,
        "scalar_metric_gates": scalar_gates,
        "profile_gates": profile_gates,
        "all_available_scalar_gates_pass": topology_matching_complete
        and bool(scalar_gates)
        and all(row["passed"] is True for row in scalar_gates),
        "all_available_profile_gates_pass": topology_matching_complete
        and bool(profile_gates)
        and all(row["passed"] is True for row in profile_gates),
        "interpretation": (
            "The 20% test applies to the first-order Richardson estimate "
            "2*|M_dt0025-M_dt005|, never to the unscaled raw difference."
        ),
    }


def profile_pairwise_changes(
    profiles: Sequence[Mapping[str, Any]],
    runs: Sequence[CoreRun],
    *,
    mode: str,
    S_scale: float = 1.0 / 3.0,
) -> list[dict[str, Any]]:
    """Same-defect radial-profile changes and their charge distributions."""
    lookup: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    run_labels = {run.label for run in runs}
    for profile in profiles:
        if profile.get("resolution") not in run_labels:
            continue
        key = (
            profile["state"],
            profile["resolution"],
            profile["z_fraction"],
            int(profile["defect_id"]),
        )
        if key in lookup:
            raise ValueError(f"duplicate radial profile for {key}")
        lookup[key] = profile
    if mode == "space":
        ordered = sorted(runs, key=lambda run: run.artifact.resolution_h, reverse=True)
    else:
        ordered = sorted(runs, key=lambda run: run.artifact.dt, reverse=True)
    individual: list[dict[str, Any]] = []
    for coarse, fine in zip(ordered, ordered[1:]):
        coarse_keys = sorted(
            (key for key in lookup if key[1] == coarse.label),
            key=lambda key: (str(key[0]), str(key[2]), int(key[3])),
        )
        for coarse_key in coarse_keys:
            state, _, z_fraction, defect_id = coarse_key
            fine_key = (state, fine.label, z_fraction, defect_id)
            if fine_key not in lookup:
                continue
            coarse_member = lookup[coarse_key]
            fine_member = lookup[fine_key]
            coarse_charge = float(coarse_member["charge"])
            fine_charge = float(fine_member["charge"])
            if coarse_charge != fine_charge:
                continue
            metrics = weighted_profile_difference(
                coarse_member, fine_member, S_scale=S_scale
            )
            relative_l2 = float(metrics["weighted_relative_l2"])
            row = {
                "mode": mode,
                "aggregation": "individual",
                "state": state,
                "z_fraction": z_fraction,
                "charge": coarse_charge,
                "defect_id": defect_id,
                "coarse": coarse.label,
                "fine": fine.label,
                **metrics,
                "within_5_percent": bool(
                    math.isfinite(relative_l2) and relative_l2 <= 0.05
                ),
                "within_10_percent": bool(
                    math.isfinite(relative_l2) and relative_l2 <= 0.10
                ),
            }
            if mode == "time":
                time_ratio = coarse.artifact.dt / fine.artifact.dt
                valid_time_ratio = math.isclose(
                    time_ratio, 2.0, rel_tol=0.0, abs_tol=1.0e-12
                )
                row["time_step_ratio"] = time_ratio
                row["richardson_model_valid"] = valid_time_ratio
                row["richardson_model_reason"] = (
                    "verified_first_order_same_matched_profile_with_dt_ratio_2"
                    if valid_time_ratio
                    else "requires_dt_ratio_2_for_prescribed_time_gate"
                )
                if valid_time_ratio:
                    coarse_values = np.asarray(coarse_member["median"], dtype=np.float64)
                    fine_values = np.asarray(fine_member["median"], dtype=np.float64)
                    richardson_values = 2.0 * fine_values - coarse_values
                    richardson_member = {
                        "radii": np.asarray(fine_member["radii"], dtype=np.float64),
                        "median": richardson_values,
                    }
                    coarse_error = weighted_profile_difference(
                        coarse_member, richardson_member, S_scale=S_scale
                    )
                    fine_error = weighted_profile_difference(
                        fine_member, richardson_member, S_scale=S_scale
                    )
                    row["richardson_extrapolated_profile_median"] = richardson_values
                    row["first_order_richardson_weighted_difference_rms"] = coarse_error[
                        "weighted_difference_rms"
                    ]
                    row["first_order_richardson_weighted_relative_l2"] = coarse_error[
                        "weighted_relative_l2"
                    ]
                    row["fine_step_remaining_weighted_difference_rms"] = fine_error[
                        "weighted_difference_rms"
                    ]
                    row["fine_step_remaining_weighted_relative_l2"] = fine_error[
                        "weighted_relative_l2"
                    ]
                else:
                    row["richardson_extrapolated_profile_median"] = None
                    row["first_order_richardson_weighted_difference_rms"] = math.nan
                    row["first_order_richardson_weighted_relative_l2"] = math.nan
                    row["fine_step_remaining_weighted_difference_rms"] = math.nan
                    row["fine_step_remaining_weighted_relative_l2"] = math.nan
            individual.append(row)

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in individual:
        base = (
            row["mode"],
            row["state"],
            row["z_fraction"],
            row["coarse"],
            row["fine"],
        )
        grouped[(*base, row["charge"])].append(row)
        grouped[(*base, "all")].append(row)
    summaries: list[dict[str, Any]] = []
    distribution_metrics = (
        "weighted_difference_rms",
        "weighted_relative_l2",
        "difference_linf",
    )
    for key, members in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        mode_value, state, z_fraction, coarse_label, fine_label, charge = key
        summary: dict[str, Any] = {
            "mode": mode_value,
            "aggregation": "paired_distribution",
            "state": state,
            "z_fraction": z_fraction,
            "charge": charge,
            "coarse": coarse_label,
            "fine": fine_label,
            "individual_count": len(members),
            "defect_ids": [int(member["defect_id"]) for member in members],
        }
        for metric in distribution_metrics:
            finite_pairs = [
                (int(member["defect_id"]), float(member[metric]))
                for member in members
                if math.isfinite(float(member[metric]))
            ]
            values = np.asarray([value for _, value in finite_pairs], dtype=np.float64)
            median = float(np.median(values)) if values.size else math.nan
            mad = (
                float(np.median(np.abs(values - median)))
                if values.size
                else math.nan
            )
            summary[metric] = median
            summary[f"{metric}_count"] = int(values.size)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values.size else math.nan
            summary[f"{metric}_std"] = float(np.std(values)) if values.size else math.nan
            summary[f"{metric}_median"] = median
            summary[f"{metric}_mad"] = mad
            summary[f"{metric}_p90"] = (
                float(np.quantile(values, 0.9)) if values.size else math.nan
            )
            summary[f"{metric}_max"] = float(np.max(values)) if values.size else math.nan
            threshold = (
                max(
                    3.0 * 1.4826 * mad,
                    0.10 * abs(median),
                    32.0 * np.finfo(np.float64).eps,
                )
                if values.size
                else math.nan
            )
            summary[f"{metric}_outlier_defect_ids"] = (
                [
                    defect_id
                    for defect_id, value in finite_pairs
                    if abs(value - median) > threshold
                ]
                if values.size
                else []
            )
        relative_l2 = float(summary["weighted_relative_l2"])
        summary["within_5_percent"] = bool(
            math.isfinite(relative_l2) and relative_l2 <= 0.05
        )
        summary["within_10_percent"] = bool(
            math.isfinite(relative_l2) and relative_l2 <= 0.10
        )
        if mode == "time":
            valid_members = [
                member for member in members if member.get("richardson_model_valid") is True
            ]
            summary["richardson_valid_count"] = len(valid_members)
            summary["richardson_invalid_count"] = len(members) - len(valid_members)
            for field in (
                "first_order_richardson_weighted_difference_rms",
                "first_order_richardson_weighted_relative_l2",
                "fine_step_remaining_weighted_difference_rms",
                "fine_step_remaining_weighted_relative_l2",
            ):
                values = np.asarray(
                    [
                        float(member[field])
                        for member in valid_members
                        if math.isfinite(float(member[field]))
                    ],
                    dtype=np.float64,
                )
                summary[field] = float(np.median(values)) if values.size else math.nan
                summary[f"{field}_count"] = int(values.size)
                summary[f"{field}_mean"] = (
                    float(np.mean(values)) if values.size else math.nan
                )
                summary[f"{field}_std"] = (
                    float(np.std(values)) if values.size else math.nan
                )
        summaries.append(summary)
    return [*individual, *summaries]


def raw_to_projected_profile_changes(
    profiles: Sequence[Mapping[str, Any]],
    *,
    S_scale: float,
) -> list[dict[str, Any]]:
    raw_lookup = {
        (profile["resolution"], int(profile["defect_id"])): profile
        for profile in profiles
        if profile.get("state") == RAW_STATE
    }
    output: list[dict[str, Any]] = []
    for projected in profiles:
        if projected.get("state") != PROJECTED_STATE:
            continue
        key = (projected["resolution"], int(projected["defect_id"]))
        raw = raw_lookup.get(key)
        if raw is None or float(raw["charge"]) != float(projected["charge"]):
            continue
        output.append(
            {
                "transition": f"{RAW_STATE}_to_{PROJECTED_STATE}",
                "resolution": projected["resolution"],
                "z_fraction": projected["z_fraction"],
                "defect_id": int(projected["defect_id"]),
                "charge": float(projected["charge"]),
                **weighted_profile_difference(raw, projected, S_scale=S_scale),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _best_effort_git(args: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "error": str(error)}
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.rstrip(),
        "stderr": completed.stderr.rstrip(),
    }


def _small_file_manifest(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    manifest: list[dict[str, Any]] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        size = path.stat().st_size
        entry: dict[str, Any] = {"path": str(path), "size_bytes": size}
        if size <= 16 * 1024 * 1024:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()
        else:
            entry["sha256"] = None
            entry["hash_reason"] = "file_exceeds_16MiB_provenance_manifest_limit"
        manifest.append(entry)
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_run_manifest(runs: Sequence[CoreRun]) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for run in runs:
        metadata = run.metadata
        identity = validate_run_identity(
            metadata, path=run.artifact.directory / "metadata.json"
        )
        manifest.append(
            {
                "resolution": run.label,
                "directory": str(run.artifact.directory),
                "shape": run.shape,
                "lengths": run.lengths,
                "dt": run.artifact.dt,
                "steps": run.artifact.steps,
                "script": identity["script"],
                "model_variant": identity["model_variant"],
                "model_stage": identity["model_stage"],
                "Q_convention": metadata["model"]["Q_convention"],
                "validation_config_sha256": identity[
                    "validation_config_sha256"
                ],
                "implementation_provenance": identity[
                    "implementation_provenance"
                ],
                "runtime_environment": metadata.get("runtime_environment"),
                "metadata_sha256": _canonical_json_sha256(metadata),
                "raw_q_sha256": metadata["initial_condition"]["raw_q_sha256"],
                "projected_q_sha256": metadata["initial_condition"][
                    "projected_q_sha256"
                ],
                "q2d_initial_sha256": _file_sha256(run.q2d_path),
                "defect_table_sha256": _file_sha256(run.defects_path),
            }
        )
    return manifest


def analysis_and_simulation_provenance(
    args: argparse.Namespace,
    runs: Sequence[CoreRun],
    output_dir: Path,
) -> dict[str, Any]:
    run_directories = [run.artifact.directory for run in runs]
    common_root = Path(os.path.commonpath([str(path) for path in run_directories]))
    if common_root in run_directories:
        common_root = common_root.parent
    provenance_dir = common_root / "provenance"
    jobs_dir = common_root / "jobs"
    package_versions = {}
    for distribution in ("numpy", "scipy", "matplotlib", "torch", "pytest"):
        try:
            package_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            package_versions[distribution] = None
    analysis_source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "scripts_plane" / "analyze_fig4_convergence.py",
        PROJECT_ROOT / "pssolver" / "models" / "active_nematics" / "__init__.py",
        PROJECT_ROOT / "pssolver" / "models" / "active_nematics" / "q_tensor.py",
    )
    analysis_source_sha256 = {
        str(source.relative_to(PROJECT_ROOT)): _file_sha256(source)
        for source in analysis_source_paths
    }
    return {
        "analysis_runtime": {
            "analysis_source_sha256": analysis_source_sha256,
            "command": shlex.join([sys.executable, *sys.argv]),
            "executable": sys.executable,
            "prefix": sys.prefix,
            "python_version": sys.version,
            "platform": platform.platform(),
            "package_versions": package_versions,
            "loaded_modules": os.environ.get("LOADEDMODULES", ""),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_head_best_effort": _best_effort_git(("rev-parse", "HEAD")),
            "git_branch_best_effort": _best_effort_git(("rev-parse", "--abbrev-ref", "HEAD")),
            "git_status_best_effort": _best_effort_git(("status", "--short", "--branch")),
        },
        "simulation_provenance": {
            "run_root": str(common_root),
            "run_directories": [str(path) for path in run_directories],
            "analysis_output_directory": str(output_dir),
            "provenance_directory": str(provenance_dir),
            "jobs_directory": str(jobs_dir),
            "provenance_manifest": _small_file_manifest(provenance_dir),
            "job_script_manifest": _small_file_manifest(jobs_dir),
            "interpretation": (
                "Saved simulation provenance is authoritative for the runs; "
                "best-effort analysis-time Git state is reported separately."
            ),
        },
    }


def save_profiles(path: Path, profiles: Sequence[Mapping[str, Any]]) -> None:
    if not profiles:
        raise ValueError(f"cannot write empty profile archive {path}")
    np.savez_compressed(
        path,
        state=np.asarray([item["state"] for item in profiles], dtype="U32"),
        resolution=np.asarray([item["resolution"] for item in profiles], dtype="U128"),
        step=np.asarray([item["step"] for item in profiles], dtype=np.int64),
        z_fraction=np.asarray(
            [math.nan if item["z_fraction"] is None else item["z_fraction"] for item in profiles],
            dtype=np.float64,
        ),
        defect_id=np.asarray([item["defect_id"] for item in profiles], dtype=np.int64),
        charge=np.asarray([item["charge"] for item in profiles], dtype=np.float64),
        radii=np.stack([item["radii"] for item in profiles]),
        profile_median=np.stack([item["median"] for item in profiles]),
        profile_mean=np.stack([item["mean"] for item in profiles]),
        profile_std=np.stack([item["std"] for item in profiles]),
        profile_count=np.stack([item["count"] for item in profiles]),
    )


def reserve_outputs(output_dir: Path) -> dict[str, Path]:
    if output_dir.exists():
        raise FileExistsError(
            f"Core-analysis output directory must be wholly new: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    names = {
        "csv": "defect_core_metrics.csv",
        "json": "defect_core_summary.json",
        "initial_npz": "initial_core_profiles.npz",
        "projected_initial_npz": "projected_initial_core_profiles.npz",
        "final_npz": "final_core_profiles.npz",
        "initial_plot": "initial_r50_convergence.png",
        "projected_initial_plot": "projected_initial_profiles_by_resolution.png",
        "final_plot": "final_core_profiles_by_charge.png",
        "change_plot": "core_metric_resolution_changes.png",
        "positive_zoom": "core_zoom_positive.png",
        "negative_zoom": "core_zoom_negative.png",
        "readme": "README.md",
    }
    paths = {key: output_dir / name for key, name in names.items()}
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing core-analysis outputs: "
            + ", ".join(str(path) for path in existing)
        )
    return paths


def _plot_initial_r50(path: Path, rows: Sequence[Mapping[str, Any]], runs: Sequence[CoreRun]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    ordered = sorted(runs, key=lambda run: run.artifact.resolution_h, reverse=True)
    labels = [run.label for run in ordered]
    for charge, color in ((0.5, "tab:red"), (-0.5, "tab:blue")):
        medians = []
        for run in ordered:
            values = [
                float(row["r50_absolute"])
                for row in rows
                if row.get("state") == RAW_STATE
                and row.get("resolution") == run.label
                and row.get("initial_charge") == charge
                and row.get("match_status") == "matched"
                and math.isfinite(float(row.get("r50_absolute", math.nan)))
            ]
            medians.append(float(np.median(values)) if values else math.nan)
        axis.plot(labels, medians, "o-", color=color, label=f"charge {charge:+.1f}")
        reference_values = [
            float(row["r50_absolute"])
            for row in rows
            if row.get("state") == "continuous_multidefect_reference"
            and row.get("initial_charge") == charge
            and math.isfinite(float(row.get("r50_absolute", math.nan)))
        ]
        if reference_values:
            axis.axhline(
                float(np.median(reference_values)),
                color=color,
                linestyle=":",
                label=f"continuous 12-core median, q={charge:+.1f}",
            )
    axis.axhline(
        1.5 * np.arctanh(0.5),
        color="black",
        linestyle="--",
        label="isolated-core sanity",
    )
    axis.set_ylabel("absolute r50 (physical length)")
    axis.set_xlabel("raw Q2D resolution")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_final_profiles(path: Path, profiles: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    groups: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)
    radii_by_group: dict[tuple[str, float], np.ndarray] = {}
    for profile in profiles:
        if profile["state"] != "final" or not math.isclose(float(profile["z_fraction"]), 0.5, abs_tol=1e-12):
            continue
        key = (str(profile["resolution"]), float(profile["charge"]))
        groups[key].append(np.asarray(profile["median"], dtype=float))
        radii_by_group[key] = np.asarray(profile["radii"], dtype=float)
    for (resolution, charge), values in sorted(groups.items()):
        axis.plot(
            radii_by_group[(resolution, charge)],
            np.nanmedian(np.stack(values), axis=0),
            label=f"{resolution}, q={charge:+.1f}",
        )
    axis.set_xlabel("radius")
    axis.set_ylabel("median angular S")
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_projected_initial_profiles(
    path: Path,
    profiles: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    groups: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)
    radii_by_group: dict[tuple[str, float], np.ndarray] = {}
    for profile in profiles:
        if profile["state"] != PROJECTED_STATE or not math.isclose(
            float(profile["z_fraction"]), 0.5, abs_tol=1e-12
        ):
            continue
        key = (str(profile["resolution"]), float(profile["charge"]))
        groups[key].append(np.asarray(profile["median"], dtype=float))
        radii_by_group[key] = np.asarray(profile["radii"], dtype=float)
    for (resolution, charge), values in sorted(groups.items()):
        axis.plot(
            radii_by_group[(resolution, charge)],
            np.nanmedian(np.stack(values), axis=0),
            label=f"{resolution}, q={charge:+.1f}",
        )
    axis.set_xlabel("radius")
    axis.set_ylabel("projected-initial median angular S")
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_changes(path: Path, changes: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row for row in changes
        if row["state"] == "final"
        and row["metric"] in ("r50_absolute", "r50_local", "core_deficit_local", "low_s_area_local")
        and row["z_fraction"] == 0.5
    ]
    fig, axis = plt.subplots(figsize=(max(8.0, 0.35 * len(selected)), 4.8))
    labels = [
        f"{row['coarse']}→{row['fine']}\n{row['metric']}\nq="
        + (
            str(row["charge"])
            if isinstance(row["charge"], str)
            else f"{float(row['charge']):+.1f}"
        )
        for row in selected
    ]
    values = [100.0 * float(row["relative_change"]) for row in selected]
    axis.bar(np.arange(len(values)), values)
    axis.axhline(5.0, color="tab:green", linestyle="--", label="5% practical threshold")
    axis.axhline(10.0, color="tab:orange", linestyle="--", label="10% practical threshold")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("relative change (%)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_zoom(
    path: Path,
    runs: Sequence[CoreRun],
    final_planes: Mapping[str, np.ndarray],
    metric_rows: Sequence[Mapping[str, Any]],
    defect_id: int,
    *,
    S_bulk: float,
    interpolation_order: int,
    z_index: int,
    z_fraction: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    center = runs[0].defect_positions[defect_id]
    extent_radius = 4.0
    coordinate = np.linspace(-extent_radius, extent_radius, 201)
    xx, yy = np.meshgrid(center[0] + coordinate, center[1] + coordinate, indexing="xy")
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    image = None
    for axis, run in zip(axes, runs):
        plane = final_planes[run.label][:, :, z_index, :]
        sampler = PeriodicQSampler(plane, run.lengths[:2], order=interpolation_order)
        S = sampler.sample_S(xx, yy)
        image = axis.imshow(
            S,
            origin="lower",
            extent=(center[0] - extent_radius, center[0] + extent_radius, center[1] - extent_radius, center[1] + extent_radius),
            vmin=0.0,
            vmax=S_bulk,
            cmap="viridis",
            interpolation="nearest",
        )
        matched_row = next(
            (
                row
                for row in metric_rows
                if row.get("state") == FINAL_STATE
                and row.get("resolution") == run.label
                and row.get("defect_id") == defect_id
                and row.get("match_status") == "matched"
                and math.isclose(
                    float(row.get("z_fraction", math.nan)),
                    z_fraction,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            None,
        )
        if matched_row is not None:
            delta = minimum_image_delta(
                np.asarray((matched_row["center_x"], matched_row["center_y"])) - center,
                run.lengths[:2],
            )
            plotted_center = center + delta
            axis.plot(
                plotted_center[0],
                plotted_center[1],
                marker="x",
                color="white",
                markersize=7,
                markeredgewidth=1.5,
            )
        axis.plot(center[0], center[1], marker="+", color="red", markersize=7)
        axis.set_title(f"{run.label}, z/H={z_fraction:g}")
        axis.set_xlabel("x")
    axes[0].set_ylabel("y")
    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), label="S", shrink=0.85)
    fig.subplots_adjust(wspace=0.08, right=0.9)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def common_reliably_matched_defect_id(
    rows: Sequence[Mapping[str, Any]],
    runs: Sequence[CoreRun],
    *,
    charge: float,
    z_fraction: float,
) -> int | None:
    label_set = {run.label for run in runs}
    matched_by_id: dict[int, set[str]] = defaultdict(set)
    for row in rows:
        defect_id = row.get("defect_id")
        if (
            row.get("state") == FINAL_STATE
            and row.get("match_status") == "matched"
            and row.get("initial_charge") == charge
            and row.get("resolution") in label_set
            and isinstance(defect_id, (int, np.integer))
            and math.isclose(
                float(row.get("z_fraction", math.nan)),
                z_fraction,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            matched_by_id[int(defect_id)].add(str(row["resolution"]))
    common_ids = sorted(
        defect_id for defect_id, labels in matched_by_id.items() if labels == label_set
    )
    return common_ids[0] if common_ids else None


def _plot_unavailable_zoom(path: Path, *, charge: float, reason: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.0, 3.8))
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        f"charge {charge:+.1f} core zoom unavailable\n{reason}",
        ha="center",
        va="center",
        transform=axis.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_readme(
    path: Path,
    *,
    args: argparse.Namespace,
    runs: Sequence[CoreRun],
    output_paths: Mapping[str, Path],
    provenance: Mapping[str, Any],
    scales: Mapping[str, Any],
    continuous_reference_dr: float,
    continuous_reference_angular_samples: int,
) -> None:
    runtime = provenance["analysis_runtime"]
    simulation = provenance["simulation_provenance"]
    scale_rows = "\n".join(
        "| {resolution} | {dx:.9g} | {dz:.9g} | {xi_S_over_dx:.4f} | "
        "{xi_S_over_dz:.4f} | {active_length_over_dx:.4f} | "
        "{active_length_over_dz:.4f} | {isolated_r50_diameter_over_dx:.4f} | "
        "{retained_q_modes} | {retained_normal_velocity_modes} |".format(**row)
        for row in scales["grids"]
    )
    output_lines = "\n".join(f"- `{item}`" for item in output_paths.values())
    package_lines = "\n".join(
        f"- {name}: `{version}`"
        for name, version in runtime["package_versions"].items()
    )
    identity = validate_run_identity(
        runs[0].metadata, path=runs[0].artifact.directory / "metadata.json"
    )
    mode = getattr(args, "mode", "space")
    if mode == "space":
        report_title = "Defect-core spatial-resolution precheck"
        study_description = (
            "Shendruk-inspired A=18 defect-core spatial-resolution precheck "
            "at T=1 under the current PSSolver model"
        )
        scope_text = (
            "It may support a tentative resolution recommendation for a later "
            "long benchmark.  It does not establish spatial convergence of "
            "long-time Fig. 4 dynamics or statistics."
        )
    elif mode == "time":
        report_title = "Defect-core R512 time-step sensitivity precheck"
        study_description = (
            "Shendruk-inspired A=18 defect-core R512 time-step sensitivity "
            "precheck at T=1 under the current PSSolver model"
        )
        scope_text = (
            "It tests the dt=0.005 core change against the R320-to-R512 "
            "spatial-error floor supplied by the space study.  It does not "
            "establish a temporal convergence order or long-time Fig. 4 "
            "dynamics and statistics."
        )
    else:
        raise ValueError(f"unsupported README mode {mode!r}")
    if identity["script"] == BERIS_EDWARDS_BENCHMARK_SCRIPT:
        model_difference_text = (
            "This model retains the complete one-constant Beris--Edwards "
            "reactive, distortion, and active nematic stresses up to an "
            "isotropic contribution absorbed into pressure.  It still differs "
            "from the cited paper through quasistatic Stokes momentum, the "
            "selected `zero_mean` free-slip plug-mode convention, and the "
            "spectral `cubic_half` discretization."
        )
    else:
        model_difference_text = (
            "The model differs from the cited paper in four important ways: "
            "it uses quasistatic Stokes rather than a complete momentum "
            "equation; the Stokes solve omits some passive elastic/reactive "
            "nematic stresses; `zero_mean` fixes the free-slip tangential plug "
            "mode in a selected reference frame; and the spectral `cubic_half` "
            "discretization differs from the paper's original scheme."
        )
    text = f"""# {report_title}

This directory reports a **{study_description}**.

{scope_text}

Model script: `{identity["script"]}`; model variant: `{identity["model_variant"]}`.

{model_difference_text}  Here
`mu_S={scales['mu_S']:.12g}`, `tau_S=gamma/mu_S={scales['tau_S']:.12g}`, and
`T/tau_S={scales['T_over_tau_S']:.6g}`.  Thus T=1 is much shorter than the
amplitude-relaxation time and the final core still strongly remembers the
analytic initialization.

## Data provenance

`Q2D_initial.npy` is used only to calibrate the unprojected analytic generator
and periodic xy interpolation.  `Q_0.npy` is the cubic-half projected state that
actually entered the evolution and is the reported projected-initial state.
Its five component streams are re-hashed in bounded x chunks and must reproduce
metadata `initial_condition.projected_q_sha256` before analysis continues.
Final metrics use the completed `Q_<step>.npy` snapshot.  Three-dimensional
files are opened with `numpy.load(..., mmap_mode="r")` and processed in x chunks
of {args.chunk_size}; a complete R512 five-component 3-D Q array is never copied
or diagonalized at once.

Run directories:
{chr(10).join(f'- `{run.artifact.directory}`' for run in runs)}

Analysis command:

```text
{runtime['command']}
```

Analysis executable/prefix: `{runtime['executable']}` / `{runtime['prefix']}`.
Loaded modules: `{runtime['loaded_modules']}`.  Saved simulation provenance is
authoritative for the simulation Git/environment/job context; the analysis-time
best-effort Git probes in JSON are intentionally separate and are not used to
infer the simulation commit.  Provenance directory: `{simulation['provenance_directory']}`.
Job-script directory: `{simulation['jobs_directory']}`.

Package versions:
{package_lines}

Outputs (all inside the newly created analysis directory):
{output_lines}

## Physical scales and retained cutoffs

`xi_S=sqrt(L1/mu_S)={scales['xi_S']:.12g}` and
`ell_a=sqrt(K/zeta)={scales['active_length']:.12g}`.  The last two columns are
validated directly against each completed run's metadata; raw collocation
points alone are not treated as proof that the cubic-half projected core is
resolved.

| grid | dx | dz | xi_S/dx | xi_S/dz | ell_a/dx | ell_a/dz | 2 r50 isolated/dx | retained Q modes | retained normal-u modes |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
{scale_rows}

## Common-coordinate evaluation

The z collocation is cell centred, `z_j/H=(j+1/2)/Nz`.  Values at z/H =
{', '.join(str(value) for value in args.z_fractions)} are evaluated from the
orthonormal DCT-II coefficients with `phi_0=1/sqrt(Nz)` and
`phi_k=sqrt(2/Nz) cos(pi k z/H)`; equal array indices and endpoint-grid
interpolation are not used.  xy samples first interpolate all five Q components
with a periodic order-{args.interpolation_order} spline (`grid-wrap`) whose
Fourier-mode and seam accuracy is tested, then compute `S=lambda_max(Q)` in
float64.  A runtime analytic periodic-mode/seam calibration is saved for every
grid and compared with each paired profile RMS difference; this calibration is
an interpolation check, not a bound on field-representation error.

The continuous raw reference is the complete multi-defect product

`S_cont(x,y) = S_initial product_j tanh(d_periodic((x,y),x_j)/core_radius)`.

The isolated value `1.5 atanh(0.5) = {1.5 * np.arctanh(0.5):.10f}` is only a
sanity reference.  It is not a required monotone-convergence target for the
multi-defect, spectrally projected initial condition.
Continuous-reference metrics use `dr={continuous_reference_dr:g}` and
{continuous_reference_angular_samples} angles, then the high-accuracy profile
is resampled to the common analysis radii.  Per-defect signed/absolute errors
and profile norms are reported separately for `raw_q2d` and `projected_q0`.

## Detection, charge, and matching

Low-S local minima are candidates only.  A candidate is assigned charge
`+/-1/2` only when director winding and `(Qxx-Qyy)+2iQxy` winding agree on two
physical rings.  Every ring must also satisfy `min(S)>=0.05*S_bulk`,
`min(abs((Qxx-Qyy)+2iQxy))>=0.05*S_bulk`, and minimum in-plane director
projection 0.15.  Candidate centres are refined before charge classification;
each winding ring must remain inside the target candidate's periodic Voronoi
half-cell, with one xy grid spacing of additional clearance from every other
candidate.  Nearest-candidate distance, required clearance, and isolation
status are recorded.  A failed condition is recorded as a reason.  Otherwise
an unreliable candidate is `ambiguous` and never inherits an initial charge.
Reliable candidates are Hungarian-matched within the same charge using
periodic minimum-image distance and a {args.match_radius:g} maximum distance.
Missing, ambiguous, and unmatched candidates remain explicit rows.

## Core metrics

Profiles use radii 0 to {args.r_max:g} with dr={args.dr:g} and
{args.angular_samples} angles.  `P(r)` is the angular median of owned samples.
At least {PROFILE_MINIMUM_ANGULAR_FRACTION:.0%} of angular samples are required
for a profile bin.  `S_far` is the median over owned samples with `3 <= r <= 4`
after excluding values below
`{S_FAR_LOW_ORDER_CUTOFF_FRACTION:g}*S_bulk`; at least
{S_FAR_MINIMUM_VALID_FRACTION:.0%} of the full ring must remain.
Its median absolute deviation and owned valid fraction are reported.  The
10th--90th percentile spread of angle-resolved absolute-r50 contours, divided
by twice their median, is reported as `r50_contour_anisotropy`.

* `r50_absolute`, `r90_absolute`: first outward crossings of
  `P(r)=0.5*S_bulk` and `P(r)=0.9*S_bulk`.
* `r50_local`, `r90_local`: first outward crossings of
  `(P(r)-S_min)/(S_far-S_min)=0.5` and `=0.9`.

The local scale is valid only when `S_far-S_min` is at least
`{LOCAL_REFERENCE_MINIMUM_SPAN_FRACTION:g}*S_bulk` (and above a float64
roundoff floor).  If not, all local radii, deficits, and low-S areas are NaN
with an explicit quality reason.  A crossing is interpolated only between
adjacent finite radial bins; missing bins are never compressed away or bridged.

No crossing, an invalid local scale, or multiple outward crossings are stored
as NaN in CSV (JSON null) together with a reason; the analysis neither chooses
an unstable branch nor invents an extrapolated radius.  Crossing counts and
profile-valid fractions remain explicit quality diagnostics.

Area integrals use midpoint polar quadrature on `0 <= r <= {args.r_max:g}`.
Each point is owned by the nearest detected core (periodic Voronoi ownership),
so overlapping core disks are not counted twice.  With `Omega_i` the owned
domain:

* `core_deficit_absolute = integral_Omega max(1-S/S_bulk,0) dA`;
* `core_deficit_local = integral_Omega max(1-S/S_far,0) dA`;
* `low_s_area_absolute = integral_Omega 1[S<0.5*S_bulk] dA`;
* `low_s_area_local = integral_Omega 1[S<0.5*S_far] dA`.

Biaxiality is `1-6 Tr(Q^3)^2/Tr(Q^2)^3` only where
`Tr(Q^2) > {BIAXIALITY_RELATIVE_TRQ2_CUTOFF:g} S_ref^2`, using `S_bulk` for raw
and projected initial states and reliable `S_far` for the final state.  Q near
zero is invalid rather than 0/0.  Masked values do not enter statistics; valid
fractions and thresholds ten times lower/higher are included.

Centres are local minima of periodically interpolated S in minimum-image local
coordinates.  CSV reports centre-fit status, absolute error/displacement,
error over h, and error over the continuous-reference r50.  Radial profile
differences are computed for the same defect ID and reliable charge with

`E = sqrt(integral |P_c-P_f|^2 r dr) / max(sqrt(integral |P_f|^2 r dr), S_scale*sqrt(integral r dr))`.

The JSON contains each defect's value, charge-conditioned/all-defect paired
distributions, MAD-based outlier IDs, and valid/missing/ambiguous/extra-candidate
counts.  Relative percentages are suppressed for near-zero S_min,
displacement, and biaxiality metrics; their absolute changes remain available.

All adjacent pairwise changes and 5%/10% practical thresholds are reported.
There is no hard requirement that the 320->512 raw change be smaller than the
256->320 change because the refinement ratios differ.  Any unequal-grid
three-level fit is auxiliary only.  In time mode the first-order Richardson
estimate is twice the difference between dt=0.005 and dt=0.0025 metrics; a 20%
gate is applied to that scaled estimate only after R512 input identity,
topology, matching, and crossing branch checks.  When the spatial difference is
below its predefined physical tolerance, the gate is marked indeterminate and
reports the absolute-tolerance comparison separately; that tolerance never
silently replaces the 20% criterion.

For a finite, monotone, same-sign three-level scalar sequence, the auxiliary
fit is `M(h)=M_inf+C*h^p`, with p obtained from the unequal-grid difference
ratio.  It is accepted only for positive p with a well-conditioned scaled
Jacobian and stable six-way input perturbation.  Candidate p, algebraic
residual, condition number, sensitivity, and every rejection reason are saved.
Because three points determine three parameters, the residual is not an
independent goodness-of-fit test and this fit is never itself a convergence
proof.

The practical 5% median and 10% individual thresholds are working tolerances,
not mathematical pass/fail theorems.  Raw sampling, projected Q0, and T=1
dynamic results must be interpreted separately.  Even a successful result is
only a T=1 short-time core-resolution gate; long-time dynamics, statistics,
and ensemble variability remain untested.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs = [load_core_run(path) for path in args.run_dirs]
    validate_core_runs(runs, args.mode)
    validate_task_contract(runs, args.mode)
    output_paths = reserve_outputs(args.output_dir.expanduser().resolve())

    reference = runs[0]
    parameters = reference.metadata["model"]["parameters"]
    S_bulk = float(parameters["S_bulk"])
    S_initial = float(parameters["S_initial"])
    generator = reference.metadata["initial_defect_gas"]
    core_radius = float(generator["core_radius"])

    all_rows: list[dict[str, Any]] = []
    initial_profiles: list[dict[str, Any]] = []
    final_profiles: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    interpolation_calibrations: list[dict[str, Any]] = []
    final_planes: dict[str, np.ndarray] = {}

    for run in runs:
        print(f"[{run.label}] loading Q2D calibration plane", flush=True)
        q2d = np.asarray(_mmap_float_array(run.q2d_path, (*run.shape[:2], 5)), dtype=np.float64)
        raw_rows, raw_profiles, _ = analyze_q_plane(
            q2d,
            run,
            state=RAW_STATE,
            step=0,
            z_fraction=None,
            S_bulk=S_bulk,
            r_max=args.r_max,
            dr=args.dr,
            angular_samples=args.angular_samples,
            match_radius=args.match_radius,
            interpolation_order=args.interpolation_order,
        )
        all_rows.extend(raw_rows)
        initial_profiles.extend(raw_profiles)
        raw_sampler = PeriodicQSampler(q2d, run.lengths[:2], order=args.interpolation_order)
        calibrations.append(
            {
                "resolution": run.label,
                **q2d_interpolation_calibration(
                    raw_sampler,
                    run.defect_positions,
                    run.lengths[:2],
                    core_radius=core_radius,
                    S_initial=S_initial,
                    r_max=args.r_max,
                    dr=args.dr,
                    angular_samples=args.angular_samples,
                ),
            }
        )
        interpolation_calibrations.append(
            {
                "resolution": run.label,
                **periodic_interpolation_synthetic_calibration(
                    run.shape[:2],
                    run.lengths[:2],
                    order=args.interpolation_order,
                ),
            }
        )

        print(f"[{run.label}] DCT-modal evaluation of Q_0", flush=True)
        initial_planes = evaluate_q_dct_planes(
            run.q0_path,
            run.shape,
            args.z_fractions,
            chunk_size=args.chunk_size,
        )
        print(f"[{run.label}] DCT-modal evaluation of final Q", flush=True)
        evolved_planes = evaluate_q_dct_planes(
            run.artifact.q_path,
            run.shape,
            args.z_fractions,
            chunk_size=args.chunk_size,
        )
        final_planes[run.label] = evolved_planes
        for z_index, z_fraction in enumerate(args.z_fractions):
            rows, profiles, _ = analyze_q_plane(
                initial_planes[:, :, z_index, :],
                run,
                state=PROJECTED_STATE,
                step=0,
                z_fraction=float(z_fraction),
                S_bulk=S_bulk,
                r_max=args.r_max,
                dr=args.dr,
                angular_samples=args.angular_samples,
                match_radius=args.match_radius,
                interpolation_order=args.interpolation_order,
            )
            all_rows.extend(rows)
            initial_profiles.extend(profiles)

            rows, profiles, _ = analyze_q_plane(
                evolved_planes[:, :, z_index, :],
                run,
                state=FINAL_STATE,
                step=run.artifact.steps,
                z_fraction=float(z_fraction),
                S_bulk=S_bulk,
                r_max=args.r_max,
                dr=args.dr,
                angular_samples=args.angular_samples,
                match_radius=args.match_radius,
                interpolation_order=args.interpolation_order,
            )
            all_rows.extend(rows)
            final_profiles.extend(profiles)

    continuous_metrics: dict[int, dict[str, Any]] = {}
    continuous_profiles: dict[int, dict[str, np.ndarray]] = {}
    continuous_reference_dr = min(args.dr / 10.0, 0.005)
    continuous_reference_angular_samples = max(512, 4 * args.angular_samples)
    common_radii = np.arange(0.0, args.r_max + 0.5 * args.dr, args.dr)
    for defect_id, charge in enumerate(reference.defect_charges):
        metrics, high_precision_profile = continuous_reference_profile(
            defect_id,
            reference.defect_positions,
            reference.lengths[:2],
            core_radius=core_radius,
            S_initial=S_initial,
            S_bulk=S_bulk,
            r_max=args.r_max,
            dr=continuous_reference_dr,
            angular_samples=continuous_reference_angular_samples,
        )
        profile = resample_profile(high_precision_profile, common_radii)
        continuous_metrics[defect_id] = metrics
        continuous_profiles[defect_id] = profile
        initial_profiles.append(
            {
                "state": "continuous_multidefect_reference",
                "resolution": "continuous",
                "step": 0,
                "z_fraction": None,
                "defect_id": defect_id,
                "charge": float(charge),
                **profile,
            }
        )
        all_rows.append(
            {
                "state": "continuous_multidefect_reference",
                "resolution": "continuous",
                "run_dir": "analytic",
                "step": 0,
                "z_fraction": None,
                "nx": math.nan,
                "ny": math.nan,
                "nz": math.nan,
                "dt": math.nan,
                "defect_id": defect_id,
                "initial_charge": float(charge),
                "detected_charge": float(charge),
                "match_status": "matched",
                "match_reason": "continuous_reference_coordinate",
                "initial_x": float(reference.defect_positions[defect_id, 0]),
                "initial_y": float(reference.defect_positions[defect_id, 1]),
                "center_x": float(reference.defect_positions[defect_id, 0]),
                "center_y": float(reference.defect_positions[defect_id, 1]),
                "center_fit_status": "analytic_exact",
                "isolated_r50_sanity": 1.5 * float(np.arctanh(0.5)),
                "r50_absolute_minus_isolated_r50": float(
                    metrics.get("r50_absolute", math.nan)
                )
                - 1.5 * float(np.arctanh(0.5)),
                **metrics,
            }
        )

    run_by_label = {run.label: run for run in runs}
    for row in all_rows:
        defect_id = row.get("defect_id")
        if not isinstance(defect_id, (int, np.integer)):
            continue
        state = row.get("state")
        reference_r50 = float(continuous_metrics[int(defect_id)].get("r50_absolute", math.nan))
        displacement = float(row.get("center_displacement", math.nan))
        row["continuous_reference_r50_absolute"] = reference_r50
        run = run_by_label.get(str(row.get("resolution")))
        if state in (RAW_STATE, PROJECTED_STATE):
            row["center_error_to_continuous"] = displacement
            row["center_error_over_h"] = (
                displacement / run.artifact.resolution_h
                if run is not None and math.isfinite(displacement)
                else math.nan
            )
            row["center_error_over_reference_r50"] = (
                displacement / reference_r50
                if math.isfinite(displacement)
                and math.isfinite(reference_r50)
                and reference_r50 > 0.0
                else math.nan
            )
        elif state == FINAL_STATE:
            row["center_displacement_from_initial"] = displacement
            row["center_displacement_over_h"] = (
                displacement / run.artifact.resolution_h
                if run is not None and math.isfinite(displacement)
                else math.nan
            )
            row["center_displacement_over_continuous_reference_r50"] = (
                displacement / reference_r50
                if math.isfinite(displacement)
                and math.isfinite(reference_r50)
                and reference_r50 > 0.0
                else math.nan
            )
        elif state == "continuous_multidefect_reference":
            row["center_error_to_continuous"] = 0.0
            row["center_error_over_reference_r50"] = 0.0

    reference_profile_errors = continuous_reference_errors(
        all_rows,
        initial_profiles,
        continuous_metrics,
        continuous_profiles,
        S_scale=S_bulk,
    )

    summaries = summarize_groups(all_rows)
    status_counts = summarize_status_counts(all_rows)
    reference_error_summaries = continuous_reference_error_summaries(all_rows)
    reference_error_trends = (
        continuous_reference_error_trends(all_rows, runs)
        if args.mode == "space"
        else []
    )
    changes = pairwise_metric_changes(summaries, runs, mode=args.mode)
    paired_changes = paired_individual_metric_changes(all_rows, runs, mode=args.mode)
    profile_changes = profile_pairwise_changes(
        [*initial_profiles, *final_profiles], runs, mode=args.mode, S_scale=S_bulk
    )
    interpolation_resolution_checks = interpolation_error_vs_profile_changes(
        interpolation_calibrations, profile_changes
    )
    spatial_fits = (
        generalized_spatial_fits(summaries, runs) if args.mode == "space" else []
    )
    representation_changes = raw_to_projected_changes(summaries)
    representation_profile_changes = raw_to_projected_profile_changes(
        initial_profiles, S_scale=S_bulk
    )
    z_variations = z_fraction_variations(summaries)
    scales = physical_scale_summary(runs)
    provenance = analysis_and_simulation_provenance(args, runs, args.output_dir.resolve())
    if args.mode == "time" and args.space_core_summary is not None:
        time_space_gate: dict[str, Any] = time_vs_space_error_gate(
            args.space_core_summary,
            paired_changes,
            profile_changes,
            runs,
            status_counts,
        )
    else:
        time_space_gate = {
            "applied": False,
            "reason": (
                "space mode"
                if args.mode == "space"
                else "provide --space-core-summary to apply the time-vs-space gate"
            ),
        }
    run_manifest = input_run_manifest(runs)
    write_csv(output_paths["csv"], all_rows)
    save_profiles(output_paths["initial_npz"], initial_profiles)
    projected_initial_profiles = [
        profile
        for profile in initial_profiles
        if profile["state"] == PROJECTED_STATE
    ]
    save_profiles(output_paths["projected_initial_npz"], projected_initial_profiles)
    save_profiles(output_paths["final_npz"], final_profiles)

    report = {
        "schema_version": 2,
        "analysis": "fig4_defect_core_convergence",
        "scope": (
            "current PSSolver model, T=1 Shendruk-inspired A=18 "
            "defect-core spatial-resolution precheck"
        ),
        "automatic_long_time_convergence_claim": False,
        "mode": args.mode,
        "run_directories": [str(run.artifact.directory) for run in runs],
        "model_identity": {
            field: run_manifest[0][field]
            for field in (
                "script",
                "model_variant",
                "model_stage",
                "Q_convention",
                "implementation_provenance",
            )
        },
        "input_run_manifest": run_manifest,
        "defect_table_consistent": True,
        "projected_q0_hash_verified_for_all_runs": True,
        "continuous_defects": {
            "positions": reference.defect_positions,
            "charges": reference.defect_charges,
        },
        "isolated_r50_sanity": 1.5 * float(np.arctanh(0.5)),
        "continuous_reference": "all-defect periodic tanh product",
        "continuous_reference_numerics": {
            "dr": continuous_reference_dr,
            "angular_samples": continuous_reference_angular_samples,
            "profiles_resampled_to_common_radii_dr": args.dr,
        },
        "physical_scales_and_retained_modes": scales,
        "provenance": provenance,
        "q2d_interpolation_calibration": calibrations,
        "periodic_interpolation_synthetic_calibration": interpolation_calibrations,
        "interpolation_error_vs_profile_changes": interpolation_resolution_checks,
        "configuration": {
            "z_fractions": args.z_fractions,
            "r_max": args.r_max,
            "dr": args.dr,
            "angular_samples": args.angular_samples,
            "match_radius": args.match_radius,
            "chunk_size": args.chunk_size,
            "interpolation": f"periodic order-{args.interpolation_order} spline, grid-wrap",
            "z_evaluation": "orthonormal cell-centred DCT-II modal evaluation",
            "S_definition": "lambda_max(Q)",
            "analysis_dtype": "float64",
            "winding_ring_min_S_over_S_bulk": 0.05,
            "winding_ring_min_in_plane_q_amplitude_over_S_bulk": 0.05,
            "winding_ring_min_director_projection": 0.15,
            "winding_ring_neighbor_clearance_multiplier": WINDING_RING_NEIGHBOR_CLEARANCE_MULTIPLIER,
            "biaxiality_relative_trQ2_cutoff": BIAXIALITY_RELATIVE_TRQ2_CUTOFF,
            "S_far_minimum_valid_fraction": S_FAR_MINIMUM_VALID_FRACTION,
            "S_far_low_order_cutoff_fraction_of_S_bulk": S_FAR_LOW_ORDER_CUTOFF_FRACTION,
            "profile_minimum_angular_fraction": PROFILE_MINIMUM_ANGULAR_FRACTION,
            "local_reference_minimum_span_fraction_of_S_bulk": LOCAL_REFERENCE_MINIMUM_SPAN_FRACTION,
        },
        "charge_conditioned_summaries": summaries,
        "status_counts": status_counts,
        "continuous_reference_profile_errors_by_defect": reference_profile_errors,
        "continuous_reference_error_summaries": reference_error_summaries,
        "continuous_reference_error_trends": reference_error_trends,
        "group_median_pairwise_changes": changes,
        "pairwise_changes": changes,
        "paired_individual_metric_changes_and_distributions": paired_changes,
        "charge_conditioned_profile_pairwise_changes": profile_changes,
        "raw_to_projected_representation_changes": representation_changes,
        "raw_to_projected_profile_changes_by_defect": representation_profile_changes,
        "z_fraction_variations": z_variations,
        "unequal_grid_three_level_fits": spatial_fits,
        "practical_thresholds": {"median_preferred": 0.05, "individual_outlier": 0.10},
        "time_error_rule": (
            "first-order Richardson estimate is 2*abs(metric_dt005-metric_dt0025)"
        ),
        "time_vs_space_error_gate": time_space_gate,
        "time_gate_absolute_tolerances": TIME_GATE_ABSOLUTE_TOLERANCES,
    }
    output_paths["json"].write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot_initial_r50(output_paths["initial_plot"], all_rows, runs)
    _plot_projected_initial_profiles(
        output_paths["projected_initial_plot"], projected_initial_profiles
    )
    _plot_final_profiles(output_paths["final_plot"], final_profiles)
    _plot_changes(output_paths["change_plot"], changes)
    positive_id = common_reliably_matched_defect_id(
        all_rows, runs, charge=0.5, z_fraction=0.5
    )
    negative_id = common_reliably_matched_defect_id(
        all_rows, runs, charge=-0.5, z_fraction=0.5
    )
    midplane_index = next(
        index
        for index, value in enumerate(args.z_fractions)
        if math.isclose(value, 0.5, rel_tol=0.0, abs_tol=1.0e-12)
    )
    if positive_id is None:
        _plot_unavailable_zoom(
            output_paths["positive_zoom"],
            charge=0.5,
            reason="no +1/2 defect is reliably matched in every run at z/H=0.5",
        )
    else:
        _plot_zoom(
            output_paths["positive_zoom"],
            runs,
            final_planes,
            all_rows,
            positive_id,
            S_bulk=S_bulk,
            interpolation_order=args.interpolation_order,
            z_index=midplane_index,
            z_fraction=0.5,
        )
    if negative_id is None:
        _plot_unavailable_zoom(
            output_paths["negative_zoom"],
            charge=-0.5,
            reason="no -1/2 defect is reliably matched in every run at z/H=0.5",
        )
    else:
        _plot_zoom(
            output_paths["negative_zoom"],
            runs,
            final_planes,
            all_rows,
            negative_id,
            S_bulk=S_bulk,
            interpolation_order=args.interpolation_order,
            z_index=midplane_index,
            z_fraction=0.5,
        )
    write_readme(
        output_paths["readme"],
        args=args,
        runs=runs,
        output_paths=output_paths,
        provenance=provenance,
        scales=scales,
        continuous_reference_dr=continuous_reference_dr,
        continuous_reference_angular_samples=continuous_reference_angular_samples,
    )
    for path in output_paths.values():
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
