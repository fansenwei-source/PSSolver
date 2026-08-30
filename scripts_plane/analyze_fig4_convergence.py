#!/usr/bin/env python3
"""Compare completed single-point Fig. 4 numerical-validation runs.

Time-step studies compare final fields on one common grid.  Spatial studies
compare resolution-independent scalar statistics only: this script deliberately
does not call a collocation-grid difference a pointwise error without a
basis-aware FFT/DCT/DST transfer.  Dealiasing studies compare final fields on
the same grid against ``cubic_half`` and report sensitivity, not convergence.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pssolver.models.active_nematics import S_from_Q
from pssolver.snapshots import validate_active_nematic_q_source


Q_COMPONENT_COUNT = 5
VELOCITY_COMPONENT_COUNT = 3
DIAGNOSTIC_FIELDS = (
    "div_max",
    "div_rms",
    "div_rel",
    "schur_iterations",
    "schur_abs_residual",
    "schur_rel_residual",
    "wall_normal_momentum_max",
    "wall_normal_momentum_rms",
)
NONCOMPARABLE_METADATA_KEYS = {
    "status",
    "validation_config_sha256",
    "completed_steps",
    "elapsed_seconds",
    "device",
    "save_start_step",
    "save_interval",
    "diagnostic_interval",
    "save_hydrodynamics",
    "retained_q_modes",
    "retained_normal_velocity_modes",
}
SUPPORTED_RUN_IDENTITIES = {
    "Plane_fig4_benchmark.py": None,
    "Plane_beris_edwards_stokes.py": (
        "beris_edwards_complete_nematic_stress_stokes"
    ),
}
DEALIAS_RULE_FRACTIONS = {
    "none": None,
    "two_thirds": 2.0 / 3.0,
    "cubic_half": 0.5,
}


@dataclass(frozen=True)
class RunArtifact:
    label: str
    directory: Path
    metadata: dict[str, Any]
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    dt: float
    steps: int
    final_time: float
    q_path: Path
    u_path: Path
    p_path: Path
    diagnostics_path: Path

    @property
    def cell_sizes(self) -> tuple[float, float, float]:
        return tuple(length / count for length, count in zip(self.lengths, self.shape))

    @property
    def resolution_h(self) -> float:
        return max(self.cell_sizes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze final Q/u/p fields and solver diagnostics from completed "
            "supported Fig. 4 numerical-validation runs."
        )
    )
    parser.add_argument(
        "--mode", choices=("time", "space", "dealias"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        default=None,
        help=(
            "Output basename. Defaults to fig4_<mode>_convergence, except "
            "dealias mode uses fig4_dealias_sensitivity."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Number of x planes processed at once. Default: 8.",
    )
    parser.add_argument(
        "--order-dt",
        type=float,
        nargs=3,
        metavar=("COARSE", "MEDIUM", "FINE"),
        default=None,
        help=(
            "Explicit geometrically refined dt triplet for the observed-order "
            "estimate. Valid only with --mode time. Without this option, the "
            "finest geometrically refined triplet of adjacent supplied dt "
            "levels is used."
        ),
    )
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        help="Two or more completed benchmark run directories.",
    )
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        parser.error("at least two run directories are required")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.order_dt is not None and args.mode != "time":
        parser.error("--order-dt is valid only with --mode time")
    return args


def validate_run_identity(
    metadata: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    """Validate one explicit script/model identity and return report metadata.

    The allowlist is intentionally narrow. In particular, the intermediate
    ``Plane_shendruk_stokes.py`` implementation is not accepted, and the
    legacy and complete-stress models remain non-comparable because ``script``
    and ``model.variant`` stay in the strict comparison signature.
    """
    script = metadata.get("script")
    if script not in SUPPORTED_RUN_IDENTITIES:
        supported = ", ".join(sorted(SUPPORTED_RUN_IDENTITIES))
        raise ValueError(
            f"{path} declares unsupported benchmark script {script!r}; "
            f"supported scripts: {supported}"
        )

    model = metadata.get("model")
    if not isinstance(model, Mapping) or model.get("name") != "active_nematics":
        raise ValueError(f"{path} must declare model.name='active_nematics'")
    expected_variant = SUPPORTED_RUN_IDENTITIES[script]
    actual_variant = model.get("variant")
    if actual_variant != expected_variant:
        raise ValueError(
            f"{path} has model.variant={actual_variant!r} for {script}; "
            f"expected {expected_variant!r}"
        )

    implementation = metadata.get("implementation_provenance")
    if script == "Plane_beris_edwards_stokes.py":
        if not isinstance(implementation, Mapping) or not implementation:
            raise ValueError(
                f"{path} requires non-empty implementation_provenance"
            )
        files = implementation.get("files")
        if not isinstance(files, Mapping) or not files:
            raise ValueError(
                f"{path} implementation_provenance.files must be a "
                "non-empty mapping"
            )
        for source_path, digest in files.items():
            if not isinstance(source_path, str) or not source_path:
                raise ValueError(
                    f"{path} implementation provenance file names must be "
                    "non-empty strings"
                )
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in digest
                )
            ):
                raise ValueError(
                    f"{path} implementation hash for {source_path!r} must be "
                    "a canonical 64-character lowercase SHA-256 hex digest"
                )
    elif implementation is not None and not isinstance(implementation, Mapping):
        raise ValueError(f"{path} implementation_provenance must be a mapping")

    validation_config_sha256 = metadata.get("validation_config_sha256")
    if script == "Plane_beris_edwards_stokes.py" and (
        not isinstance(validation_config_sha256, str)
        or len(validation_config_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in validation_config_sha256
        )
    ):
        raise ValueError(
            f"{path} requires validation_config_sha256 to be a canonical "
            "64-character lowercase SHA-256 hex digest"
        )
    return {
        "script": script,
        "model_name": model["name"],
        "model_variant": actual_variant,
        "model_stage": model.get("stage"),
        "implementation_provenance": (
            None if implementation is None else copy.deepcopy(dict(implementation))
        ),
        "validation_config_sha256": validation_config_sha256,
    }


def _positive_float(value: Any, *, name: str, path: Path) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{path} has invalid {name}={value!r}")
    return float(value)


def _positive_int(value: Any, *, name: str, path: Path) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path} has invalid {name}={value!r}")
    return value


def _load_array_header(
    path: Path,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required convergence field {path}; rerun with "
            "--save-hydrodynamics when u/p are absent."
        )
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != expected_shape:
        raise ValueError(
            f"{path} has shape {values.shape}, expected {expected_shape}"
        )
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{path} must contain floating values, got {values.dtype}")
    return values


def load_run(directory: str | Path) -> RunArtifact:
    directory = Path(directory).expanduser().resolve()
    metadata_path = directory / "metadata.json"
    complete_path = directory / "COMPLETE"
    if not complete_path.is_file():
        raise FileNotFoundError(f"Missing completion marker: {complete_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing benchmark metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError(f"{metadata_path} must contain schema_version=1 metadata")
    if metadata.get("status") != "complete":
        raise ValueError(f"{metadata_path} must declare status='complete'")
    validate_run_identity(metadata, path=metadata_path)
    validate_active_nematic_q_source(directory, require_S_initial=True)

    solver = metadata.get("solver")
    if not isinstance(solver, dict):
        raise ValueError(f"{metadata_path} is missing solver metadata")
    raw_shape = solver.get("shape")
    if (
        not isinstance(raw_shape, (list, tuple))
        or len(raw_shape) != 3
    ):
        raise ValueError(f"{metadata_path} has invalid solver.shape={raw_shape!r}")
    shape = tuple(
        _positive_int(value, name=f"solver.shape[{axis}]", path=metadata_path)
        for axis, value in enumerate(raw_shape)
    )
    raw_lengths = solver.get("lengths")
    if (
        not isinstance(raw_lengths, (list, tuple))
        or len(raw_lengths) != 3
    ):
        raise ValueError(
            f"{metadata_path} has invalid solver.lengths={raw_lengths!r}"
        )
    lengths = tuple(
        _positive_float(value, name=f"solver.lengths[{axis}]", path=metadata_path)
        for axis, value in enumerate(raw_lengths)
    )
    dt = _positive_float(solver.get("dt"), name="solver.dt", path=metadata_path)
    steps = _positive_int(
        solver.get("steps"), name="solver.steps", path=metadata_path
    )
    completed_steps = _positive_int(
        metadata.get("completed_steps"),
        name="completed_steps",
        path=metadata_path,
    )
    if completed_steps != steps:
        raise ValueError(
            f"{metadata_path} completed_steps={completed_steps} differs from "
            f"solver.steps={steps}"
        )

    q_path = directory / f"Q_{completed_steps}.npy"
    u_path = directory / f"u_{completed_steps}.npy"
    p_path = directory / f"p_{completed_steps}.npy"
    _load_array_header(q_path, (*shape, Q_COMPONENT_COUNT))
    _load_array_header(u_path, (*shape, VELOCITY_COMPONENT_COUNT))
    _load_array_header(p_path, shape)
    diagnostics_path = directory / "diagnostics.npy"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"Missing required diagnostics: {diagnostics_path}; rerun with "
            "--diagnostics."
        )

    return RunArtifact(
        label=directory.name,
        directory=directory,
        metadata=metadata,
        shape=shape,
        lengths=lengths,
        dt=dt,
        steps=steps,
        final_time=dt * steps,
        q_path=q_path,
        u_path=u_path,
        p_path=p_path,
        diagnostics_path=diagnostics_path,
    )


def _comparison_signature(metadata: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Return metadata that must agree within one numerical-validation study."""
    if mode == "dealias":
        _dealias_configuration(metadata)

    signature = copy.deepcopy(dict(metadata))
    for key in NONCOMPARABLE_METADATA_KEYS:
        signature.pop(key, None)

    solver = signature.get("solver")
    if not isinstance(solver, dict):
        raise ValueError("metadata is missing solver")
    solver.pop("save_interval", None)
    if mode == "time":
        solver.pop("dt", None)
        solver.pop("steps", None)
        signature.pop("dt", None)
        signature.pop("steps", None)
        _normalize_time_spectral_refresh_signature(signature)
    elif mode == "space":
        solver.pop("shape", None)
        signature.pop("shape", None)
        initial_condition = signature.get("initial_condition")
        if isinstance(initial_condition, dict):
            # These hashes certify the exact discrete arrays. They must differ
            # when the grid changes, while all generator parameters stay in
            # the strict comparison signature.
            initial_condition.pop("raw_q_sha256", None)
            initial_condition.pop("projected_q_sha256", None)
    elif mode == "dealias":
        signature.pop("dealias_rule")
        signature.pop("dealias_fraction")

        numerics = signature["numerics"]
        dealiasing = numerics["dealiasing"]
        dealiasing.pop("rule")
        dealiasing.pop("fraction")

        initial_condition = signature["initial_condition"]
        # Projection changes the discrete starting array by construction. The
        # raw hash remains strict and therefore certifies a common unprojected
        # initial condition.
        initial_condition.pop("projected_q_sha256")
    else:  # pragma: no cover - protected by argparse and caller validation
        raise ValueError(f"unknown validation mode {mode!r}")
    return signature


def _sha256_string(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"metadata has invalid {name}={value!r}")
    return value


def _retained_mode_counts(value: Any, *, name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(count) is not int or count <= 0 for count in value)
    ):
        raise ValueError(
            f"metadata {name} must contain three positive integer mode counts"
        )
    return tuple(value)


def _dealias_configuration(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate duplicated dealias provenance and return its configuration."""
    if "dealias_rule" not in metadata or "dealias_fraction" not in metadata:
        raise ValueError(
            "dealias sensitivity requires top-level dealias_rule and "
            "dealias_fraction metadata"
        )
    top_rule = metadata["dealias_rule"]
    if top_rule not in DEALIAS_RULE_FRACTIONS:
        raise ValueError(f"metadata has unsupported dealias_rule={top_rule!r}")
    top_fraction = metadata["dealias_fraction"]

    numerics = metadata.get("numerics")
    if not isinstance(numerics, Mapping):
        raise ValueError("dealias sensitivity requires numerics metadata")
    dealiasing = numerics.get("dealiasing")
    if not isinstance(dealiasing, Mapping):
        raise ValueError(
            "dealias sensitivity requires numerics.dealiasing metadata"
        )
    if "rule" not in dealiasing or "fraction" not in dealiasing:
        raise ValueError(
            "numerics.dealiasing must declare both rule and fraction"
        )
    nested_rule = dealiasing["rule"]
    nested_fraction = dealiasing["fraction"]
    if nested_rule != top_rule or nested_fraction != top_fraction:
        raise ValueError(
            "top-level and numerics.dealiasing rule/fraction metadata disagree"
        )

    expected_fraction = DEALIAS_RULE_FRACTIONS[top_rule]
    if expected_fraction is None:
        fraction_matches = top_fraction is None
    else:
        fraction_matches = (
            not isinstance(top_fraction, bool)
            and isinstance(top_fraction, (int, float))
            and math.isfinite(top_fraction)
            and math.isclose(
                float(top_fraction),
                expected_fraction,
                rel_tol=1.0e-15,
                abs_tol=0.0,
            )
        )
    if not fraction_matches:
        raise ValueError(
            f"dealias rule {top_rule!r} requires fraction={expected_fraction!r}, "
            f"got {top_fraction!r}"
        )

    initial_condition = metadata.get("initial_condition")
    if not isinstance(initial_condition, Mapping):
        raise ValueError(
            "dealias sensitivity requires initial_condition metadata"
        )
    raw_hash = _sha256_string(
        initial_condition.get("raw_q_sha256"),
        name="initial_condition.raw_q_sha256",
    )
    projected_hash = _sha256_string(
        initial_condition.get("projected_q_sha256"),
        name="initial_condition.projected_q_sha256",
    )
    retained_q_modes = _retained_mode_counts(
        metadata.get("retained_q_modes"),
        name="retained_q_modes",
    )
    retained_normal_velocity_modes = _retained_mode_counts(
        metadata.get("retained_normal_velocity_modes"),
        name="retained_normal_velocity_modes",
    )
    return {
        "rule": top_rule,
        "fraction": top_fraction,
        "raw_q_sha256": raw_hash,
        "projected_q_sha256": projected_hash,
        "retained_q_modes": retained_q_modes,
        "retained_normal_velocity_modes": retained_normal_velocity_modes,
    }


def _normalize_time_spectral_refresh_signature(
    signature: dict[str, Any],
) -> None:
    """Normalize only the step count implied by one physical refresh period.

    A fixed physical refresh period necessarily corresponds to a different
    integer step interval when ``dt`` changes.  That derived integer is not a
    model/numerics difference in a time-step study.  Every other refresh field
    remains in the comparison signature, so mode, requested/effective physical
    time, phase, and any future provenance fields must still agree exactly.
    """
    numerics = signature.get("numerics")
    if numerics is None:
        return
    if not isinstance(numerics, dict):
        raise ValueError("metadata numerics must be a mapping")
    refresh = numerics.get("spectral_refresh")
    if refresh is None:
        return
    if not isinstance(refresh, dict):
        raise ValueError("metadata numerics.spectral_refresh must be a mapping")

    required = {
        "mode",
        "requested_interval_time",
        "requested_interval_steps",
        "effective_interval_steps",
        "effective_interval_time",
        "phase_origin_step",
    }
    missing = required - set(refresh)
    if missing:
        raise ValueError(
            "metadata numerics.spectral_refresh is missing fields "
            f"{sorted(missing)}"
        )

    refresh_mode = refresh["mode"]
    if (
        type(refresh["phase_origin_step"]) is not int
        or refresh["phase_origin_step"] != 0
    ):
        raise ValueError(
            "metadata numerics.spectral_refresh.phase_origin_step must be 0"
        )
    if refresh_mode == "physical_time":
        if refresh["requested_interval_steps"] is not None:
            raise ValueError(
                "physical_time spectral refresh must have "
                "requested_interval_steps=null"
            )
        _positive_float_value(
            refresh["requested_interval_time"],
            name="numerics.spectral_refresh.requested_interval_time",
        )
        _positive_int_value(
            refresh["effective_interval_steps"],
            name="numerics.spectral_refresh.effective_interval_steps",
        )
        _positive_float_value(
            refresh["effective_interval_time"],
            name="numerics.spectral_refresh.effective_interval_time",
        )
        refresh.pop("effective_interval_steps")
    elif refresh_mode == "steps":
        if refresh["requested_interval_time"] is not None:
            raise ValueError(
                "steps spectral refresh must have requested_interval_time=null"
            )
        _positive_int_value(
            refresh["requested_interval_steps"],
            name="numerics.spectral_refresh.requested_interval_steps",
        )
        _positive_int_value(
            refresh["effective_interval_steps"],
            name="numerics.spectral_refresh.effective_interval_steps",
        )
        _positive_float_value(
            refresh["effective_interval_time"],
            name="numerics.spectral_refresh.effective_interval_time",
        )
    elif refresh_mode == "disabled":
        interval_fields = (
            "requested_interval_time",
            "requested_interval_steps",
            "effective_interval_steps",
            "effective_interval_time",
        )
        if any(refresh[name] is not None for name in interval_fields):
            raise ValueError(
                "disabled spectral refresh must have null interval fields"
            )
    else:
        raise ValueError(
            "metadata numerics.spectral_refresh.mode must be one of "
            "'physical_time', 'steps', or 'disabled'"
        )


def _positive_float_value(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"metadata has invalid {name}={value!r}")
    return float(value)


def _positive_int_value(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"metadata has invalid {name}={value!r}")
    return value


def validate_comparability(runs: Sequence[RunArtifact], mode: str) -> None:
    if mode not in {"time", "space", "dealias"}:
        raise ValueError(f"unknown validation mode {mode!r}")
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"run directory basenames must be unique, got {labels}")

    reference = runs[0]
    if mode == "dealias":
        if any(run.shape != reference.shape for run in runs[1:]):
            raise ValueError(
                "dealias sensitivity requires one common grid shape"
            )
        if any(run.dt != reference.dt for run in runs[1:]):
            raise ValueError("dealias sensitivity requires one common dt")
        time_scale = max(reference.final_time, 1.0)
        for run in runs[1:]:
            if not math.isclose(
                run.final_time,
                reference.final_time,
                rel_tol=1e-12,
                abs_tol=1e-12 * time_scale,
            ):
                raise ValueError(
                    "all comparison runs must end at one physical time; got "
                    f"{reference.label}: {reference.final_time} and "
                    f"{run.label}: {run.final_time}"
                )

    reference_signature = _comparison_signature(reference.metadata, mode)
    for run in runs[1:]:
        if _comparison_signature(run.metadata, mode) != reference_signature:
            study = (
                "dealias-sensitivity"
                if mode == "dealias"
                else f"{mode}-convergence"
            )
            raise ValueError(
                f"{run.directory} differs from {reference.directory} in metadata "
                f"outside the fields allowed for a {study} study"
            )

    time_scale = max(reference.final_time, 1.0)
    for run in runs[1:]:
        if not math.isclose(
            run.final_time,
            reference.final_time,
            rel_tol=1e-12,
            abs_tol=1e-12 * time_scale,
        ):
            raise ValueError(
                "all comparison runs must end at one physical time; got "
                f"{reference.label}: {reference.final_time} and "
                f"{run.label}: {run.final_time}"
            )

    if mode == "time":
        if any(run.shape != reference.shape for run in runs[1:]):
            raise ValueError("time-step convergence requires one common grid shape")
        if len({run.dt for run in runs}) != len(runs):
            raise ValueError("time-step convergence requires distinct dt values")
    elif mode == "space":
        if len({run.shape for run in runs}) != len(runs):
            raise ValueError("spatial convergence requires distinct grid shapes")
    else:
        configurations = [
            _dealias_configuration(run.metadata) for run in runs
        ]
        rules = [configuration["rule"] for configuration in configurations]
        if len(set(rules)) != len(rules):
            raise ValueError(
                "dealias sensitivity requires distinct dealias rules"
            )
        if rules.count("cubic_half") != 1:
            raise ValueError(
                "dealias sensitivity requires exactly one cubic_half reference"
            )


def compact_q_frobenius_squared(q: np.ndarray) -> np.ndarray:
    """Return Q:Q from [Qxx,Qxy,Qxz,Qyy,Qyz] without losing Qzz weights."""
    q = np.asarray(q)
    if q.ndim < 1 or q.shape[-1] != Q_COMPONENT_COUNT:
        raise ValueError(f"compact Q must have final dimension 5, got {q.shape}")
    qxx, qxy, qxz, qyy, qyz = np.moveaxis(q, -1, 0)
    qzz = -(qxx + qyy)
    return (
        qxx * qxx
        + qyy * qyy
        + qzz * qzz
        + 2.0 * (qxy * qxy + qxz * qxz + qyz * qyz)
    )


def _x_slices(nx: int, chunk_size: int):
    for start in range(0, nx, chunk_size):
        yield slice(start, min(start + chunk_size, nx))


def _finite_chunk(values: np.ndarray, *, path: Path) -> np.ndarray:
    chunk = np.asarray(values)
    if not np.isfinite(chunk).all():
        raise ValueError(f"{path} contains non-finite values")
    return chunk


def q_scalar_metrics(path: Path, shape: tuple[int, int, int], chunk_size: int) -> dict[str, float]:
    q = _load_array_header(path, (*shape, Q_COMPONENT_COUNT))
    S_values = np.empty(math.prod(shape), dtype=np.float64)
    offset = 0
    frobenius_sum = 0.0
    for section in _x_slices(shape[0], chunk_size):
        chunk = _finite_chunk(q[section], path=path)
        S_chunk = np.asarray(S_from_Q(chunk), dtype=np.float64).reshape(-1)
        S_values[offset : offset + S_chunk.size] = S_chunk
        offset += S_chunk.size
        frobenius_sum += float(
            np.sum(
                compact_q_frobenius_squared(
                    np.asarray(chunk, dtype=np.float64)
                ),
                dtype=np.float64,
            )
        )
    quantiles = np.quantile(S_values, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "q_frobenius_rms": math.sqrt(frobenius_sum / math.prod(shape)),
        "S_mean": float(np.mean(S_values, dtype=np.float64)),
        "S_std": float(np.std(S_values, dtype=np.float64)),
        "S_q05": float(quantiles[0]),
        "S_q25": float(quantiles[1]),
        "S_q50": float(quantiles[2]),
        "S_q75": float(quantiles[3]),
        "S_q95": float(quantiles[4]),
    }


def velocity_scalar_metrics(
    path: Path,
    shape: tuple[int, int, int],
    chunk_size: int,
) -> dict[str, float]:
    velocity = _load_array_header(path, (*shape, VELOCITY_COMPONENT_COUNT))
    component_sum = np.zeros(VELOCITY_COMPONENT_COUNT, dtype=np.float64)
    component_square_sum = np.zeros(VELOCITY_COMPONENT_COUNT, dtype=np.float64)
    for section in _x_slices(shape[0], chunk_size):
        chunk = _finite_chunk(velocity[section], path=path)
        component_sum += np.sum(chunk, axis=(0, 1, 2), dtype=np.float64)
        component_square_sum += np.sum(
            np.square(chunk, dtype=np.float64), axis=(0, 1, 2), dtype=np.float64
        )
    count = math.prod(shape)
    component_mean = component_sum / count
    component_rms = np.sqrt(component_square_sum / count)
    return {
        "u_rms": math.sqrt(float(np.sum(component_square_sum)) / count),
        "ux_mean": float(component_mean[0]),
        "uy_mean": float(component_mean[1]),
        "uz_mean": float(component_mean[2]),
        "ux_rms": float(component_rms[0]),
        "uy_rms": float(component_rms[1]),
        "uz_rms": float(component_rms[2]),
    }


def pressure_scalar_metrics(
    path: Path,
    shape: tuple[int, int, int],
    chunk_size: int,
) -> dict[str, float]:
    pressure = _load_array_header(path, shape)
    total = 0.0
    for section in _x_slices(shape[0], chunk_size):
        chunk = _finite_chunk(pressure[section], path=path)
        total += float(np.sum(chunk, dtype=np.float64))
    count = math.prod(shape)
    mean = total / count
    square_sum = 0.0
    for section in _x_slices(shape[0], chunk_size):
        chunk = np.asarray(pressure[section], dtype=np.float64)
        square_sum += float(np.sum((chunk - mean) ** 2, dtype=np.float64))
    return {
        "p_mean": mean,
        "p_demeaned_rms": math.sqrt(square_sum / count),
    }


def diagnostics_metrics(path: Path, final_step: int) -> dict[str, float]:
    diagnostics = np.load(path, mmap_mode="r", allow_pickle=False)
    if diagnostics.ndim != 1 or diagnostics.dtype.names is None or len(diagnostics) == 0:
        raise ValueError(f"{path} must contain a nonempty structured 1D array")
    required = {"step", *DIAGNOSTIC_FIELDS}
    missing = required - set(diagnostics.dtype.names)
    if missing:
        raise ValueError(f"{path} is missing diagnostic fields {sorted(missing)}")
    steps = np.asarray(diagnostics["step"])
    if int(steps[-1]) != final_step:
        raise ValueError(
            f"{path} ends at diagnostic step {int(steps[-1])}, expected {final_step}"
        )
    result: dict[str, float] = {
        "diagnostic_samples": int(len(diagnostics)),
    }
    for name in DIAGNOSTIC_FIELDS:
        values = np.asarray(diagnostics[name], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path} diagnostic field {name} contains non-finite values")
        result[f"diag_final_{name}"] = float(values[-1])
        result[f"diag_max_{name}"] = float(np.max(values))
    return result


def run_scalar_metrics(run: RunArtifact, chunk_size: int) -> dict[str, Any]:
    dx, dy, dz = run.cell_sizes
    row: dict[str, Any] = {
        "label": run.label,
        "run_dir": str(run.directory),
        "nx": run.shape[0],
        "ny": run.shape[1],
        "nz": run.shape[2],
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "resolution_h": run.resolution_h,
        "dt": run.dt,
        "steps": run.steps,
        "final_time": run.final_time,
    }
    row.update(q_scalar_metrics(run.q_path, run.shape, chunk_size))
    row.update(velocity_scalar_metrics(run.u_path, run.shape, chunk_size))
    row.update(pressure_scalar_metrics(run.p_path, run.shape, chunk_size))
    row.update(diagnostics_metrics(run.diagnostics_path, run.steps))
    return row


def _relative_norm(numerator_square: float, denominator_square: float) -> float | None:
    if denominator_square > 0.0:
        return math.sqrt(numerator_square / denominator_square)
    if numerator_square == 0.0:
        return 0.0
    return None


def q_error_metrics(
    candidate_path: Path,
    reference_path: Path,
    shape: tuple[int, int, int],
    chunk_size: int,
) -> dict[str, float | None]:
    candidate = _load_array_header(candidate_path, (*shape, Q_COMPONENT_COUNT))
    reference = _load_array_header(reference_path, (*shape, Q_COMPONENT_COUNT))
    difference_square_sum = 0.0
    reference_square_sum = 0.0
    difference_max = 0.0
    reference_max = 0.0
    for section in _x_slices(shape[0], chunk_size):
        candidate_chunk = _finite_chunk(candidate[section], path=candidate_path)
        reference_chunk = _finite_chunk(reference[section], path=reference_path)
        difference_square = compact_q_frobenius_squared(
            np.asarray(candidate_chunk, dtype=np.float64)
            - np.asarray(reference_chunk, dtype=np.float64)
        )
        reference_square = compact_q_frobenius_squared(
            np.asarray(reference_chunk, dtype=np.float64)
        )
        difference_square_sum += float(np.sum(difference_square, dtype=np.float64))
        reference_square_sum += float(np.sum(reference_square, dtype=np.float64))
        difference_max = max(difference_max, math.sqrt(float(np.max(difference_square))))
        reference_max = max(reference_max, math.sqrt(float(np.max(reference_square))))
    relative_l2 = _relative_norm(difference_square_sum, reference_square_sum)
    relative_linf = (
        difference_max / reference_max
        if reference_max > 0.0
        else 0.0 if difference_max == 0.0 else None
    )
    return {
        "relative_l2": relative_l2,
        "relative_linf": relative_linf,
        "difference_rms": math.sqrt(difference_square_sum / math.prod(shape)),
        "difference_linf": difference_max,
    }


def q_relative_errors(
    candidate_path: Path,
    reference_path: Path,
    shape: tuple[int, int, int],
    chunk_size: int,
) -> tuple[float | None, float | None]:
    metrics = q_error_metrics(candidate_path, reference_path, shape, chunk_size)
    return metrics["relative_l2"], metrics["relative_linf"]


def velocity_error_metrics(
    candidate_path: Path,
    reference_path: Path,
    shape: tuple[int, int, int],
    chunk_size: int,
) -> dict[str, float | None]:
    candidate = _load_array_header(
        candidate_path, (*shape, VELOCITY_COMPONENT_COUNT)
    )
    reference = _load_array_header(
        reference_path, (*shape, VELOCITY_COMPONENT_COUNT)
    )
    difference_square_sum = 0.0
    reference_square_sum = 0.0
    for section in _x_slices(shape[0], chunk_size):
        candidate_chunk = _finite_chunk(candidate[section], path=candidate_path)
        reference_chunk = _finite_chunk(reference[section], path=reference_path)
        difference = (
            np.asarray(candidate_chunk, dtype=np.float64)
            - np.asarray(reference_chunk, dtype=np.float64)
        )
        difference_square_sum += float(np.sum(difference * difference, dtype=np.float64))
        reference_values = np.asarray(reference_chunk, dtype=np.float64)
        reference_square_sum += float(
            np.sum(reference_values * reference_values, dtype=np.float64)
        )
    return {
        "relative_l2": _relative_norm(difference_square_sum, reference_square_sum),
        "difference_rms": math.sqrt(difference_square_sum / math.prod(shape)),
    }


def velocity_relative_l2(
    candidate_path: Path,
    reference_path: Path,
    shape: tuple[int, int, int],
    chunk_size: int,
) -> float | None:
    return velocity_error_metrics(candidate_path, reference_path, shape, chunk_size)["relative_l2"]


def demeaned_relative_l2(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> float | None:
    """Gauge-invariant relative L2 error for small in-memory scalar fields."""
    candidate_values = np.asarray(candidate, dtype=np.float64)
    reference_values = np.asarray(reference, dtype=np.float64)
    if candidate_values.shape != reference_values.shape:
        raise ValueError("candidate and reference must have the same shape")
    if not np.isfinite(candidate_values).all() or not np.isfinite(reference_values).all():
        raise ValueError("candidate and reference must contain finite values")
    candidate_centered = candidate_values - np.mean(candidate_values)
    reference_centered = reference_values - np.mean(reference_values)
    difference = candidate_centered - reference_centered
    return _relative_norm(
        float(np.sum(difference * difference, dtype=np.float64)),
        float(np.sum(reference_centered * reference_centered, dtype=np.float64)),
    )


def pressure_error_metrics(
    candidate_path: Path,
    reference_path: Path,
    shape: tuple[int, int, int],
    chunk_size: int,
) -> dict[str, float | None]:
    candidate = _load_array_header(candidate_path, shape)
    reference = _load_array_header(reference_path, shape)
    candidate_sum = 0.0
    reference_sum = 0.0
    for section in _x_slices(shape[0], chunk_size):
        candidate_chunk = _finite_chunk(candidate[section], path=candidate_path)
        reference_chunk = _finite_chunk(reference[section], path=reference_path)
        candidate_sum += float(np.sum(candidate_chunk, dtype=np.float64))
        reference_sum += float(np.sum(reference_chunk, dtype=np.float64))
    count = math.prod(shape)
    candidate_mean = candidate_sum / count
    reference_mean = reference_sum / count

    difference_square_sum = 0.0
    reference_square_sum = 0.0
    for section in _x_slices(shape[0], chunk_size):
        candidate_centered = np.asarray(candidate[section], dtype=np.float64) - candidate_mean
        reference_centered = np.asarray(reference[section], dtype=np.float64) - reference_mean
        difference = candidate_centered - reference_centered
        difference_square_sum += float(np.sum(difference * difference, dtype=np.float64))
        reference_square_sum += float(
            np.sum(reference_centered * reference_centered, dtype=np.float64)
        )
    return {
        "relative_l2": _relative_norm(difference_square_sum, reference_square_sum),
        "difference_rms": math.sqrt(difference_square_sum / math.prod(shape)),
    }


def pressure_relative_l2(
    candidate_path: Path,
    reference_path: Path,
    shape: tuple[int, int, int],
    chunk_size: int,
) -> float | None:
    return pressure_error_metrics(candidate_path, reference_path, shape, chunk_size)["relative_l2"]


def field_error_row(
    candidate: RunArtifact,
    reference: RunArtifact,
    chunk_size: int,
) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise ValueError("pointwise field errors require one common grid shape")
    q_errors = q_error_metrics(
        candidate.q_path, reference.q_path, candidate.shape, chunk_size
    )
    u_errors = velocity_error_metrics(
        candidate.u_path, reference.u_path, candidate.shape, chunk_size
    )
    p_errors = pressure_error_metrics(
        candidate.p_path, reference.p_path, candidate.shape, chunk_size
    )
    return {
        "candidate": candidate.label,
        "reference": reference.label,
        "candidate_dt": candidate.dt,
        "reference_dt": reference.dt,
        "q_rel_l2": q_errors["relative_l2"],
        "q_rel_linf": q_errors["relative_linf"],
        "q_difference_rms": q_errors["difference_rms"],
        "q_difference_linf": q_errors["difference_linf"],
        "u_rel_l2": u_errors["relative_l2"],
        "u_difference_rms": u_errors["difference_rms"],
        "p_demeaned_rel_l2": p_errors["relative_l2"],
        "p_demeaned_difference_rms": p_errors["difference_rms"],
    }


def observed_order(
    coarse_medium_error: float | None,
    medium_fine_error: float | None,
    refinement_ratio: float,
) -> float | None:
    if (
        coarse_medium_error is None
        or medium_fine_error is None
        or not math.isfinite(coarse_medium_error)
        or not math.isfinite(medium_fine_error)
        or coarse_medium_error <= 0.0
        or medium_fine_error <= 0.0
        or not math.isfinite(refinement_ratio)
        or refinement_ratio <= 1.0
    ):
        return None
    return math.log(coarse_medium_error / medium_fine_error) / math.log(
        refinement_ratio
    )

def select_geometric_dt_triplet(
    dt_values: Sequence[float],
    requested: Sequence[float] | None = None,
) -> tuple[float, float, float] | None:
    """Resolve an explicit triplet or select the finest adjacent one.

    Automatic selection deliberately considers only consecutive dt levels in
    descending order.  This prevents a heterogeneous list such as several
    local refinement studies from accidentally producing a cross-decade
    triplet merely because three non-adjacent values share a common ratio.
    """
    ordered = sorted({float(value) for value in dt_values}, reverse=True)
    if any(not math.isfinite(value) or value <= 0.0 for value in ordered):
        raise ValueError("all dt values must be finite and positive")

    if requested is not None:
        if len(requested) != 3:
            raise ValueError("an explicit order-dt selection must contain three values")
        selected = tuple(float(value) for value in requested)
        if any(not math.isfinite(value) or value <= 0.0 for value in selected):
            raise ValueError("explicit order-dt values must be finite and positive")
        if not selected[0] > selected[1] > selected[2]:
            raise ValueError(
                "explicit order-dt values must be ordered COARSE > MEDIUM > FINE"
            )
        resolved = []
        missing = []
        for value in selected:
            matching = next(
                (
                    available
                    for available in ordered
                    if math.isclose(
                        value, available, rel_tol=1.0e-12, abs_tol=0.0
                    )
                ),
                None,
            )
            if matching is None:
                missing.append(value)
            else:
                resolved.append(matching)
        if missing:
            raise ValueError(
                f"explicit order-dt values are absent from the supplied runs: {missing}"
            )
        selected = tuple(resolved)
        if not _is_geometric_dt_triplet(selected):
            raise ValueError(
                "explicit order-dt values must form a geometrically refined triplet"
            )
        return selected

    if len(ordered) < 3:
        return None

    candidates: list[tuple[float, float, float]] = []
    for first in range(len(ordered) - 2):
        candidate = tuple(ordered[first : first + 3])
        if _is_geometric_dt_triplet(candidate):
            candidates.append(candidate)
    if not candidates:
        return None
    return min(candidates, key=lambda values: values[-1])


def _is_geometric_dt_triplet(values: Sequence[float]) -> bool:
    coarse, medium, fine = values
    ratio_1 = coarse / medium
    ratio_2 = medium / fine
    return ratio_1 > 1.0 and math.isclose(
        ratio_1,
        ratio_2,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    )

def _relative_scalar_change(coarse: float, fine: float) -> float | None:
    difference = abs(float(coarse) - float(fine))
    denominator = abs(float(fine))
    if denominator > 0.0:
        return difference / denominator
    return 0.0 if difference == 0.0 else None


SPACE_SCALAR_FIELDS = (
    "q_frobenius_rms",
    "S_mean",
    "S_std",
    "S_q05",
    "S_q25",
    "S_q50",
    "S_q75",
    "S_q95",
    "u_rms",
    "ux_rms",
    "uy_rms",
    "uz_rms",
    "p_demeaned_rms",
    "diag_final_div_rel",
    "diag_final_schur_rel_residual",
    "diag_final_wall_normal_momentum_rms",
)


def space_pair_row(
    coarse_run: RunArtifact,
    fine_run: RunArtifact,
    scalar_by_label: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    coarse = scalar_by_label[coarse_run.label]
    fine = scalar_by_label[fine_run.label]
    row: dict[str, Any] = {
        "coarse": coarse_run.label,
        "fine": fine_run.label,
        "coarse_shape": "x".join(str(value) for value in coarse_run.shape),
        "fine_shape": "x".join(str(value) for value in fine_run.shape),
        "coarse_h": coarse_run.resolution_h,
        "fine_h": fine_run.resolution_h,
        "refinement_ratio_h": coarse_run.resolution_h / fine_run.resolution_h,
        "pointwise_field_errors_computed": False,
    }
    for name in SPACE_SCALAR_FIELDS:
        row[f"{name}_relative_change"] = _relative_scalar_change(
            float(coarse[name]), float(fine[name])
        )
    return row


def _csv_fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    return fields


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fields(rows))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze_time(
    runs: Sequence[RunArtifact],
    scalar_rows: list[dict[str, Any]],
    chunk_size: int,
    order_dt: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(runs, key=lambda run: run.dt, reverse=True)
    reference = min(runs, key=lambda run: run.dt)
    errors_to_reference = {
        run.label: field_error_row(run, reference, chunk_size)
        for run in runs
    }
    for row in scalar_rows:
        errors = errors_to_reference[row["label"]]
        row.update(
            reference_label=reference.label,
            q_rel_l2_to_reference=errors["q_rel_l2"],
            q_rel_linf_to_reference=errors["q_rel_linf"],
            u_rel_l2_to_reference=errors["u_rel_l2"],
            p_demeaned_rel_l2_to_reference=errors["p_demeaned_rel_l2"],
        )

    adjacent_errors = [
        field_error_row(coarse, fine, chunk_size)
        for coarse, fine in zip(ordered, ordered[1:])
    ]
    order_summary: dict[str, Any] = {
        "estimated": False,
        "reason": "no geometrically refined three-dt subset is available",
    }
    triplet_dts = select_geometric_dt_triplet(
        [run.dt for run in ordered], requested=order_dt
    )
    if triplet_dts is not None:
        run_by_dt = {run.dt: run for run in ordered}
        triplet_runs = [run_by_dt[value] for value in triplet_dts]
        adjacent_by_pair = {
            (row["candidate"], row["reference"]): row
            for row in adjacent_errors
        }
        order_errors = []
        for coarse, fine in zip(triplet_runs, triplet_runs[1:]):
            key = (coarse.label, fine.label)
            errors = adjacent_by_pair.get(key)
            if errors is None:
                errors = field_error_row(coarse, fine, chunk_size)
            order_errors.append(errors)
        ratio = triplet_dts[0] / triplet_dts[1]
        order_summary = {
            "estimated": True,
            "selection": "explicit" if order_dt is not None else "adjacent_automatic",
            "refinement_ratio": ratio,
            "dt_values": list(triplet_dts),
            "q_l2": observed_order(
                order_errors[0]["q_difference_rms"],
                order_errors[1]["q_difference_rms"],
                ratio,
            ),
            "q_linf": observed_order(
                order_errors[0]["q_difference_linf"],
                order_errors[1]["q_difference_linf"],
                ratio,
            ),
            "u_l2": observed_order(
                order_errors[0]["u_difference_rms"],
                order_errors[1]["u_difference_rms"],
                ratio,
            ),
            "p_demeaned_l2": observed_order(
                order_errors[0]["p_demeaned_difference_rms"],
                order_errors[1]["p_demeaned_difference_rms"],
                ratio,
            ),
        }
    return adjacent_errors, {
        "reference_run": reference.label,
        "reference_dt": reference.dt,
        "observed_order": order_summary,
        "automatic_convergence_claim": False,
        "interpretation": (
            "Finite field differences or one observed-order estimate are evidence "
            "for the user to assess, not an automatic convergence declaration."
        ),
        "pointwise_error_definition": {
            "Q": (
                "relative L2 and Linf of the full symmetric traceless tensor; "
                "Qzz=-(Qxx+Qyy) and off-diagonal terms have multiplicity two"
            ),
            "u": "componentwise vector relative L2",
            "p": "relative L2 after independently removing each pressure mean",
            "observed_order": (
                "log(error_coarse-medium/error_medium-fine)/log(refinement); "
                "absolute RMS/Linf differences are used rather than relative "
                "errors with changing denominators"
            ),
        },
    }


def analyze_space(
    runs: Sequence[RunArtifact],
    scalar_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(runs, key=lambda run: run.resolution_h, reverse=True)
    for coarse, fine in zip(ordered, ordered[1:]):
        if any(fine_count < coarse_count for coarse_count, fine_count in zip(coarse.shape, fine.shape)):
            raise ValueError(
                f"spatial grids are not componentwise refined: {coarse.shape} -> {fine.shape}"
            )
    scalar_by_label = {row["label"]: row for row in scalar_rows}
    pairs = [
        space_pair_row(coarse, fine, scalar_by_label)
        for coarse, fine in zip(ordered, ordered[1:])
    ]
    return pairs, {
        "pointwise_field_errors_computed": False,
        "automatic_convergence_claim": False,
        "interpretation": (
            "Finite adjacent scalar changes are resolution-sensitivity evidence, "
            "not an automatic spatial-convergence declaration."
        ),
        "reason": (
            "The solver mixes periodic FFT, Neumann DCT, and Dirichlet DST "
            "collocation fields. No basis-aware spectral restriction is applied, "
            "so this report does not label cross-grid point differences as errors."
        ),
        "comparison": "resolution-independent final-state scalar statistics",
    }


def analyze_dealias(
    runs: Sequence[RunArtifact],
    scalar_rows: list[dict[str, Any]],
    chunk_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare equal-grid dealias rules against the cubic-half reference."""
    configurations = {
        run.label: _dealias_configuration(run.metadata) for run in runs
    }
    reference = next(
        run
        for run in runs
        if configurations[run.label]["rule"] == "cubic_half"
    )
    errors_to_reference = {
        reference.label: {
            "q_rel_l2": 0.0,
            "q_rel_linf": 0.0,
            "u_rel_l2": 0.0,
            "p_demeaned_rel_l2": 0.0,
        }
    }
    errors_to_reference.update(
        {
            run.label: field_error_row(run, reference, chunk_size)
            for run in runs
            if run.label != reference.label
        }
    )

    for row in scalar_rows:
        configuration = configurations[row["label"]]
        errors = errors_to_reference[row["label"]]
        row.update(
            dealias_rule=configuration["rule"],
            dealias_fraction=configuration["fraction"],
            retained_q_modes="x".join(
                str(value) for value in configuration["retained_q_modes"]
            ),
            retained_normal_velocity_modes="x".join(
                str(value)
                for value in configuration[
                    "retained_normal_velocity_modes"
                ]
            ),
            raw_q_sha256=configuration["raw_q_sha256"],
            projected_q_sha256=configuration["projected_q_sha256"],
            reference_label=reference.label,
            is_cubic_half_reference=(row["label"] == reference.label),
            q_rel_l2_to_reference=errors["q_rel_l2"],
            q_rel_linf_to_reference=errors["q_rel_linf"],
            u_rel_l2_to_reference=errors["u_rel_l2"],
            p_demeaned_rel_l2_to_reference=errors["p_demeaned_rel_l2"],
        )

    pair_rows = []
    candidates = sorted(
        (run for run in runs if run.label != reference.label),
        key=lambda run: configurations[run.label]["rule"],
    )
    for candidate in candidates:
        row = errors_to_reference[candidate.label]
        row.update(
            candidate_dealias_rule=configurations[candidate.label]["rule"],
            candidate_dealias_fraction=configurations[candidate.label][
                "fraction"
            ],
            reference_dealias_rule="cubic_half",
            reference_dealias_fraction=configurations[reference.label][
                "fraction"
            ],
        )
        pair_rows.append(row)

    return pair_rows, {
        "analysis_kind": "dealias_sensitivity",
        "reference_run": reference.label,
        "reference_rule": "cubic_half",
        "pointwise_field_errors_computed": True,
        "automatic_convergence_claim": False,
        "interpretation": (
            "These equal-grid differences measure sensitivity to the spectral "
            "dealiasing rule. They are not a convergence estimate and do not "
            "establish that either rule is physically correct."
        ),
        "comparison": (
            "final Q, velocity, and pressure fields plus final-state global "
            "scalar statistics"
        ),
        "allowed_metadata_differences": [
            "dealias_rule and dealias_fraction",
            "numerics.dealiasing.rule and numerics.dealiasing.fraction",
            "initial_condition.projected_q_sha256",
            "retained_q_modes",
            "retained_normal_velocity_modes",
        ],
        "pointwise_error_definition": {
            "Q": (
                "relative L2 and Linf of the full symmetric traceless tensor; "
                "Qzz=-(Qxx+Qyy) and off-diagonal terms have multiplicity two"
            ),
            "u": "componentwise vector relative L2",
            "p": "relative L2 after independently removing each pressure mean",
        },
    }


def reserve_output_paths(output_dir: Path, prefix: str) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = output_dir / f"{prefix}_runs.csv"
    pair_csv = output_dir / f"{prefix}_pairs.csv"
    json_path = output_dir / f"{prefix}.json"
    existing = [path for path in (run_csv, pair_csv, json_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing validation outputs: "
            + ", ".join(str(path) for path in existing)
        )
    return run_csv, pair_csv, json_path


def main() -> int:
    args = parse_args()
    runs = [load_run(path) for path in args.run_dirs]
    validate_comparability(runs, args.mode)
    study_label = (
        "dealias-sensitivity"
        if args.mode == "dealias"
        else f"{args.mode}-convergence"
    )
    print(f"Validated {len(runs)} completed {study_label} runs.")

    scalar_rows = []
    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] scalar metrics: {run.directory}", flush=True)
        scalar_rows.append(run_scalar_metrics(run, args.chunk_size))

    if args.mode == "time":
        pair_rows, method = analyze_time(
            runs,
            scalar_rows,
            args.chunk_size,
            order_dt=args.order_dt,
        )
        scalar_rows.sort(key=lambda row: float(row["dt"]), reverse=True)
    elif args.mode == "space":
        pair_rows, method = analyze_space(runs, scalar_rows)
        scalar_rows.sort(key=lambda row: float(row["resolution_h"]), reverse=True)
    else:
        pair_rows, method = analyze_dealias(
            runs, scalar_rows, args.chunk_size
        )
        scalar_rows.sort(key=lambda row: str(row["dealias_rule"]))

    default_prefix = (
        "fig4_dealias_sensitivity"
        if args.mode == "dealias"
        else f"fig4_{args.mode}_convergence"
    )
    prefix = args.output_prefix or default_prefix
    run_csv, pair_csv, json_path = reserve_output_paths(
        args.output_dir.expanduser().resolve(), prefix
    )
    write_csv(run_csv, scalar_rows)
    write_csv(pair_csv, pair_rows)
    run_identity = validate_run_identity(
        runs[0].metadata,
        path=runs[0].directory / "metadata.json",
    )
    signature_json = json.dumps(
        _json_safe(_comparison_signature(runs[0].metadata, args.mode)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    report = {
        "schema_version": 1,
        "analysis": (
            "fig4_dealias_sensitivity"
            if args.mode == "dealias"
            else "fig4_convergence"
        ),
        "mode": args.mode,
        "run_identity": run_identity,
        "comparison_signature_sha256": hashlib.sha256(
            signature_json.encode("utf-8")
        ).hexdigest(),
        "final_time": runs[0].final_time,
        "runs": scalar_rows,
        (
            "sensitivity_pairs"
            if args.mode == "dealias"
            else "adjacent_pairs"
        ): pair_rows,
        "method": method,
    }
    json_path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {run_csv}")
    print(f"Wrote {pair_csv}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
