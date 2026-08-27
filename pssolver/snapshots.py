"""Load and restore saved Q, velocity, and pressure snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .models.active_nematics import (
    Q_convention_metadata,
    Q_magnitude,
    S_from_Q,
)


Q_FIELD_NAMES = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
VELOCITY_FIELD_NAMES = ("ux", "uy", "uz")
SNAPSHOT_MODES = ("resume", "branch")
_EXPECTED_S_BULK_UNSET = object()
REPRESENTATIVE_ORDERED_S_DEFINITION = (
    "median(S=lambda_max(Q)) over points with Q_magnitude at or above "
    "the field's 75th percentile"
)


@dataclass(frozen=True)
class ActiveNematicQSourceMetadata:
    """Validated provenance for Q data produced by active nematics."""

    directory: Path
    path: Path
    metadata: dict[str, Any]
    S_initial: float | None
    S_bulk: float | None


@dataclass(frozen=True)
class QTensorSnapshot:
    """One validated Q-only snapshot and its source provenance."""

    directory: Path
    path: Path
    shape: tuple[int, int, int]
    values: np.ndarray
    source_metadata: ActiveNematicQSourceMetadata


@dataclass(frozen=True)
class SimulationSnapshot:
    """One saved spatial state from a Q-tensor simulation."""

    directory: Path
    step: int
    shape: tuple[int, int, int]
    q_fields: dict[str, torch.Tensor]
    velocity_fields: dict[str, torch.Tensor]
    pressure: torch.Tensor
    source_metadata: ActiveNematicQSourceMetadata


_CONTINUATION_SIGNATURE_FIELDS = (
    "script",
    "solver",
    "model",
    "boundary_conditions",
    "numerics",
)


def _continuation_signature(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Extract trajectory-defining metadata for a metadata-compatible resume."""
    if not isinstance(metadata, Mapping):
        raise ValueError("run metadata must be a mapping")

    script = metadata.get("script")
    if not isinstance(script, str) or not script:
        raise ValueError("resume metadata is missing a nonempty script")

    solver_metadata = metadata.get("solver")
    if not isinstance(solver_metadata, Mapping):
        raise ValueError("resume metadata is missing solver")
    solver_signature = {}
    for name in ("shape", "lengths", "dt"):
        if name not in solver_metadata:
            raise ValueError(f"resume metadata is missing solver.{name}")
        solver_signature[name] = solver_metadata[name]

    model_metadata = metadata.get("model")
    if not isinstance(model_metadata, Mapping):
        raise ValueError("resume metadata is missing model")
    model_signature = {}
    for name in ("name", "Q_convention", "parameters"):
        if name not in model_metadata:
            raise ValueError(f"resume metadata is missing model.{name}")
        model_signature[name] = model_metadata[name]

    signature = {
        "script": script,
        "solver": solver_signature,
        "model": model_signature,
    }
    for name in ("boundary_conditions", "numerics"):
        value = metadata.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"resume metadata is missing {name}")
        signature[name] = value
    try:
        return json.loads(json.dumps(signature, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "trajectory-defining resume metadata must be JSON serializable"
        ) from error


def validate_resume_compatibility(
    source: ActiveNematicQSourceMetadata,
    current_metadata: Mapping[str, Any],
) -> None:
    """Require identical trajectory-defining metadata for compatible continuation.

    Final step count, save cadence, diagnostics, status, and initial-condition
    provenance may differ. Shape, lengths, dt, model parameters, boundary
    conditions, numerical mode, and script identity may not. Use a branch when
    any of those trajectory-defining values changes.
    """
    try:
        source_signature = _continuation_signature(source.metadata)
    except ValueError as error:
        raise ValueError(
            f"{source.path} cannot use metadata-compatible resume: {error}. "
            "Use mode='branch'."
        ) from error
    try:
        current_signature = _continuation_signature(current_metadata)
    except ValueError as error:
        raise ValueError(
            f"current run cannot use metadata-compatible resume: {error}. "
            "Use mode='branch'."
        ) from error
    if source_signature != current_signature:
        differing = [
            name
            for name in _CONTINUATION_SIGNATURE_FIELDS
            if source_signature[name] != current_signature[name]
        ]
        raise ValueError(
            "trajectory-defining metadata differs for "
            f"{differing}; use mode='branch' when parameters, dt, boundary "
            "conditions, numerical mode, or script changed."
        )


def _positive_finite_parameter(value, *, name: str, metadata_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{metadata_path} has non-numeric model.parameters.{name}={value!r}."
        )
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(
            f"{metadata_path} requires model.parameters.{name} to be positive "
            f"and finite, got {value!r}."
        )
    return result


def validate_active_nematic_q_source(
    directory: str | Path,
    *,
    expected_S_bulk=_EXPECTED_S_BULK_UNSET,
    require_S_initial: bool = False,
    S_bulk_rtol: float = 1.0e-9,
    S_bulk_atol: float = 1.0e-12,
) -> ActiveNematicQSourceMetadata:
    """Require canonical active-nematic provenance for externally loaded Q."""
    directory = Path(directory).expanduser().resolve()
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Canonical active-nematic Q metadata is required at "
            f"{metadata_path}; legacy convention fallback is not supported."
        )

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{metadata_path} does not contain valid JSON.") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object.")

    if metadata.get("schema_version") != 1:
        raise ValueError(
            f"{metadata_path} must declare schema_version=1."
        )

    model = metadata.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{metadata_path} is missing model metadata.")
    if model.get("name") != "active_nematics":
        raise ValueError(
            f"{metadata_path} must declare model.name='active_nematics'."
        )
    expected_convention = Q_convention_metadata()
    if model.get("Q_convention") != expected_convention:
        raise ValueError(
            f"{metadata_path} does not declare the complete canonical Q "
            f"convention {expected_convention!r}."
        )

    parameters = model.get("parameters")
    if not isinstance(parameters, dict) or "S_bulk" not in parameters:
        raise ValueError(f"{metadata_path} is missing model.parameters.S_bulk.")
    raw_S_bulk = parameters["S_bulk"]
    S_bulk = (
        None
        if raw_S_bulk is None
        else _positive_finite_parameter(
            raw_S_bulk,
            name="S_bulk",
            metadata_path=metadata_path,
        )
    )

    raw_S_initial = parameters.get("S_initial", _EXPECTED_S_BULK_UNSET)
    if raw_S_initial is _EXPECTED_S_BULK_UNSET:
        if require_S_initial:
            raise ValueError(
                f"{metadata_path} is missing model.parameters.S_initial."
            )
        S_initial = None
    else:
        S_initial = _positive_finite_parameter(
            raw_S_initial,
            name="S_initial",
            metadata_path=metadata_path,
        )

    if expected_S_bulk is not _EXPECTED_S_BULK_UNSET:
        if expected_S_bulk is None:
            if S_bulk is not None:
                raise ValueError(
                    f"{metadata_path} declares S_bulk={S_bulk:.12g}, but the "
                    "current model has no positive uniform nematic equilibrium "
                    "and requires S_bulk=null."
                )
        else:
            expected_value = _positive_finite_parameter(
                expected_S_bulk,
                name="expected_S_bulk",
                metadata_path=metadata_path,
            )
            if S_bulk is None or not np.isclose(
                S_bulk,
                expected_value,
                rtol=S_bulk_rtol,
                atol=S_bulk_atol,
            ):
                raise ValueError(
                    f"{metadata_path} declares S_bulk={S_bulk!r}, but the current "
                    f"model requires S_bulk={expected_value:.12g}."
                )

    return ActiveNematicQSourceMetadata(
        directory=directory,
        path=metadata_path,
        metadata=metadata,
        S_initial=S_initial,
        S_bulk=S_bulk,
    )


def _load_floating_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"snapshot file does not exist: {path}")

    values = np.load(path, allow_pickle=False)
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(
            f"snapshot array must use a floating dtype, got {values.dtype} in {path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"snapshot array contains NaN or Inf values: {path}")
    return np.ascontiguousarray(values, dtype=np.float32)


def representative_ordered_S(values) -> float:
    """Return a reproducible scalar descriptor for an externally used Q field."""
    magnitudes = Q_magnitude(values)
    S_values = S_from_Q(values)
    if torch.is_tensor(magnitudes):
        magnitudes = magnitudes.detach().cpu().numpy()
        S_values = S_values.detach().cpu().numpy()
    else:
        magnitudes = np.asarray(magnitudes)
        S_values = np.asarray(S_values)
    cutoff = float(np.quantile(magnitudes, 0.75))
    ordered = S_values[magnitudes >= cutoff]
    result = float(np.median(ordered))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(
            "Q snapshot must have a positive finite ordered-region median S"
        )
    return result


def load_q_snapshot(
    path: str | Path,
    *,
    expected_shape: tuple[int, int, int] | None = None,
    metadata_directory: str | Path | None = None,
    expected_S_bulk=_EXPECTED_S_BULK_UNSET,
    require_S_initial: bool = False,
) -> QTensorSnapshot:
    """Load Q only after validating its canonical source metadata."""
    path = Path(path).expanduser().resolve()
    directory = (
        path.parent
        if metadata_directory is None
        else Path(metadata_directory).expanduser().resolve()
    )
    if directory != path.parent:
        raise ValueError(
            "Q snapshot metadata must be colocated with the Q file; "
            f"got Q in {path.parent} and metadata_directory={directory}."
        )
    source_metadata = validate_active_nematic_q_source(
        directory,
        expected_S_bulk=expected_S_bulk,
        require_S_initial=require_S_initial,
    )
    values = _load_floating_array(path)
    if values.ndim != 4 or values.shape[-1] != len(Q_FIELD_NAMES):
        raise ValueError(
            f"{path.name} must have shape (Nx, Ny, Nz, 5), got {values.shape}"
        )
    shape = tuple(values.shape[:-1])
    if expected_shape is not None and shape != tuple(expected_shape):
        raise ValueError(
            f"snapshot spatial shape {shape} does not match solver shape "
            f"{tuple(expected_shape)}"
        )
    return QTensorSnapshot(
        directory=directory,
        path=path,
        shape=shape,
        values=values,
        source_metadata=source_metadata,
    )


def load_snapshot(
    directory: str | Path,
    step: int,
    *,
    expected_shape: tuple[int, int, int] | None = None,
    expected_S_bulk=_EXPECTED_S_BULK_UNSET,
    require_S_initial: bool = False,
) -> SimulationSnapshot:
    """Load a matching ``Q_<step>``, ``u_<step>``, ``p_<step>`` triplet."""
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError(f"snapshot step must be a non-negative integer, got {step!r}")

    directory = Path(directory).expanduser().resolve()
    q_snapshot = load_q_snapshot(
        directory / f"Q_{step}.npy",
        expected_shape=expected_shape,
        metadata_directory=directory,
        expected_S_bulk=expected_S_bulk,
        require_S_initial=require_S_initial,
    )
    q_values = q_snapshot.values
    u_values = _load_floating_array(directory / f"u_{step}.npy")
    p_values = _load_floating_array(directory / f"p_{step}.npy")

    shape = q_snapshot.shape
    expected_u_shape = (*shape, len(VELOCITY_FIELD_NAMES))
    if u_values.shape != expected_u_shape:
        raise ValueError(
            f"u_{step}.npy must have shape {expected_u_shape}, got {u_values.shape}"
        )
    if p_values.shape != shape:
        raise ValueError(
            f"p_{step}.npy must have shape {shape}, got {p_values.shape}"
        )

    return SimulationSnapshot(
        directory=directory,
        step=step,
        shape=shape,
        q_fields={
            name: torch.from_numpy(q_values[..., component])
            for component, name in enumerate(Q_FIELD_NAMES)
        },
        velocity_fields={
            name: torch.from_numpy(u_values[..., component])
            for component, name in enumerate(VELOCITY_FIELD_NAMES)
        },
        pressure=torch.from_numpy(p_values),
        source_metadata=q_snapshot.source_metadata,
    )


def _batched_field(values: torch.Tensor, solver) -> torch.Tensor:
    values = values.to(device=solver.device, dtype=solver.model.fields.dtype)
    return values.unsqueeze(0).repeat(
        solver.batchsize,
        *([1] * len(solver.shape)),
    )


def apply_snapshot_to_solver(
    solver,
    snapshot: SimulationSnapshot,
    *,
    mode: str,
    current_run_metadata: Mapping[str, Any] | None = None,
    validate_resume_static: bool = True,
    resume_static_rtol: float = 1e-3,
    resume_static_atol: float = 1e-6,
) -> int:
    """Apply a loaded snapshot after ``solver.build()``.

    ``resume`` restores Q/u/p and consumes the restored static fields on the
    next integrator step. ``branch`` restores only Q, so u/p are recomputed
    from the branch parameters before Q advances.

    Returns the global step from which output numbering should start.
    """
    if mode not in SNAPSHOT_MODES:
        raise ValueError(f"snapshot mode must be one of {SNAPSHOT_MODES}, got {mode!r}")
    if tuple(solver.shape) != snapshot.shape:
        raise ValueError(
            f"snapshot spatial shape {snapshot.shape} does not match solver shape "
            f"{tuple(solver.shape)}"
        )
    if not hasattr(solver, "integrator"):
        raise RuntimeError("solver must be built before applying a snapshot")

    fields = solver.model.fields
    for name, values in snapshot.q_fields.items():
        fields[name] = _batched_field(values, solver)

    if mode == "branch":
        return 0

    if current_run_metadata is None:
        raise ValueError(
            "current_run_metadata is required for mode='resume'; "
            "use mode='branch' when a metadata-compatible continuation cannot "
            "be verified."
        )
    validate_resume_compatibility(
        snapshot.source_metadata,
        current_run_metadata,
    )

    if validate_resume_static:
        computed_static_hats = solver.model.compute_static()
        saved_static_fields = {
            **snapshot.velocity_fields,
            "p": snapshot.pressure,
        }
        inconsistent_fields = []
        for offset, name in enumerate((*VELOCITY_FIELD_NAMES, "p")):
            computed = fields.inverse_transform(
                name,
                spectral=computed_static_hats[offset],
            )
            saved = _batched_field(saved_static_fields[name], solver)
            if name == "p":
                computed = computed - computed.mean()
                saved = saved - saved.mean()
            difference_norm = torch.linalg.vector_norm(computed - saved)
            reference_norm = torch.maximum(
                torch.linalg.vector_norm(computed),
                torch.linalg.vector_norm(saved),
            )
            relative_error = (
                difference_norm / reference_norm.clamp_min(resume_static_atol)
            ).item()
            maximum_error = (computed - saved).abs().max().item()
            if relative_error > resume_static_rtol and maximum_error > resume_static_atol:
                inconsistent_fields.append(
                    f"{name}(relative={relative_error:.3e}, max={maximum_error:.3e})"
                )
        if inconsistent_fields:
            details = ", ".join(inconsistent_fields)
            raise ValueError(
                "saved u/p are inconsistent with the current solver configuration: "
                f"{details}. Use mode='branch' when parameters have changed."
            )

    for name, values in snapshot.velocity_fields.items():
        fields[name] = _batched_field(values, solver)
    fields["p"] = _batched_field(snapshot.pressure, solver)

    static_model = solver.model.static_model
    if not hasattr(static_model, "pressure_guess"):
        raise TypeError(
            "resume mode requires a static model with a pressure_guess attribute"
        )
    pressure_hat = fields["p.hat"].detach().clone()
    static_model.pressure_guess = pressure_hat
    if hasattr(static_model, "last_pressure_hat"):
        static_model.last_pressure_hat = pressure_hat

    solver.integrator.restore_progress(
        snapshot.step,
        static_fields_are_current=True,
    )
    return snapshot.step


def require_distinct_output_directory(
    snapshot: SimulationSnapshot,
    output_directory: str | Path,
) -> Path:
    """Reject snapshot output that would overwrite the source run."""
    output_directory = Path(output_directory).expanduser().resolve()
    if output_directory == snapshot.directory:
        raise ValueError(
            "snapshot output directory must differ from the source directory: "
            f"{output_directory}"
        )
    return output_directory


__all__ = [
    "ActiveNematicQSourceMetadata",
    "Q_FIELD_NAMES",
    "QTensorSnapshot",
    "REPRESENTATIVE_ORDERED_S_DEFINITION",
    "SNAPSHOT_MODES",
    "SimulationSnapshot",
    "VELOCITY_FIELD_NAMES",
    "apply_snapshot_to_solver",
    "load_q_snapshot",
    "load_snapshot",
    "representative_ordered_S",
    "require_distinct_output_directory",
    "validate_active_nematic_q_source",
    "validate_resume_compatibility",
]
