"""Caller-owned Plane observation capture, independent of execution scheduling."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pssolver.engine.field_view import FieldView
from pssolver.errors import SimulationExecutionError, SimulationStateError
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.run_metadata import write_run_metadata


VELOCITY_COMPONENTS = ("ux", "uy", "uz")


@dataclass(frozen=True, slots=True)
class PlaneObservation:
    """One backend-independent Q/u/p observation in production layout."""

    step: int
    q: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 0
        ):
            raise ValueError("observation step must be non-negative")
        expected = {
            "q": (self.q, 4, len(Q_COMPONENTS)),
            "velocity": (self.velocity, 4, len(VELOCITY_COMPONENTS)),
            "pressure": (self.pressure, 3, None),
        }
        for description, (value, ndim, last) in expected.items():
            if not isinstance(value, np.ndarray) or value.ndim != ndim:
                raise ValueError(f"{description} observation shape is invalid")
            if last is not None and value.shape[-1] != last:
                raise ValueError(f"{description} component count is invalid")
            if not np.issubdtype(value.dtype, np.floating):
                raise TypeError(f"{description} observation must be floating")
            if not np.isfinite(value).all():
                raise SimulationStateError(
                    f"Plane {description} observation is non-finite at step {self.step}",
                    application_family="plane_complete_stress",
                    stage="observation",
                    step=self.step,
                    details={"field": description},
                )
        if self.velocity.shape[:-1] != self.q.shape[:-1]:
            raise ValueError("velocity grid does not match Q")
        if self.pressure.shape != self.q.shape[:-1]:
            raise ValueError("pressure grid does not match Q")
        object.__setattr__(self, "q", np.array(self.q, copy=True))
        object.__setattr__(
            self,
            "velocity",
            np.array(self.velocity, copy=True),
        )
        object.__setattr__(
            self,
            "pressure",
            np.array(self.pressure, copy=True),
        )


@dataclass(frozen=True, slots=True)
class PlaneDiagnostic:
    """Existing production diagnostic schema, independent of runtime path."""

    step: int
    div_max: float
    div_rms: float
    div_rel: float
    schur_iterations: float
    schur_abs_residual: float
    schur_rel_residual: float
    wall_normal_momentum_max: float
    wall_normal_momentum_rms: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step, int)
            or isinstance(self.step, bool)
            or self.step < 0
        ):
            raise ValueError("diagnostic step must be non-negative")
        values = tuple(
            float(getattr(self, name))
            for name in (
                "div_max",
                "div_rms",
                "div_rel",
                "schur_iterations",
                "schur_abs_residual",
                "schur_rel_residual",
                "wall_normal_momentum_max",
                "wall_normal_momentum_rms",
            )
        )
        if not all(math.isfinite(value) for value in values):
            raise SimulationStateError(
                f"Plane diagnostic is non-finite at step {self.step}",
                application_family="plane_complete_stress",
                stage="diagnostic",
                step=self.step,
            )

    def as_tuple(self) -> tuple[float | int, ...]:
        return (
            self.step,
            self.div_max,
            self.div_rms,
            self.div_rel,
            self.schur_iterations,
            self.schur_abs_residual,
            self.schur_rel_residual,
            self.wall_normal_momentum_max,
            self.wall_normal_momentum_rms,
        )


DIAGNOSTIC_DTYPE = np.dtype(
    [
        ("step", np.int64),
        ("div_max", np.float64),
        ("div_rms", np.float64),
        ("div_rel", np.float64),
        ("schur_iterations", np.float64),
        ("schur_abs_residual", np.float64),
        ("schur_rel_residual", np.float64),
        ("wall_normal_momentum_max", np.float64),
        ("wall_normal_momentum_rms", np.float64),
    ]
)
DIAGNOSTIC_HEADER = (
    "step,div_max,div_rms,div_rel,schur_iterations,schur_abs_residual,"
    "schur_rel_residual,wall_normal_momentum_max,wall_normal_momentum_rms"
)


def capture_plane_observation(
    fields: FieldView,
    *,
    step: int,
) -> PlaneObservation:
    """Capture Q/u/p without changing the established NumPy layout."""

    if not isinstance(fields, FieldView):
        raise TypeError("fields must be a FieldView")
    actual_step = step
    if fields.spatial.shape[1] != 1:
        raise ValueError("Plane production observations require batch_size=1")

    def cpu_array(name: str, *, evolved: bool = False) -> np.ndarray:
        value = fields[name] if evolved else fields[name]
        return value[0].detach().to(device="cpu").contiguous().numpy()

    return PlaneObservation(
        step=actual_step,
        q=np.stack(
            [cpu_array(name, evolved=True) for name in Q_COMPONENTS],
            axis=-1,
        ),
        velocity=np.stack(
            [cpu_array(name) for name in VELOCITY_COMPONENTS],
            axis=-1,
        ),
        pressure=cpu_array("p"),
    )


def _atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary observation exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def write_plane_observation(
    directory: str | Path,
    observation: PlaneObservation,
    *,
    save_hydrodynamics: bool,
) -> tuple[Path, ...]:
    """Write canonical filenames and refuse every partial overwrite."""

    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Plane output directory is missing: {directory}")
    values = {"Q": observation.q}
    if save_hydrodynamics:
        values.update(u=observation.velocity, p=observation.pressure)
    paths = tuple(
        directory / f"{prefix}_{observation.step}.npy" for prefix in values
    )
    if any(path.exists() for path in paths):
        raise FileExistsError(
            "refusing to overwrite Plane observation files: "
            f"{tuple(path for path in paths if path.exists())!r}"
        )
    for path, value in zip(paths, values.values(), strict=True):
        _atomic_save(path, value)
    return paths


def capture_plane_diagnostic(
    fields: FieldView,
    *,
    step: int,
    viscosity: float,
    friction: float,
    constraint_diagnostics: Mapping[str, object],
) -> PlaneDiagnostic:
    """Compute the same divergence, Schur, and wall-momentum diagnostics."""

    div_u = (
        fields.gradient("ux", axis=0)
        + fields.gradient("uy", axis=1)
        + fields.gradient("uz", axis=2)
    )
    div_abs = div_u.abs()
    div_max = div_abs.max().item()
    div_rms = torch.sqrt(torch.mean(div_abs.square())).item()
    grad_u_sq = sum(
        fields.gradient(name, axis=axis).abs().square()
        for name in VELOCITY_COMPONENTS
        for axis in range(3)
    )
    grad_u_rms = torch.sqrt(torch.mean(grad_u_sq)).item()
    div_rel = div_rms / max(grad_u_rms, 1.0e-30)

    normal_force = _projected_normal_force(constraint_diagnostics)
    residual = fields.gradient("p", axis=2) - (
        normal_force + viscosity * fields.laplacian("uz") - friction * fields["uz"]
    )
    wall = torch.stack((residual[..., 0], residual[..., -1]), dim=-1).abs()
    flow = _flow_diagnostics(constraint_diagnostics)
    return PlaneDiagnostic(
        step=step,
        div_max=div_max,
        div_rms=div_rms,
        div_rel=div_rel,
        schur_iterations=float(flow["last_pressure_iterations"]),
        schur_abs_residual=float(flow["last_pressure_residual"]),
        schur_rel_residual=float(flow["last_pressure_relative_residual"]),
        wall_normal_momentum_max=wall.max().item(),
        wall_normal_momentum_rms=torch.sqrt(torch.mean(wall.square())).item(),
    )


def write_plane_diagnostics(
    directory: str | Path,
    diagnostics: tuple[PlaneDiagnostic, ...],
) -> tuple[Path, Path]:
    """Write the established structured NPY and CSV diagnostics."""

    directory = Path(directory).expanduser().resolve()
    array = np.array([value.as_tuple() for value in diagnostics], dtype=DIAGNOSTIC_DTYPE)
    npy_path = directory / "diagnostics.npy"
    csv_path = directory / "diagnostics.csv"
    if npy_path.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite Plane diagnostics")
    _atomic_save(npy_path, array)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            np.savetxt(
                handle,
                array,
                delimiter=",",
                header=DIAGNOSTIC_HEADER,
                comments="",
            )
        temporary.replace(csv_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return npy_path, csv_path


def _projected_normal_force(constraint_diagnostics):
    value = constraint_diagnostics.get("last_projected_normal_force")
    if value is None:
        raise RuntimeError("legacy normal-force diagnostics are unavailable")
    return value


def _flow_diagnostics(constraint_diagnostics):
    return {
        "last_pressure_iterations": int(constraint_diagnostics["last_pressure_iterations"]),
        "last_pressure_residual": float(constraint_diagnostics["last_pressure_residual"]),
        "last_pressure_relative_residual": float(
            constraint_diagnostics["last_pressure_relative_residual"]
        ),
    }


@dataclass(frozen=True, slots=True)
class PlaneCapture:
    """Plane observation recipe; diagnostics belong to the calling Runner."""

    hydrodynamics: bool
    runtime_path: str
    viscosity: float
    friction: float
    application_family: str = "plane_complete_stress"
    wants_constraint_diagnostics: bool = True
    observation = staticmethod(capture_plane_observation)
    diagnostic_dtype = DIAGNOSTIC_DTYPE
    diagnostic_header = DIAGNOSTIC_HEADER
    progress_order_error = "progress iterator must yield consecutive local steps"
    progress_length_error = "progress iterator length differs from requested steps"

    @staticmethod
    def begin_run(directory, metadata):
        if (directory / "COMPLETE").exists():
            raise FileExistsError("Plane output directory is already complete")
        write_run_metadata(directory, metadata, status="running")

    @staticmethod
    def progress_diagnostic(progress, value):
        setter = getattr(progress, "set_postfix", None)
        if callable(setter):
            setter(
                div_max=f"{value.div_max:.2e}",
                div_rms=f"{value.div_rms:.2e}",
                div_rel=f"{value.div_rel:.2e}",
                schur_it=int(value.schur_iterations),
                schur_rel=f"{value.schur_rel_residual:.2e}",
                wall_n_rms=f"{value.wall_normal_momentum_rms:.2e}",
            )

    @staticmethod
    def record_failure(directory, metadata, error, *, state, saved, checkpoints, elapsed):
        public_error = (
            error
            if isinstance(error, SimulationExecutionError)
            else SimulationExecutionError(
                str(error) or type(error).__name__,
                application_family="plane_complete_stress_beris_edwards",
                stage="workflow_execution",
                step=int(state.completed_steps),
                details={"legacy_exception_type": type(error).__name__},
            )
        )
        metadata.update(completed_steps=int(state.completed_steps), elapsed_seconds=elapsed)
        metadata["workflow"].update(
            saved_steps=list(saved),
            checkpoint_steps=list(checkpoints),
            failed_step=int(state.completed_steps),
        )
        metadata["failure"] = public_error.to_metadata()
        write_run_metadata(directory, metadata, status="failed")

    @staticmethod
    def checkpoint_provenance(metadata):
        return {"legacy_configuration": metadata["configuration"]}

    @staticmethod
    def finalize_metadata(metadata, *, state, saved, checkpoints):
        metadata["numerics"]["spectral_refresh"]["actual_count"] = int(state.refresh_phase[1])
        metadata["workflow"].update(
            saved_steps=list(saved),
            checkpoint_steps=list(checkpoints),
            final_step=state.completed_steps,
        )

    def diagnostic(self, fields, *, step, constraint_diagnostics):
        return capture_plane_diagnostic(
            fields,
            step=step,
            viscosity=self.viscosity,
            friction=self.friction,
            constraint_diagnostics=constraint_diagnostics,
        )
