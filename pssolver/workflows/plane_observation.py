"""Canonical observations and diagnostics for the Plane workflow."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS
from pssolver.runtime import PlaneRuntimeAdapterProtocol


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
                raise ValueError(f"{description} observation must be finite")
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
            raise ValueError("diagnostic values must be finite")

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
    adapter: PlaneRuntimeAdapterProtocol,
    *,
    step: int | None = None,
    synchronize: bool = False,
) -> PlaneObservation:
    """Capture Q/u/p without changing the established NumPy layout."""

    if not isinstance(adapter, PlaneRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PlaneRuntimeAdapterProtocol")
    if synchronize:
        adapter.synchronize_for_observation()
    actual_step = adapter.completed_steps if step is None else step
    fields = adapter.fields
    if adapter.solver.batchsize != 1:
        raise ValueError("Plane production observations require batch_size=1")

    def cpu_array(name: str) -> np.ndarray:
        return (
            fields[name][0]
            .detach()
            .to(device="cpu")
            .contiguous()
            .numpy()
        )

    return PlaneObservation(
        step=actual_step,
        q=np.stack([cpu_array(name) for name in Q_COMPONENTS], axis=-1),
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
    adapter: PlaneRuntimeAdapterProtocol,
    *,
    step: int,
    viscosity: float,
    friction: float,
) -> PlaneDiagnostic:
    """Compute the same divergence, Schur, and wall-momentum diagnostics."""

    fields = adapter.fields
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

    normal_force = adapter.projected_normal_force()
    residual = fields.gradient("p", axis=2) - (
        normal_force
        + viscosity * fields.laplacian("uz")
        - friction * fields["uz"]
    )
    wall = torch.stack((residual[..., 0], residual[..., -1]), dim=-1).abs()
    flow = adapter.flow_diagnostics()
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


__all__ = [
    "DIAGNOSTIC_DTYPE",
    "DIAGNOSTIC_HEADER",
    "PlaneDiagnostic",
    "PlaneObservation",
    "capture_plane_diagnostic",
    "capture_plane_observation",
    "write_plane_diagnostics",
    "write_plane_observation",
]
