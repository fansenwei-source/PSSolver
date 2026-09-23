"""Canonical observations and diagnostics for the rectangular Channel."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch

from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.channel_active_nematics import (
    ChannelRuntimeAdapterProtocol,
)


VELOCITY_COMPONENTS = ("ux", "uy", "uz")


@dataclass(frozen=True, slots=True)
class ChannelObservation:
    step: int
    q: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("observation step must be non-negative")
        expected = ((self.q, 4, 5), (self.velocity, 4, 3), (self.pressure, 3, None))
        for value, ndim, components in expected:
            if not isinstance(value, np.ndarray) or value.ndim != ndim:
                raise ValueError("Channel observation shape is invalid")
            if components is not None and value.shape[-1] != components:
                raise ValueError("Channel observation component count is invalid")
            if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
                raise ValueError("Channel observations must be finite floating arrays")
        if self.velocity.shape[:-1] != self.q.shape[:-1] or self.pressure.shape != self.q.shape[:-1]:
            raise ValueError("Channel observation grids differ")
        object.__setattr__(self, "q", np.array(self.q, copy=True))
        object.__setattr__(self, "velocity", np.array(self.velocity, copy=True))
        object.__setattr__(self, "pressure", np.array(self.pressure, copy=True))


@dataclass(frozen=True, slots=True)
class ChannelDiagnostic:
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
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("diagnostic step must be non-negative")
        if not all(math.isfinite(float(value)) for value in self.as_tuple()[1:]):
            raise ValueError("diagnostic values must be finite")

    def as_tuple(self) -> tuple[float | int, ...]:
        return (
            self.step, self.div_max, self.div_rms, self.div_rel,
            self.schur_iterations, self.schur_abs_residual,
            self.schur_rel_residual, self.wall_normal_momentum_max,
            self.wall_normal_momentum_rms,
        )


DIAGNOSTIC_DTYPE = np.dtype([
    ("step", np.int64), ("div_max", np.float64),
    ("div_rms", np.float64), ("div_rel", np.float64),
    ("schur_iterations", np.float64),
    ("schur_abs_residual", np.float64),
    ("schur_rel_residual", np.float64),
    ("wall_normal_momentum_max", np.float64),
    ("wall_normal_momentum_rms", np.float64),
])
DIAGNOSTIC_HEADER = (
    "step,div_max,div_rms,div_rel,schur_iterations,schur_abs_residual,"
    "schur_rel_residual,wall_normal_momentum_max,wall_normal_momentum_rms"
)


def capture_channel_observation(
    adapter: ChannelRuntimeAdapterProtocol,
    *,
    step: int | None = None,
    synchronize: bool = False,
) -> ChannelObservation:
    if not isinstance(adapter, ChannelRuntimeAdapterProtocol):
        raise TypeError("adapter must implement ChannelRuntimeAdapterProtocol")
    if adapter.solver.batchsize != 1:
        raise ValueError("Channel production observations require batch_size=1")
    if synchronize:
        adapter.synchronize_for_observation()
    views = adapter.output_views
    return ChannelObservation(
        step=adapter.completed_steps if step is None else step,
        q=views.q[:, 0].movedim(0, -1).detach().cpu().contiguous().numpy(),
        velocity=views.velocity[:, 0].movedim(0, -1).detach().cpu().contiguous().numpy(),
        pressure=views.pressure[0].detach().cpu().contiguous().numpy(),
    )


def _atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite Channel output: {path}")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def write_channel_observation(directory: str | Path, value: ChannelObservation) -> tuple[Path, ...]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Channel output directory is missing: {directory}")
    paths = tuple(directory / f"{name}_{value.step}.npy" for name in ("Q", "u", "p"))
    if any(path.exists() for path in paths):
        raise FileExistsError("refusing to overwrite Channel observation files")
    for path, array in zip(paths, (value.q, value.velocity, value.pressure), strict=True):
        _atomic_save(path, array)
    return paths


def capture_channel_diagnostic(
    adapter: ChannelRuntimeAdapterProtocol,
    *,
    step: int,
    beta: float,
    viscosity: float,
    friction: float,
) -> ChannelDiagnostic:
    fields = adapter.fields
    div_u = sum(fields.gradient(name, axis=axis) for name, axis in zip(VELOCITY_COMPONENTS, range(3), strict=True))
    div_abs = div_u.abs()
    div_rms = torch.sqrt(torch.mean(div_abs.square())).item()
    grad_u_sq = sum(
        fields.gradient(name, axis=axis).abs().square()
        for name in VELOCITY_COMPONENTS for axis in range(3)
    )
    div_rel = div_rms / max(torch.sqrt(torch.mean(grad_u_sq)).item(), 1.0e-30)
    force_prefactor = beta * adapter.solver.parameters["alpha"]
    force_y = force_prefactor * (
        fields.gradient("Qxy", axis=0)
        + fields.gradient("Qyy", axis=1)
        + fields.gradient("Qyz", axis=2)
    )
    force_z = force_prefactor * (
        fields.gradient("Qxz", axis=0)
        + fields.gradient("Qyz", axis=1)
        - fields.gradient("Qxx", axis=2)
        - fields.gradient("Qyy", axis=2)
    )
    residuals = []
    for component, axis, force_component in (
        ("uy", 1, force_y),
        ("uz", 2, force_z),
    ):
        residual = fields.gradient("p", axis=axis) - (
            force_component
            + viscosity * fields.laplacian(component)
            - friction * fields[component]
        )
        residuals.extend((residual.select(-2 if axis == 1 else -1, 0).reshape(-1), residual.select(-2 if axis == 1 else -1, -1).reshape(-1)))
    wall = torch.cat(residuals).abs()
    static = adapter.solver.model.static_model
    return ChannelDiagnostic(
        step=step,
        div_max=div_abs.max().item(), div_rms=div_rms, div_rel=div_rel,
        schur_iterations=float(static.last_pressure_iterations),
        schur_abs_residual=float(static.last_pressure_residual),
        schur_rel_residual=float(static.last_pressure_relative_residual),
        wall_normal_momentum_max=wall.max().item(),
        wall_normal_momentum_rms=torch.sqrt(torch.mean(wall.square())).item(),
    )


def write_channel_diagnostics(directory: str | Path, values: tuple[ChannelDiagnostic, ...]) -> tuple[Path, Path]:
    directory = Path(directory).expanduser().resolve()
    array = np.array([value.as_tuple() for value in values], dtype=DIAGNOSTIC_DTYPE)
    npy_path, csv_path = directory / "diagnostics.npy", directory / "diagnostics.csv"
    if npy_path.exists() or csv_path.exists():
        raise FileExistsError("refusing to overwrite Channel diagnostics")
    _atomic_save(npy_path, array)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            np.savetxt(handle, array, delimiter=",", header=DIAGNOSTIC_HEADER, comments="")
        temporary.replace(csv_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return npy_path, csv_path


__all__ = [
    "ChannelDiagnostic", "ChannelObservation", "DIAGNOSTIC_DTYPE",
    "DIAGNOSTIC_HEADER", "capture_channel_diagnostic",
    "capture_channel_observation", "write_channel_diagnostics",
    "write_channel_observation",
]
