"""Opt-in observation and run coordination for the migrated Plane model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS
from pssolver.run_metadata import (
    prepare_new_run_directory,
    write_run_metadata,
)

from ._shadow_support import (
    completed_steps,
    evolved_names,
    ordered_tensor_sha256,
    require_nonnegative_integer,
    require_plane_beris_edwards_runtime,
)
from .checkpointing import (
    ShadowRunCheckpoint,
    capture_shadow_checkpoint,
    restore_shadow_checkpoint,
    write_shadow_checkpoint,
)
from .model_execution import ExperimentalModelRuntime
from .shadow_metadata import build_shadow_run_metadata


@dataclass(frozen=True, slots=True)
class ShadowObservation:
    """One synchronized, production-layout Plane observation."""

    step: int
    q: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray

    def __post_init__(self) -> None:
        require_nonnegative_integer(self.step, "step")
        for value, description in (
            (self.q, "q"),
            (self.velocity, "velocity"),
            (self.pressure, "pressure"),
        ):
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{description} must be a NumPy array")
            if not np.issubdtype(value.dtype, np.floating):
                raise TypeError(f"{description} must have a floating dtype")
            if not np.isfinite(value).all():
                raise ValueError(f"{description} must be finite")
        if self.q.ndim != 4 or self.q.shape[-1] != len(Q_COMPONENTS):
            raise ValueError("q must have shape (Nx, Ny, Nz, 5)")
        if self.velocity.shape != (*self.q.shape[:-1], 3):
            raise ValueError("velocity must have shape (Nx, Ny, Nz, 3)")
        if self.pressure.shape != self.q.shape[:-1]:
            raise ValueError("pressure shape must match the physical grid")
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


def capture_shadow_observation(
    runtime: ExperimentalModelRuntime,
) -> ShadowObservation:
    """Synchronize u/p and return the production Q/u/p array layout."""

    require_plane_beris_edwards_runtime(runtime)
    if runtime.solver.batchsize != 1:
        raise ValueError("production-layout shadow snapshots require batch_size=1")
    runtime.synchronize_algebraic_for_observation()
    fields = runtime.solver.fields

    def cpu_array(name: str) -> np.ndarray:
        return (
            fields[name][0]
            .detach()
            .to(device="cpu")
            .contiguous()
            .numpy()
        )

    q = np.stack([cpu_array(name) for name in Q_COMPONENTS], axis=-1)
    velocity = np.stack(
        [cpu_array(name) for name in VELOCITY_COMPONENTS],
        axis=-1,
    )
    return ShadowObservation(
        step=completed_steps(runtime),
        q=q,
        velocity=velocity,
        pressure=cpu_array("p"),
    )


def _atomic_write_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale temporary snapshot file exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def write_shadow_observation(
    directory: str | Path,
    observation: ShadowObservation,
) -> tuple[Path, Path, Path]:
    """Write Q/u/p using the established Plane snapshot filenames."""

    if not isinstance(observation, ShadowObservation):
        raise TypeError("observation must be a ShadowObservation")
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"shadow output directory is missing: {directory}")
    paths = (
        directory / f"Q_{observation.step}.npy",
        directory / f"u_{observation.step}.npy",
        directory / f"p_{observation.step}.npy",
    )
    existing = tuple(path for path in paths if path.exists())
    if existing:
        raise FileExistsError(
            "refusing to overwrite shadow observation files: "
            f"{existing!r}"
        )
    for path, values in zip(
        paths,
        (observation.q, observation.velocity, observation.pressure),
        strict=True,
    ):
        _atomic_write_array(path, values)
    return paths


class ExperimentalPlaneShadowRun:
    """Small opt-in driver boundary around a coupled Plane runtime."""

    def __init__(
        self,
        runtime: ExperimentalModelRuntime,
        output_directory: str | Path,
        *,
        initial_values: Mapping[str, torch.Tensor],
        initial_condition_metadata: Mapping[str, object],
    ) -> None:
        require_plane_beris_edwards_runtime(runtime)
        if runtime.solver.batchsize != 1:
            raise ValueError("Plane shadow runs currently require batch_size=1")
        if completed_steps(runtime) != 0:
            raise ValueError("new shadow runs require an unadvanced runtime")
        if not isinstance(initial_values, Mapping):
            raise TypeError("initial_values must be a mapping")
        values = dict(initial_values)
        expected = evolved_names(runtime)
        if tuple(values) != expected:
            raise ValueError(
                "initial_values must contain evolved components in declared order"
            )
        allowed_shapes = {
            runtime.context.physical_shape,
            (
                runtime.context.batch_size,
                *runtime.context.physical_shape,
            ),
        }
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"initial value for {name!r} must be a tensor")
            if tuple(value.shape) not in allowed_shapes:
                raise ValueError(
                    f"initial value for {name!r} has invalid shape "
                    f"{tuple(value.shape)!r}"
                )
            if value.dtype != runtime.context.real_dtype:
                raise ValueError(
                    f"initial value for {name!r} has dtype {value.dtype}; "
                    f"expected {runtime.context.real_dtype}"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"initial value for {name!r} must be finite")
        if not isinstance(initial_condition_metadata, Mapping):
            raise TypeError("initial_condition_metadata must be a mapping")
        initial_metadata = dict(initial_condition_metadata)
        try:
            json.dumps(initial_metadata, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "initial-condition metadata must be JSON-compatible"
            ) from exc

        raw_hash = ordered_tensor_sha256(values)
        output_path = prepare_new_run_directory(output_directory)
        runtime.reset(values)
        projected = {
            name: runtime.solver.fields[name] for name in expected
        }
        projected_hash = ordered_tensor_sha256(projected)
        self.runtime = runtime
        self.output_directory = output_path
        self._metadata = build_shadow_run_metadata(
            runtime,
            initial_condition=initial_metadata,
            raw_q_sha256=raw_hash,
            projected_q_sha256=projected_hash,
        )
        self._complete = False
        write_run_metadata(
            self.output_directory,
            self._metadata,
            status="running",
        )

    @classmethod
    def from_checkpoint(
        cls,
        runtime: ExperimentalModelRuntime,
        output_directory: str | Path,
        checkpoint: ShadowRunCheckpoint,
    ) -> "ExperimentalPlaneShadowRun":
        """Start a distinct continuation run from a complete checkpoint."""

        require_plane_beris_edwards_runtime(runtime)
        if runtime.solver.batchsize != 1:
            raise ValueError("Plane shadow runs currently require batch_size=1")
        if completed_steps(runtime) != 0:
            raise ValueError("checkpoint target runtime must be unadvanced")
        output_path = prepare_new_run_directory(output_directory)
        completed = restore_shadow_checkpoint(runtime, checkpoint)
        checkpoint_q = {
            name: checkpoint.evolved_spatial[name]
            for name in evolved_names(runtime)
        }
        q_hash = ordered_tensor_sha256(checkpoint_q)
        self = cls.__new__(cls)
        self.runtime = runtime
        self.output_directory = output_path
        self._metadata = build_shadow_run_metadata(
            runtime,
            initial_condition={
                "name": "complete_shadow_checkpoint",
                "source_runtime_identity_sha256": (
                    checkpoint.runtime_identity_sha256
                ),
                "source_completed_steps": checkpoint.completed_steps,
            },
            raw_q_sha256=q_hash,
            projected_q_sha256=q_hash,
        )
        self._metadata["restart"] = {
            "kind": "complete_shadow_checkpoint",
            "source_runtime_identity_sha256": (
                checkpoint.runtime_identity_sha256
            ),
            "source_completed_steps": checkpoint.completed_steps,
        }
        self._metadata["completed_steps"] = completed
        self._complete = False
        write_run_metadata(
            self.output_directory,
            self._metadata,
            status="running",
        )
        return self

    @property
    def completed_steps(self) -> int:
        return completed_steps(self.runtime)

    def _require_open(self) -> None:
        if self._complete:
            raise RuntimeError("shadow run is already complete")

    def advance(self, steps: int) -> int:
        self._require_open()
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        self.runtime.solver.run(steps)
        return self.completed_steps

    def save_observation(self) -> ShadowObservation:
        self._require_open()
        observation = capture_shadow_observation(self.runtime)
        write_shadow_observation(self.output_directory, observation)
        return observation

    def save_checkpoint(self) -> Path:
        self._require_open()
        checkpoint = capture_shadow_checkpoint(self.runtime)
        return write_shadow_checkpoint(
            self.output_directory / f"checkpoint_{checkpoint.completed_steps}",
            checkpoint,
        )

    def complete(self) -> ShadowObservation:
        self._require_open()
        observation = capture_shadow_observation(self.runtime)
        paths = tuple(
            self.output_directory / f"{prefix}_{observation.step}.npy"
            for prefix in ("Q", "u", "p")
        )
        if all(path.exists() for path in paths):
            saved_values = tuple(
                np.load(path, allow_pickle=False) for path in paths
            )
            expected_values = (
                observation.q,
                observation.velocity,
                observation.pressure,
            )
            if not all(
                saved.dtype == expected.dtype
                and saved.shape == expected.shape
                and np.array_equal(saved, expected)
                for saved, expected in zip(
                    saved_values,
                    expected_values,
                    strict=True,
                )
            ):
                raise RuntimeError(
                    "saved final observation differs from current state"
                )
        else:
            if any(path.exists() for path in paths):
                raise RuntimeError("final shadow observation is incomplete")
            write_shadow_observation(self.output_directory, observation)
        self._metadata["completed_steps"] = observation.step
        self._metadata["spectral_refresh"] = {
            "interval_steps": (
                self.runtime.solver.integrator.spectral_refresh_interval
            ),
            "actual_count": self.runtime.solver.integrator.refresh_count,
            "phase_step_count": self.runtime.solver.integrator.step_count,
        }
        write_run_metadata(
            self.output_directory,
            self._metadata,
            status="complete",
        )
        (self.output_directory / "COMPLETE").write_text(
            "complete\n",
            encoding="utf-8",
        )
        self._complete = True
        return observation


__all__ = [
    "ExperimentalPlaneShadowRun",
    "ShadowObservation",
    "capture_shadow_observation",
    "write_shadow_observation",
]
