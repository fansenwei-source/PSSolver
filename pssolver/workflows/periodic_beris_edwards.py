"""Observation and exact-restart workflow for the P8.2 periodic runtime."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS
from pssolver.runtime.periodic_beris_edwards import (
    PeriodicRuntimeAdapterProtocol,
)
from .periodic_checkpoint import (
    capture_periodic_checkpoint,
    load_periodic_checkpoint,
    read_periodic_checkpoint_header,
    restore_periodic_checkpoint,
    write_periodic_checkpoint,
)
from .diagnostic_io import write_structured_diagnostics


@dataclass(frozen=True, slots=True)
class PeriodicObservation:
    step: int
    q: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray


@dataclass(frozen=True, slots=True)
class PeriodicDiagnostic:
    step: int
    divergence_max: float
    divergence_rms: float
    pressure_mean: float
    velocity_mean_norm: float

    def as_tuple(self) -> tuple[int | float, ...]:
        return (
            self.step,
            self.divergence_max,
            self.divergence_rms,
            self.pressure_mean,
            self.velocity_mean_norm,
        )


PERIODIC_DIAGNOSTIC_DTYPE = np.dtype(
    [
        ("step", np.int64),
        ("divergence_max", np.float64),
        ("divergence_rms", np.float64),
        ("pressure_mean", np.float64),
        ("velocity_mean_norm", np.float64),
    ]
)
PERIODIC_DIAGNOSTIC_HEADER = (
    "step,divergence_max,divergence_rms,pressure_mean,velocity_mean_norm"
)


@dataclass(frozen=True, slots=True)
class PeriodicWorkflowResult:
    start_step: int
    final_step: int
    elapsed_seconds: float
    saved_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    final_observation: PeriodicObservation
    diagnostics: tuple[PeriodicDiagnostic, ...]


def capture_periodic_observation(adapter, *, step=None):
    if not isinstance(adapter, PeriodicRuntimeAdapterProtocol):
        raise TypeError("adapter must implement PeriodicRuntimeAdapterProtocol")
    fields = adapter.fields

    def array(name):
        return fields[name][0].detach().cpu().contiguous().numpy()

    return PeriodicObservation(
        adapter.completed_steps if step is None else step,
        np.stack(tuple(array(name) for name in Q_COMPONENTS), axis=-1),
        np.stack(tuple(array(name) for name in VELOCITY_COMPONENTS), axis=-1),
        array("p"),
    )


def capture_periodic_diagnostic(adapter, *, step):
    fields = adapter.fields
    divergence = sum(
        fields.gradient(name, axis=axis)
        for axis, name in enumerate(VELOCITY_COMPONENTS)
    )
    velocity_means = torch.stack(
        tuple(fields[name].mean() for name in VELOCITY_COMPONENTS)
    )
    return PeriodicDiagnostic(
        step=step,
        divergence_max=float(divergence.abs().max().item()),
        divergence_rms=float(torch.sqrt(divergence.square().mean()).item()),
        pressure_mean=float(fields["p"].mean().item()),
        velocity_mean_norm=float(torch.linalg.vector_norm(velocity_means).item()),
    )


def _save_observation(directory, observation, *, hydrodynamics):
    values = {"Q": observation.q}
    if hydrodynamics:
        values.update(u=observation.velocity, p=observation.pressure)
    for prefix, value in values.items():
        path = directory / f"{prefix}_{observation.step}.npy"
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        with path.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)


def _write_checkpoint(directory, adapter, *, run_spec):
    return write_periodic_checkpoint(
        directory,
        capture_periodic_checkpoint(
            adapter,
            run_spec=run_spec,
        ),
    )


def _load_checkpoint(directory, adapter, *, run_spec):
    header = read_periodic_checkpoint_header(directory)
    if header.compatibility_identity is None:
        raise ValueError(
            "legacy periodic checkpoint requires explicit identity upgrade"
        )
    checkpoint = load_periodic_checkpoint(directory)
    return restore_periodic_checkpoint(
        adapter,
        checkpoint,
        run_spec=run_spec,
    )


class PeriodicBerisEdwardsWorkflow:
    def __init__(self, adapter, run_spec, output_directory, metadata):
        if not isinstance(adapter, PeriodicRuntimeAdapterProtocol):
            raise TypeError("adapter must implement PeriodicRuntimeAdapterProtocol")
        self.adapter = adapter
        self.run_spec = run_spec
        self.output_directory = Path(output_directory).expanduser().resolve()
        self.metadata = dict(metadata)
        self.options = dict(run_spec.simulation.workflow.options)
        self.saved = []
        self.checkpoints = []
        self.diagnostics = []

    def run(self, *, progress=None):
        restart = self.options["restart_from"]
        start = 0 if restart is None else _load_checkpoint(
            restart,
            self.adapter,
            run_spec=self.run_spec,
        )
        steps = self.run_spec.simulation.workflow.steps
        if self.options["save_start_step"] > start + steps:
            raise ValueError(
                "save_start_step must not exceed the absolute final step"
            )
        iterator = range(steps) if progress is None else progress
        started = time.time()
        executed = 0
        for local_step in iterator:
            if local_step != executed:
                raise ValueError(
                    "progress must yield consecutive local steps starting at zero"
                )
            absolute = start + local_step
            diagnostic_due = (
                self.options["diagnostics"]
                and absolute % self.options["diagnostic_interval"] == 0
            )
            save_due = (
                absolute >= self.options["save_start_step"]
                and absolute % self.options["save_interval"] == 0
            )
            if diagnostic_due or save_due:
                self.adapter.synchronize_for_observation()
            if diagnostic_due:
                self.diagnostics.append(
                    capture_periodic_diagnostic(self.adapter, step=absolute)
                )
            if save_due:
                observation = capture_periodic_observation(self.adapter, step=absolute)
                _save_observation(
                    self.output_directory,
                    observation,
                    hydrodynamics=self.options["save_hydrodynamics"],
                )
                self.saved.append(absolute)
            self.adapter.advance(1)
            executed += 1
            interval = self.options["checkpoint_interval"]
            if interval is not None and self.adapter.completed_steps % interval == 0:
                _write_checkpoint(
                    self.output_directory / f"checkpoint_{self.adapter.completed_steps}",
                    self.adapter,
                    run_spec=self.run_spec,
                )
                self.checkpoints.append(self.adapter.completed_steps)
        if executed != steps:
            raise ValueError(
                f"progress yielded {executed} steps; expected exactly {steps}"
            )
        final_step = start + steps
        self.adapter.synchronize_for_observation()
        final = capture_periodic_observation(self.adapter, step=final_step)
        if final_step not in self.saved:
            _save_observation(
                self.output_directory,
                final,
                hydrodynamics=self.options["save_hydrodynamics"],
            )
            self.saved.append(final_step)
        if self.options["diagnostics"]:
            self.diagnostics.append(
                capture_periodic_diagnostic(self.adapter, step=final_step)
            )
            write_structured_diagnostics(
                self.output_directory,
                [value.as_tuple() for value in self.diagnostics],
                dtype=PERIODIC_DIAGNOSTIC_DTYPE,
                header=PERIODIC_DIAGNOSTIC_HEADER,
            )
        elapsed = time.time() - started
        self.metadata.update(
            status="complete",
            completed_steps=final_step,
            elapsed_seconds=elapsed,
            saved_steps=self.saved,
            checkpoint_steps=self.checkpoints,
            diagnostics={
                "enabled": self.options["diagnostics"],
                "count": len(self.diagnostics),
                "steps": [value.step for value in self.diagnostics],
                "npy": (
                    "diagnostics.npy" if self.options["diagnostics"] else None
                ),
                "csv": (
                    "diagnostics.csv" if self.options["diagnostics"] else None
                ),
            },
        )
        (self.output_directory / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (self.output_directory / "COMPLETE").write_text("complete\n", encoding="utf-8")
        return PeriodicWorkflowResult(
            start,
            final_step,
            elapsed,
            tuple(self.saved),
            tuple(self.checkpoints),
            final,
            tuple(self.diagnostics),
        )


__all__ = [
    "PeriodicBerisEdwardsWorkflow",
    "PeriodicDiagnostic",
    "PERIODIC_DIAGNOSTIC_DTYPE",
    "PERIODIC_DIAGNOSTIC_HEADER",
    "PeriodicObservation",
    "PeriodicWorkflowResult",
    "capture_periodic_diagnostic",
    "capture_periodic_observation",
]
