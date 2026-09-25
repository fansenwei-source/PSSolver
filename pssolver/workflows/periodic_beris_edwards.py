"""Observation and exact-restart workflow for the P8.2 periodic runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS
from pssolver.runtime.periodic_beris_edwards import (
    PERIODIC_RUNTIME_PATH,
    PeriodicRuntimeAdapterProtocol,
)


CHECKPOINT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _write_checkpoint(directory, adapter, *, runtime_identity_sha256):
    if directory.exists():
        raise FileExistsError(f"checkpoint already exists: {directory}")
    directory.mkdir(parents=True)
    adapter.synchronize_for_observation()
    fields = adapter.fields
    records = {"spatial": {}, "spectral": {}}
    for kind, suffix in (("spatial", ""), ("spectral", ".hat")):
        for name in Q_COMPONENTS:
            path = directory / f"{kind}__{name}.npy"
            values = fields[f"{name}{suffix}"].detach().cpu().numpy()
            with path.open("xb") as handle:
                np.save(handle, values, allow_pickle=False)
            records[kind][name] = {
                "file": path.name,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "sha256": _sha256(path),
            }
    integrator = adapter.solver.integrator
    metadata = {
        "format_version": CHECKPOINT_VERSION,
        "runtime_path": PERIODIC_RUNTIME_PATH,
        "runtime_identity_sha256": runtime_identity_sha256,
        "completed_steps": adapter.completed_steps,
        "integrator": {
            "spectral_refresh_interval": integrator.spectral_refresh_interval,
            "step_count": int(integrator.step_count),
            "refresh_count": int(integrator.refresh_count),
        },
        "backend_restart": adapter.backend_restart_metadata(),
        "tensor_files": records,
    }
    (directory / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_checkpoint(directory, adapter, *, runtime_identity_sha256):
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads(
        (directory / "checkpoint.json").read_text(encoding="utf-8")
    )
    if metadata.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported periodic checkpoint version")
    if metadata.get("runtime_path") != PERIODIC_RUNTIME_PATH:
        raise ValueError("checkpoint runtime identity does not match target")
    if metadata.get("runtime_identity_sha256") != runtime_identity_sha256:
        raise ValueError("checkpoint scientific/runtime identity does not match")
    if metadata.get("backend_restart") != adapter.backend_restart_metadata():
        raise ValueError("checkpoint backend contract does not match target")
    fields = adapter.fields
    manifest = metadata.get("tensor_files")
    for kind, suffix in (("spatial", ""), ("spectral", ".hat")):
        records = manifest.get(kind, {})
        if tuple(records) != Q_COMPONENTS:
            raise ValueError("periodic checkpoint tensor manifest is incomplete")
        for name in Q_COMPONENTS:
            record = records[name]
            path = directory / record["file"]
            if _sha256(path) != record["sha256"]:
                raise ValueError("periodic checkpoint tensor checksum mismatch")
            values = np.load(path, allow_pickle=False)
            target = fields[f"{name}{suffix}"]
            if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
                raise ValueError("periodic checkpoint tensor metadata differs")
            if tuple(values.shape) != tuple(target.shape):
                raise ValueError("periodic checkpoint tensor shape differs")
            if str(values.dtype) != str(target.detach().cpu().numpy().dtype):
                raise ValueError("periodic checkpoint tensor dtype differs")
            if not np.isfinite(values).all():
                raise ValueError("periodic checkpoint tensor is not finite")
            target.copy_(torch.from_numpy(values).to(device=target.device))
    integrator = metadata["integrator"]
    adapter.synchronize_for_observation()
    adapter.restore_progress(
        completed_steps=metadata["completed_steps"],
        spectral_refresh_interval=integrator["spectral_refresh_interval"],
        integrator_step_count=integrator["step_count"],
        integrator_refresh_count=integrator["refresh_count"],
    )
    return int(metadata["completed_steps"])


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
            runtime_identity_sha256=self.run_spec.runtime_identity_sha256(),
        )
        steps = self.run_spec.simulation.workflow.steps
        iterator = range(steps) if progress is None else progress
        started = time.time()
        executed = 0
        for local_step in iterator:
            if local_step != executed:
                raise ValueError(
                    "progress must yield consecutive local steps starting at zero"
                )
            absolute = start + local_step
            if self.options["diagnostics"] and absolute % self.options["diagnostic_interval"] == 0:
                self.adapter.synchronize_for_observation()
                self.diagnostics.append(
                    capture_periodic_diagnostic(self.adapter, step=absolute)
                )
            if absolute >= self.options["save_start_step"] and absolute % self.options["save_interval"] == 0:
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
                    runtime_identity_sha256=self.run_spec.runtime_identity_sha256(),
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
        elapsed = time.time() - started
        self.metadata.update(
            status="complete",
            completed_steps=final_step,
            elapsed_seconds=elapsed,
            saved_steps=self.saved,
            checkpoint_steps=self.checkpoints,
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
    "PeriodicObservation",
    "PeriodicWorkflowResult",
    "capture_periodic_diagnostic",
    "capture_periodic_observation",
]
