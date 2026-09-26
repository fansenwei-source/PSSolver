"""Observation and exact-restart workflow for the P8.3 Channel runtime."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS
from pssolver.runtime.channel_beris_edwards import (
    CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
    ChannelBerisEdwardsRuntimeAdapterProtocol,
)


CHECKPOINT_VERSION = 1
_STATE_COMPONENTS = (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsObservation:
    step: int
    q: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsDiagnostic:
    step: int
    divergence_max: float
    divergence_rms: float
    pressure_mean: float
    pressure_iterations: int
    pressure_relative_residual: float


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsWorkflowResult:
    start_step: int
    final_step: int
    elapsed_seconds: float
    saved_steps: tuple[int, ...]
    checkpoint_steps: tuple[int, ...]
    final_observation: ChannelBerisEdwardsObservation
    diagnostics: tuple[ChannelBerisEdwardsDiagnostic, ...]


def capture_channel_beris_edwards_observation(adapter, *, step=None):
    if not isinstance(adapter, ChannelBerisEdwardsRuntimeAdapterProtocol):
        raise TypeError(
            "adapter must implement ChannelBerisEdwardsRuntimeAdapterProtocol"
        )
    views = adapter.output_views
    return ChannelBerisEdwardsObservation(
        adapter.completed_steps if step is None else step,
        np.moveaxis(views.q[:, 0].detach().cpu().numpy(), 0, -1),
        np.moveaxis(views.velocity[:, 0].detach().cpu().numpy(), 0, -1),
        views.pressure[0].detach().cpu().numpy(),
    )


def capture_channel_beris_edwards_diagnostic(adapter, *, step):
    fields = adapter.fields
    divergence = sum(
        fields.gradient(name, axis=axis, projector=adapter.projector)
        for axis, name in enumerate(VELOCITY_COMPONENTS)
    )
    flow = adapter.flow_diagnostics()
    return ChannelBerisEdwardsDiagnostic(
        step=step,
        divergence_max=float(divergence.abs().max().item()),
        divergence_rms=float(torch.sqrt(divergence.square().mean()).item()),
        pressure_mean=float(fields["p"].mean().item()),
        pressure_iterations=flow["last_pressure_iterations"],
        pressure_relative_residual=flow["last_pressure_relative_residual"],
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


def _tensor_record(path, values):
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": _sha256(path),
    }


def _write_checkpoint(directory, adapter, *, runtime_identity_sha256):
    if directory.exists():
        raise FileExistsError(f"checkpoint already exists: {directory}")
    directory.mkdir(parents=True)
    adapter.synchronize_for_observation()
    fields = adapter.fields
    records = {"spatial": {}, "spectral": {}}
    for kind, suffix in (("spatial", ""), ("spectral", ".hat")):
        for name in _STATE_COMPONENTS:
            values = fields[f"{name}{suffix}"].detach().cpu().numpy()
            path = directory / f"{kind}__{name}.npy"
            records[kind][name] = _tensor_record(path, values)
    pressure_guess = adapter.capture_pressure_guess().cpu().numpy()
    pressure_record = _tensor_record(
        directory / "backend__pressure_guess.npy",
        pressure_guess,
    )
    integrator = adapter.solver.integrator
    metadata = {
        "format_version": CHECKPOINT_VERSION,
        "runtime_path": CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
        "runtime_identity_sha256": runtime_identity_sha256,
        "completed_steps": adapter.completed_steps,
        "integrator": {
            "spectral_refresh_interval": integrator.spectral_refresh_interval,
            "step_count": int(integrator.step_count),
            "refresh_count": int(integrator.refresh_count),
        },
        "backend_restart": adapter.backend_restart_metadata(),
        "tensor_files": records,
        "backend_files": {"pressure_guess": pressure_record},
    }
    (directory / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validated_array(directory, record, target, description):
    if set(record) != {"file", "shape", "dtype", "sha256"}:
        raise ValueError(f"{description} record schema differs")
    path = directory / record["file"]
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"{description} checksum mismatch")
    values = np.load(path, allow_pickle=False)
    if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
        raise ValueError(f"{description} metadata differs")
    if tuple(values.shape) != tuple(target.shape):
        raise ValueError(f"{description} shape differs")
    if str(values.dtype) != str(target.detach().cpu().numpy().dtype):
        raise ValueError(f"{description} dtype differs")
    if not np.isfinite(values).all():
        raise ValueError(f"{description} is not finite")
    return values


def _load_checkpoint(directory, adapter, *, runtime_identity_sha256):
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads(
        (directory / "checkpoint.json").read_text(encoding="utf-8")
    )
    if metadata.get("format_version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported complete-stress Channel checkpoint")
    if metadata.get("runtime_path") != CHANNEL_COMPLETE_STRESS_RUNTIME_PATH:
        raise ValueError("checkpoint runtime identity does not match target")
    if metadata.get("runtime_identity_sha256") != runtime_identity_sha256:
        raise ValueError("checkpoint scientific/runtime identity does not match")
    if metadata.get("backend_restart") != adapter.backend_restart_metadata():
        raise ValueError("checkpoint backend contract does not match target")

    fields = adapter.fields
    pending = []
    manifest = metadata.get("tensor_files", {})
    for kind, suffix in (("spatial", ""), ("spectral", ".hat")):
        records = manifest.get(kind, {})
        if set(records) != set(_STATE_COMPONENTS):
            raise ValueError("Channel checkpoint tensor manifest is incomplete")
        for name in _STATE_COMPONENTS:
            target = fields[f"{name}{suffix}"]
            values = _validated_array(
                directory,
                records[name],
                target,
                f"{kind} {name}",
            )
            pending.append((target, values))
    pressure_target = fields["p.hat"]
    pressure_record = metadata.get("backend_files", {}).get("pressure_guess")
    pressure_values = _validated_array(
        directory,
        pressure_record,
        pressure_target,
        "pressure guess",
    )

    # Mutation starts only after every identity, file, shape, dtype, checksum,
    # and finite-value gate has passed.
    for target, values in pending:
        target.copy_(torch.from_numpy(values).to(device=target.device))
    adapter.restore_pressure_guess(
        torch.from_numpy(pressure_values).to(device=pressure_target.device)
    )
    integrator = metadata["integrator"]
    adapter.restore_progress(
        completed_steps=metadata["completed_steps"],
        spectral_refresh_interval=integrator["spectral_refresh_interval"],
        integrator_step_count=integrator["step_count"],
        integrator_refresh_count=integrator["refresh_count"],
    )
    adapter.restore_derived_state()
    return int(metadata["completed_steps"])


class ChannelBerisEdwardsWorkflow:
    def __init__(self, adapter, run_spec, output_directory, metadata):
        if not isinstance(adapter, ChannelBerisEdwardsRuntimeAdapterProtocol):
            raise TypeError(
                "adapter must implement "
                "ChannelBerisEdwardsRuntimeAdapterProtocol"
            )
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
                    "progress must yield consecutive local steps from zero"
                )
            absolute = start + local_step
            if (
                self.options["diagnostics"]
                and absolute % self.options["diagnostic_interval"] == 0
            ):
                self.adapter.synchronize_for_observation()
                self.diagnostics.append(
                    capture_channel_beris_edwards_diagnostic(
                        self.adapter,
                        step=absolute,
                    )
                )
            if (
                absolute >= self.options["save_start_step"]
                and absolute % self.options["save_interval"] == 0
            ):
                observation = capture_channel_beris_edwards_observation(
                    self.adapter,
                    step=absolute,
                )
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
                    self.output_directory
                    / f"checkpoint_{self.adapter.completed_steps}",
                    self.adapter,
                    runtime_identity_sha256=(
                        self.run_spec.runtime_identity_sha256()
                    ),
                )
                self.checkpoints.append(self.adapter.completed_steps)
        if executed != steps:
            raise ValueError(
                f"progress yielded {executed} steps; expected {steps}"
            )
        final_step = start + steps
        self.adapter.synchronize_for_observation()
        final = capture_channel_beris_edwards_observation(
            self.adapter,
            step=final_step,
        )
        if final_step not in self.saved:
            _save_observation(
                self.output_directory,
                final,
                hydrodynamics=self.options["save_hydrodynamics"],
            )
            self.saved.append(final_step)
        if self.options["diagnostics"]:
            self.diagnostics.append(
                capture_channel_beris_edwards_diagnostic(
                    self.adapter,
                    step=final_step,
                )
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
        (self.output_directory / "COMPLETE").write_text(
            "complete\n",
            encoding="utf-8",
        )
        return ChannelBerisEdwardsWorkflowResult(
            start,
            final_step,
            elapsed,
            tuple(self.saved),
            tuple(self.checkpoints),
            final,
            tuple(self.diagnostics),
        )


__all__ = [
    "ChannelBerisEdwardsDiagnostic",
    "ChannelBerisEdwardsObservation",
    "ChannelBerisEdwardsWorkflow",
    "ChannelBerisEdwardsWorkflowResult",
    "capture_channel_beris_edwards_diagnostic",
    "capture_channel_beris_edwards_observation",
]
