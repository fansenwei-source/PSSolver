"""Test-only carrier for exercising the original v0.4 Channel writer."""

import json

import numpy as np
import torch

from pssolver.configuration.legacy_channel import (
    _STATE_COMPONENTS,
    CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
    CHECKPOINT_VERSION,
    _sha256,
    build_channel_checkpoint_compatibility_identity,
    load_channel_initial_q,
    translate_channel,
)
from pssolver.engine.engine import Engine
from pssolver.io.checkpoint import checkpoint_directory_transaction, seal_checkpoint_metadata
from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS
from pssolver.runtime.legacy_entry import _ChannelCheckpointDestination, restore_channel_state
from pssolver.runtime.state_validation import require_finite_tensors


class ChannelProductionFixture(_ChannelCheckpointDestination):
    """Expose just the fixture writer and assertions on a new Engine state."""

    def __init__(self, engine, state, report):
        super().__init__(engine, state)
        self.report = report

    @property
    def completed_steps(self):
        return self.state.completed_steps

    def _progress(self):
        integrator = self.solver.integrator
        integrator.step_count, integrator.refresh_count = self.state.refresh_phase

    def advance(self, steps):
        for _ in range(steps):
            self.engine.step_(self.state, diagnostics=self.report)
        self._progress()

    def synchronize_for_observation(self):
        self.engine.prepare_constraints_(self.state, diagnostics=self.report)
        self._progress()

    def flow_diagnostics(self):
        return {"schema_version": 2, "pressure": dict(self.report)}


def build_channel_production_fixture(run_spec, *, device="cpu"):
    initial, _, _ = load_channel_initial_q(run_spec)
    dtype = getattr(torch, run_spec.simulation.numerics.precision.value)
    engine = Engine(translate_channel(run_spec), device=device, dtype=dtype)
    report = {}
    state = engine.initial_state({key: value.to(device=device) for key, value in initial.items()},
                                 diagnostics=report)
    return ChannelProductionFixture(engine, state, report)


def restore_channel_checkpoint(directory, adapter, *, run_spec):
    return restore_channel_state(adapter.engine, adapter.state, directory, run_spec=run_spec)


def _tensor_record(path, values):
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
    return {
        "file": path.name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": _sha256(path),
    }


def write_channel_beris_edwards_checkpoint(
    directory,
    adapter,
    *,
    run_spec,
):
    """Write the complete-stress Channel production checkpoint format."""

    adapter.synchronize_for_observation()
    fields = adapter.fields
    pressure_guess_tensor = adapter.capture_pressure_guess()
    state_names = (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p")
    require_finite_tensors(
        (
            (f"{name}{suffix}", fields[f"{name}{suffix}"])
            for suffix in ("", ".hat")
            for name in state_names
        ),
        application_family="channel_complete_stress",
        stage="checkpoint",
        step=adapter.completed_steps,
    )
    require_finite_tensors(
        (("pressure_guess", pressure_guess_tensor),),
        application_family="channel_complete_stress",
        stage="checkpoint",
        step=adapter.completed_steps,
    )
    with checkpoint_directory_transaction(
        directory,
        operation="complete_stress_channel_checkpoint_export",
    ) as (staging, _target):
        records = {"spatial": {}, "spectral": {}}
        for kind, suffix in (("spatial", ""), ("spectral", ".hat")):
            for name in _STATE_COMPONENTS:
                values = fields[f"{name}{suffix}"].detach().cpu().numpy()
                path = staging / f"{kind}__{name}.npy"
                records[kind][name] = _tensor_record(path, values)
        pressure_guess = pressure_guess_tensor.cpu().numpy()
        pressure_record = _tensor_record(
            staging / "backend__pressure_guess.npy",
            pressure_guess,
        )
        integrator = adapter.solver.integrator
        backend_restart = adapter.backend_restart_metadata()
        compatibility_identity = build_channel_checkpoint_compatibility_identity(
            run_spec,
            fields=fields,
            pressure_guess=pressure_guess_tensor,
            backend_restart=backend_restart,
        )
        metadata = {
            "format_version": CHECKPOINT_VERSION,
            "runtime_path": CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
            "run_provenance": {
                "legacy_runtime_identity_sha256": (
                    run_spec.runtime_identity_sha256()
                ),
            },
            "compatibility_identity": compatibility_identity.to_metadata(),
            "compatibility_identity_sha256": (
                compatibility_identity.canonical_sha256()
            ),
            "completed_steps": adapter.completed_steps,
            "integrator": {
                "spectral_refresh_interval": integrator.spectral_refresh_interval,
                "step_count": int(integrator.step_count),
                "refresh_count": int(integrator.refresh_count),
            },
            "backend_restart": backend_restart,
            "tensor_files": records,
            "backend_files": {"pressure_guess": pressure_record},
        }
        metadata = seal_checkpoint_metadata(metadata)
        (staging / "checkpoint.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
