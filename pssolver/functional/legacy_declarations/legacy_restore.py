"""Frozen v0.4 payload restore routines for the Engine checkpoint bridge."""

import hashlib
from types import SimpleNamespace

import numpy as np
import torch

from pssolver.engine.engine import Engine
from pssolver.engine.progress import (
    _refresh_counters_are_consistent,
    _refresh_counters_from_completed_steps,
)
from pssolver.engine.state import State
from pssolver.functional.legacy_declarations.legacy_channel import (
    _load_checkpoint as load_v04_channel_checkpoint,
)
from pssolver.functional.legacy_declarations.legacy_periodic import validate_periodic_restore
from pssolver.functional.legacy_declarations.legacy_plane import (
    plane_backend_restart,
    validate_plane_restore,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS


def restore_periodic_state(engine, state, checkpoint, *, run_spec):
    """Validate and materialize all legacy Q tensors before mutating caller state."""
    fields = engine.field_view(state)
    spatial = {name: fields[name] for name in Q_COMPONENTS}
    spectral = {name: fields[f"{name}.hat"] for name in Q_COMPONENTS}
    validate_periodic_restore(
        checkpoint,
        run_spec=run_spec,
        target_spatial=spatial,
        target_spectral=spectral,
        target_refresh_interval=engine.plan.integrator.refresh_interval,
        backend_restart={"kind": "periodic_stokes_stateless", "state_keys": []},
    )
    pending = tuple(
        (target[name], source[name].to(device=engine.device))
        for name in Q_COMPONENTS
        for target, source in (
            (spatial, checkpoint.evolved_spatial),
            (spectral, checkpoint.evolved_spectral),
        )
    )
    for target, source in pending:
        target.copy_(source)
    state.spatial[len(Q_COMPONENTS) :].zero_()
    state.spectral[len(Q_COMPONENTS) :].zero_()
    state.constrained_current = False
    engine.prepare_constraints_(state)
    state.completed_steps = checkpoint.completed_steps
    state.refresh_phase = (
        checkpoint.integrator_step_count,
        checkpoint.integrator_refresh_count,
    )
    if not _refresh_counters_are_consistent(
        completed_steps=state.completed_steps,
        refresh_interval=engine.plan.integrator.refresh_interval,
        refresh_step_count=state.refresh_phase[0],
        refresh_count=state.refresh_phase[1],
    ):
        raise ValueError("restored refresh counters are inconsistent")
    return state.completed_steps


class _ChannelCheckpointDestination:
    """View of an Engine State for the unchanged format-3 checkpoint reader."""

    def __init__(self, engine, state):
        self.engine = engine
        self.state = state
        self.fields = engine.field_view(state)
        self.solver = SimpleNamespace(
            integrator=SimpleNamespace(spectral_refresh_interval=engine.plan.integrator.refresh_interval)
        )

    def capture_pressure_guess(self):
        return self.state.solver["pressure_guess"]

    def restore_pressure_guess(self, value):
        target = self.fields["p.hat"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("pressure_guess must be a tensor")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError("pressure_guess does not match the Channel runtime")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("pressure_guess must be finite")
        self.state.solver = {
            "pressure_guess": value.detach().clone(),
            "pressure_guess_action": None,
            "pressure_guess_action_exponent": None,
        }

    def backend_restart_metadata(self):
        cache = self.engine.explicit.q_gradient_cache
        return {
            "kind": "channel_complete_stress_pressure_pcg",
            "state_keys": ["pressure_guess"],
            "reconstructed_state_keys": ["q_gradient_cache"] if cache is not None else [],
        }

    def restore_progress(
        self, *, completed_steps, spectral_refresh_interval, integrator_step_count, integrator_refresh_count
    ):
        expected = _refresh_counters_from_completed_steps(
            completed_steps, self.engine.plan.integrator.refresh_interval
        )
        if (
            spectral_refresh_interval != self.engine.plan.integrator.refresh_interval
            or (integrator_step_count, integrator_refresh_count) != expected
        ):
            raise ValueError("restored Channel refresh counters differ")
        self.state.completed_steps = completed_steps
        self.state.refresh_phase = expected
        self.state.constrained_current = True

    def restore_derived_state(self):
        return self.engine.constraint.term.rebuild_q_gradient_cache(self.fields)


def restore_channel_state(engine, state, checkpoint, *, run_spec):
    """Restore stored Q, velocity, pressure and guess without a new solve."""
    if not isinstance(engine, Engine) or engine.plan.geometry.kind != "rectangular_channel":
        raise TypeError("Channel restore requires a Channel Engine")
    if not isinstance(state, State):
        raise TypeError("Channel restore requires an Engine State")
    return load_v04_channel_checkpoint(
        checkpoint, _ChannelCheckpointDestination(engine, state), run_spec=run_spec
    )


def tensor_sha256(tensors):
    """Hash an ordered collection of contiguous CPU tensor byte strings."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _restore_plane_progress(state, plan_interval, completed_steps,
                            spectral_refresh_interval, step_count, refresh_count):
    """Restore the v0.4 refresh clock, retaining its defensive error text."""
    if spectral_refresh_interval != plan_interval:
        raise RuntimeError("restored spectral-refresh counters are inconsistent")
    expected = _refresh_counters_from_completed_steps(completed_steps, plan_interval)
    if expected != (step_count, refresh_count):
        raise RuntimeError("restored spectral-refresh counters are inconsistent")
    state.completed_steps = completed_steps
    state.refresh_phase = expected


def restore_plane_state(engine, state, checkpoint, *, run_spec, diagnostics=None):
    """Validate every legacy tensor before copying it into the engine state."""
    if not isinstance(engine, Engine):
        raise TypeError("engine must be an Engine")
    if not isinstance(state, State):
        raise TypeError("state must be a State")
    fields = engine.field_view(state)
    validate_plane_restore(
        checkpoint, run_spec,
        target_spatial={name: fields[name] for name in Q_COMPONENTS},
        target_spectral={name: fields[f"{name}.hat"] for name in Q_COMPONENTS},
        backend_restart=plane_backend_restart(run_spec),
    )
    targets: list[tuple[torch.Tensor, torch.Tensor, str]] = []
    for name in Q_COMPONENTS:
        for suffix, source in (
            ("", checkpoint.evolved_spatial[name]),
            (".hat", checkpoint.evolved_spectral[name]),
        ):
            target = fields[f"{name}{suffix}"]
            if target.shape != source.shape:
                raise ValueError(f"checkpoint shape for {name}{suffix} differs")
            if target.dtype != source.dtype:
                raise ValueError(f"checkpoint dtype for {name}{suffix} differs")
            targets.append((target, source, f"{name}{suffix}"))
    pending = tuple(
        (target, source.to(device=target.device))
        for target, source, _description in targets
    )
    for target, source in pending:
        target.copy_(source)
    state.constrained_current = False
    engine.prepare_constraints_(state, diagnostics=diagnostics)
    _restore_plane_progress(
        state, engine.plan.integrator.refresh_interval,
        checkpoint.completed_steps, checkpoint.spectral_refresh_interval,
        checkpoint.integrator_step_count, checkpoint.integrator_refresh_count,
    )
    if state.completed_steps != checkpoint.completed_steps:
        raise RuntimeError("restored completed-step count is inconsistent")
    return checkpoint.completed_steps
