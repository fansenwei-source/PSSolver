"""Temporary v0.4 declaration entrance to Engine and Runner; retired at S-28."""

from __future__ import annotations

import json
from collections.abc import Iterable

import torch

from pssolver.configuration.execution_device import (
    configure_tf32_execution,
    execution_device_normalization,
    resolve_execution_device,
    scoped_tf32_execution,
)
from pssolver.configuration.legacy_periodic import (
    PeriodicBerisEdwardsRunSpec,
    _reject_restart,
    load_periodic_initial_q,
    preflight_periodic_restart,
    translate_periodic,
    validate_periodic_restore,
)
from pssolver.engine.engine import Engine
from pssolver.engine.progress import _refresh_counters_are_consistent
from pssolver.errors import SimulationRestartError
from pssolver.io import engine_state
from pssolver.io.checkpoint import CheckpointError
from pssolver.io.legacy.periodic_workflow import load_periodic_checkpoint
from pssolver.models.active_nematics.fields import Q_COMPONENTS

from .observations import PeriodicCapture
from .runner import Runner, RunSchedule


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


def _run_v04_translation_owned(
    run_spec: PeriodicBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
    tf32_metadata: dict[str, object],
    public_compilation_metadata: dict[str, object] | None = None,
):
    """Execute one frozen P8.2 application with no runtime fallback."""

    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be PeriodicBerisEdwardsRunSpec")
    spec = run_spec.simulation
    stokes = spec.equation_system.parameters["stokes"]["parameters"]
    uniform_mode_action = (
        "remove_all_uniform_velocity_and_force"
        if stokes["tangential_zero_mode_policy"] == "zero_mean"
        else "retain_all_uniform_velocity_resolved_by_friction"
    )
    metadata = {
        "schema_version": 1,
        "application": "periodic_complete_stress_beris_edwards",
        "configuration": spec.to_metadata(),
        "configuration_sha256": spec.canonical_sha256(),
        "runtime_identity_sha256": run_spec.runtime_identity_sha256(),
        "runtime_selection": {
            "requested": run_spec.runtime_path,
            "effective": run_spec.runtime_path,
            "fallback_allowed": False,
            "fallback_used": False,
        },
        "normalization": execution_device_normalization(spec.execution.options["device"]),
        "pressure_gauge": "zero_mean",
        "tangential_zero_mode_policy": stokes["tangential_zero_mode_policy"],
        "friction": stokes["friction"],
        "uniform_velocity_mode_action": uniform_mode_action,
    }
    if public_compilation_metadata is not None:
        metadata["public_compilation"] = dict(public_compilation_metadata)
    if emit_metadata:
        print(json.dumps(metadata, indent=2, sort_keys=True))
    if run_spec.dry_run:
        return None

    device = resolve_execution_device(spec.execution.options["device"])
    real_dtype = getattr(torch, spec.numerics.precision.value)
    metadata["precision"] = {
        "real_dtype": spec.numerics.precision.value,
        "spectral_dtype": ("complex64" if real_dtype == torch.float32 else "complex128"),
        **tf32_metadata,
    }
    translation = translate_periodic(spec, flavour="production", load_initial=False)
    restart = spec.workflow.options["restart_from"]
    restart_kind = None
    start = 0
    if restart is not None:
        try:
            restart_kind, header = preflight_periodic_restart(
                restart, run_spec=run_spec, plan=translation.plan, dtype=real_dtype
            )
            start = (
                header["completed_steps"]
                if restart_kind == "engine_state"
                else header.completed_steps
            )
        except CheckpointError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            _reject_restart(
                str(exc) or "periodic checkpoint preflight failed",
                reason="checkpoint_preflight_failed",
                details={"legacy_exception_type": type(exc).__name__},
                exception=exc,
            )
        if spec.workflow.options["save_start_step"] > start + spec.workflow.steps:
            _reject_restart(
                "save_start_step must not exceed the absolute final step",
                reason="save_start_step_after_final_step",
                step=start,
                details={
                    "save_start_step": spec.workflow.options["save_start_step"],
                    "absolute_final_step": start + spec.workflow.steps,
                },
            )
    initial_values, snapshot_path, snapshot_sha256 = load_periodic_initial_q(run_spec)
    metadata["initial_condition"] = {
        "source": str(snapshot_path),
        "sha256": snapshot_sha256,
        "logical_mode": spec.initial_condition.parameters["mode"],
    }
    output = run_spec.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty periodic output: {output}")
    engine = Engine(translation.plan, device=device, dtype=real_dtype)
    state = engine.initial_state(
        {name: value.to(device=device) for name, value in initial_values.items()}
    )
    metadata["runtime_construction"] = {
        "plan_identity_sha256": engine.plan.identity_sha256,
        "engine": "pssolver.engine.engine.Engine",
        "operation_order": [
            "prepare_constraints",
            "explicit",
            "euler",
            "project",
            "inverse",
        ],
    }
    metadata["runtime"] = {
        "requested": run_spec.runtime_path,
        "effective": run_spec.runtime_path,
        "adapter": "pssolver.engine.engine.Engine",
        "fallback_used": False,
        "pressure_gauge": "zero_mean",
        "hermitian_state_projection": engine.projector.execution_metadata()[
            "hermitian_state_projection"
        ],
        "tangential_zero_mode_policy": stokes["tangential_zero_mode_policy"],
        "friction": float(stokes["friction"]),
        "uniform_velocity_mode_action": uniform_mode_action,
    }
    if restart is not None:
        try:
            if restart_kind == "engine_state":
                state = engine_state.read(
                    restart,
                    plan_identity_sha256=engine.plan.identity_sha256,
                    dtype=real_dtype,
                    batch=1,
                    device=device,
                )
            else:
                restore_periodic_state(
                    engine, state, load_periodic_checkpoint(restart), run_spec=run_spec
                )
        except (SimulationRestartError, CheckpointError):
            raise
        except Exception as exc:
            raise SimulationRestartError(
                str(exc) or "periodic checkpoint restore failed",
                application_family="periodic_complete_stress_beris_edwards",
                stage="restart_restore",
                step=state.completed_steps,
                details={
                    "reason": "checkpoint_restore_failed",
                    "legacy_exception_type": type(exc).__name__,
                },
            ) from exc
    options = spec.workflow.options
    workflow = Runner(
        engine,
        schedule=RunSchedule(
            steps=spec.workflow.steps,
            save_interval=options["save_interval"],
            save_start_step=options["save_start_step"],
            diagnostics=options["diagnostics"],
            diagnostic_interval=options["diagnostic_interval"],
            checkpoint_interval=options["checkpoint_interval"],
        ),
        output_directory=output,
        capture=PeriodicCapture(hydrodynamics=options["save_hydrodynamics"]),
        metadata=metadata,
    )
    return workflow.run(state, start_step=start, progress=progress)


def run_v04_translation(
    run_spec: PeriodicBerisEdwardsRunSpec,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
    public_compilation_metadata: dict[str, object] | None = None,
):
    """Execute one periodic run inside a dynamically scoped TF32 policy."""

    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be PeriodicBerisEdwardsRunSpec")
    spec = run_spec.simulation
    real_dtype = getattr(torch, spec.numerics.precision.value)
    requested = spec.execution.options["tf32"]
    if run_spec.dry_run:
        return _run_v04_translation_owned(
            run_spec,
            progress=progress,
            emit_metadata=emit_metadata,
            tf32_metadata=configure_tf32_execution(
                requested,
                real_dtype=real_dtype,
                device=spec.execution.options["device"],
            ),
            public_compilation_metadata=public_compilation_metadata,
        )
    device = resolve_execution_device(spec.execution.options["device"])
    with scoped_tf32_execution(
        requested,
        real_dtype=real_dtype,
        device=device,
    ) as tf32_metadata:
        return _run_v04_translation_owned(
            run_spec,
            progress=progress,
            emit_metadata=emit_metadata,
            tf32_metadata=tf32_metadata,
            public_compilation_metadata=public_compilation_metadata,
        )
