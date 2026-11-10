"""Test-only v0.4 checkpoint capture; not a production compatibility adapter."""

from pssolver.configuration.legacy_periodic import (
    PeriodicBerisEdwardsRunSpec,
    build_periodic_checkpoint_compatibility_identity,
)
from pssolver.io.legacy.periodic_workflow import (
    PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PeriodicWorkflowCheckpoint,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.runtime.state_validation import require_finite_tensors


def production_handle(simulation):
    """Construct explicit test state without retaining a production adapter."""
    import torch

    from pssolver.configuration.legacy_periodic import translate_periodic
    from pssolver.engine.engine import Engine
    from pssolver.functional.validation import PeriodicProductionHandle

    translation = translate_periodic(
        simulation.specification, flavour="production", load_initial=True
    )
    engine = Engine(
        translation.plan,
        device="cpu",
        dtype=getattr(torch, simulation.specification.numerics.precision.value),
    )
    return translation.run_spec, PeriodicProductionHandle(
        engine, engine.initial_state(translation.initial_values)
    )


def capture_periodic_checkpoint(production, *, run_spec):
    return periodic_v04_checkpoint_from_state(
        production.engine,
        production.state,
        run_spec,
        backend_restart={"kind": "periodic_stokes_stateless", "state_keys": []},
    )


def restore_periodic_checkpoint(production, checkpoint, *, run_spec):
    from pssolver.runtime.legacy_entry import restore_periodic_state

    return restore_periodic_state(
        production.engine, production.state, checkpoint, run_spec=run_spec
    )


def periodic_v04_checkpoint_from_state(engine, state, run_spec, *, backend_restart):
    """Capture the frozen v0.4 carrier from an explicit, caller-owned state."""

    if not isinstance(run_spec, PeriodicBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PeriodicBerisEdwardsRunSpec")
    engine.prepare_constraints_(state)
    fields = engine.field_view(state)
    spatial = {name: fields[name] for name in Q_COMPONENTS}
    spectral = {name: fields[f"{name}.hat"] for name in Q_COMPONENTS}
    require_finite_tensors(
        (
            *((name, value) for name, value in spatial.items()),
            *((f"{name}.hat", value) for name, value in spectral.items()),
        ),
        application_family="periodic_complete_stress",
        stage="checkpoint",
        step=state.completed_steps,
    )
    backend = backend_restart
    return PeriodicWorkflowCheckpoint(
        format_version=PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_identity_sha256=run_spec.runtime_identity_sha256(),
        completed_steps=state.completed_steps,
        spectral_refresh_interval=engine.plan.integrator.refresh_interval,
        integrator_step_count=int(state.refresh_phase[0]),
        integrator_refresh_count=int(state.refresh_phase[1]),
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        backend_restart=backend,
        compatibility_identity=build_periodic_checkpoint_compatibility_identity(
            run_spec,
            evolved_spatial=spatial,
            evolved_spectral=spectral,
            backend_restart=backend,
        ),
    )
