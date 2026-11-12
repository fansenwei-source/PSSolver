"""Test-only v0.4 carrier capture; production writes engine_state exclusively."""

from dataclasses import dataclass

from pssolver.engine.engine import Engine
from pssolver.engine.state import State
from pssolver.functional.legacy_declarations.legacy_plane import (
    PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PlaneBerisEdwardsRunSpec,
    PlaneWorkflowCheckpoint,
    build_plane_checkpoint_compatibility_identity,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.runtime.state_validation import require_finite_tensors


@dataclass
class PlaneTestState:
    """Test ownership only: the engine and state are never a production adapter."""

    engine: object
    state: object

    @property
    def fields(self):
        return self.engine.field_view(self.state)

    @property
    def completed_steps(self):
        return self.state.completed_steps


def capture_plane_checkpoint(production, *, run_spec):
    from pssolver.functional.legacy_declarations.legacy_plane import plane_backend_restart

    return plane_v04_checkpoint_from_state(
        production.engine, production.state, run_spec,
        backend_restart=plane_backend_restart(run_spec),
    )


def restore_plane_checkpoint(production, checkpoint, *, run_spec):
    from pssolver.functional.legacy_declarations.legacy_restore import restore_plane_state

    return restore_plane_state(production.engine, production.state, checkpoint, run_spec=run_spec)


def plane_v04_checkpoint_from_state(
    engine: Engine,
    state: State,
    run_spec: PlaneBerisEdwardsRunSpec,
    *,
    backend_restart,
) -> PlaneWorkflowCheckpoint:
    """Synchronize algebraic fields and capture layered format-v2 state."""

    if not isinstance(engine, Engine):
        raise TypeError("engine must be an Engine")
    if not isinstance(state, State):
        raise TypeError("state must be a State")
    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PlaneBerisEdwardsRunSpec")
    runtime_identity_sha256 = run_spec.runtime_identity_sha256()
    engine.prepare_constraints_(state)
    fields = engine.field_view(state)
    spatial = {name: fields[name] for name in Q_COMPONENTS}
    spectral = {name: fields[f"{name}.hat"] for name in Q_COMPONENTS}
    require_finite_tensors(
        (
            *((name, value) for name, value in spatial.items()),
            *((f"{name}.hat", value) for name, value in spectral.items()),
        ),
        application_family="plane_complete_stress",
        stage="checkpoint",
        step=state.completed_steps,
    )
    backend = backend_restart
    lifting = None
    compatibility_identity = build_plane_checkpoint_compatibility_identity(
        run_spec,
        runtime_path=run_spec.runtime_path,
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        backend_restart=backend,
        lifting_restart=lifting,
    )
    fresh_construction = None
    return PlaneWorkflowCheckpoint(
        format_version=PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        runtime_path=run_spec.runtime_path,
        runtime_identity_sha256=runtime_identity_sha256,
        completed_steps=state.completed_steps,
        spectral_refresh_interval=engine.plan.integrator.refresh_interval,
        integrator_step_count=int(state.refresh_phase[0]),
        integrator_refresh_count=int(state.refresh_phase[1]),
        evolved_spatial=spatial,
        evolved_spectral=spectral,
        backend_restart=backend,
        lifting_restart=lifting,
        compatibility_identity=compatibility_identity,
        fresh_construction_provenance=fresh_construction,
    )
