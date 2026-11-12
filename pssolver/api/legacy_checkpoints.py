"""Read v0.4 checkpoint compatibility data only on an explicit restart."""

from dataclasses import asdict


def _legacy_refresh(time):
    refresh = time.refresh
    if refresh is None:
        return {"mode": "disabled"}
    if "steps" in refresh:
        return {"mode": "steps", "interval_steps": refresh["steps"]}
    return {"mode": "physical_time", "interval_time": refresh["physical_time"]}


def _legacy_options(source):
    options = {
        "tf32": "off",
        "molecular_field_linear_space": "spectral",
        "stress_divergence_sum_space": "spectral",
        "pointwise_execution": "eager",
        "disable_q_gradient_reuse": False,
    }
    options.update(source.execution.options)
    return options


def downgrade_simulation(source, *, runtime_path=None):
    """Recreate the v0.4 declaration used solely for compatibility identity."""
    from pssolver.functional import legacy_declarations as old
    from pssolver.functional.legacy_declarations.plane_beris_edwards import (
        create_plane_beris_edwards_run_spec,
    )

    from .declarations import GeneratedInitialCondition
    from .models import ShendrukPlaneBerisEdwards

    kind = source.geometry.name
    if runtime_path is None:
        runtime_path = {
            "periodic_box": "periodic_spectral",
            "plane_slab": "legacy_production",
            "rectangular_channel": "channel_complete_stress",
        }[kind]
    options = _legacy_options(source)
    output = source.output
    numeric = source.numerics
    time = source.time
    initial = source.initial_condition
    refresh = _legacy_refresh(time)
    if isinstance(source.model, ShendrukPlaneBerisEdwards):
        nx, ny, nz = source.geometry.domain.shape
        lx, ly, height = source.geometry.domain.lengths
        return create_plane_beris_edwards_run_spec(
            **asdict(source.model),
            **options,
            **initial.parameters,
            output_dir=output.directory,
            nx=nx,
            ny=ny,
            nz=nz,
            lx=lx,
            ly=ly,
            height=height,
            dt=time.dt,
            steps=output.steps,
            save_interval=output.save_interval,
            save_start_step=output.save_start_step,
            diagnostics=output.diagnostics,
            diagnostic_interval=output.diagnostic_interval,
            save_hydrodynamics=output.save_hydrodynamics,
            checkpoint_interval=output.checkpoint_interval,
            restart_from=output.restart_from,
            dry_run=output.dry_run,
            runtime_path=runtime_path,
            dtype=numeric.dtype,
            validation_config_sha256=output.validation_config_sha256,
            device=source.execution.device,
            dealias_rule=numeric.dealias_rule.value,
            projected_transform_execution=numeric.projected_transform_execution.value,
            spectral_storage=numeric.spectral_storage.value,
            transform_execution_order=numeric.transform_execution_order.value,
            disable_spectral_refresh=time.refresh is None,
            spectral_refresh_time=refresh.get("interval_time"),
            spectral_refresh_steps=refresh.get("interval_steps"),
        )
    model = old.CompleteStressBerisEdwards(**asdict(source.model))
    policies = source.boundaries
    if not policies:
        policies = {
            "Q": old.neumann_q(),
            "velocity": old.free_slip_velocity(),
            "pressure": old.neumann_pressure_compatibility(),
        }
    old_initial = (
        old.GeneratedInitialCondition(initial.family, parameters=dict(initial.parameters))
        if isinstance(initial, GeneratedInitialCondition)
        else old.SnapshotInitialCondition(initial.directory, step=initial.step, mode=initial.mode)
    )
    return old.Simulation(
        model=model,
        geometry=source.geometry,
        boundaries=old.assign_boundaries(model=model, geometry=source.geometry, policies=policies),
        numerics=old.SpectralNumerics(
            dtype=numeric.dtype,
            spectral_storage=numeric.spectral_storage.value,
            dealias_rule=numeric.dealias_rule.value,
            hermitian_axis=numeric.hermitian_axis,
            projected_transform_execution=numeric.projected_transform_execution.value,
            transform_execution_order=numeric.transform_execution_order.value,
        ),
        time=old.TimeStepping(dt=time.dt, refresh=refresh),
        initial_condition=old_initial,
        execution=old.TorchSpectralExecution(
            runtime_path=runtime_path,
            device=source.execution.device,
            options=options,
        ),
        output=old.Output(
            directory=output.directory,
            steps=output.steps,
            save_interval=output.save_interval,
            save_start_step=output.save_start_step,
            diagnostics=output.diagnostics,
            diagnostic_interval=output.diagnostic_interval,
            save_hydrodynamics=output.save_hydrodynamics,
            checkpoint_interval=output.checkpoint_interval,
            restart_from=output.restart_from,
        ),
        discretization=(
            {}
            if numeric.pressure_solver is None
            else {"pressure_solver": dict(numeric.pressure_solver)}
        ),
        invocation=old.InvocationSpec(
            options={
                "dry_run": output.dry_run,
                **(
                    {}
                    if output.validation_config_sha256 is None
                    else {"validation_config_sha256": output.validation_config_sha256}
                ),
            }
        ),
    ).specification


def _plane_preflight(header, run_spec):
    from pssolver.errors import SimulationRestartError
    from pssolver.functional.legacy_declarations.legacy_plane import (
        PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
        plane_run_spec_dynamics_identity,
    )

    def reject(message, reason):
        raise SimulationRestartError(
            message,
            application_family="plane_complete_stress_beris_edwards",
            stage="restart_validation",
            step=header.completed_steps,
            details={"reason": reason},
        )

    if header.format_version == PLANE_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
        reject(
            "legacy Plane checkpoint requires explicit identity upgrade",
            "legacy_identity_upgrade_required",
        )
    identity = header.compatibility_identity
    if identity is None:
        reject("checkpoint compatibility identity is missing", "compatibility_identity_missing")
    payload = identity.forward_dynamics.to_metadata()["payload"]
    if payload.get("run_spec_dynamics") != plane_run_spec_dynamics_identity(run_spec).to_metadata():
        reject("checkpoint forward dynamics does not match target", "forward_dynamics_mismatch")
    lifting = payload.get("lifting_dynamics")
    if isinstance(lifting, dict) and isinstance(lifting.get("lifting"), dict):
        if lifting["lifting"].get("plan_sha256") is not None:
            reject(
                "checkpoint static lifting dynamics does not match target",
                "static_lifting_dynamics_mismatch",
            )


def continue_from_v04_checkpoint(compiled, *, dtype):
    """Return header-authenticated progress and a deferred frozen restore."""
    from pssolver.functional.legacy_declarations.legacy_channel import (
        read_channel_beris_edwards_checkpoint_progress,
    )
    from pssolver.functional.legacy_declarations.legacy_periodic import preflight_periodic_restart
    from pssolver.functional.legacy_declarations.legacy_plane import (
        load_plane_checkpoint,
        read_plane_checkpoint_header,
    )
    from pssolver.functional.legacy_declarations.legacy_restore import (
        restore_channel_state,
        restore_periodic_state,
        restore_plane_state,
    )
    from pssolver.functional.legacy_declarations.plane_beris_edwards import PlaneBerisEdwardsRunSpec
    from pssolver.functional.legacy_declarations.public_simulation_runner import (
        compile_public_simulation,
    )
    from pssolver.io.legacy.periodic_workflow import load_periodic_checkpoint

    source = compiled.source
    path = source.output.restart_from
    kind = compiled.plan.geometry.kind
    header = read_plane_checkpoint_header(path) if kind == "plane_slab" else None
    old = downgrade_simulation(source, runtime_path=None if header is None else header.runtime_path)
    run_spec = (
        old
        if isinstance(old, PlaneBerisEdwardsRunSpec)
        else compile_public_simulation(old).run_spec
    )
    if kind == "periodic_box":
        _, header = preflight_periodic_restart(
            path, run_spec=run_spec, plan=compiled.plan, dtype=dtype
        )

        def restore(engine, state, _diagnostics):
            step = restore_periodic_state(
                engine,
                state,
                load_periodic_checkpoint(path),
                run_spec=run_spec,
            )
            return step, None

        return header.completed_steps, restore
    if kind == "rectangular_channel":
        start = read_channel_beris_edwards_checkpoint_progress(path)

        def restore(engine, state, _diagnostics):
            return restore_channel_state(engine, state, path, run_spec=run_spec), None

        return start, restore
    _plane_preflight(header, run_spec)

    def restore(engine, state, diagnostics):
        checkpoint = load_plane_checkpoint(path)
        step = restore_plane_state(
            engine,
            state,
            checkpoint,
            run_spec=run_spec,
            diagnostics=diagnostics,
        )
        return step, checkpoint.runtime_identity_sha256

    return header.completed_steps, restore
