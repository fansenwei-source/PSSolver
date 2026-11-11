"""One operation order for pure and in-place Periodic execution."""

from dataclasses import replace

import torch

from .groups import select_group, store_group
from .progress import _advance_refresh_counters, _refresh_due_after_next_step


def _evolved_transform(engine, value, operation, *, inplace=False):
    """Execute each evolved transform group in the plan's declared order."""
    groups = [group for group in engine.transform_groups if group[0][0] < engine.evolved_count]
    method = getattr(engine.projector, operation)
    if len(groups) == 1:
        result = method(value, groups[0][1])
        return value if inplace else result
    result = value if inplace else None
    for indices, boundaries, _indexing in groups:
        selected = select_group(value, indices, engine.component_indices,
                                len(engine.components), "contiguous_slice")
        transformed = method(selected, boundaries)
        if inplace:
            transformed = selected
        if result is None:
            result = torch.empty((value.shape[0], *transformed.shape[1:]),
                                 dtype=transformed.dtype, device=transformed.device)
        store_group(result, indices, transformed, engine.component_indices,
                    len(engine.components), "contiguous_slice")
    return result


def _constraint_physical(engine, hats):
    if engine.plan.geometry.kind == "periodic_box":
        return engine.projector.inverse_transform(hats, engine.boundaries)
    physical = None
    inverse = (
        engine.projector.inverse_transform_full
        if engine.plan.geometry.kind == "rectangular_channel"
        else engine.projector.inverse_transform
    )
    for indices, bc, _indexing in engine.transform_groups:
        if indices[0] < engine.evolved_count:
            continue
        relative = [index - engine.evolved_count for index in indices]
        value = inverse(
            select_group(hats, relative, engine.component_indices,
                         len(engine.components), "contiguous_slice"), bc
        )
        if physical is None:
            physical = torch.empty((hats.shape[0], engine.batch, *engine.shape),
                                   dtype=value.dtype, device=value.device)
        store_group(physical, relative, value, engine.component_indices,
                    len(engine.components), "contiguous_slice")
    return physical


def prepare(engine, state, view, controls, *, out=None, diagnostics=None):
    if state.constrained_current:
        return state
    if engine.constraint is None:
        if out is None:
            return replace(state, constrained_current=True)
        # Reuse the one reviewed State publication below; no tensor work.
        return _mark_constraints_current(out)
    channel = engine.plan.geometry.kind == "rectangular_channel"
    if channel:
        channel_report = {} if diagnostics is None else diagnostics
        hats, solver = engine._constraint_hats(
            view, controls, solver_state=state.solver, diagnostics=channel_report
        )
    else:
        hats = engine._constraint_hats(view, controls, diagnostics=diagnostics)
        solver = state.solver
    evolved = engine.evolved_count
    if out is None:
        q_physical, q_spectral = (
            state.pure_evolved
            if state.pure_evolved is not None
            else (state.spatial[:evolved], state.spectral[:evolved])
        )
        if isinstance(hats, tuple):
            # Keep the functional velocity and pressure on independent graph
            # paths, as in the legacy functional step.
            velocity_hat, pressure_hat = hats
            inverse = engine.projector.inverse_transform_full
            velocity = inverse(velocity_hat, engine.component_boundaries[evolved])
            pressure = inverse(pressure_hat, engine.component_boundaries[-1])
            physical = torch.cat((velocity, pressure.unsqueeze(0)), dim=0)
            hats = torch.cat((velocity_hat, pressure_hat.unsqueeze(0)), dim=0)
        else:
            physical = _constraint_physical(engine, hats)
        spatial = torch.cat((q_physical, physical), dim=0)
        spectral = torch.cat((q_spectral, hats), dim=0)
        result = replace(
            state,
            spectral=spectral,
            spatial=spatial,
            pure_evolved=(q_physical, q_spectral),
            constraint_physical=physical,
            solver=solver,
            constrained_current=True,
        )
    else:
        spectral = out.spectral
        spectral[evolved:].copy_(hats)
        if engine.plan.geometry.kind == "periodic_box":
            physical = engine.projector.inverse_transform(spectral[evolved:], engine.boundaries)
            out.spatial[evolved:].copy_(physical)
        else:
            inverse = (
                engine.projector.inverse_transform_full if channel
                else engine.projector.inverse_transform
            )
            for indices, bc, _indexing in engine.transform_groups:
                if indices[0] < evolved:
                    continue
                physical = inverse(
                    select_group(spectral, indices, engine.component_indices,
                                 len(engine.components), "contiguous_slice"), bc
                )
                store_group(out.spatial, indices, physical, engine.component_indices,
                            len(engine.components), "contiguous_slice")
        _mark_constraints_current(out)
        if channel:
            out.solver = solver
        result = out
    view._bind(result.spatial, result.spectral)
    engine.constraint.term.after_static_fields_updated(view)
    return result


def _mark_constraints_current(out):
    out.constrained_current = True
    return out


def advance(engine, state, view, controls, *, out=None, diagnostics=None):
    current = prepare(engine, state, view, controls, out=out, diagnostics=diagnostics)
    explicit = engine._explicit_at(current.completed_steps)
    rhs = explicit(view, controls)
    evolved = engine.evolved_count
    source = (
        current.pure_evolved[1]
        if out is None and current.pure_evolved is not None
        else current.spectral[:evolved]
    )
    target = source if out is not None else None
    spectral = engine.scheme.explicit_update(
        source,
        rhs,
        current.history,
        engine.dt,
        out=target,
    )
    if out is None:
        spectral = spectral / engine.linear.denominator
        spectral = _evolved_transform(engine, spectral, "project_real_spectrum")
    else:
        spectral.div_(engine.linear.denominator)
        _evolved_transform(engine, spectral, "project_real_spectrum_", inplace=True)
    physical = _evolved_transform(engine, spectral, "inverse_transform")
    interval = engine.plan.integrator.refresh_interval
    refreshed = _refresh_due_after_next_step(
        refresh_interval=interval, refresh_step_count=current.refresh_phase[0]
    )
    if refreshed:
        refreshed_spectral = _evolved_transform(engine, physical, "forward_transform")
        if out is None:
            spectral = refreshed_spectral
        else:
            spectral.copy_(refreshed_spectral)
        physical = _evolved_transform(engine, spectral, "inverse_transform")
    steps, refresh_steps, refresh_count = _advance_refresh_counters(
        completed_steps=current.completed_steps,
        refresh_interval=interval,
        refresh_step_count=current.refresh_phase[0],
        refresh_count=current.refresh_phase[1],
        refreshed=refreshed,
    )
    if out is None:
        return replace(
            current,
            spectral=torch.cat((spectral, current.spectral[evolved:])),
            spatial=torch.cat((physical, current.spatial[evolved:])),
            pure_evolved=(physical, spectral),
            completed_steps=steps,
            refresh_phase=(refresh_steps, refresh_count),
            constrained_current=False,
        )
    out.spatial[:evolved].copy_(physical)
    out.completed_steps = steps
    out.refresh_phase = (refresh_steps, refresh_count)
    out.constrained_current = False
    return out
