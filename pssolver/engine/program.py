"""One operation order for pure and in-place Periodic execution."""

from dataclasses import replace

import torch

from .progress import _advance_refresh_counters


def prepare(engine, state, view, alpha, *, out=None, diagnostics=None):
    if state.constrained_current:
        return state
    hats = engine.constraint.constrained_hats(view, alpha, diagnostics=diagnostics)
    evolved = engine.evolved_count
    if out is None:
        q_physical, q_spectral = (
            state.pure_evolved
            if state.pure_evolved is not None
            else (state.spatial[:evolved], state.spectral[:evolved])
        )
        physical = engine.projector.inverse_transform(hats, engine.boundaries)
        spatial = torch.cat((q_physical, physical), dim=0)
        spectral = torch.cat((q_spectral, hats), dim=0)
        result = replace(
            state,
            spectral=spectral,
            spatial=spatial,
            pure_evolved=(q_physical, q_spectral),
            constraint_physical=physical,
            constrained_current=True,
        )
    else:
        spectral = out.spectral
        spectral[evolved:].copy_(hats)
        physical = engine.projector.inverse_transform(spectral[evolved:], engine.boundaries)
        out.spatial[evolved:].copy_(physical)
        out.constrained_current = True
        result = out
    view._bind(result.spatial, result.spectral)
    engine.constraint.term.after_static_fields_updated(view)
    return result


def advance(engine, state, view, alpha, *, out=None, diagnostics=None):
    current = prepare(engine, state, view, alpha, out=out, diagnostics=diagnostics)
    rhs = engine.explicit(view, {})
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
        spectral = engine.projector.project_real_spectrum(spectral, engine.boundaries)
    else:
        spectral.div_(engine.linear.denominator)
        engine.projector.project_real_spectrum_(spectral, engine.boundaries)
    physical = engine.projector.inverse_transform(spectral, engine.boundaries)
    steps, refresh_steps, refresh_count = _advance_refresh_counters(
        completed_steps=current.completed_steps,
        refresh_interval=None,
        refresh_step_count=current.refresh_phase[0],
        refresh_count=current.refresh_phase[1],
        refreshed=False,
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
