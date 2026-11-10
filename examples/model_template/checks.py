"""Reusable CPU checks for component models; no relaxation of model contracts."""

import hashlib
import math

import numpy as np
import sympy as sp
import torch


def manufactured_source(solution, *, coordinates, time, diffusion, reaction):
    """Derive a trusted symbolic forcing and record its exact expression identity."""
    laplacian = sum(sp.diff(solution, axis, 2) for axis in coordinates)
    source = sp.diff(solution, time) - diffusion * laplacian - reaction(solution)
    expression = sp.srepr(source)
    return source, {
        "sympy_version": sp.__version__,
        "expression": expression,
        "sha256": hashlib.sha256(expression.encode()).hexdigest(),
    }


def fitted_order(steps, errors):
    """Fit log(error) against log(step); reject invalid samples rather than drop them."""
    if not all(math.isfinite(value) and value > 0 for value in (*steps, *errors)):
        raise ValueError("convergence samples must be finite and positive")
    return float(np.polyfit(np.log(steps), np.log(errors), 1)[0])


def spatial_mms(grids, measure, *, ratio, bound):
    """Run every registered grid and enforce monotonic, ratio and final-error gates."""
    errors = [float(measure(size)) for size in grids]
    assert all(math.isfinite(value) and value > 0 for value in errors)
    assert all(a > b for a, b in zip(errors, errors[1:], strict=False))
    assert errors[-2] / errors[-1] >= ratio
    assert errors[-1] <= bound
    return errors


def energy_monotone(energies):
    """Check each step against the fixed relative roundoff allowance."""
    assert all(math.isfinite(value) for value in energies)
    assert all(b <= a + 1e-12 * abs(a) for a, b in zip(energies, energies[1:], strict=False))


def raw_equal(left, right):
    """Compare exact tensor bytes, including signed zeros and complex components."""
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.detach().contiguous().numpy().tobytes()
        == right.detach().contiguous().numpy().tobytes()
    )


def restart_check(engine, initial, path):
    """Compare ten continuous steps with a five/checkpoint/five split."""
    from pssolver.io import engine_state

    continuous = engine.initial_state(initial)
    split = engine.initial_state(initial)
    for _ in range(10):
        engine.step_(continuous)
    for _ in range(5):
        engine.step_(split)
    engine_state.write(
        path,
        split,
        plan_identity_sha256=engine.plan.identity_sha256,
        plan_metadata=engine.plan.to_metadata(),
        run_provenance={},
        refresh_interval=None,
    )
    restored = engine_state.read(
        path,
        plan_identity_sha256=engine.plan.identity_sha256,
        dtype=engine.dtype,
        device=engine.device,
        batch=engine.batch,
    )
    for _ in range(5):
        engine.step_(restored)
    assert raw_equal(continuous.spatial, restored.spatial)
    assert raw_equal(continuous.spectral, restored.spectral)
    assert continuous.completed_steps == restored.completed_steps
    assert continuous.refresh_phase == restored.refresh_phase
    return continuous, restored


def taylor_check(objective, value, direction, steps):
    """Return the first-order Taylor remainder slope with every sample retained."""
    value = value.detach().clone().requires_grad_()
    result = objective(value)
    (gradient,) = torch.autograd.grad(result, value)
    derivative = (gradient * direction).sum()
    errors = [
        abs(float((objective(value + h * direction) - result - h * derivative).detach()))
        for h in steps
    ]
    return fitted_order(steps, errors), errors
