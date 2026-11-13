"""Numerical proof of the private tiers and history through the shared program."""

import hashlib
import json
from dataclasses import fields
from math import isfinite, log2, pi

import numpy as np
import pytest
import scipy
import torch

from examples.model_template.checks import raw_equal, taylor_check
from pssolver.core.integrators import IntegratorScheme
from pssolver.engine import program
from pssolver.engine.linear import DiagonalOperator, ModalBlockOperator
from pssolver.integrators.reference_qualification import measure_temporal_convergence
from pssolver.integrators.scalar_reference import ScalarPeriodicReactionDiffusion
from pssolver.io import engine_state
from pssolver.io.checkpoint import CheckpointCompatibilityError
from qualification.identity.scenarios import scenarios
from qualification.identity.scenes_v05 import simulation_v05
from tests.oracles.s29_diffusion import coefficients, coupled_exact, variable_diffusion_expressions
from tests.s29_reference import VariableDiffusion, block_engine, scalar_engine, variable_engine
from tests.test_phase4_modal_block_operator import _bound, _random_rhs


DT = (.04, .02, .01, .005)
FINAL = .4
DTYPES = (torch.float64, torch.float32)


def report(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def record(value):
    if isinstance(value, torch.Tensor):
        return [str(value.dtype), list(value.shape), hashlib.sha256(
            value.detach().contiguous().numpy().tobytes(),
        ).hexdigest()]
    if isinstance(value, dict):
        return {key: record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [record(item) for item in value]
    return value


def state_record(state):
    return {field.name: record(getattr(state, field.name)) for field in fields(state)}


def steps(engine, state, count, *, inplace=False, observations=None):
    for _ in range(count):
        row = {} if observations is not None else None
        if inplace:
            engine.step_(state, diagnostics=row)
        else:
            state = engine.step(state, diagnostics=row)
        if observations is not None:
            observations.append(row)
    return state


def order(errors, threshold):
    assert all(isfinite(e) and e > 0 for e in errors)
    assert all(a > b for a, b in zip(errors[:-1], errors[1:], strict=True))
    result = [log2(a / b) for a, b in zip(errors[:-1], errors[1:], strict=True)]
    assert min(result) >= threshold
    return result


def scalar_values(model, dtype):
    value = model.initial_physical(dtype=dtype, device=torch.device("cpu"))
    return value[:, None, None].expand(32, 8, 8).contiguous()


def write_state(directory, engine, state):
    return engine_state.write(
        directory, state, plan_identity_sha256=engine.plan.identity_sha256,
        plan_metadata=engine.plan.to_metadata(), refresh_interval=None, run_provenance={},
    )


def read_state(directory, engine, dt=None):
    return engine_state.read(
        directory, plan_identity_sha256=engine.plan.identity_sha256,
        dtype=engine.dtype, batch=1, device="cpu",
        _reference=("sbdf2_test", engine.dt if dt is None else dt),
    )


def test_modal_two_matches_p44_reference(tmp_path):
    reference = _bound(implementation="torch_linalg_reference")
    rhs = _random_rhs(4)
    expected = reference.solve(rhs, alpha=3.75)
    actual = ModalBlockOperator(reference.coefficients).solve(
        rhs.movedim(-1, 0), shift=3.75, dt=1, out=None, diagnostics=None,
    ).movedim(0, -1)
    assert torch.allclose(actual, expected, rtol=2e-15, atol=2e-15)
    report(tmp_path / "modal_two.json", {
        "bitwise": raw_equal(actual, expected), "expected": record(expected),
        "actual": record(actual), "rhs": record(rhs), "shift": 3.75,
    })


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("sbdf2", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_coupled_diffusion_oracle(tmp_path, n, sbdf2, dtype):
    coupling, diffusion = coefficients(n)
    initial = torch.as_tensor(coupled_exact(coupling, diffusion, 0), dtype=dtype)
    exact = torch.as_tensor(coupled_exact(coupling, diffusion, FINAL))
    errors, residuals = [], []
    for dt in DT:
        engine = block_engine(dt, dtype, sbdf2, coupling, diffusion)
        state = engine.initial_state(dict(zip(engine.components, initial, strict=True)))
        values = []
        state = steps(engine, state, round(FINAL / dt), inplace=True, observations=values)
        assert torch.isfinite(state.spatial).all()
        errors.append(float((state.spatial[:, 0].double() - exact).square().mean().sqrt()))
        residuals.append([
            float(row["linear_residual"].abs().max()) for row in values
        ])
    payload = {"n": n, "scheme": "sbdf2_test" if sbdf2 else "Euler", "dtype": str(dtype),
               "scipy": scipy.__version__, "initial": record(initial), "dt": DT,
               "errors": errors, "linf_residuals": residuals}
    report(tmp_path / "coupled.json", payload)
    if dtype == torch.float64:
        payload["orders"] = order(errors, 1.8 if sbdf2 else .9)
        report(tmp_path / "coupled.json", payload)


def test_variable_diffusion_mms(tmp_path):
    exact, source, provenance = variable_diffusion_expressions()

    def solve(n, dt, *, maxiter=128, precondition=None):
        shape = (n, 8, 8)
        x = (np.arange(n, dtype=np.float64) * (2 * pi / n)).reshape(n, 1, 1)

        def value(function, time):
            return torch.from_numpy(np.broadcast_to(function(x, time), shape).copy())

        engine = variable_engine(
            dt, shape, lambda time: value(source, time).reshape(1, 1, *shape),
            maxiter=maxiter, precondition=precondition,
        )
        state = engine.initial_state({"u": value(exact, 0)})
        rows = []
        state = steps(engine, state, round(FINAL / dt), observations=rows)
        traces = [row["linear_solve"] for row in rows]
        assert all(row.converged and 0 < row.iterations <= maxiter for row in traces)
        return {
            "grid": shape, "dt": dt,
            "l2": float((state.spatial[0, 0] - value(exact, FINAL)).square().mean().sqrt()),
            "iterations": [row.iterations for row in traces],
            "residual_history": [row.residuals for row in traces],
        }

    rows = [solve(32, dt) for dt in DT]
    refined = solve(64, DT[-1])
    identity = solve(32, DT[0], precondition=lambda x: x)
    assert identity == rows[0]
    with pytest.raises(RuntimeError, match="^test diffusion CG did not converge$"):
        solve(32, DT[0], maxiter=0)
    report(tmp_path / "mms.json", {
        "rows": rows, "refined": refined, "provenance": provenance,
        "orders": order([row["l2"] for row in rows], .9),
        "refinement_delta_fraction": abs(refined["l2"] - rows[-1]["l2"]) / rows[-1]["l2"],
    })


def test_same_advance_and_solve_for_three_tiers(tmp_path, monkeypatch):
    import pssolver as pub

    calls = []

    def trace(cls):
        implementation = cls.solve

        def solve(self, rhs, **kwargs):
            calls.append((cls.__name__, kwargs["shift"]))
            return implementation(self, rhs, **kwargs)

        monkeypatch.setattr(cls, "solve", solve)

    for cls in (DiagonalOperator, ModalBlockOperator, VariableDiffusion):
        trace(cls)
    assert program.advance.__module__ == "pssolver.engine.program"
    for family in ("periodic", "plane", "channel"):
        scene = next(row for row in scenarios() if row["family"] == family
                     and row["dtype"] == "float64" and not row["options"])
        compiled = pub.compile_simulation(simulation_v05(scene, tmp_path / family))
        engine = compiled._run.assembled.engine(device="cpu", dtype=torch.float64)
        state = engine.initial_state({field: torch.zeros(engine.shape, dtype=torch.float64)
                                      for field in engine.components[:engine.evolved_count]})
        engine.step_(state)
    c, d = coefficients(2)
    engine = block_engine(.04, torch.float64, True, c, d)
    state = engine.initial_state(dict(zip(engine.components,
                                          torch.from_numpy(coupled_exact(c, d, 0)), strict=True)))
    steps(engine, state, 2)
    engine = variable_engine(.04, (32, 8, 8), lambda _: torch.ones(1, 1, 32, 8, 8, dtype=torch.float64))
    state = engine.initial_state({"u": torch.ones(engine.shape, dtype=torch.float64)})
    steps(engine, state, 1)
    assert calls == [("DiagonalOperator", 1.)] * 3 + [
        ("ModalBlockOperator", 1.), ("ModalBlockOperator", 1.5), ("VariableDiffusion", 1.),
    ]
    report(tmp_path / "shared_solve.json", calls)


def test_euler_start_then_sbdf2_on_same_program():
    model = ScalarPeriodicReactionDiffusion()
    engines = [scalar_engine(.04, torch.float64, use) for use in (False, True)]
    states = [engine.initial_state({"u": scalar_values(model, torch.float64)}) for engine in engines]
    updated = [engine.step(state) for engine, state in zip(engines, states, strict=True)]
    assert raw_equal(updated[0].spectral, updated[1].spectral)
    assert raw_equal(updated[0].spatial, updated[1].spatial)
    assert len(updated[0].history) == 0 and len(updated[1].history) == 1
    assert engines[1].scheme.implicit_shift(updated[1].history, .04) == 1.5


@pytest.mark.parametrize("sbdf2", [False, True])
def test_scalar_time_ladder(tmp_path, sbdf2):
    model = ScalarPeriodicReactionDiffusion()
    exact = model.exact_physical(FINAL, dtype=torch.float64, device=torch.device("cpu"))
    errors = []
    for dt in DT:
        engine = scalar_engine(dt, torch.float64, sbdf2)
        state = engine.initial_state({"u": scalar_values(model, torch.float64)})
        state = steps(engine, state, round(FINAL / dt))
        errors.append(float((state.spatial[0, 0, :, 0, 0] - exact).square().mean().sqrt()))
    scheme = IntegratorScheme.SBDF2 if sbdf2 else IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    report(tmp_path / "order.json", {
        "dt": DT, "errors": errors, "orders": order(errors, 1.8 if sbdf2 else .9),
        "p43_reference": measure_temporal_convergence(model=model, scheme=scheme).to_metadata(),
    })


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_single_history_slot_and_checkpoint(tmp_path, dtype, inplace):
    engine = scalar_engine(.04, dtype, True)
    state = engine.initial_state({"u": scalar_values(ScalarPeriodicReactionDiffusion(), dtype)})
    assert not state.history
    for step in range(1, 4):
        state = steps(engine, state, 1, inplace=inplace)
        assert len(state.history) == 1
        checkpoint = write_state(tmp_path / f"step_{step}", engine, state)
        header = json.loads((checkpoint / "checkpoint.json").read_bytes())
        assert header["plan"]["integrator"]["scheme"] == "sbdf2_test"
        assert header["history_length"] == 1 and "history_0" in header["tensor_files"]
        assert state_record(read_state(checkpoint, engine)) == state_record(state)


@pytest.mark.parametrize("cut", [1, 2, 23])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("inplace", [False, True])
def test_sbdf2_restart_exact(tmp_path, cut, dtype, inplace):
    initial = scalar_values(ScalarPeriodicReactionDiffusion(), dtype)
    engine = scalar_engine(.01, dtype, True)
    prefix = steps(engine, engine.initial_state({"u": initial}), cut, inplace=inplace)
    path = write_state(tmp_path / "cut", engine, prefix)
    restarted = scalar_engine(.01, dtype, True)
    restored = read_state(path, restarted)
    assert state_record(prefix) == state_record(restored)
    resumed = steps(restarted, restored, 40 - cut, inplace=inplace)
    continuous = steps(engine, prefix, 40 - cut, inplace=inplace)
    assert state_record(resumed) == state_record(continuous)
    opposite = steps(engine, engine.initial_state({"u": initial}), 40, inplace=not inplace)
    assert state_record(resumed) == state_record(opposite)
    report(tmp_path / "restart.json", state_record(resumed))


def test_history_rejects_changed_dt(tmp_path, monkeypatch):
    engine = scalar_engine(.04, torch.float64, True)
    initial = scalar_values(ScalarPeriodicReactionDiffusion(), torch.float64)
    state = engine.step(engine.initial_state({"u": initial}))
    path = write_state(tmp_path / "cut", engine, state)
    before = state_record(state)

    def unexpected(*args, **kwargs):
        pytest.fail("history mismatch reached state preparation or payload")

    monkeypatch.setattr(engine_state, "load_checkpoint_numpy", unexpected)
    monkeypatch.setattr(program, "prepare", unexpected)
    with pytest.raises(CheckpointCompatibilityError, match="^SBDF2 history requires unchanged dt$"):
        read_state(path, engine, dt=.02)
    engine.dt = .02
    with pytest.raises(ValueError, match="^SBDF2 history requires unchanged dt$"):
        engine.step_(state)
    assert state_record(state) == before


def test_sbdf2_taylor_including_startup(tmp_path):
    engine = scalar_engine(.04, torch.float64, True)
    initial = scalar_values(ScalarPeriodicReactionDiffusion(), torch.float64)
    index = torch.arange(initial.numel(), dtype=initial.dtype).reshape(initial.shape)
    direction, weight = index.sin(), (.31 * index).cos()

    def objective(value):
        state = steps(engine, engine.initial_state({"u": value}), 4)
        return (weight * state.spatial[0, 0].square()).sum()

    value, errors = taylor_check(objective, initial, direction, [1e-2, 1e-3, 1e-4, 1e-5])
    report(tmp_path / "taylor.json", {"order": value, "errors": errors})
    assert 1.9 <= value <= 2.1
