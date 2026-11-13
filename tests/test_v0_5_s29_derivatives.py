"""Step ownership and numerical invariants for Q and generic derivatives."""

import ast
import subprocess
from pathlib import Path

import pytest
import torch

import pssolver as pub
from pssolver.core.terms import EvolvedField, Explicit, Linear, Model, d, lap
from pssolver.engine import engine as engine_module
from pssolver.engine.derivatives import StepDerivativeTable
from pssolver.io import engine_state
from qualification.identity.scenarios import scenarios
from qualification.identity.scenes_v05 import simulation_v05
from tests.s29_reference import assemble_periodic
from tests.test_v0_5_s29_linear_imex import record, state_record


ROOT = Path(__file__).resolve().parents[1]
BASE = "7267dcd00d58c11069a1ee205c181fa3f62ba745"


class DuplicateGradient(Model):
    u = EvolvedField()

    def terms(self, f, p):
        yield Linear(f.u, .1 * lap(f.u))
        yield Explicit(f.u, lambda grad: grad, inputs=(d(f.u, 0),))
        yield Explicit(f.u, lambda grad: grad.square(), inputs=(d(f.u, 0),))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_generic_derivatives_once(monkeypatch, dtype):
    engine = assemble_periodic(DuplicateGradient(), .01, (8, 8, 8)).engine(
        device="cpu", dtype=dtype,
    )
    initial = torch.randn(engine.shape, dtype=dtype, generator=torch.Generator().manual_seed(29))
    a = engine.initial_state({"u": initial})
    b = engine.initial_state({"u": initial})
    table_type = StepDerivativeTable
    calls = []

    def resolve(self, key, compute):
        calls.append((key, key in self))
        return table_type.resolve(self, key, compute)

    class Tracked(table_type):
        pass

    Tracked.resolve = resolve
    monkeypatch.setattr(engine_module, "StepDerivativeTable", Tracked)
    for _ in range(2):
        a = engine.step(a)
    assert [hit for _, hit in calls] == [False, True] * 2

    class Disabled(table_type):
        def resolve(self, key, compute):
            calls.append((key, False))
            return compute()

    monkeypatch.setattr(engine_module, "StepDerivativeTable", Disabled)
    calls.clear()
    for _ in range(2):
        b = engine.step(b)
    assert [hit for _, hit in calls] == [False] * 4
    assert state_record(a) == state_record(b)


def test_derivative_key():
    table = StepDerivativeTable()
    keys = [
        ("u", (1, 0, 0), ("periodic",) * 3),
        ("v", (1, 0, 0), ("periodic",) * 3),
        ("u", (2, 0, 0), ("periodic",) * 3),
        ("u", (0, 1, 0), ("periodic",) * 3),
        ("u", (1, 0, 0), ("periodic", "periodic", "neumann")),
    ]
    for n, key in enumerate(keys):
        assert table.resolve(key, lambda n=n: n) == n
        assert table.resolve(key, lambda: None) == n
    assert tuple(table) == tuple(keys)


def test_no_derivative_lease_after_step(monkeypatch):
    tables = []

    class Tracked(StepDerivativeTable):
        def __init__(self):
            super().__init__()
            tables.append(self)

    monkeypatch.setattr(engine_module, "StepDerivativeTable", Tracked)
    engine = assemble_periodic(DuplicateGradient(), .01, (8, 8, 8)).engine(
        device="cpu", dtype=torch.float64,
    )
    state = engine.initial_state({"u": torch.ones(engine.shape, dtype=torch.float64)})
    engine.step(state)
    assert len(tables) == 1 and not tables[0]

    def fail(*args, **kwargs):
        assert tables[-1]
        raise RuntimeError("derivative caller failed")

    monkeypatch.setattr(engine.projector, "forward_transform", fail)
    with pytest.raises(RuntimeError, match="^derivative caller failed$"):
        engine.step(state)
    assert len(tables) == 2 and not any(tables)


def _scene(family, dtype):
    return next(row for row in scenarios() if row["family"] == family
                and row["dtype"] == str(dtype).removeprefix("torch.") and not row["options"])


def _case(tmp_path, family, dtype, *, disable):
    scene = _scene(family, dtype)
    scene = {**scene, "options": {**scene["options"], "disable_q_gradient_reuse": disable}}
    compiled = pub.compile_simulation(simulation_v05(scene, tmp_path))
    engine = compiled._run.assembled.engine(device="cpu", dtype=dtype)
    grid = torch.arange(engine.shape[0], dtype=dtype).view(-1, 1, 1)
    q = (.00001 * grid).expand(engine.shape).contiguous()
    initial = {name: (i + 1) * q + (.2 if i == 0 else -.1 if i == 3 else 0)
               for i, name in enumerate(engine.components[:engine.evolved_count])}
    return engine, engine.initial_state(initial)


def _empty(engine):
    cache = engine.explicit.q_gradient_cache
    assert cache is not None
    assert all(getattr(cache, attr) is None for attr in (
        "_owner", "_gradients", "_pending_owner", "_pending_gradients",
        "_spatial_version", "_spectral_version",
    ))


@pytest.mark.parametrize("family", ["periodic", "plane", "channel"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("inplace", [False, True])
def test_q_reuse_equivalence(tmp_path, family, dtype, inplace):
    e_on, a = _case(tmp_path / "on", family, dtype, disable=False)
    e_off, b = _case(tmp_path / "off", family, dtype, disable=True)
    for i in range(64):
        if inplace:
            e_on.prepare_constraints_(a)
            e_off.prepare_constraints_(b)
        else:
            a, b = e_on.prepare_constraints(a), e_off.prepare_constraints(b)
        assert record(e_on.observe(a)) == record(e_off.observe(b))
        if i in (23, 40):
            def restore(engine, state, root):
                plan = engine.plan
                engine_state.write(root, state, plan_identity_sha256=plan.identity_sha256,
                                   plan_metadata=plan.to_metadata(), run_provenance={},
                                   refresh_interval=plan.integrator.refresh_interval)
                restored = engine_state.read(root, plan_identity_sha256=plan.identity_sha256,
                                             dtype=dtype, batch=1, device="cpu")
                if family == "channel":
                    engine._restore_channel_state_layout(restored)
                return restored

            a = restore(e_on, a, tmp_path / f"on_{i}")
            b = restore(e_off, b, tmp_path / f"off_{i}")
        if inplace:
            e_on.step_(a)
            e_off.step_(b)
        else:
            a, b = e_on.step(a), e_off.step(b)
        _empty(e_on)
        assert state_record(a) == state_record(b), (family, dtype, inplace, i)


@pytest.mark.parametrize("family", ["periodic", "plane", "channel"])
def test_no_cross_step_hit(tmp_path, family):
    e_on, a = _case(tmp_path / "on", family, torch.float64, disable=False)
    e_off, b = _case(tmp_path / "off", family, torch.float64, disable=True)
    initial = dict(zip(e_on.components[:e_on.evolved_count],
                       a.spatial[:e_on.evolved_count], strict=True))
    shifted = {name: value + .001 for name, value in initial.items()}
    a2, b2 = e_on.initial_state(shifted), e_off.initial_state(shifted)
    assert a.spectral._version == a2.spectral._version
    assert record(a.spectral) != record(a2.spectral)
    for lhs, rhs in ((a2, b2), (a, b), (a2, b2)):
        assert state_record(e_on.step(lhs)) == state_record(e_off.step(rhs))
        _empty(e_on)


@pytest.mark.parametrize("family", ["periodic", "plane", "channel"])
def test_cache_empty_on_step_exit(tmp_path, monkeypatch, family):
    engine, state = _case(tmp_path / "on", family, torch.float64, disable=False)
    prepared = engine.prepare_constraints(state)
    engine.step(prepared)
    _empty(engine)

    def fail(*args, **kwargs):
        raise RuntimeError("explicit caller failed")

    monkeypatch.setattr(engine, "_explicit_at", fail)
    with pytest.raises(RuntimeError, match="^explicit caller failed$"):
        engine.step(state)
    _empty(engine)


def test_version_scan_and_retirement():
    allowed = ROOT / "pssolver/models/active_nematics/beris_edwards.py"
    occurrences = {path for path in (ROOT / "pssolver").rglob("*.py")
                   if any(isinstance(node, ast.Attribute) and node.attr == "_version"
                          for node in ast.walk(ast.parse(path.read_bytes())))}
    assert occurrences == {allowed}
    from pssolver.experimental import representations

    assert not hasattr(representations, "AlgebraicRepresentationCache")


def test_channel_cache_test_sources():
    file = "tests/test_v0_5_s28a_solver_state.py"
    assert (ROOT / file).read_bytes() == subprocess.check_output(
        ["git", "show", f"{BASE}:{file}"], cwd=ROOT,
    )
