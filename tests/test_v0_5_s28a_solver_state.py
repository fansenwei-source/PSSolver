"""Test explicit Channel cache ownership, member transfers and persistence."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver.configuration.legacy_channel import load_channel_initial_q, translate_channel
from pssolver.configuration.public_simulation_runner import compile_public_simulation
from pssolver.core._canonical import canonical_sha256
from pssolver.engine import members
from pssolver.engine.engine import Engine
from pssolver.engine.state import State
from pssolver.io import engine_state
from pssolver.io.checkpoint import seal_checkpoint_metadata
from qualification.identity.channel_comparison import ChannelEngineCase
from tests.test_phase9_p974_channel_functional_runtime import _simulation
from tests.test_v0_3_s09_pressure_operator_cache import _cache, _rhs, _solve, _solver


def test_explicit_warm_start_invalidation_matches_reference_action_count(monkeypatch):
    spectral, solver = _solver()
    _, state, _ = _solve(solver, _rhs(spectral, solver, seed=9201), _cache())
    rhs = _rhs(spectral, solver, seed=9202)
    original = solver._pressure_operator
    count = 0

    def count_action(value):
        nonlocal count
        count += 1
        return original(value)

    monkeypatch.setattr(solver, "_pressure_operator", count_action)
    cached, _, _ = _solve(solver, rhs, state)
    assert count == 9
    count = 0
    direct, _, _ = _solve(solver, rhs, _cache(state["pressure_guess"].clone()))
    assert count == 10
    assert cached.numpy().tobytes() == direct.numpy().tobytes()


def _checkpoint_state(solver, batch):
    spectral = torch.zeros((1, batch, 3, 2, 2), dtype=torch.complex128)
    spatial = torch.zeros(spectral.shape, dtype=torch.float64)
    return State(spectral, spatial, (), solver, 0, (0, 0), True)


@pytest.mark.parametrize("batch", [1, 2])
def test_checkpoint_keeps_exact_action_and_python_working_exponent(tmp_path, batch):
    guess = torch.full((1, 3, 2, 2), 1 + 2j, dtype=torch.complex128)
    first = dict(_cache(guess), pressure_guess_action=guess / 2,
                 pressure_guess_action_exponent=-597)
    solver = first if batch == 1 else members._join_solver([first, _cache(guess)])
    state = _checkpoint_state(solver, batch)
    metadata = {"name": "test"}
    plan_identity = canonical_sha256(metadata)
    root = tmp_path / "saved"
    engine_state.write(
        root, state, plan_identity_sha256=plan_identity, plan_metadata=metadata,
        refresh_interval=None, run_provenance={},
    )
    restored = engine_state.read(
        root, plan_identity_sha256=plan_identity, dtype=torch.float64, batch=batch, device="cpu",
    )
    for key in ("pressure_guess", "pressure_guess_action"):
        assert restored.solver[key].numpy().tobytes() == solver[key].numpy().tobytes()
    assert restored.solver["pressure_guess_action_exponent"] == (
        -597 if batch == 1 else (-597, None)
    )
    if batch == 2:
        assert restored.solver["pressure_guess_action"][1].count_nonzero() == 0

    # Old engine_state headers did not have this optional key. Their action
    # cannot be reused without a working exponent.
    header = root / "checkpoint.json"
    old = json.loads(header.read_text())
    old.pop("solver_scalars")
    old.pop("metadata_sha256")
    header.write_text(json.dumps(seal_checkpoint_metadata(old)))
    restored_old = engine_state.read(
        root, plan_identity_sha256=plan_identity, dtype=torch.float64, batch=batch, device="cpu",
    )
    assert restored_old.solver["pressure_guess_action"] is None


def test_empty_solver_does_not_change_legacy_engine_state_header(tmp_path):
    plan = {}
    identity = canonical_sha256(plan)
    target = tmp_path / "saved"
    engine_state.write(
        target, _checkpoint_state({}, 1), plan_identity_sha256=identity,
        plan_metadata=plan, refresh_interval=None, run_provenance={},
    )
    assert "solver_scalars" not in json.loads((target / "checkpoint.json").read_text())


@pytest.mark.parametrize("dtype,shape", [("float32", [16, 12, 12]), ("float64", [17, 13, 11])])
def test_channel_restart_recovers_fft_layout_without_solving(tmp_path, monkeypatch, dtype, shape):
    scene = {
        "family": "channel", "shape": shape, "steps": 2, "dtype": dtype,
        "options": {"fixed_iterations": 12}, "observe": True, "restart_steps": [1],
    }
    continuous = ChannelEngineCase(scene, tmp_path)
    continuous.step()
    checkpoint = tmp_path / "saved"
    continuous.export(checkpoint)
    saved = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()}
    resumed = ChannelEngineCase(scene, tmp_path)

    def unexpected_solve(*args, **kwargs):
        pytest.fail("checkpoint restore unexpectedly solves the pressure constraint")

    with monkeypatch.context() as patch:
        patch.setattr(resumed.engine.constraint.term.pressure_solver, "solve", unexpected_solve)
        resumed.restore(checkpoint, 1)
    assert resumed.state.completed_steps == continuous.state.completed_steps == 1
    for name, tensor in continuous.state.solver.items():
        if isinstance(tensor, torch.Tensor):
            assert tensor.numpy().tobytes() == resumed.state.solver[name].numpy().tobytes()
        else:
            assert tensor == resumed.state.solver[name]
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()} == saved
    assert resumed.state.spectral.stride() == continuous.state.spectral.stride()
    continuous.step()
    resumed.step()
    assert resumed.observe() == continuous.observe()
    assert resumed.record() == continuous.record()
    assert all(torch.isfinite(tensor).all() for tensor in (resumed.state.spatial, resumed.state.spectral))


@pytest.mark.parametrize("inplace", [False, True])
def test_channel_member_copy_preserves_coherent_warm_states(tmp_path, inplace):
    spec = compile_public_simulation(_simulation(tmp_path).specification).run_spec
    plan = translate_channel(spec)
    initial, _, _ = load_channel_initial_q(spec)
    batch = {name: torch.stack((value, value * 0.9)) for name, value in initial.items()}
    aggregate = Engine(plan, device="cpu", dtype=torch.float64, batch=2)
    singles = [Engine(plan, device="cpu", dtype=torch.float64) for _ in range(2)]
    states = [
        engine.initial_state({name: value[i] for name, value in batch.items()})
        for i, engine in enumerate(singles)
    ]
    joined = aggregate.initial_state(batch)

    def verify():
        for attr in ("spatial", "spectral"):
            expected = torch.cat([getattr(state, attr) for state in states], dim=1)
            assert getattr(joined, attr).contiguous().numpy().tobytes() == (
                expected.contiguous().numpy().tobytes()
            )
        assert joined.solver["pressure_guess_action_exponent"] == tuple(
            state.solver["pressure_guess_action_exponent"] for state in states
        )

    verify()
    for _ in range(2):
        if inplace:
            aggregate.step_(joined)
            for engine, state in zip(singles, states, strict=True):
                engine.step_(state)
        else:
            joined = aggregate.step(joined)
            states = [engine.step(state) for engine, state in zip(singles, states, strict=True)]
        verify()


def test_engine_does_not_modify_solver_tensor_slots_in_place():
    root = Path(__file__).resolve().parents[1] / "pssolver"
    for rel in ("engine", "io/engine_state.py", "runtime/legacy_entry.py",
                "linear_solvers/stokes/channel_no_slip.py"):
        target = root / rel
        for path in target.glob("*.py") if target.is_dir() else (target,):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    receiver = ast.unparse(node.func.value)
                    assert not (".solver[" in receiver and node.func.attr.endswith("_")), path
                if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
                    assert ".solver[" not in ast.unparse(node.value), path
                if rel.endswith("channel_no_slip.py") and isinstance(node, ast.Attribute):
                    assert node.attr != "_version"
