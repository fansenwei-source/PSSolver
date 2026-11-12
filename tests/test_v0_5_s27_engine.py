"""Plane grouped execution and frozen Stokes operations: S-27 U4/U5/U6/U11."""

import math
from dataclasses import replace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import pssolver
from pssolver.engine.engine import Engine
from pssolver.io import engine_state
from pssolver.linear_solvers.stokes.plane_free_slip_stateless import solve_force_hats_stateless
from qualification.identity.scenarios import scenarios
from qualification.identity.scenes_v05 import simulation_v05


def plane_engine(tmp_path, dtype=torch.float64, *, batch=1, diagnostics=True):
    scene = next(
        item for item in scenarios()
        if item["id"] == f"plane_legacy_production_{str(dtype).split('.')[-1]}_default"
    )
    scene = {**scene, "shape": [8, 8, 8], "options": {"spectral_refresh_steps": 20}}
    declaration = simulation_v05(scene, tmp_path, steps=24)
    declaration = replace(
        declaration,
        geometry=pssolver.PlaneSlab(shape=(8, 8, 8), lengths=(8.0, 9.0, 20.0)),
        output=replace(declaration.output, diagnostics=diagnostics),
    )
    engine = Engine(pssolver.compile_simulation(declaration).plan, device="cpu", dtype=dtype, batch=batch)
    generator = torch.Generator().manual_seed(42)
    values = {name: torch.randn((batch, *engine.shape), dtype=dtype, generator=generator) * .01
              for name in engine.components[:engine.evolved_count]}
    return declaration, engine, values


def same_state(left, right):
    assert torch.equal(left.spatial, right.spatial)
    assert torch.equal(left.spectral, right.spectral)
    assert left.completed_steps == right.completed_steps
    assert left.refresh_phase == right.refresh_phase
    assert left.constrained_current == right.constrained_current


class Operations(TorchDispatchMode):
    def __init__(self):
        self.operations = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations.append(str(func))
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("diagnostics", [True, False])
@pytest.mark.parametrize("zero", [False, True])
def test_u5_stateless_stokes_exact_operations_and_report(tmp_path, dtype, diagnostics, zero):
    _, engine, _ = plane_engine(tmp_path, dtype, diagnostics=diagnostics)
    solver = engine.constraint.term
    generator = torch.Generator().manual_seed(19)
    forces = [torch.randn((1, *engine.shape), dtype=dtype, generator=generator)
              for _ in range(3)]
    forces = [engine.transform_backend.forward(value * (not zero), engine.component_boundaries[i])
              for value, i in zip(forces, (5, 6, 7), strict=True)]
    before = dict(vars(solver))
    actual_ops, expected_ops = Operations(), Operations()
    with actual_ops:
        *actual, report = solve_force_hats_stateless(solver, *forces, pressure_diagnostics=diagnostics)
    assert vars(solver).keys() == before.keys()
    assert all(vars(solver)[key] is value for key, value in before.items())
    with expected_ops:
        expected = solver.solve_force_hats(*forces)
    assert actual_ops.operations == expected_ops.operations
    assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
    for key, value in report.items():
        frozen = getattr(solver, key)
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, frozen)
        elif isinstance(value, float) and math.isnan(value):
            assert math.isnan(frozen)
        else:
            assert value == frozen


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_u4_pure_and_inplace_cross_refresh_bitwise(tmp_path, dtype):
    _, engine, values = plane_engine(tmp_path, dtype)
    pure, inplace = engine.initial_state(values), engine.initial_state(values)
    for _ in range(24):
        before = (pure.spatial.clone(), pure.spectral.clone(), pure.completed_steps)
        following = engine.step(pure)
        assert torch.equal(pure.spatial, before[0])
        assert torch.equal(pure.spectral, before[1])
        assert pure.completed_steps == before[2]
        engine.step_(inplace)
        same_state(following, inplace)
        pure = following


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_u6_engine_restart_at_refresh_boundary(tmp_path, dtype):
    _, engine, values = plane_engine(tmp_path, dtype)
    continuous = engine.initial_state(values)
    for _ in range(19):
        engine.step_(continuous)
    engine.prepare_constraints_(continuous)
    path = tmp_path / "checkpoint"
    engine_state.write(path, continuous, plan_identity_sha256=engine.plan.identity_sha256,
                       plan_metadata=engine.plan.to_metadata(), refresh_interval=20,
                       run_provenance={})
    restored = engine_state.read(path, plan_identity_sha256=engine.plan.identity_sha256,
                                 dtype=dtype, batch=1, device="cpu")
    for _ in range(5):
        engine.step_(continuous)
        engine.step_(restored)
        same_state(continuous, restored)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("batch", [2, 3])
def test_u11_batch_members_match_singletons(tmp_path, dtype, batch):
    _, engine, values = plane_engine(tmp_path, dtype, batch=batch)
    state = engine.initial_state(values)
    singles = []
    for index in range(batch):
        single = Engine(engine.plan, device="cpu", dtype=dtype)
        singles.append((single, single.initial_state({name: value[index] for name, value in values.items()})))
    for _ in range(24):
        engine.step_(state)
        for index, (single, other) in enumerate(singles):
            single.step_(other)
            assert torch.equal(state.spatial[:, index:index+1], other.spatial)
            assert torch.equal(state.spectral[:, index:index+1], other.spectral)


def test_plane_plan_groups_and_diagnostics_are_declared(tmp_path):
    spec, engine, _ = plane_engine(tmp_path)
    assert [g.indexing for g in engine.plan.groups] == ["contiguous_slice", "advanced", "contiguous_slice"]
    assert engine.plan.constraints[0].arguments["distortion_odd_boundary_conditions"] == ("periodic", "periodic", "dirichlet")
    other = pssolver.compile_simulation(
        replace(spec, output=replace(spec.output, directory=tmp_path / "different"))
    ).plan
    assert engine.plan.to_metadata() == other.to_metadata()
