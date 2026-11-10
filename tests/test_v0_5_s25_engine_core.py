"""Direct execution checks before changing the legacy Periodic entry points."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from pssolver.configuration.legacy_periodic import translate_periodic
from pssolver.engine import Engine
from qualification.identity.records import tensor_record
from qualification.identity.runtime_case import complete_simulation, initial_values
from qualification.identity.scenarios import scenarios


def _setup(tmp_path, dtype="float64", variant="default", flavour="production", batch=1):
    scene = next(row for row in scenarios() if row["id"] == f"periodic_default_{dtype}_{variant}")
    spec = complete_simulation(scene, tmp_path).specification
    plan = translate_periodic(spec, flavour=flavour, load_initial=False).plan
    engine = Engine(plan, device="cpu", dtype=getattr(torch, dtype), batch=batch)
    raw = initial_values(scene)
    initial = {
        name: torch.tensor(raw[..., index], dtype=getattr(torch, dtype))
        for index, name in enumerate(engine.components[:5])
    }
    reference = json.loads(
        (Path(__file__).parent / "baselines/s25/periodic_comparison.json").read_text()
    )
    return reference["core"][dtype], engine, initial


def _snapshot(root):
    seen = {}

    def walk(value):
        if isinstance(value, torch.Tensor):
            return {"id": id(value), "version": value._version, "tensor": tensor_record(value)}
        if isinstance(value, (str, int, float, bool, type(None))):
            return repr(value)
        if isinstance(value, (tuple, list)):
            return [walk(item) for item in value]
        if isinstance(value, Mapping):
            return {repr(key): walk(item) for key, item in value.items()}
        if id(value) in seen:
            return {"reference": seen[id(value)]}
        if type(value).__module__.startswith("pssolver"):
            seen[id(value)] = len(seen)
            attributes = dict(vars(value)) if hasattr(value, "__dict__") else {}
            if is_dataclass(value):
                attributes.update(
                    {field.name: getattr(value, field.name) for field in fields(value)}
                )
            return {
                "type": type(value).__qualname__,
                "attributes": {key: walk(item) for key, item in attributes.items()},
            }
        return {"type": str(type(value)), "id": id(value)}

    return walk(root)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_periodic_initial_and_inplace_20_steps_match_legacy(tmp_path, dtype):
    legacy, engine, initial = _setup(tmp_path, dtype)
    diagnostics = {}
    before = _snapshot(engine)
    state = engine.initial_state(initial, diagnostics=diagnostics)
    assert _snapshot(engine) == before
    assert diagnostics["pressure_iterations"] == 1
    assert torch.count_nonzero(state.spatial[5:]) == 0
    assert tensor_record(state.spatial) == legacy["states"][0]["spatial"]
    assert tensor_record(state.spectral) == legacy["states"][0]["spectral"]
    for index in range(20):
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("host scalar")):
            engine.step_(state, diagnostics=diagnostics)
        assert tensor_record(state.spatial) == legacy["states"][index + 1]["spatial"]
        assert tensor_record(state.spectral) == legacy["states"][index + 1]["spectral"]
        assert _snapshot(engine) == before


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_periodic_pure_and_inplace_have_same_values_and_do_not_change_engine(tmp_path, dtype):
    _, engine, initial = _setup(tmp_path, dtype)
    pure = engine.initial_state(initial)
    inplace = engine.initial_state(initial)
    before = _snapshot(engine)
    for _ in range(4):
        previous = (tensor_record(pure.spatial), tensor_record(pure.spectral))
        result = engine.step(pure)
        assert (tensor_record(pure.spatial), tensor_record(pure.spectral)) == previous
        pure = result
        engine.step_(inplace)
        assert tensor_record(pure.spatial) == tensor_record(inplace.spatial)
        assert tensor_record(pure.spectral) == tensor_record(inplace.spectral)
        assert _snapshot(engine) == before


def test_prepare_does_not_resolve_again_in_step(tmp_path):
    _, engine, initial = _setup(tmp_path)
    state = engine.initial_state(initial)
    before = _snapshot(engine)
    solve = engine.constraint.term.solve_force_hats_with_diagnostics
    with patch.object(
        engine.constraint.term, "solve_force_hats_with_diagnostics", wraps=solve
    ) as spy:
        engine.prepare_constraints_(state)
        engine.observe(state)
        engine.step_(state)
        assert spy.call_count == 1
    assert _snapshot(engine) == before
