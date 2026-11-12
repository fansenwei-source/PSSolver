"""Registry parity, retaining the reachable pre-switch binding as control."""

import ast
import importlib
import inspect
from dataclasses import FrozenInstanceError
from enum import Enum
from pathlib import Path

import pytest

from pssolver.functional.legacy_declarations import application_registry as shadow


ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def _binding_table():
    # Evaluate only the existing literal table (enum members and constants),
    # never execute a numerical kernel or a copy of the binding dispatcher.
    from tests.v0_5_s24_registry_reference import retained_runtime_bindings

    return retained_runtime_bindings()


def test_shadow_matches_literal_binding_table_keys_values_and_order():
    table = _binding_table()
    assert len(shadow.RUNTIMES) == len(shadow.RUNTIME_BY_KEY) == len(shadow.RUNTIME_BY_KIND) == 4
    assert tuple(shadow.RUNTIME_BY_KEY) == tuple(table)
    for entry in shadow.RUNTIMES:
        values = tuple(x.value if isinstance(x, Enum) else x for x in table[entry.key])
        assert entry.binding_fields() == values
        assert shadow.RUNTIME_BY_KIND[entry.kind] is entry


def test_shadow_is_immutable_tensor_free_and_not_public():
    for mapping in (shadow.APPLICATION_BY_PAIR, shadow.RUNTIME_BY_KEY, shadow.RUNTIME_BY_KIND):
        with pytest.raises(TypeError):
            mapping["injected"] = object()
    with pytest.raises(FrozenInstanceError):
        shadow.APPLICATIONS[0].application = "injected"
    with pytest.raises(FrozenInstanceError):
        shadow.RUNTIMES[0].runtime_path = "injected"
    assert shadow.__all__ == []
    tree = ast.parse(inspect.getsource(shadow))
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == ["dataclasses", "types"]
    assert not any(isinstance(node, ast.Import) for node in ast.walk(tree))


@pytest.mark.parametrize("pair", [
    ("unknown", "plane_slab"),
    ("complete_stress_beris_edwards", "unknown"),
    ("legacy_active_force_active_nematics", "periodic_box"),
    ("legacy_active_force_active_nematics", "plane_slab"),
])
def test_unregistered_shadow_keys_fail_closed_without_fallback(pair):
    assert shadow.APPLICATION_BY_PAIR.get(pair) is None
    with pytest.raises(KeyError):
        shadow.APPLICATION_BY_PAIR[pair]
    for runtime in shadow.RUNTIMES:
        with pytest.raises(KeyError):
            shadow.RUNTIME_BY_KEY[(*pair, runtime.runtime_path)]
