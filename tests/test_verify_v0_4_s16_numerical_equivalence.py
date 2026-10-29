"""Unit gates for the S-16 numerical-equivalence support helper."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "verify_v0_4_s16_numerical_equivalence.py"


def _module():
    spec = importlib.util.spec_from_file_location("v0_4_s16_equivalence", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s16_equivalence_support_covers_every_frozen_runtime_family():
    module = _module()
    assert module.FAMILIES == (
        "plane_legacy_production",
        "plane_compiled_v2",
        "periodic_complete_stress",
        "channel_legacy",
        "channel_compiled_v2",
        "channel_complete_stress",
        "periodic_functional",
        "channel_functional",
    )


def test_s16_equivalence_support_does_not_import_a_provider_at_module_load():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "pssolver" not in imports
    assert "torch" not in imports
