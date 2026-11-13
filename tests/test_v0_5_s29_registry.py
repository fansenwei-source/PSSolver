"""The package-external registry must cite executable qualification checks."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import pssolver
from pssolver.engine.execution_device import _resolve_tf32_execution


ROOT = Path(__file__).resolve().parents[1]
BASE = "7267dcd00d58c11069a1ee205c181fa3f62ba745"
REGISTRY = ROOT / "qualification/s29_optimizations.json"


def test_registry_is_single_and_complete():
    rows = json.loads(REGISTRY.read_bytes())
    assert set(rows) == {"compiled_v2", "pointwise_compile", "projected_truncation",
                         "q_gradient_reuse", "tf32", "closed_form_2x2"}
    assert all(set(row) == {"state", "reference", "implementation", "relation",
                           "default_source", "switch_source", "assertion_source",
                           "collected_nodes"} for row in rows.values())
    assert rows["compiled_v2"]["state"] == "retired_S27"


def _ast_reference(row):
    tree = ast.parse((ROOT / row["path"]).read_bytes())
    name = row["name"]
    unit, = (
        node for node in ast.walk(tree)
        if (isinstance(node, ast.FunctionDef) and node.name == name) or
        (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                              for t in node.targets))
    )
    if "key" in row:
        return ast.literal_eval(unit.value)[row["key"]]
    if isinstance(unit, ast.Assign):
        if isinstance(unit.value, ast.Attribute):
            enum_member = unit.value.value
            assert unit.value.attr == "value" and isinstance(enum_member, ast.Attribute)
            enum_name = enum_member.value.id
            enum, = (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == enum_name)
            value, = (n.value for n in enum.body if isinstance(n, ast.Assign)
                      and any(getattr(t, "id", None) == enum_member.attr for t in n.targets))
            return ast.literal_eval(value)
        return ast.literal_eval(unit.value)
    return {
        node.value for node in ast.walk(unit) if isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    }


def test_switch_default_limit_and_execution_nodes():
    for row in json.loads(REGISTRY.read_bytes()).values():
        for entry in row["default_source"]:
            assert _ast_reference(entry) == entry["value"]
        for entry in row["switch_source"]:
            tree = ast.parse((ROOT / entry["path"]).read_bytes())
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.add(node.id)
                elif isinstance(node, ast.arg):
                    names.add(node.arg)
                elif isinstance(node, ast.Attribute):
                    names.add(node.attr)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    names.add(node.value)
            assert entry["token"] in names
        for entry in row["assertion_source"]:
            assert set(entry["values"]) <= _ast_reference(entry)
        for node in row["collected_nodes"]:
            path, function = node.split("::")
            assert any(
                isinstance(unit, ast.FunctionDef) and unit.name == function
                for unit in ast.parse((ROOT / path).read_bytes()).body
            )


def test_retained_execution_tests_match_basis_blobs():
    for path in (
        "tests/test_projected_transform_truncation.py",
        "tests/test_phase4_modal_block_operator.py",
        "tests/test_phase4_combined_sbdf2_modal_block.py",
        "tests/test_phase4_scalar_sbdf2_reference.py",
        "tests/test_v0_5_s26_allen_cahn.py",
    ):
        assert (ROOT / path).read_bytes() == subprocess.check_output(
            ["git", "show", f"{BASE}:{path}"], cwd=ROOT,
        )


def test_catalog_does_not_publish_registry():
    catalog = json.dumps(pssolver.capability_catalog().to_metadata(), sort_keys=True)
    assert "s29_optimizations" not in catalog
    assert "sbdf2_test" not in catalog
    assert "matrix_free_test" not in catalog
    assert "modal_block_test" not in catalog


def test_retired_compiled_v2_cli_switch():
    code = (
        "import sys;sys.path.insert(0,sys.argv.pop(1));"
        "from pssolver.cli.plane import main;raise SystemExit(main())"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(ROOT), "--runtime-path", "compiled_v2"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 2
    assert result.stderr.splitlines()[-1].endswith(
        ": error: --runtime-path was removed; see docs/migration_0_4_to_0_5.md",
    )


def test_t2_gate_raw_ratio_and_matrix():
    from qualification.s29_t2 import COUNTS, GRIDS, KERNELS, SCENES, SEEDS, compare

    assert len(KERNELS) * 2 * len(GRIDS) * len(SCENES) == 252
    assert sum(COUNTS) * 2 * len(GRIDS) * len(SCENES) * len(SEEDS) == 13536
    dtype = torch.float64
    eps = torch.finfo(dtype).eps
    a = torch.tensor([1], dtype=dtype)
    assert compare(torch, a, a + 16 * eps, dtype)["passed"]
    assert not compare(torch, a, a + 17 * eps, dtype)["passed"]
    zero = torch.zeros(1, dtype=dtype)
    assert not compare(torch, zero, -zero, dtype)["passed"]
    assert not compare(torch, a, torch.full_like(a, float("nan")), dtype)["passed"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tf32_scope_without_device_initialization(dtype, device):
    for policy in ("off", "on"):
        effective, resolved = _resolve_tf32_execution(policy, real_dtype=dtype, device=device)
        assert effective is (policy == "on" and device == "cuda" and dtype == torch.float32)
        assert resolved.type == device
    assert not torch.cuda.is_initialized()
