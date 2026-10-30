"""Forward contract for the S-16 V-55 rework and binding errata."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "notes/PSSolver_v0_4_0_s16_r8_benchmark_harness_sharing_v2.json"
CORRECTIONS = ROOT / "notes/PSSolver_v0_4_0_s16_r8_binding_reference_corrections.json"
RATCHET_TEST = ROOT / "tests/test_v0_4_s16_r8_benchmark_harness_sharing.py"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _function(path: Path, name: str) -> ast.FunctionDef:
    module = ast.parse(path.read_text(encoding="utf-8"))
    return next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_pruning_contract_reads_no_repository_external_path() -> None:
    node = _function(RATCHET_TEST, "test_v2_records_mode_change_and_confirmed_temp_pruning")
    calls = {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    }
    assert calls.isdisjoint({"exists", "is_dir", "is_file", "lstat", "stat"})


def test_binding_reference_corrections_are_exact_and_forward_only() -> None:
    inventory = _read(INVENTORY)
    corrections = _read(CORRECTIONS)
    assert corrections["source_inventory"] == {
        "path": "notes/PSSolver_v0_4_0_s16_r8_benchmark_harness_sharing_v2.json",
        "sha256": _sha256(INVENTORY),
        "rewritten": False,
    }
    original = {row["path"]: row for row in inventory["additional_scripts"]}
    fixed = {row["path"]: row for row in corrections["corrections"]}
    assert set(fixed) == {
        "benchmarks/shendruk_v1_dynamic_velocity/fig4_timestep_convergence.py",
        "benchmarks/v0_2_0_checkpoint_fixtures.py",
    }
    for relative, row in fixed.items():
        assert row["historical_disposition"] == original[relative]["disposition"]
        assert row["historical_binding_references"] == original[relative][
            "binding_references"
        ]
        assert row["baseline_sha256"] == _sha256(ROOT / relative)
    assert fixed[
        "benchmarks/shendruk_v1_dynamic_velocity/fig4_timestep_convergence.py"
    ]["corrected_binding_references"] == []
    assert fixed["benchmarks/v0_2_0_checkpoint_fixtures.py"][
        "corrected_binding_references"
    ] == ["notes/PSSolver_v0_3_0_s03_successor_56_s10_r0033_candidate.json"]


def test_binding_corrections_change_no_runtime_or_gate() -> None:
    assert all(value is False for value in _read(CORRECTIONS)["scope"].values())
