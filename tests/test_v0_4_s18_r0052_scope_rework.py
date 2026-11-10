"""Static scope ratchet for the R-0052 S-18 design rework."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_4_0_s18_r2_r4_r5_r1_design_proposal.json"
ZH = ROOT / "notes/PSSolver_v0_4_0_s18_r2_r4_r5_r1_design_proposal_zh.md"
DISPOSITIONS = ("batch_change", "read_only_frozen", "out_of_scope")


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _classified(batch: str) -> dict[str, str]:
    scope = _read()["scope_classification_v2"][batch]
    result: dict[str, str] = {}
    for disposition in DISPOSITIONS:
        for item in scope[disposition]:
            assert item["path"] not in result
            assert item["reason"].strip()
            result[item["path"]] = disposition
    return result


def _definition_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _checkpoint_inventory() -> set[str]:
    helpers = {"_sha256_bytes", "_sha256_file", "_identity_sha256", "_write_tensor"}
    found: set[str] = set()
    for path in (ROOT / "pssolver").rglob("*.py"):
        names = _definition_names(path)
        if any("checkpoint" in name.lower() for name in names) or helpers & names:
            found.add(path.relative_to(ROOT).as_posix())
    return found


def _spectral_refresh_inventory() -> set[str]:
    control_names = {
        "step",
        "restore_progress",
        "_refresh_dynamic_spectra",
        "_advance_spectral_refresh_clock",
    }
    found: set[str] = set()
    revision = "4c12228aba01e8483e80d28a1471ab72384d5fda"
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision, "pssolver"], cwd=ROOT, text=True).splitlines()
    for relative in paths:
        if not relative.endswith(".py"):
            continue
        text = subprocess.check_output(["git", "show", revision + ":" + relative], cwd=ROOT, text=True)
        if not any(token in text for token in ("spectral_refresh", "refresh_count", "_refresh_dynamic_spectra")):
            continue
        names = {node.name for node in ast.walk(ast.parse(text)) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        if control_names & names or any("checkpoint" in name.lower() for name in names):
            found.add(relative)
    return found


def test_r4_code_scan_and_claude_inventory_have_no_unclassified_module() -> None:
    classified = _classified("R4")
    assert _spectral_refresh_inventory() <= set(classified)
    claude_inventory = {
        "pssolver/configuration/plane_beris_edwards.py",
        "pssolver/execution/state.py",
        "pssolver/experimental/_shadow_support.py",
        "pssolver/experimental/channel_compiled_v2.py",
        "pssolver/experimental/checkpointing.py",
        "pssolver/experimental/h100_shadow_qualification.py",
        "pssolver/experimental/integrators.py",
        "pssolver/experimental/modal_block_checkpoint.py",
        "pssolver/experimental/modal_block_reference.py",
        "pssolver/experimental/plane_shadow_driver.py",
        "pssolver/experimental/shadow_metadata.py",
        "pssolver/experimental/shadow_run.py",
        "pssolver/functional/channel_activity_runtime.py",
        "pssolver/functional/channel_checkpoint.py",
        "pssolver/functional/periodic_activity.py",
        "pssolver/functional/periodic_checkpoint.py",
        "pssolver/integrator.py",
        "pssolver/integrators/reference_checkpoint.py",
        "pssolver/integrators/scalar_reference.py",
        "pssolver/integrators/state_backed.py",
        "pssolver/planning/channel_compiled_v2.py",
        "pssolver/planning/plane_compiled_v2.py",
        "pssolver/runtime/channel_active_nematics.py",
        "pssolver/runtime/channel_beris_edwards.py",
        "pssolver/runtime/periodic_beris_edwards.py",
        "pssolver/runtime/plane_beris_edwards.py",
        "pssolver/runtime/plane_compiled_v2_binding.py",
        "pssolver/runtime/plane_legacy.py",
        "pssolver/runtime/robin_scalar.py",
        "pssolver/workflows/channel_beris_edwards.py",
        "pssolver/workflows/channel_checkpoint.py",
        "pssolver/workflows/periodic_checkpoint.py",
        "pssolver/workflows/plane_beris_edwards.py",
        "pssolver/workflows/plane_checkpoint.py",
        "pssolver/workflows/plane_compiled_v2.py",
    }
    assert claude_inventory <= set(classified)


def test_control_functional_runtimes_are_frozen_but_still_gate_control_main() -> None:
    classified = _classified("R4")
    for relative in (
        "pssolver/functional/periodic_activity.py",
        "pssolver/functional/channel_activity_runtime.py",
    ):
        assert classified[relative] == "read_only_frozen"
    record = _read()
    assert "complete PSSolver-Control main CPU suite" in (
        record["common_cpu_acceptance"]["control_main_full_cpu_gate"]
    )
    assert "complete Control main CPU suite" in (
        record["scope_classification_v2"]["R4"]["control_relation"]
    )


def test_r1_classifies_every_missing_per_combination_module_as_frozen() -> None:
    classified = _classified("R1")
    required = {
        "pssolver/configuration/channel_active_nematics.py",
        "pssolver/configuration/channel_beris_edwards.py",
        "pssolver/configuration/periodic_beris_edwards.py",
        "pssolver/configuration/plane_beris_edwards.py",
        "pssolver/configuration/plane_beris_edwards_builders.py",
        "pssolver/configuration/plane_beris_edwards_component_graph.py",
        "pssolver/configuration/plane_beris_edwards_components.py",
        "pssolver/configuration/plane_beris_edwards_schema_v1.py",
        "pssolver/planning/channel_compiled_v2.py",
        "pssolver/planning/plane_compiled_v2.py",
        "pssolver/runtime/channel_active_nematics.py",
        "pssolver/runtime/channel_beris_edwards.py",
        "pssolver/runtime/periodic_beris_edwards.py",
        "pssolver/runtime/plane_beris_edwards.py",
        "pssolver/runtime/plane_compiled_v2_binding.py",
        "pssolver/runtime/plane_compiled_v2_step.py",
        "pssolver/workflows/plane_compiled_v2.py",
    }
    assert {classified[path] for path in required} == {"read_only_frozen"}


def test_r2_real_nfs_trigger_is_exact_and_design_zh_names_all_categories() -> None:
    trigger = _read()["scope_classification_v2"]["R2"]["real_nfs_trigger"]
    assert "rename/renameat2" in trigger["required_if"]
    assert "fsync" in trigger["required_if"]
    assert "publication call graph" in trigger["not_triggered_by"]
    zh = ZH.read_text(encoding="utf-8")
    for label in ("本批修改", "只读冻结", "范围外"):
        assert label in zh
    for missing_from_first_design in (
        "pssolver/integrators/reference_checkpoint.py",
        "pssolver/workflows/finite_q_anchoring.py",
        "pssolver/integrator.py",
        "pssolver/runtime/plane_legacy.py",
    ):
        assert missing_from_first_design in zh
