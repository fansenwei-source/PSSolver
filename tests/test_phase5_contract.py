"""Phase 5.0 planning, scope, and frozen Plane baseline tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "notes/architecture_v0_2/phase_5_contract.json"
INVENTORY = ROOT / "notes/architecture_v0_2/phase_5_plane_inventory.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha256(commit: str, relative: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def test_phase5_planning_is_authorized_separately_from_implementation():
    contract = _load(CONTRACT)
    assert contract["status"] == (
        "P5_0_PLANNING_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED"
    )
    assert contract["baseline"] == {
        "phase_4_classification": (
            "PASS_PHASE4_CLOSURE_WITH_MEMORY_MEASUREMENT_RECOVERY"
        ),
        "phase_4_record_commit": (
            "e54db2f7e5be16d0e84b5d854e5232d6bb7fb72b"
        ),
        "phase_4_complete": True,
        "eligible_for_phase_5_planning": True,
    }
    assert contract["authorization"] == {
        "phase_5_planning_authorized": True,
        "phase_5_implementation_authorized": False,
        "phase_6_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }


def test_phase5_freezes_explicit_runtime_identity_without_fallback():
    runtime = _load(CONTRACT)["runtime_paths"]
    assert runtime == {
        "current_default": "legacy_production",
        "current_diagnostic": "separated_canary",
        "planned_opt_in": "compiled_v2",
        "planned_selector_introduction_slice": "P5.5",
        "fallback_allowed": False,
    }
    inventory = _load(INVENTORY)
    assert inventory["current_runtime_paths"] == {
        "default": "legacy_production",
        "diagnostic": "separated_canary",
        "compiled_v2_present": False,
    }


def test_phase5_parity_does_not_mix_architecture_and_integrator_changes():
    target = _load(CONTRACT)["parity_target"]
    assert target == {
        "integrator": "projected_semi_implicit_euler",
        "scientific_reference": "legacy_production",
        "reuse_high_performance_tensor_layout": True,
        "reuse_qualified_numerical_kernels": True,
        "reuse_separated_dynamic_scheduler": False,
        "plane_checkpoint_format": 1,
        "sbdf2_connected_to_plane": False,
    }


def test_phase5_hot_loop_forbids_dynamic_control_work():
    forbidden = set(_load(CONTRACT)["hot_loop_forbidden"])
    assert forbidden == {
        "registry_lookup",
        "capability_lookup",
        "string_field_discovery",
        "json_parsing_or_serialization",
        "cli_or_configuration_parsing",
        "global_device_or_dtype_inference",
        "unbounded_tensor_allocation",
        "implicit_runtime_fallback",
    }


def test_phase5_slices_delay_public_selector_until_runtime_exists():
    slices = _load(CONTRACT)["phase_slices"]
    assert slices == [
        "P5.0_plan_inventory_and_oracle_freeze",
        "P5.1_disconnected_compiled_declarations",
        "P5.2_construction_time_binding_and_dataflow_audit",
        "P5.3_compiled_euler_step_program",
        "P5.4_observation_diagnostics_and_checkpoint_adapter",
        "P5.5_opt_in_application_connection",
        "P5.6_local_closure_and_phase6_handoff",
    ]


def test_phase5_defers_formal_gpu_and_default_decisions_to_phase6():
    deferred = set(_load(CONTRACT)["deferred_to_phase_6"])
    assert {
        "formal_r128_r320_h100_balanced_ab",
        "formal_peak_memory_and_transform_call_gate",
        "formal_100_step_q_u_p_gate",
        "formal_restart_output_and_provenance_gate",
        "long_time_scientific_benchmark",
        "implicit_default_smoke",
        "production_default_promotion_decision",
    } == deferred


def test_phase5_inventory_hashes_the_exact_plane_oracle():
    inventory = _load(INVENTORY)
    assert inventory["baseline_commit"] == (
        "e54db2f7e5be16d0e84b5d854e5232d6bb7fb72b"
    )
    source_oracle = inventory["source_oracle"]
    assert len(source_oracle) == 14
    for relative, expected in source_oracle.items():
        assert _git_blob_sha256(inventory["baseline_commit"], relative) == (
            expected
        )


def test_phase5_planning_does_not_claim_plane_or_default_changes():
    contract = _load(CONTRACT)
    assert "plane_sbdf2_activation" in contract["out_of_scope"]
    assert "strong_anchoring" in contract["out_of_scope"]
    assert contract["authorization"]["production_default_changed"] is False
    assert contract["authorization"]["separated_canary_promoted"] is False
