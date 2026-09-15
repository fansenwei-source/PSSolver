from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from benchmarks.stage_o42_runtime import _plain_diagnostic_snapshot
from pssolver.diagnostics import compare_tensor_inventories
from pssolver.experimental.stage_o42_diagnostics import (
    analyze_stage_o42_diagnostics,
)
from pssolver.experimental.stage_o42_plan import build_stage_o42_h100_plan


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compare_inventories(before: dict, after: dict):
    return compare_tensor_inventories(
        before,
        after,
        maximum_reported_differences=max(
            1,
            before["tensor_reference_count"] + after["tensor_reference_count"],
        ),
    )


def _inventory(*, unique: int, allocated: int, categories: dict[str, int]):
    inventory = {
        "schema_version": 1,
        "scope": "explicit_runtime_object_graph",
        "device_filter": "cuda:0",
        "truncated": False,
        "visited_object_count": 3,
        "tensor_reference_count": 1,
        "unique_storage_count": 1,
        "logical_tensor_bytes": unique,
        "unique_storage_bytes": unique,
        "raw_storage_range_count": 1,
        "raw_storage_bytes_before_overlap_coalescing": unique,
        "overlap_collapsed_bytes": 0,
        "aliased_tensor_reference_count": 0,
        "exclusive_storage_bytes_by_category": categories,
        "shared_storage_bytes_by_category": {},
        "unique_storage_bytes_by_dtype": {"torch.float64": unique},
        "tensor_references": [
            {
                "path": "runtime.model.value",
                "category": "model",
                "shape": [unique // 8],
                "stride": [1],
                "dtype": "torch.float64",
                "device": "cuda:0",
                "logical_bytes": unique,
                "storage_offset": 0,
                "is_view": False,
                "requires_grad": False,
            }
        ],
        "storages": [
            {
                "device": "cuda:0",
                "data_ptr": 4096,
                "storage_bytes": unique,
                "paths": ["runtime.model.value"],
                "categories": ["model"],
                "dtypes": ["torch.float64"],
                "tensor_references": 1,
                "source_storage_ranges": [
                    {"data_ptr": 4096, "storage_bytes": unique}
                ],
                "source_storage_range_count": 1,
            }
        ],
        "reported_storage_count": 1,
        "all_storages_reported": True,
    }
    priming_inventory = copy.deepcopy(inventory)
    measured_inventory = copy.deepcopy(inventory)
    verification_inventory = copy.deepcopy(inventory)
    identity_comparison = _compare_inventories(
        priming_inventory,
        measured_inventory,
    )
    repeated_comparison = _compare_inventories(
        measured_inventory,
        verification_inventory,
    )
    return {
        "allocator": {
            "allocated_bytes": allocated,
            "reserved_bytes": allocated * 2,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": allocated * 2,
        },
        "inventory": inventory,
        "allocator_minus_inventory_bytes": allocated - unique,
        "inventory_exceeds_allocator_bytes": max(0, unique - allocated),
        "storage_accounting_consistent": unique <= allocated,
        "allocator_counters_stable": True,
        "storage_identity_priming": {
            "tensor_reference_count": 1,
            "unique_storage_count": 1,
            "unique_storage_bytes": unique,
            "truncated": False,
            "all_storages_reported": True,
            "identical_to_measured_inventory": True,
            "allocated_delta_bytes": 0,
            "reserved_delta_bytes": 0,
        },
        "storage_identity_priming_inventory": priming_inventory,
        "storage_identity_measured_inventory": measured_inventory,
        "storage_identity_verification_inventory": verification_inventory,
        "storage_identity_inventory_comparison": identity_comparison,
        "storage_identity_repeated_measurement_comparison": repeated_comparison,
        "storage_identity_inventory_stable": True,
        "storage_identity_set_stable": True,
        "storage_identity_post_priming_stable": True,
        "storage_identity_observer_effect_explained": True,
        "allocator_block_reconciliation": {
            "schema_version": 1,
            "classification": "fully_reconciled_with_active_allocator_blocks",
            "accounting_reconciled": True,
            "allocator_counter_matches_active_block_bytes": True,
            "all_unmatched_storages_reported": True,
            "inventory_bytes_outside_active_allocator_blocks": 0,
            "unmatched_storages": [],
        },
    }


def _profile(role: str, *, scale: int):
    config = {
        "shape": [320, 320, 80],
        "lengths": [100.0, 100.0, 20.0],
        "dtype": "float64",
        "dt": 0.005,
        "dealias_rule": "cubic_half",
        "projected_transform_execution": "truncated",
        "spectral_storage": "hermitian_half",
        "spectral_refresh_interval": 2,
    }
    phase = _inventory(
        unique=80 * scale,
        allocated=100 * scale,
        categories={"solver_fields": 50 * scale, "model": 30 * scale},
    )
    operator = {
        "schema_version": 1,
        "steps": 2,
        "operators": {
            "aten::cat": {
                "calls": scale,
                "self_device_memory_bytes": 10 * scale,
            },
            "aten::stack": {
                "calls": 2 * scale,
                "self_device_memory_bytes": 20 * scale,
            },
        },
    }
    return {
        "schema_version": 1,
        "qualification_stage": "O.4.2",
        "measurement_role": role,
        "classification": "DIAGNOSTIC_COMPLETE",
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "finite": True,
        "production_metadata_sha256": "a" * 64,
        "production_initial_q_sha256": "b" * 64,
        "configuration": config,
        "diagnostic_contract": {
            "shape": [320, 320, 80],
            "diagnostic_steps": 2,
            "global_gc_traversal": False,
            "runtime_roots_only": True,
            "allocator_block_reconciliation": True,
            "allocator_counter_stability_checked": True,
            "storage_identity_priming_pass": True,
            "repeated_inventory_identity_checked": True,
            "priming_inventory_preserved": True,
            "canonical_inventory_hashes_preserved": True,
            "path_level_inventory_diff_preserved": True,
            "post_priming_inventory_verification": True,
            "custom_mapping_getitem_forbidden": True,
        },
        "residency_phases": {
            "after_warmup": copy.deepcopy(phase),
            "after_timestep": copy.deepcopy(phase),
            "after_observation": copy.deepcopy(phase),
        },
        "operator_audit": operator,
        "semantic_regions": {
            "steps": 2,
            "regions": {
                "timestep.total": {
                    "calls": 2,
                    "total_seconds": 0.2 * scale,
                }
            },
        },
    }


def _o41():
    return {
        "qualification_stage": "O.4.1",
        "classification": "B_neutral",
        "gate_failures": [
            "r320_performance_non_regression",
            "r320_phase_aware_memory_non_regression",
        ],
    }


def test_stage_o42_read_only_runtime_snapshots_cross_json_boundary():
    reuse = MappingProxyType(
        {
            "schema_version": 1,
            "generation": 4,
            "retained_pairs_after_generation": 0,
        }
    )
    materialization = MappingProxyType(
        {
            "schema_version": 1,
            "generation": 4,
            "physical_materializations": 29,
            "physical_materialization_batches": 4,
        }
    )

    report_fragment = {
        "algebraic_representation_reuse": _plain_diagnostic_snapshot(reuse),
        "algebraic_physical_materialization": _plain_diagnostic_snapshot(
            materialization
        ),
        "disabled_snapshot": _plain_diagnostic_snapshot(None),
    }

    assert json.loads(json.dumps(report_fragment, allow_nan=False)) == {
        "algebraic_representation_reuse": dict(reuse),
        "algebraic_physical_materialization": dict(materialization),
        "disabled_snapshot": None,
    }


def test_stage_o42_analysis_ranks_measured_owner_and_operator_deltas(tmp_path):
    legacy = _write_json(tmp_path / "legacy.json", _profile("legacy", scale=1))
    canary = _write_json(tmp_path / "canary.json", _profile("canary", scale=2))
    o41 = _write_json(tmp_path / "o41.json", _o41())

    report = analyze_stage_o42_diagnostics(
        legacy,
        canary,
        stage_o41_report=o41,
        expected_stage_o41_sha256=_sha256(o41),
    )

    assert report["classification"] == "DIAGNOSTIC_COMPLETE"
    assert report["allocator_block_reconciliation_complete"] is True
    assert report["allocator_block_accounting_reconciled"] is True
    assert report["storage_identity_priming_stable"] is True
    assert report["accounting_ready_for_optimization"] is True
    assert report["eligible_for_stage_o43_optimization_design"] is True
    assert report["eligible_for_production_promotion"] is False
    phase = report["phase_comparison"]["after_warmup"]
    assert phase["allocator_ratio_canary_over_legacy"] == 2.0
    assert phase["ranked_positive_category_deltas"] == [
        "solver_fields",
        "model",
    ]
    assert report["ranked_positive_operator_deltas"][0]["operator"] == ("aten::stack")


def test_stage_o42_analysis_rejects_truncated_inventory(tmp_path):
    legacy_report = _profile("legacy", scale=1)
    legacy_report["residency_phases"]["after_timestep"]["inventory"]["truncated"] = True
    legacy = _write_json(tmp_path / "legacy.json", legacy_report)
    canary = _write_json(tmp_path / "canary.json", _profile("canary", scale=2))
    o41 = _write_json(tmp_path / "o41.json", _o41())

    with pytest.raises(ValueError, match="after_timestep"):
        analyze_stage_o42_diagnostics(
            legacy,
            canary,
            stage_o41_report=o41,
            expected_stage_o41_sha256=_sha256(o41),
        )


def test_stage_o42_analysis_rejects_unstable_repeated_inventory(tmp_path):
    legacy_report = _profile("legacy", scale=1)
    phase = legacy_report["residency_phases"]["after_warmup"]
    phase["storage_identity_priming"][
        "identical_to_measured_inventory"
    ] = False
    legacy = _write_json(tmp_path / "legacy.json", legacy_report)
    canary = _write_json(tmp_path / "canary.json", _profile("canary", scale=2))
    o41 = _write_json(tmp_path / "o41.json", _o41())

    with pytest.raises(ValueError, match="after_warmup"):
        analyze_stage_o42_diagnostics(
            legacy,
            canary,
            stage_o41_report=o41,
            expected_stage_o41_sha256=_sha256(o41),
        )


def test_stage_o42_analysis_reports_reference_expansion_without_eligibility(tmp_path):
    canary_report = _profile("canary", scale=2)
    phase = canary_report["residency_phases"]["after_warmup"]
    priming = phase["storage_identity_priming_inventory"]
    measured = copy.deepcopy(priming)
    added = copy.deepcopy(measured["tensor_references"][0])
    added["path"] = "runtime.representation.transient_cache.alias"
    added["category"] = "algebraic_representations"
    measured["tensor_references"].append(added)
    measured["tensor_references"].sort(key=lambda item: item["path"])
    measured["tensor_reference_count"] += 1
    measured["logical_tensor_bytes"] += added["logical_bytes"]
    measured["aliased_tensor_reference_count"] += 1
    measured["storages"][0]["paths"].append(added["path"])
    measured["storages"][0]["paths"].sort()
    measured["storages"][0]["categories"].append(added["category"])
    measured["storages"][0]["categories"].sort()
    measured["storages"][0]["tensor_references"] += 1
    measured["exclusive_storage_bytes_by_category"] = {}
    measured["shared_storage_bytes_by_category"] = {
        "algebraic_representations": measured["unique_storage_bytes"],
        "model": measured["unique_storage_bytes"],
    }
    verification = copy.deepcopy(measured)
    phase["inventory"] = verification
    phase["storage_identity_measured_inventory"] = measured
    phase["storage_identity_verification_inventory"] = verification
    comparison = _compare_inventories(priming, measured)
    repeated = _compare_inventories(measured, verification)
    phase["storage_identity_inventory_comparison"] = comparison
    phase["storage_identity_repeated_measurement_comparison"] = repeated
    phase["storage_identity_priming"][
        "identical_to_measured_inventory"
    ] = False
    phase["storage_identity_inventory_stable"] = False
    phase["storage_identity_set_stable"] = True
    phase["storage_identity_post_priming_stable"] = True
    phase["storage_identity_observer_effect_explained"] = False

    legacy = _write_json(tmp_path / "legacy.json", _profile("legacy", scale=1))
    canary = _write_json(tmp_path / "canary.json", canary_report)
    o41 = _write_json(tmp_path / "o41.json", _o41())
    report = analyze_stage_o42_diagnostics(
        legacy,
        canary,
        stage_o41_report=o41,
        expected_stage_o41_sha256=_sha256(o41),
    )

    assert report["storage_identity_priming_stable"] is False
    assert report["storage_identity_set_stable"] is True
    assert report["storage_identity_reference_graph_stable"] is False
    assert report["storage_identity_observer_effect_explained"] is False
    assert report["accounting_ready_for_optimization"] is False
    assert report["eligible_for_stage_o43_optimization_design"] is False
    comparison = report["phase_comparison"]["after_warmup"][
        "canary_storage_identity_inventory_comparison"
    ]
    assert comparison["classification"] == (
        "transient_cache_reference_expansion"
    )
    assert comparison["added_tensor_reference_count"] == 1


def test_stage_o42_analysis_keeps_unexplained_reference_change_non_eligible(
    tmp_path,
):
    canary_report = _profile("canary", scale=2)
    phase = canary_report["residency_phases"]["after_warmup"]
    priming = phase["storage_identity_priming_inventory"]
    measured = copy.deepcopy(priming)
    added = copy.deepcopy(measured["tensor_references"][0])
    added["path"] = "runtime.model.unexplained_alias"
    measured["tensor_references"].append(added)
    measured["tensor_references"].sort(key=lambda item: item["path"])
    measured["tensor_reference_count"] += 1
    measured["logical_tensor_bytes"] += added["logical_bytes"]
    measured["aliased_tensor_reference_count"] += 1
    measured["storages"][0]["paths"].append(added["path"])
    measured["storages"][0]["paths"].sort()
    measured["storages"][0]["tensor_references"] += 1
    verification = copy.deepcopy(measured)
    phase["inventory"] = verification
    phase["storage_identity_measured_inventory"] = measured
    phase["storage_identity_verification_inventory"] = verification
    comparison = _compare_inventories(priming, measured)
    repeated = _compare_inventories(measured, verification)
    phase["storage_identity_inventory_comparison"] = comparison
    phase["storage_identity_repeated_measurement_comparison"] = repeated
    phase["storage_identity_priming"][
        "identical_to_measured_inventory"
    ] = False
    phase["storage_identity_inventory_stable"] = False
    phase["storage_identity_set_stable"] = True
    phase["storage_identity_post_priming_stable"] = True
    phase["storage_identity_observer_effect_explained"] = False

    legacy = _write_json(tmp_path / "legacy.json", _profile("legacy", scale=1))
    canary = _write_json(tmp_path / "canary.json", canary_report)
    o41 = _write_json(tmp_path / "o41.json", _o41())
    report = analyze_stage_o42_diagnostics(
        legacy,
        canary,
        stage_o41_report=o41,
        expected_stage_o41_sha256=_sha256(o41),
    )

    assert report["storage_identity_set_stable"] is True
    assert report["storage_identity_observer_effect_explained"] is False
    assert report["accounting_ready_for_optimization"] is False
    assert report["eligible_for_stage_o43_optimization_design"] is False


def test_stage_o42_analysis_persists_accounting_ambiguity_without_promotion(
    tmp_path,
):
    legacy = _write_json(tmp_path / "legacy.json", _profile("legacy", scale=1))
    canary_report = _profile("canary", scale=2)
    phase = canary_report["residency_phases"]["after_warmup"]
    phase["allocator_minus_inventory_bytes"] = -25
    phase["inventory_exceeds_allocator_bytes"] = 25
    phase["storage_accounting_consistent"] = False
    canary = _write_json(tmp_path / "canary.json", canary_report)
    o41 = _write_json(tmp_path / "o41.json", _o41())

    report = analyze_stage_o42_diagnostics(
        legacy,
        canary,
        stage_o41_report=o41,
        expected_stage_o41_sha256=_sha256(o41),
    )

    assert report["classification"] == "DIAGNOSTIC_COMPLETE"
    assert report["storage_accounting_consistent"] is False
    assert report["accounting_ready_for_optimization"] is False
    assert report["eligible_for_stage_o43_optimization_design"] is False
    assert report["architecture_decision"] == (
        "refine_storage_accounting_before_optimization"
    )


def test_stage_o42_plan_is_bounded_read_only_and_non_promoting(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    o41 = _write_json(tmp_path / "o41.json", _o41())

    plan = build_stage_o42_h100_plan(
        project_root=project,
        control_root=tmp_path / "control",
        python=python,
        production_reference_dir=reference,
        expected_commit="c" * 40,
        stage_o41_report=o41,
        expected_stage_o41_sha256=_sha256(o41),
    )

    assert plan["planning_only"] is True
    assert plan["diagnostic_contract"]["profile_count"] == 2
    assert plan["diagnostic_contract"]["runtime_roots_only"] is True
    assert plan["diagnostic_contract"]["allocator_block_reconciliation"] is True
    assert plan["diagnostic_contract"]["allocator_counter_stability_checked"] is True
    assert plan["diagnostic_contract"]["storage_identity_priming_pass"] is True
    assert plan["diagnostic_contract"]["repeated_inventory_identity_checked"] is True
    assert plan["diagnostic_contract"]["priming_inventory_preserved"] is True
    assert plan["diagnostic_contract"][
        "canonical_inventory_hashes_preserved"
    ] is True
    assert plan["diagnostic_contract"]["path_level_inventory_diff_preserved"] is True
    assert plan["diagnostic_contract"][
        "post_priming_inventory_verification"
    ] is True
    assert plan["diagnostic_contract"]["custom_mapping_getitem_forbidden"] is True
    assert plan["diagnostic_contract"]["global_gc_traversal"] is False
    assert plan["diagnostic_contract"]["production_default_may_change"] is False
    assert [item["role"] for item in plan["commands"]["profiles"]] == [
        "legacy",
        "canary",
    ]
    assert "--stage-o41-report" in plan["commands"]["analysis"]
