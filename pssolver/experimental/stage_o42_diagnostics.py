"""Stage O.4.2 tensor-residency and operator-allocation diagnostics.

This stage is diagnostic only.  It compares the already qualified legacy and
separated Plane runtimes at R320 without changing either runtime or declaring
a production promotion.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

from .h100_shadow_qualification import _write_new_json
from .stage_o4_qualification import _load_json, _sha256

STAGE_O42_SHAPE = (320, 320, 80)


def _validated_profile(path: str | Path, role: str) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    report = _load_json(resolved, f"Stage O.4.2 {role} diagnostic")
    if (
        report.get("qualification_stage") != "O.4.2"
        or report.get("measurement_role") != role
        or report.get("classification") != "DIAGNOSTIC_COMPLETE"
        or report.get("finite") is not True
        or report.get("eligible_for_production_promotion") is not False
        or report.get("production_default_changed") is not False
    ):
        raise ValueError(f"Stage O.4.2 {role} diagnostic is incomplete")
    try:
        config_valid = tuple(report["configuration"]["shape"]) == STAGE_O42_SHAPE
        contract_valid = (
            tuple(report["diagnostic_contract"]["shape"]) == STAGE_O42_SHAPE
            and report["diagnostic_contract"]["global_gc_traversal"] is False
            and report["diagnostic_contract"]["runtime_roots_only"] is True
        )
        audit_valid = int(report["operator_audit"]["steps"]) > 0 and isinstance(
            report["operator_audit"]["operators"], Mapping
        )
    except (KeyError, TypeError, ValueError):
        config_valid = contract_valid = audit_valid = False
    if not config_valid or not contract_valid or not audit_valid:
        raise ValueError(f"Stage O.4.2 {role} diagnostic contract differs")
    phases = report.get("residency_phases")
    if not isinstance(phases, Mapping):
        raise ValueError(f"Stage O.4.2 {role} residency phases are missing")
    for name in ("after_warmup", "after_timestep", "after_observation"):
        try:
            inventory = phases[name]["inventory"]
            gap = int(phases[name]["allocator_minus_inventory_bytes"])
            exceeds = int(phases[name]["inventory_exceeds_allocator_bytes"])
            consistent = phases[name]["storage_accounting_consistent"]
            valid = (
                inventory["schema_version"] == 1
                and inventory["truncated"] is False
                and inventory["all_storages_reported"] is True
                and isinstance(consistent, bool)
                and exceeds == max(0, -gap)
                and consistent is (gap >= 0)
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"Stage O.4.2 {role} phase {name!r} is invalid")
    return resolved, report


def _configuration_identity(report: Mapping[str, object]) -> str:
    config = report["configuration"]
    normalized = {
        "shape": config["shape"],
        "lengths": config["lengths"],
        "dtype": config["dtype"],
        "dt": config["dt"],
        "dealias_rule": config["dealias_rule"],
        "projected_transform_execution": config["projected_transform_execution"],
        "spectral_storage": config["spectral_storage"],
        "spectral_refresh_interval": config["spectral_refresh_interval"],
    }
    return json.dumps(normalized, allow_nan=False, sort_keys=True)


def _phase_comparison(
    legacy: Mapping[str, object],
    canary: Mapping[str, object],
) -> dict[str, object]:
    result = {}
    for name in ("after_warmup", "after_timestep", "after_observation"):
        l_phase = legacy["residency_phases"][name]
        c_phase = canary["residency_phases"][name]
        l_alloc = int(l_phase["allocator"]["allocated_bytes"])
        c_alloc = int(c_phase["allocator"]["allocated_bytes"])
        l_unique = int(l_phase["inventory"]["unique_storage_bytes"])
        c_unique = int(c_phase["inventory"]["unique_storage_bytes"])
        l_inventory = l_phase["inventory"]
        c_inventory = c_phase["inventory"]
        categories = sorted(
            set(l_inventory["exclusive_storage_bytes_by_category"])
            | set(c_inventory["exclusive_storage_bytes_by_category"])
            | set(l_inventory.get("shared_storage_bytes_by_category", {}))
            | set(c_inventory.get("shared_storage_bytes_by_category", {}))
        )

        def referenced(inventory: Mapping[str, object], category: str) -> int:
            return int(
                inventory["exclusive_storage_bytes_by_category"].get(category, 0)
            ) + int(
                inventory.get("shared_storage_bytes_by_category", {}).get(category, 0)
            )

        category_deltas = {
            category: referenced(c_inventory, category)
            - referenced(l_inventory, category)
            for category in categories
        }
        result[name] = {
            "legacy_allocator_allocated_bytes": l_alloc,
            "canary_allocator_allocated_bytes": c_alloc,
            "allocator_delta_bytes": c_alloc - l_alloc,
            "allocator_ratio_canary_over_legacy": c_alloc / l_alloc,
            "legacy_inventory_unique_storage_bytes": l_unique,
            "canary_inventory_unique_storage_bytes": c_unique,
            "inventory_delta_bytes": c_unique - l_unique,
            "inventory_ratio_canary_over_legacy": c_unique / l_unique,
            "legacy_storage_accounting_consistent": l_phase[
                "storage_accounting_consistent"
            ],
            "canary_storage_accounting_consistent": c_phase[
                "storage_accounting_consistent"
            ],
            "legacy_inventory_exceeds_allocator_bytes": l_phase[
                "inventory_exceeds_allocator_bytes"
            ],
            "canary_inventory_exceeds_allocator_bytes": c_phase[
                "inventory_exceeds_allocator_bytes"
            ],
            "category_referenced_storage_delta_bytes": category_deltas,
            "ranked_positive_category_deltas": [
                category
                for category, delta in sorted(
                    category_deltas.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if delta > 0
            ],
        }
    return result


def _semantic_ranking(
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    semantic = report["semantic_regions"]
    regions = semantic["regions"]
    steps = int(
        semantic.get("steps", report["diagnostic_contract"]["diagnostic_steps"])
    )
    ranked = []
    for name, raw in regions.items():
        total = float(raw["total_seconds"])
        if not math.isfinite(total) or total < 0.0:
            raise ValueError(f"semantic region {name!r} has an invalid duration")
        ranked.append(
            {
                "region": name,
                "calls": int(raw["calls"]),
                "seconds_per_diagnostic_step": total / steps,
            }
        )
    ranked.sort(
        key=lambda item: (-float(item["seconds_per_diagnostic_step"]), item["region"])
    )
    return ranked


def _operator_comparison(
    legacy: Mapping[str, object],
    canary: Mapping[str, object],
) -> dict[str, object]:
    l_audit = legacy["operator_audit"]
    c_audit = canary["operator_audit"]
    names = sorted(set(l_audit["operators"]) | set(c_audit["operators"]))
    result = {}
    for name in names:
        l = l_audit["operators"][name]
        c = c_audit["operators"][name]
        l_steps = int(l_audit["steps"])
        c_steps = int(c_audit["steps"])
        l_calls = int(l["calls"]) / l_steps
        c_calls = int(c["calls"]) / c_steps
        l_memory = int(l["self_device_memory_bytes"]) / l_steps
        c_memory = int(c["self_device_memory_bytes"]) / c_steps
        result[name] = {
            "legacy_calls_per_step": l_calls,
            "canary_calls_per_step": c_calls,
            "call_delta_per_step": c_calls - l_calls,
            "legacy_self_device_memory_bytes_per_step": l_memory,
            "canary_self_device_memory_bytes_per_step": c_memory,
            "self_device_memory_delta_bytes_per_step": c_memory - l_memory,
        }
    return result


def analyze_stage_o42_diagnostics(
    legacy_profile: str | Path,
    canary_profile: str | Path,
    *,
    stage_o41_report: str | Path,
    expected_stage_o41_sha256: str,
) -> dict[str, object]:
    """Compare the two read-only diagnostics and rank measured deltas."""

    o41_path = Path(stage_o41_report).expanduser().resolve()
    if _sha256(o41_path) != expected_stage_o41_sha256:
        raise ValueError("Stage O.4.1 report SHA-256 differs from expectation")
    o41 = _load_json(o41_path, "Stage O.4.1 qualification")
    if (
        o41.get("qualification_stage") != "O.4.1"
        or o41.get("classification") != "B_neutral"
        or set(o41.get("gate_failures", ()))
        != {
            "r320_performance_non_regression",
            "r320_phase_aware_memory_non_regression",
        }
    ):
        raise ValueError("input is not the expected Stage O.4.1 evidence")
    legacy_path, legacy = _validated_profile(legacy_profile, "legacy")
    canary_path, canary = _validated_profile(canary_profile, "canary")
    identities = {
        (
            report["production_metadata_sha256"],
            report["production_initial_q_sha256"],
            _configuration_identity(report),
        )
        for report in (legacy, canary)
    }
    if len(identities) != 1:
        raise ValueError("Stage O.4.2 diagnostic identities differ")
    phase_comparison = _phase_comparison(legacy, canary)
    storage_accounting_consistent = all(
        phase["legacy_storage_accounting_consistent"] is True
        and phase["canary_storage_accounting_consistent"] is True
        for phase in phase_comparison.values()
    )
    operator_comparison = _operator_comparison(legacy, canary)
    category_totals: dict[str, int] = {}
    for phase in phase_comparison.values():
        for category, delta in phase["category_referenced_storage_delta_bytes"].items():
            category_totals[category] = category_totals.get(category, 0) + int(delta)
    ranked_categories = [
        {"category": name, "summed_phase_delta_bytes": delta}
        for name, delta in sorted(
            category_totals.items(), key=lambda item: (-item[1], item[0])
        )
        if delta > 0
    ]
    ranked_operators = [
        {
            "operator": name,
            "call_delta_per_step": values["call_delta_per_step"],
            "self_device_memory_delta_bytes_per_step": values[
                "self_device_memory_delta_bytes_per_step"
            ],
        }
        for name, values in sorted(
            operator_comparison.items(),
            key=lambda item: (
                -item[1]["self_device_memory_delta_bytes_per_step"],
                -item[1]["call_delta_per_step"],
                item[0],
            ),
        )
        if values["call_delta_per_step"] > 0
        or values["self_device_memory_delta_bytes_per_step"] > 0
    ]
    input_hashes = {
        "stage_o41": expected_stage_o41_sha256,
        "legacy": _sha256(legacy_path),
        "canary": _sha256(canary_path),
    }
    if (
        _sha256(o41_path) != input_hashes["stage_o41"]
        or _sha256(legacy_path) != input_hashes["legacy"]
        or _sha256(canary_path) != input_hashes["canary"]
    ):
        raise RuntimeError("Stage O.4.2 input changed during analysis")
    return {
        "schema_version": 1,
        "qualification_stage": "O.4.2",
        "classification": "DIAGNOSTIC_COMPLETE",
        "architecture_decision": (
            "identify_owner_before_optimization"
            if storage_accounting_consistent
            else "refine_storage_accounting_before_optimization"
        ),
        "eligible_for_stage_o43_optimization_design": (storage_accounting_consistent),
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "stage_o41_original_classification_unchanged": True,
        "storage_accounting_consistent": storage_accounting_consistent,
        "phase_comparison": phase_comparison,
        "operator_comparison": operator_comparison,
        "ranked_positive_residency_categories": ranked_categories,
        "ranked_positive_operator_deltas": ranked_operators,
        "semantic_regions": {
            "legacy": legacy["semantic_regions"],
            "canary": canary["semantic_regions"],
        },
        "ranked_semantic_regions": {
            "legacy": _semantic_ranking(legacy),
            "canary": _semantic_ranking(canary),
        },
        "interpretation_constraints": {
            "operator_memory_is_allocator_effect_not_total_traffic": True,
            "inventory_covers_explicit_runtime_roots_only": True,
            "shared_storages_are_not_assigned_to_one_owner": True,
            "no_optimization_or_promotion_is_authorized": True,
        },
        "inputs": {
            "stage_o41": {
                "path": str(o41_path),
                "sha256": input_hashes["stage_o41"],
            },
            "legacy": {
                "path": str(legacy_path),
                "sha256": input_hashes["legacy"],
            },
            "canary": {
                "path": str(canary_path),
                "sha256": input_hashes["canary"],
            },
        },
    }


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Stage O.4.2 diagnostics")
    parser.add_argument("--legacy-profile", type=Path, required=True)
    parser.add_argument("--canary-profile", type=Path, required=True)
    parser.add_argument("--stage-o41-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o41-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_o42_diagnostics(
        args.legacy_profile,
        args.canary_profile,
        stage_o41_report=args.stage_o41_report,
        expected_stage_o41_sha256=args.expected_stage_o41_sha256,
    )
    _write_new_json(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "STAGE_O42_SHAPE",
    "analysis_main",
    "analyze_stage_o42_diagnostics",
]
