from __future__ import annotations

import hashlib
import json
from pathlib import Path
import copy

import pytest

from pssolver.experimental.stage_o42_diagnostics import (
    analyze_stage_o42_diagnostics,
)
from pssolver.experimental.stage_o42_plan import build_stage_o42_h100_plan


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(*, unique: int, allocated: int, categories: dict[str, int]):
    return {
        "allocator": {
            "allocated_bytes": allocated,
            "reserved_bytes": allocated * 2,
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": allocated * 2,
        },
        "inventory": {
            "schema_version": 1,
            "truncated": False,
            "all_storages_reported": True,
            "unique_storage_bytes": unique,
            "exclusive_storage_bytes_by_category": categories,
        },
        "allocator_minus_inventory_bytes": allocated - unique,
        "inventory_exceeds_allocator_bytes": max(0, unique - allocated),
        "storage_accounting_consistent": unique <= allocated,
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
    assert plan["diagnostic_contract"]["global_gc_traversal"] is False
    assert plan["diagnostic_contract"]["production_default_may_change"] is False
    assert [item["role"] for item in plan["commands"]["profiles"]] == [
        "legacy",
        "canary",
    ]
    assert "--stage-o41-report" in plan["commands"]["analysis"]
