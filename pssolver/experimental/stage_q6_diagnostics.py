"""Read-only Stage Q.6 map of algebraic data movement and layout contracts.

Stage Q.6 consumes the frozen Q.5 decision and the O.4.3.4 attribution
report.  It joins measured copy sources to static producer, scheduler, and
consumer contracts.  The module never imports a production entry point,
constructs a solver, reads trajectory arrays, or authorizes an implementation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

from ._shadow_support import file_sha256
from .h100_shadow_qualification import _write_new_json
from .stage_q2_diagnostics import _read_unchanged_json
from .stage_q5_diagnostics import (
    STAGE_Q5_CLASSIFICATION,
    STAGE_Q5_PRIMARY_TARGET,
)


STAGE_Q6_CLASSIFICATION = "DATA_MOVEMENT_MAP_COMPLETE"
STAGE_Q6_ARCHITECTURE_DECISION = (
    "boundary_signature_native_handoff_design_review"
)
STAGE_Q6_REQUIRED_DISTINCTIONS = (
    "allocation",
    "tensor_copy",
    "view_or_alias",
    "transform_input_layout",
    "consumer_required_layout",
    "operator_launch_fragmentation",
)


_SOURCE_CONTRACTS = MappingProxyType(
    {
        "algebraic.nematic_stress.dependencies": {
            "producer": {
                "owner": "AlgebraicGenerationState",
                "representation": (
                    "named spectral component tensors for physical-cache "
                    "misses among Q, H, and Q-gradient dependencies"
                ),
                "storage_layout": (
                    "independent component tensors; shared contiguous "
                    "boundary-packed storage is not guaranteed"
                ),
            },
            "handoff": {
                "scheduler": "AlgebraicPhysicalIslandScheduler",
                "scheduler_entry": "materialize_dependencies",
                "direction": "inverse_projected",
                "grouping": "homogeneous projected-boundary signature",
                "assembly": "copy_cat",
                "transform_input_layout": (
                    "leading dimension packs all cache-miss components in "
                    "one boundary-signature group"
                ),
                "post_transform_layout": (
                    "split views, one physical tensor per named component"
                ),
            },
            "consumer": {
                "owner": "_StressSolver",
                "required_layout": (
                    "name-addressable physical Q, H, and Q-gradient tensors"
                ),
                "lifetime": "one active algebraic generation",
            },
            "allocation": {
                "kind": "logical transient packed-batch output",
                "interpretation": (
                    "copy_cat creates a packed result tensor; allocator reuse "
                    "must not be confused with eliminating tensor movement"
                ),
            },
            "tensor_copy": {
                "present": True,
                "reason": (
                    "cache-miss spectra are separate tensors and are copied "
                    "into the boundary-signature transform batch"
                ),
            },
            "view_or_alias": {
                "before_transform": "not_proven_from_current_storage",
                "after_transform": "split_views_of_transformed_batch",
                "zero_copy_currently_exercised": False,
            },
            "operator_launch_fragmentation": {
                "source": (
                    "one copy/cat operation for each multi-component "
                    "boundary-signature batch, followed by projected transforms"
                ),
                "timing_is_nonadditive": True,
            },
            "code_contracts": (
                "model_execution.LegacyAlgebraicFieldsAdapter.forward",
                "representations.AlgebraicGenerationState.prefetch_physical",
                "projected_scheduler.AlgebraicPhysicalIslandScheduler."
                "materialize_dependencies",
                "projected_scheduler.BoundarySignatureTransformScheduler."
                "_assemble_batch",
                "beris_edwards._StressSolver.solve_spectral",
            ),
        },
        "explicit_rhs.dependencies": {
            "producer": {
                "owner": "AlgebraicGenerationState",
                "representation": (
                    "named transient algebraic spectra required by the "
                    "physical explicit-RHS island"
                ),
                "storage_layout": (
                    "independent component tensors; some producers may expose "
                    "packed views, but this source still records copy_cat"
                ),
            },
            "handoff": {
                "scheduler": "AlgebraicPhysicalIslandScheduler",
                "scheduler_entry": "materialize_dependencies",
                "direction": "inverse_projected",
                "grouping": "homogeneous projected-boundary signature",
                "assembly": "copy_cat",
                "transform_input_layout": (
                    "leading dimension packs missing transient spectra by "
                    "boundary signature"
                ),
                "post_transform_layout": (
                    "split views cached and exposed by component name"
                ),
            },
            "consumer": {
                "owner": "LegacyExplicitRHSAdapter",
                "required_layout": (
                    "name-addressable physical state combining evolved, "
                    "static, and transient fields"
                ),
                "lifetime": "current explicit-RHS evaluation/generation",
            },
            "allocation": {
                "kind": "logical transient packed-batch output",
                "interpretation": (
                    "the packed tensor is temporary even when the allocator "
                    "reuses cached device memory"
                ),
            },
            "tensor_copy": {
                "present": True,
                "reason": (
                    "separate transient spectra are assembled before their "
                    "batched inverse projected transform"
                ),
            },
            "view_or_alias": {
                "before_transform": "not_proven_from_current_storage",
                "after_transform": "split_views_cached_by_component_name",
                "zero_copy_currently_exercised": False,
            },
            "operator_launch_fragmentation": {
                "source": (
                    "copy/cat plus inverse-transform batches are partitioned "
                    "by projected-boundary signature"
                ),
                "timing_is_nonadditive": True,
            },
            "code_contracts": (
                "model_execution.LegacyAlgebraicFieldsAdapter."
                "prefetch_physical_components",
                "representations.AlgebraicGenerationState.prefetch_physical",
                "projected_scheduler.AlgebraicPhysicalIslandScheduler."
                "materialize_dependencies",
                "projected_scheduler.BoundarySignatureTransformScheduler."
                "_assemble_batch",
                "model_execution.LegacyExplicitRHSAdapter.forward",
            ),
        },
        "explicit_rhs.outputs": {
            "producer": {
                "owner": "explicit-RHS evaluator",
                "representation": "one physical tensor per evolved component",
                "storage_layout": (
                    "component mapping/tuple in model order without a required "
                    "shared boundary-packed base allocation"
                ),
            },
            "handoff": {
                "scheduler": "AlgebraicPhysicalIslandScheduler",
                "scheduler_entry": "project_output_values",
                "direction": "forward_projected",
                "grouping": "homogeneous projected-boundary signature",
                "assembly": "copy_cat",
                "transform_input_layout": (
                    "leading dimension packs physical RHS outputs by boundary "
                    "signature"
                ),
                "post_transform_layout": (
                    "split spectral views reordered to model component order"
                ),
            },
            "consumer": {
                "owner": "projected semi-implicit integrator",
                "required_layout": (
                    "one stacked spectral RHS tensor in evolved-component order"
                ),
                "lifetime": "one explicit-RHS/integrator handoff",
            },
            "allocation": {
                "kind": "logical transient packed-batch output",
                "interpretation": (
                    "batch assembly and final spectral publication are distinct "
                    "layout boundaries"
                ),
            },
            "tensor_copy": {
                "present": True,
                "reason": (
                    "separate physical RHS outputs are assembled before their "
                    "batched forward projected transform"
                ),
            },
            "view_or_alias": {
                "before_transform": "not_proven_from_current_storage",
                "after_transform": "split_views_then_stacked_for_publication",
                "zero_copy_currently_exercised": False,
            },
            "operator_launch_fragmentation": {
                "source": (
                    "copy/cat and forward-transform batches are partitioned by "
                    "boundary signature; publication is a separate boundary"
                ),
                "timing_is_nonadditive": True,
            },
            "code_contracts": (
                "model_execution.LegacyExplicitRHSAdapter.forward",
                "projected_scheduler.AlgebraicPhysicalIslandScheduler."
                "project_output_values",
                "projected_scheduler.BoundarySignatureTransformScheduler."
                "_assemble_batch",
                "model_execution.LegacyExplicitRHSAdapter.forward:"
                "publish_spectral",
            ),
        },
    }
)


def _finite_nonnegative(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{description} must be finite and nonnegative")
    return float(value)


def _load_frozen_report(
    path: str | Path,
    *,
    expected_sha256: str,
    description: str,
) -> tuple[Path, dict[str, object]]:
    resolved, digest, report = _read_unchanged_json(path, description)
    if digest != expected_sha256:
        raise ValueError(f"{description} SHA-256 differs")
    return resolved, report


def _contract_identity() -> str:
    payload = json.dumps(
        dict(_SOURCE_CONTRACTS),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_q5_contract(
    report: Mapping[str, object],
    *,
    expected_stage_o434_sha256: str,
) -> tuple[str, ...]:
    try:
        scope = report["stage_q6_scope"]
        sources = tuple(scope["must_cover_sources"])
        distinctions = tuple(scope["must_distinguish"])
        o434_inputs = [
            value
            for value in report["inputs"]
            if value.get("kind") == "stage_o434"
        ]
        valid = bool(
            report["qualification_stage"] == "Q.5"
            and report["classification"] == STAGE_Q5_CLASSIFICATION
            and report["primary_target"] == STAGE_Q5_PRIMARY_TARGET
            and scope["action"]
            == "define_one_read_only_end_to_end_data_movement_map"
            and set(distinctions) == set(STAGE_Q6_REQUIRED_DISTINCTIONS)
            and len(distinctions) == len(STAGE_Q6_REQUIRED_DISTINCTIONS)
            and scope["may_reuse_existing_profiles"] is True
            and scope["may_execute_solver"] is False
            and scope["may_implement_candidate"] is False
            and scope["may_change_production_default"] is False
            and report["eligible_for_stage_q6_diagnostic_design"] is True
            and report["eligible_for_stage_q6_candidate_implementation"]
            is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
            and len(o434_inputs) == 1
            and o434_inputs[0]["sha256"] == expected_stage_o434_sha256
        )
    except (KeyError, TypeError):
        valid = False
        sources = ()
    if not valid:
        raise ValueError("Stage Q.6 Q.5 evidence contract differs")
    if len(sources) != len(set(sources)):
        raise ValueError("Stage Q.5 source scope contains duplicates")
    if set(sources) != set(_SOURCE_CONTRACTS):
        raise ValueError("Stage Q.5 source scope is not fully mapped")
    return sources


def _require_o434_contract(
    report: Mapping[str, object],
    *,
    expected_sources: tuple[str, ...],
) -> list[dict[str, object]]:
    try:
        rows = report["remaining_copy_sources"]
        valid = bool(
            report["qualification_stage"] == "O.4.3.4"
            and report["classification"] == "DIAGNOSTIC_COMPLETE"
            and report["architecture_decision"]
            == "remaining_materialization_attribution"
            and report["dominant_source_meets_screening_signal"] is False
            and report["eligible_for_target_selection_review"] is True
            and report["eligible_for_new_optimization_candidate"] is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
            and isinstance(rows, Sequence)
            and not isinstance(rows, (str, bytes))
            and bool(rows)
        )
    except (KeyError, TypeError):
        valid = False
        rows = ()
    if not valid:
        raise ValueError("Stage Q.6 O.4.3.4 evidence contract differs")

    normalized: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Stage O.4.3.4 source row is invalid")
        source = row.get("source")
        if not isinstance(source, str):
            raise ValueError("Stage O.4.3.4 source name is invalid")
        copy_batches = _finite_nonnegative(
            row.get("copy_cat_batches_per_step"),
            f"{source} copy batches",
        )
        copy_components = _finite_nonnegative(
            row.get("copy_cat_components_per_step"),
            f"{source} copy components",
        )
        copy_bytes = _finite_nonnegative(
            row.get("copy_cat_materialized_output_bytes_per_step"),
            f"{source} materialized bytes",
        )
        if min(copy_batches, copy_components, copy_bytes) <= 0.0:
            raise ValueError(f"{source} is not a remaining copy source")
        normalized.append(
            {
                "source": source,
                "rank": len(normalized) + 1,
                "mean_milliseconds_per_step": _finite_nonnegative(
                    row.get("mean_milliseconds_per_step"),
                    f"{source} mean time",
                ),
                "mean_fraction_of_timestep": _finite_nonnegative(
                    row.get("mean_fraction_of_timestep"),
                    f"{source} timestep fraction",
                ),
                "copy_cat_batches_per_step": copy_batches,
                "copy_cat_components_per_step": copy_components,
                "copy_cat_materialized_output_bytes_per_step": copy_bytes,
                "contiguous_view_batches_per_step": _finite_nonnegative(
                    row.get("contiguous_view_batches_per_step", 0.0),
                    f"{source} contiguous-view batches",
                ),
                "fallback_reasons": dict(row.get("fallback_reasons", {})),
            }
        )
    actual = tuple(row["source"] for row in normalized)
    if len(actual) != len(set(actual)) or set(actual) != set(expected_sources):
        raise ValueError("Stage O.4.3.4 source set differs from Stage Q.5")
    return normalized


def analyze_stage_q6_data_movement_map(
    *,
    stage_q5_report: str | Path,
    expected_stage_q5_sha256: str,
    stage_o434_report: str | Path,
    expected_stage_o434_sha256: str,
) -> dict[str, object]:
    """Join measured copy sources to static runtime layout contracts."""

    q5_path, q5 = _load_frozen_report(
        stage_q5_report,
        expected_sha256=expected_stage_q5_sha256,
        description="Stage Q.5 report",
    )
    o434_path, o434 = _load_frozen_report(
        stage_o434_report,
        expected_sha256=expected_stage_o434_sha256,
        description="Stage O.4.3.4 report",
    )
    q5_sources = _require_q5_contract(
        q5,
        expected_stage_o434_sha256=expected_stage_o434_sha256,
    )
    measurements = _require_o434_contract(
        o434,
        expected_sources=q5_sources,
    )

    source_maps = []
    for measurement in measurements:
        source = str(measurement["source"])
        contract = deepcopy(_SOURCE_CONTRACTS[source])
        contract.update(
            source=source,
            measurement=measurement,
            transform_input_layout=contract["handoff"][
                "transform_input_layout"
            ],
            consumer_required_layout=contract["consumer"][
                "required_layout"
            ],
        )
        source_maps.append(contract)

    total_time = math.fsum(
        float(value["measurement"]["mean_milliseconds_per_step"])
        for value in source_maps
    )
    total_fraction = math.fsum(
        float(value["measurement"]["mean_fraction_of_timestep"])
        for value in source_maps
    )
    total_bytes = math.fsum(
        float(
            value["measurement"][
                "copy_cat_materialized_output_bytes_per_step"
            ]
        )
        for value in source_maps
    )

    return {
        "schema_version": 1,
        "qualification_stage": "Q.6",
        "classification": STAGE_Q6_CLASSIFICATION,
        "architecture_decision": STAGE_Q6_ARCHITECTURE_DECISION,
        "primary_target": STAGE_Q5_PRIMARY_TARGET,
        "source_contract_identity_sha256": _contract_identity(),
        "required_distinctions": list(STAGE_Q6_REQUIRED_DISTINCTIONS),
        "data_movement_map": source_maps,
        "aggregate_measured_evidence": {
            "source_count": len(source_maps),
            "mean_milliseconds_per_step": total_time,
            "mean_fraction_of_timestep": total_fraction,
            "copy_cat_materialized_output_bytes_per_step": total_bytes,
            "measured_subregions_are_not_additive_speedup": True,
            "no_speedup_prediction_is_made": True,
        },
        "cross_source_findings": {
            "shared_root_mechanism": (
                "boundary-signature transform batches are assembled from "
                "independently owned component tensors"
            ),
            "shared_scheduler": "BoundarySignatureTransformScheduler",
            "allocation_only_is_not_the_root_fix": True,
            "copy_cat_is_present_at_every_remaining_source": True,
            "pre_transform_zero_copy_view_is_not_proven": True,
            "projected_transforms_remain_required_numerical_operations": True,
            "consumer_contracts_differ": True,
            "directions_present": ["forward_projected", "inverse_projected"],
            "boundary_signatures_must_remain_separate": True,
            "operator_and_kernel_durations_may_overlap": True,
        },
        "stage_q61_design_boundary": {
            "design_target": "boundary_signature_native_handoff_contract",
            "goal": (
                "determine whether producers can expose generation-safe, "
                "boundary-signature-packed storage that each consumer accepts "
                "without copy_cat or semantic reordering"
            ),
            "must_preserve": [
                "equations and coefficients",
                "projected transform definitions and boundary parity",
                "component naming and consumer-visible order",
                "generation invalidation and restart semantics",
                "float64 numerical qualification path",
            ],
            "must_prove_before_implementation": [
                "one ownership and lifetime rule for packed storage",
                "no packing across incompatible boundary signatures",
                "consumer access without hidden repacking",
                "explicit-RHS publication order without an added copy",
                "bounded peak-memory behavior",
            ],
            "may_execute_solver": False,
            "may_implement_candidate": False,
            "may_change_production_default": False,
        },
        "eligible_for_stage_q61_design": True,
        "eligible_for_stage_q61_candidate_implementation": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "changes_equations": False,
        "changes_runtime_implementation": False,
        "inputs": [
            {
                "kind": "stage_q5_report",
                "path": str(q5_path),
                "sha256": file_sha256(q5_path),
            },
            {
                "kind": "stage_o434_report",
                "path": str(o434_path),
                "sha256": file_sha256(o434_path),
            },
        ],
    }


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the read-only Stage Q.6 algebraic data-movement map"
    )
    parser.add_argument("--stage-q5-report", type=Path, required=True)
    parser.add_argument("--expected-stage-q5-sha256", required=True)
    parser.add_argument("--stage-o434-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o434-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_q6_data_movement_map(
        stage_q5_report=args.stage_q5_report,
        expected_stage_q5_sha256=args.expected_stage_q5_sha256,
        stage_o434_report=args.stage_o434_report,
        expected_stage_o434_sha256=args.expected_stage_o434_sha256,
    )
    _write_new_json(args.output, report)
    return 0


__all__ = [
    "STAGE_Q6_ARCHITECTURE_DECISION",
    "STAGE_Q6_CLASSIFICATION",
    "STAGE_Q6_REQUIRED_DISTINCTIONS",
    "analysis_main",
    "analyze_stage_q6_data_movement_map",
]
