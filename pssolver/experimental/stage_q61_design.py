"""Stage Q.6.1 design of a boundary-signature native handoff contract.

The stage is static and analysis-only.  It turns the frozen Stage Q.6 map into
one bounded experimental contract.  It does not define tensor-bearing runtime
types, construct a solver, or change an execution path.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from ._shadow_support import file_sha256
from .h100_shadow_qualification import _write_new_json
from .stage_q2_diagnostics import _read_unchanged_json
from .stage_q6_diagnostics import (
    STAGE_Q6_ARCHITECTURE_DECISION,
    STAGE_Q6_CLASSIFICATION,
)


STAGE_Q61_CLASSIFICATION = "NATIVE_HANDOFF_CONTRACT_COMPLETE"
STAGE_Q61_ARCHITECTURE_DECISION = (
    "producer_owned_boundary_signature_native_segments"
)
STAGE_Q61_REQUIRED_SOURCES = (
    "algebraic.nematic_stress.dependencies",
    "explicit_rhs.dependencies",
    "explicit_rhs.outputs",
)


_NATIVE_SEGMENT_CONTRACT = {
    "schema_version": 1,
    "abstraction": "BoundarySignatureNativeSegment",
    "purpose": (
        "expose a producer-owned contiguous component run directly to one "
        "projected transform without copy_cat"
    ),
    "identity_fields": [
        "producer_id",
        "generation_token",
        "representation_space",
        "projected_boundary_signature",
        "component_names",
        "component_to_segment_index",
    ],
    "tensor_layout": {
        "logical_shape": "component,batch,grid",
        "transform_view": "component_times_batch,grid",
        "strided_layout_required": True,
        "contiguous_required": True,
        "conjugate_or_negative_views_forbidden": True,
        "copy_to_construct_forbidden": True,
    },
    "ownership_and_lifetime": {
        "owner": "producing_operator_or_current_generation",
        "consumer_access": "read_only_component_views",
        "cross_generation_reuse": False,
        "checkpointed": False,
        "invalidated_with_generation": True,
        "scheduler_retains_tensor_references": False,
    },
    "scheduler_rules": {
        "may_transform_each_native_segment_independently": True,
        "may_restore_consumer_order_with_index_metadata": True,
        "may_return_split_views": True,
        "may_concatenate_native_segments": False,
        "may_copy_into_workspace": False,
        "may_pack_incompatible_boundary_signatures": False,
        "silent_fallback_to_copy_cat": False,
    },
    "semantic_invariants": [
        "component names are unchanged",
        "consumer-visible component order is unchanged",
        "projected boundary parity is unchanged",
        "forward and inverse transform definitions are unchanged",
        "equations, coefficients, and timestep policy are unchanged",
        "restart and generation invalidation semantics are unchanged",
    ],
}


_SOURCE_DESIGNS = {
    "algebraic.nematic_stress.dependencies": {
        "native_segment_producers": [
            "molecular_field",
            "q_gradient",
        ],
        "representation_space": "spectral",
        "transform_direction": "inverse_projected",
        "consumer": "_StressSolver",
        "design_action": (
            "retain producer-native spectral blocks in the active generation "
            "and transform each boundary-compatible segment without coalescing "
            "H and Q-gradient ownership"
        ),
        "consumer_handoff": (
            "publish transformed split views under the existing dependency names"
        ),
        "known_tradeoff": (
            "removing the two cross-owner copy batches may increase inverse "
            "transform batch and kernel-launch counts"
        ),
        "single_owner_for_each_segment": True,
        "single_owner_for_entire_source": False,
    },
    "explicit_rhs.dependencies": {
        "native_segment_producers": ["velocity_gradient"],
        "representation_space": "spectral",
        "transform_direction": "inverse_projected",
        "consumer": "LegacyExplicitRHSAdapter",
        "design_action": (
            "retain the velocity-gradient producer block used by the frozen "
            "execution order and expose boundary-compatible native segments"
        ),
        "consumer_handoff": (
            "cache transformed split views in the current generation mapping"
        ),
        "known_tradeoff": (
            "native segment boundaries may differ from the current two scheduled "
            "inverse batches"
        ),
        "single_owner_for_each_segment": True,
        "single_owner_for_entire_source": True,
    },
    "explicit_rhs.outputs": {
        "native_segment_producers": ["compiled_explicit_rhs_evaluator"],
        "representation_space": "physical",
        "transform_direction": "forward_projected",
        "consumer": "projected_semi_implicit_integrator",
        "design_action": (
            "produce physical RHS components in boundary-signature-compatible "
            "native segments and restore model order after transformation"
        ),
        "consumer_handoff": (
            "preserve the existing final spectral publication boundary without "
            "adding another repack"
        ),
        "known_tradeoff": (
            "the existing final spectral stack remains outside the measured "
            "copy_cat source and must not be claimed as eliminated"
        ),
        "single_owner_for_each_segment": True,
        "single_owner_for_entire_source": True,
    },
}


def _finite_positive(value: object, description: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{description} must be finite and positive")
    return float(value)


def _load_q6_report(
    path: str | Path,
    *,
    expected_sha256: str,
) -> tuple[Path, dict[str, object]]:
    resolved, digest, report = _read_unchanged_json(
        path,
        "Stage Q.6 data-movement report",
    )
    if digest != expected_sha256:
        raise ValueError("Stage Q.6 report SHA-256 differs")
    return resolved, report


def _validate_q6_contract(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    try:
        design = report["stage_q61_design_boundary"]
        rows = report["data_movement_map"]
        valid = bool(
            report["qualification_stage"] == "Q.6"
            and report["classification"] == STAGE_Q6_CLASSIFICATION
            and report["architecture_decision"]
            == STAGE_Q6_ARCHITECTURE_DECISION
            and report["eligible_for_stage_q61_design"] is True
            and report["eligible_for_stage_q61_candidate_implementation"]
            is False
            and report["eligible_for_production_promotion"] is False
            and report["production_default_changed"] is False
            and report["changes_equations"] is False
            and report["changes_runtime_implementation"] is False
            and design["design_target"]
            == "boundary_signature_native_handoff_contract"
            and design["may_execute_solver"] is False
            and design["may_implement_candidate"] is False
            and design["may_change_production_default"] is False
            and report["cross_source_findings"][
                "copy_cat_is_present_at_every_remaining_source"
            ]
            is True
            and report["cross_source_findings"][
                "projected_transforms_remain_required_numerical_operations"
            ]
            is True
            and report["cross_source_findings"][
                "boundary_signatures_must_remain_separate"
            ]
            is True
            and isinstance(rows, Sequence)
            and not isinstance(rows, (str, bytes))
        )
    except (KeyError, TypeError):
        valid = False
        rows = ()
    if not valid:
        raise ValueError("Stage Q.6 does not authorize the Q.6.1 design")

    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Stage Q.6 data-movement row is invalid")
        source = row.get("source")
        if not isinstance(source, str):
            raise ValueError("Stage Q.6 source name is invalid")
        try:
            present = row["tensor_copy"]["present"] is True
            zero_copy = row["view_or_alias"][
                "zero_copy_currently_exercised"
            ]
            measurement = row["measurement"]
            _finite_positive(
                measurement["copy_cat_batches_per_step"],
                f"{source} copy batches",
            )
            _finite_positive(
                measurement["copy_cat_materialized_output_bytes_per_step"],
                f"{source} copy bytes",
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Stage Q.6 source {source!r} is incomplete") from exc
        if not present or zero_copy is not False:
            raise ValueError(f"Stage Q.6 source {source!r} is not designable")
        normalized.append(row)
    sources = tuple(str(row["source"]) for row in normalized)
    if len(sources) != len(set(sources)) or set(sources) != set(
        STAGE_Q61_REQUIRED_SOURCES
    ):
        raise ValueError("Stage Q.6.1 requires the complete three-source map")
    return normalized


def _contract_identity() -> str:
    payload = json.dumps(
        {
            "native_segment_contract": _NATIVE_SEGMENT_CONTRACT,
            "source_designs": _SOURCE_DESIGNS,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def analyze_stage_q61_native_handoff_design(
    *,
    stage_q6_report: str | Path,
    expected_stage_q6_sha256: str,
) -> dict[str, object]:
    """Freeze one candidate contract without implementing its runtime path."""

    q6_path, q6 = _load_q6_report(
        stage_q6_report,
        expected_sha256=expected_stage_q6_sha256,
    )
    rows = _validate_q6_contract(q6)
    by_source = {str(row["source"]): row for row in rows}

    source_plans = []
    for source in STAGE_Q61_REQUIRED_SOURCES:
        evidence = by_source[source]
        plan = deepcopy(_SOURCE_DESIGNS[source])
        plan.update(
            source=source,
            measured_copy_cat_batches_per_step=evidence["measurement"][
                "copy_cat_batches_per_step"
            ],
            measured_copy_cat_components_per_step=evidence["measurement"][
                "copy_cat_components_per_step"
            ],
            measured_materialized_bytes_per_step=evidence["measurement"][
                "copy_cat_materialized_output_bytes_per_step"
            ],
            measured_milliseconds_per_step=evidence["measurement"][
                "mean_milliseconds_per_step"
            ],
            baseline_transform_input_layout=evidence[
                "transform_input_layout"
            ],
            consumer_required_layout=evidence[
                "consumer_required_layout"
            ],
        )
        source_plans.append(plan)

    aggregate = q6["aggregate_measured_evidence"]
    if int(aggregate["source_count"]) != len(STAGE_Q61_REQUIRED_SOURCES):
        raise ValueError("Stage Q.6 aggregate source count differs")
    if aggregate["no_speedup_prediction_is_made"] is not True:
        raise ValueError("Stage Q.6 speedup interpretation differs")

    return {
        "schema_version": 1,
        "qualification_stage": "Q.6.1",
        "classification": STAGE_Q61_CLASSIFICATION,
        "architecture_decision": STAGE_Q61_ARCHITECTURE_DECISION,
        "contract_identity_sha256": _contract_identity(),
        "native_segment_contract": deepcopy(_NATIVE_SEGMENT_CONTRACT),
        "source_adaptation_plans": source_plans,
        "design_resolution": {
            "one_shared_runtime_abstraction_is_feasible": True,
            "shared_arena_or_preallocated_workspace_required": False,
            "cross_producer_copy_required_by_contract": False,
            "current_projected_transforms_are_preserved": True,
            "current_consumer_named_views_are_preserved": True,
            "transform_batch_fragmentation_is_primary_risk": True,
            "final_explicit_rhs_stack_is_not_claimed_as_eliminated": True,
            "all_three_sources_must_be_implemented_together": True,
            "single_source_candidate_remains_closed": True,
        },
        "stage_q62_candidate": {
            "name": "plane_native_segment_handoff_shadow_candidate",
            "scope": "experimental_plane_shadow_runtime_only",
            "required_sources": list(STAGE_Q61_REQUIRED_SOURCES),
            "baseline": "copy_cat_boundary_signature_batches",
            "candidate": "producer_owned_native_segments_without_copy_cat",
            "fallback_policy": "fail_closed_no_copy_cat_fallback",
            "must_not_change": [
                "production runtime path",
                "equations or physical parameters",
                "component names or consumer-visible order",
                "boundary signatures or transform definitions",
                "integrator or restart semantics",
                "generic solver or Channel defaults",
            ],
            "required_observability": [
                "native segment count by source",
                "native segment components and bytes by source",
                "copy_cat batches and bytes by source",
                "forward and inverse transform calls per step",
                "operator and kernel launch counts",
                "peak allocated and reserved bytes",
                "fallback count and reason",
            ],
            "qualification_gates": {
                "all_three_sources_exercised": True,
                "all_three_sources_copy_cat_batches_per_step": 0,
                "all_three_sources_copy_cat_materialized_bytes_per_step": 0,
                "fallback_count": 0,
                "finite_outputs": True,
                "six_step_relative_l2_maximum": 1e-12,
                "peak_allocated_ratio_maximum": 1.05,
                "peak_reserved_ratio_maximum": 1.05,
                "balanced_r320_profile_trials_per_role": 3,
                "performance_must_be_measured_not_predicted": True,
            },
        },
        "risk_register": [
            {
                "risk": "transform_launch_fragmentation",
                "severity": "primary",
                "mitigation": (
                    "measure transform calls and H100 operator/kernel launches; "
                    "reject even when copy counters reach zero if throughput "
                    "does not improve"
                ),
            },
            {
                "risk": "hidden_repacking_at_consumer_order_restore",
                "severity": "hard_gate",
                "mitigation": (
                    "order restoration must use metadata and views; preserve the "
                    "existing final RHS publication boundary without adding copy"
                ),
            },
            {
                "risk": "generation_lifetime_escape",
                "severity": "hard_gate",
                "mitigation": (
                    "invalidate segment handles with AlgebraicGenerationState "
                    "and retain no scheduler tensor references"
                ),
            },
            {
                "risk": "scope_expansion_into_cross_geometry_storage",
                "severity": "hard_gate",
                "mitigation": (
                    "keep the candidate Plane-shadow-only; generic solver and "
                    "Channel remain unchanged"
                ),
            },
        ],
        "measured_evidence_context": {
            "mean_milliseconds_per_step": aggregate[
                "mean_milliseconds_per_step"
            ],
            "mean_fraction_of_timestep": aggregate[
                "mean_fraction_of_timestep"
            ],
            "copy_cat_materialized_output_bytes_per_step": aggregate[
                "copy_cat_materialized_output_bytes_per_step"
            ],
            "not_a_predicted_speedup": True,
        },
        "eligible_for_stage_q62_candidate_implementation": True,
        "eligible_for_stage_q62_production_promotion": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "changes_equations": False,
        "changes_runtime_implementation": False,
        "inputs": [
            {
                "kind": "stage_q6_report",
                "path": str(q6_path),
                "sha256": file_sha256(q6_path),
            }
        ],
    }


def analysis_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the design-only Stage Q.6.1 native handoff contract"
    )
    parser.add_argument("--stage-q6-report", type=Path, required=True)
    parser.add_argument("--expected-stage-q6-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_stage_q61_native_handoff_design(
        stage_q6_report=args.stage_q6_report,
        expected_stage_q6_sha256=args.expected_stage_q6_sha256,
    )
    _write_new_json(args.output, report)
    return 0


__all__ = [
    "STAGE_Q61_ARCHITECTURE_DECISION",
    "STAGE_Q61_CLASSIFICATION",
    "STAGE_Q61_REQUIRED_SOURCES",
    "analysis_main",
    "analyze_stage_q61_native_handoff_design",
]
