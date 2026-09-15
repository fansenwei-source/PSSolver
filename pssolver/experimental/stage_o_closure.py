"""Read-only closure decision for the Plane Stage O migration study.

The decision records the already completed, checksummed H100 evidence.  It
does not load a solver, replay an experiment, or change a runtime default.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Sequence

from ._shadow_support import file_sha256, require_sha256


STAGE_O_CLOSURE_SCHEMA_VERSION = 1


def _require_git_sha(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a full lowercase Git SHA")
    return value


class StageOPathDisposition(str, Enum):
    """Final treatment of one Stage O implementation path."""

    RETAIN_PRODUCTION = "retain_production"
    RETAIN_EXPERIMENTAL_ORACLE = "retain_experimental_oracle"
    ARCHIVE_EXPERIMENTAL_EVIDENCE = "archive_experimental_evidence"
    CLOSE_OPTIMIZATION_LINE = "close_optimization_line"


@dataclass(frozen=True, slots=True)
class StageOClosureEvidence:
    """Identity and bounded facts from one authoritative Stage O artifact."""

    stage: str
    commit: str
    classification: str
    architecture_decision: str
    report_sha256: str
    facts: tuple[tuple[str, bool | int | float | str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.startswith("O.4"):
            raise ValueError("closure evidence must belong to Stage O.4")
        _require_git_sha(self.commit, "evidence commit")
        require_sha256(self.report_sha256, "evidence report SHA-256")
        for value, description in (
            (self.classification, "classification"),
            (self.architecture_decision, "architecture decision"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"evidence {description} must not be empty")
        facts = tuple(self.facts)
        if len({name for name, _ in facts}) != len(facts):
            raise ValueError("evidence fact names must be unique")
        for name, value in facts:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError("evidence fact names must be identifiers")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("evidence floating-point facts must be finite")
            if not isinstance(value, (bool, int, float, str)):
                raise TypeError("evidence facts must be scalar JSON values")
        object.__setattr__(self, "facts", facts)

    def to_metadata(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "commit": self.commit,
            "classification": self.classification,
            "architecture_decision": self.architecture_decision,
            "report_sha256": self.report_sha256,
            "facts": dict(self.facts),
        }


@dataclass(frozen=True, slots=True)
class StageOClosureDecision:
    """Immutable Stage O disposition and the boundary of the next study."""

    project_root: str
    evidence: tuple[StageOClosureEvidence, ...]
    implementation_file_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        project = Path(self.project_root).expanduser().resolve()
        if not project.is_dir():
            raise FileNotFoundError(f"project root is missing: {project}")
        evidence = tuple(self.evidence)
        if not all(isinstance(item, StageOClosureEvidence) for item in evidence):
            raise TypeError("Stage O closure evidence has an invalid record")
        expected_stages = (
            "O.4",
            "O.4.1",
            "O.4.2.7",
            "O.4.3",
            "O.4.3.1",
            "O.4.3.3",
            "O.4.3.4",
        )
        if tuple(item.stage for item in evidence) != expected_stages:
            raise ValueError("Stage O closure evidence is incomplete or unordered")
        identities = tuple(self.implementation_file_sha256)
        if not identities or len({path for path, _ in identities}) != len(
            identities
        ):
            raise ValueError("closure implementation identities are incomplete")
        for path, digest in identities:
            if not isinstance(path, str) or not path:
                raise ValueError("closure identity path must not be empty")
            require_sha256(digest, "closure implementation SHA-256")
        object.__setattr__(self, "project_root", str(project))
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "implementation_file_sha256", identities)

    def to_metadata(self) -> dict[str, object]:
        """Return a tensor-free, command-free architecture decision."""

        return {
            "schema_version": STAGE_O_CLOSURE_SCHEMA_VERSION,
            "qualification_stage": "O-closure",
            "planning_only": True,
            "classification": "CLOSED_RETAIN_LEGACY_PRODUCTION",
            "architecture_decision": (
                "retain_legacy_production_and_experimental_separated_oracle"
            ),
            "evidence": [item.to_metadata() for item in self.evidence],
            "path_dispositions": {
                "legacy_production": (
                    StageOPathDisposition.RETAIN_PRODUCTION.value
                ),
                "separated_canary": (
                    StageOPathDisposition.RETAIN_EXPERIMENTAL_ORACLE.value
                ),
                "packed_generation_publication": (
                    StageOPathDisposition.ARCHIVE_EXPERIMENTAL_EVIDENCE.value
                ),
                "natural_storage_views": (
                    StageOPathDisposition.ARCHIVE_EXPERIMENTAL_EVIDENCE.value
                ),
                "producer_owned_boundary_packing": (
                    StageOPathDisposition.ARCHIVE_EXPERIMENTAL_EVIDENCE.value
                ),
                "projected_materialization_optimization": (
                    StageOPathDisposition.CLOSE_OPTIMIZATION_LINE.value
                ),
            },
            "scientific_conclusion": {
                "separated_equations_and_timestep_are_qualified": True,
                "six_and_one_hundred_step_equivalence_passed": True,
                "same_backend_restart_passed": True,
                "initial_q_identity_passed": True,
                "scientific_failure_observed": False,
            },
            "production_conclusion": {
                "default_runtime": "legacy_production",
                "separated_canary_is_production_replacement": False,
                "stage_o5_entered": False,
                "stage_o44_authorized": False,
                "production_default_changed": False,
                "channel_changed": False,
                "generic_solver_changed": False,
            },
            "materialization_line_closure": {
                "closed": True,
                "reason": (
                    "no_remaining_individual_copy_source_reaches_the_frozen_"
                    "three_percent_screening_signal"
                ),
                "dominant_remaining_source": (
                    "algebraic.nematic_stress.dependencies"
                ),
                "dominant_fraction_of_timestep": 0.01997209883,
                "minimum_screening_fraction": 0.03,
                "new_layout_candidate_authorized": False,
            },
            "retention_strategy": {
                "legacy_role": "production_and_rollback_reference",
                "separated_role": (
                    "numerical_oracle_architecture_laboratory_and_diagnostic_"
                    "comparison_target"
                ),
                "experimental_candidates_remain_opt_in": True,
                "rejected_or_neutral_evidence_remains_immutable": True,
            },
            "next_stage_boundary": {
                "stage": "P",
                "authorized_action": (
                    "design_operator_and_kernel_level_production_canary_"
                    "gap_diagnostic"
                ),
                "optimization_authorized": False,
                "h100_execution_authorized": False,
                "materialization_layout_work_reopened": False,
                "focus": [
                    "transform_scheduling",
                    "kernel_launch_structure",
                    "compiled_graph_boundaries",
                    "repeated_spectral_operations",
                ],
            },
            "authorizations": {
                "stage_o_closed": True,
                "stage_p_design": True,
                "stage_p_implementation": False,
                "stage_p_h100_execution": False,
                "stage_o44": False,
                "stage_o5": False,
                "production_promotion": False,
                "default_change": False,
                "benchmark_run": False,
            },
            "implementation_file_sha256": dict(
                self.implementation_file_sha256
            ),
        }


def _authoritative_evidence() -> tuple[StageOClosureEvidence, ...]:
    return (
        StageOClosureEvidence(
            stage="O.4",
            commit="12551bea681a8f02bd52cb0c7e4d6ca4b0154c63",
            classification="B_neutral",
            architecture_decision="bounded_production_canary_qualification",
            report_sha256=(
                "e2ebc2e3b6a9fab3f302cf89fb9bde4aee972af6859ef0720af1aa08457f7efe"
            ),
            facts=(
                ("maximum_relative_l2", 3.169995072241099e-15),
                ("mean_timestep_ratio", 0.999958893270793),
                ("peak_allocated_ratio", 1.551815655390447),
                ("same_backend_restart_exact", True),
            ),
        ),
        StageOClosureEvidence(
            stage="O.4.1",
            commit="6defce87eecf0717a40c66a2eb823a0c656c3bd6",
            classification="B_neutral",
            architecture_decision="architecture_aware_requalification",
            report_sha256=(
                "46da6912fdc019418982c44986ac99c97d2fe2b04914c9d65f4ce9913a30293f"
            ),
            facts=(
                ("r320_mean_timestep_ratio", 1.429558205567479),
                ("r320_peak_allocated_ratio", 1.589530529807474),
                ("r320_peak_reserved_ratio", 1.328679331654841),
                ("retained_allocation_growth", 0),
            ),
        ),
        StageOClosureEvidence(
            stage="O.4.2.7",
            commit="7e49123132be1c09775d2c30896b7527135b3e0a",
            classification="DIAGNOSTIC_COMPLETE",
            architecture_decision="identify_owner_before_optimization",
            report_sha256=(
                "1878089621983b216b0ed64db6f06cc6bc62017fbb37cbc5e240a4cf83146a3e"
            ),
            facts=(
                ("accounting_ready_for_optimization", True),
                ("unmatched_storage_count", 0),
                ("on_demand_physical_materializations", 0),
                ("inventory_observer_effect", False),
            ),
        ),
        StageOClosureEvidence(
            stage="O.4.3",
            commit="19ae28e30a4623dbc431bf1b78811ce91a89d1f1",
            classification="C_rejected",
            architecture_decision="packed_generation_storage_with_safe_views",
            report_sha256=(
                "1888511621cbb84be79ed17ceeeeb4f16172e5212c59bf2682ec753689d3d613"
            ),
            facts=(
                ("maximum_relative_l2", 0.0),
                ("r320_mean_timestep_ratio", 1.030159030821),
                ("r320_peak_allocated_ratio", 1.262997971102),
                ("r320_peak_reserved_ratio", 1.281481481481),
            ),
        ),
        StageOClosureEvidence(
            stage="O.4.3.1",
            commit="85e6a9a074187999de72b1aa38edb97cbdb837ca",
            classification="B_neutral",
            architecture_decision=(
                "opportunistic_natural_storage_views_without_republication"
            ),
            report_sha256=(
                "ea2a4977dfb8a4a8ef5325f976100202e603e3dcda45006e4f6669022e2c8fdb"
            ),
            facts=(
                ("maximum_relative_l2", 0.0),
                ("r320_mean_timestep_ratio", 1.000066046754),
                ("r320_peak_allocated_ratio", 1.0),
                ("copy_cat_batches_per_twenty_steps", 160),
            ),
        ),
        StageOClosureEvidence(
            stage="O.4.3.3",
            commit="f6ad4dc9f83d402147ff9d1ae07f5156603e717c",
            classification="B_neutral",
            architecture_decision="producer_owned_boundary_packed_h_and_stress",
            report_sha256=(
                "feced4d3702926974aeb1fe30082699711184dd18f7e9449d4511cf9e0c68abb"
            ),
            facts=(
                ("maximum_relative_l2", 3.1862555431836906e-15),
                ("r320_mean_timestep_ratio", 0.982766967768),
                ("required_mean_timestep_ratio", 0.98),
                ("copy_cat_batches_per_twenty_steps", 100),
            ),
        ),
        StageOClosureEvidence(
            stage="O.4.3.4",
            commit="502c2b25f9e37dcaeacc9bf916bebde293ea4149",
            classification="DIAGNOSTIC_COMPLETE",
            architecture_decision="remaining_materialization_attribution",
            report_sha256=(
                "ea4ca8c19bbcf86f9ab0ba140b59193ae61b79470ea28a7f7999f958545561b5"
            ),
            facts=(
                ("dominant_source_fraction", 0.01997209883),
                ("screening_fraction", 0.03),
                ("all_sources_attributed", True),
                ("retained_tensor_references", 0),
            ),
        ),
    )


def build_stage_o_closure_decision(
    project_root: str | Path,
) -> StageOClosureDecision:
    """Build the fixed closure record without constructing numerical state."""

    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"project root is missing: {project}")
    identity_paths = (
        "pssolver/configuration/plane_beris_edwards.py",
        "pssolver/runtime/plane_beris_edwards.py",
        "pssolver/workflows/plane_beris_edwards.py",
        "pssolver/experimental/model_execution.py",
        "pssolver/experimental/projected_scheduler.py",
    )
    identities = []
    for relative in identity_paths:
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(f"closure identity file is missing: {path}")
        identities.append((relative, file_sha256(path)))
    return StageOClosureDecision(
        project_root=str(project),
        evidence=_authoritative_evidence(),
        implementation_file_sha256=tuple(identities),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the read-only Plane Stage O closure decision."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    decision = build_stage_o_closure_decision(args.project_root)
    print(json.dumps(decision.to_metadata(), allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "STAGE_O_CLOSURE_SCHEMA_VERSION",
    "StageOClosureDecision",
    "StageOClosureEvidence",
    "StageOPathDisposition",
    "build_stage_o_closure_decision",
    "main",
]
