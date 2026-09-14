"""Read-only Stage O design for migrating the qualified Plane runtime.

Stage O is deliberately a design boundary.  It records how the qualified
separated Plane Beris--Edwards runtime may enter production behind an opt-in
canary path, while the current production implementation remains the default
and rollback oracle.  This module does not construct a solver or mutate a run.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path

from ._shadow_support import file_sha256, require_sha256


STAGE_O_SCHEMA_VERSION = 1


def _require_git_sha(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a full lowercase Git SHA")
    return value


class PlaneRuntimePath(str, Enum):
    """Mutually exclusive Plane production migration paths."""

    LEGACY_PRODUCTION = "legacy_production"
    SEPARATED_CANARY = "separated_canary"


class MigrationDisposition(str, Enum):
    """How one responsibility is treated during migration."""

    PRESERVE = "preserve"
    EXTRACT_SHARED = "extract_shared"
    ADAPT = "adapt"
    DUAL_PATH = "dual_path"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class MigrationResponsibility:
    """One explicit ownership transfer or preservation decision."""

    name: str
    current_owner: str
    target_owner: str
    disposition: MigrationDisposition
    first_phase: str
    invariant: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("migration responsibility name must be an identifier")
        for value, description in (
            (self.current_owner, "current owner"),
            (self.target_owner, "target owner"),
            (self.first_phase, "first phase"),
            (self.invariant, "invariant"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"migration {description} must not be empty")
        if not isinstance(self.disposition, MigrationDisposition):
            raise TypeError("migration disposition is invalid")

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "current_owner": self.current_owner,
            "target_owner": self.target_owner,
            "disposition": self.disposition.value,
            "first_phase": self.first_phase,
            "invariant": self.invariant,
        }


@dataclass(frozen=True, slots=True)
class MigrationPhase:
    """One bounded future implementation phase in the Stage O design."""

    name: str
    objective: str
    allowed_mutations: tuple[str, ...]
    exit_gates: tuple[str, ...]
    production_default_may_change: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.startswith("O."):
            raise ValueError("migration phase name must start with 'O.'")
        if not isinstance(self.objective, str) or not self.objective.strip():
            raise ValueError("migration phase objective must not be empty")
        mutations = tuple(self.allowed_mutations)
        gates = tuple(self.exit_gates)
        if not mutations or any(
            not isinstance(value, str) or not value.strip()
            for value in mutations
        ):
            raise ValueError("migration phase mutations must be nonempty strings")
        if not gates or any(
            not isinstance(value, str) or not value.strip() for value in gates
        ):
            raise ValueError("migration phase gates must be nonempty strings")
        if not isinstance(self.production_default_may_change, bool):
            raise TypeError("production-default permission must be a bool")
        object.__setattr__(self, "allowed_mutations", mutations)
        object.__setattr__(self, "exit_gates", gates)

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "objective": self.objective,
            "allowed_mutations": list(self.allowed_mutations),
            "exit_gates": list(self.exit_gates),
            "production_default_may_change": (
                self.production_default_may_change
            ),
        }


@dataclass(frozen=True, slots=True)
class StageOQualificationEvidence:
    """Minimal immutable evidence authorizing migration design only."""

    path: str
    sha256: str
    maximum_relative_l2: float
    mean_timestep_ratio: float
    peak_allocated_ratio: float
    peak_reserved_ratio: float

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"qualification report is missing: {path}")
        require_sha256(self.sha256, "qualification report SHA-256")
        if file_sha256(path) != self.sha256:
            raise ValueError("qualification report SHA-256 does not match")
        for value, description in (
            (self.maximum_relative_l2, "maximum relative-L2 error"),
            (self.mean_timestep_ratio, "mean timestep ratio"),
            (self.peak_allocated_ratio, "peak allocated ratio"),
            (self.peak_reserved_ratio, "peak reserved ratio"),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{description} must be finite and non-negative")
        object.__setattr__(self, "path", str(path))

    def to_metadata(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "maximum_relative_l2": self.maximum_relative_l2,
            "mean_timestep_ratio_candidate_over_control": (
                self.mean_timestep_ratio
            ),
            "peak_allocated_ratio_candidate_over_control": (
                self.peak_allocated_ratio
            ),
            "peak_reserved_ratio_candidate_over_control": (
                self.peak_reserved_ratio
            ),
        }


@dataclass(frozen=True, slots=True)
class StageOMigrationDesign:
    """Complete tensor-free and command-free production migration design."""

    project_root: str
    architecture_source_commit: str
    production_baseline_commit: str
    qualification: StageOQualificationEvidence
    responsibilities: tuple[MigrationResponsibility, ...]
    phases: tuple[MigrationPhase, ...]
    implementation_file_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        project = Path(self.project_root).expanduser().resolve()
        if not project.is_dir():
            raise FileNotFoundError(f"project root is missing: {project}")
        _require_git_sha(
            self.architecture_source_commit,
            "architecture source commit",
        )
        _require_git_sha(
            self.production_baseline_commit,
            "production baseline commit",
        )
        if not isinstance(self.qualification, StageOQualificationEvidence):
            raise TypeError("qualification must be StageOQualificationEvidence")
        responsibilities = tuple(self.responsibilities)
        phases = tuple(self.phases)
        identities = tuple(self.implementation_file_sha256)
        if not responsibilities or not all(
            isinstance(value, MigrationResponsibility)
            for value in responsibilities
        ):
            raise TypeError("responsibilities must contain migration decisions")
        if len({value.name for value in responsibilities}) != len(
            responsibilities
        ):
            raise ValueError("migration responsibility names must be unique")
        if not phases or not all(isinstance(value, MigrationPhase) for value in phases):
            raise TypeError("phases must contain MigrationPhase values")
        expected_phase_names = tuple(f"O.{index}" for index in range(1, 6))
        if tuple(value.name for value in phases) != expected_phase_names:
            raise ValueError("Stage O phases must be ordered O.1 through O.5")
        if any(phase.production_default_may_change for phase in phases[:-1]):
            raise ValueError("only O.5 may authorize a production default change")
        if phases[-1].production_default_may_change:
            raise ValueError(
                "Stage O design cannot itself authorize a default change"
            )
        if not identities or any(
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            for path, digest in identities
        ):
            raise ValueError("implementation identities are incomplete")
        if len({path for path, _ in identities}) != len(identities):
            raise ValueError("implementation identity paths must be unique")
        for _, digest in identities:
            require_sha256(digest, "implementation file SHA-256")
        object.__setattr__(self, "project_root", str(project))
        object.__setattr__(self, "responsibilities", responsibilities)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "implementation_file_sha256", identities)

    def to_metadata(self) -> dict[str, object]:
        """Return the auditable design without constructing runtime state."""

        return {
            "schema_version": STAGE_O_SCHEMA_VERSION,
            "qualification_stage": "O",
            "planning_only": True,
            "architecture_decision": (
                "bounded_plane_beris_edwards_dual_path_migration"
            ),
            "architecture_source_commit": self.architecture_source_commit,
            "production_baseline_commit": self.production_baseline_commit,
            "qualification_evidence": self.qualification.to_metadata(),
            "scope": {
                "geometry": "plane_slab",
                "model": "beris_edwards_complete_nematic_stress_stokes",
                "production_driver": "Plane_beris_edwards_stokes.py",
                "first_candidate_only": True,
                "excluded_geometries": ["periodic_box", "rectangular_channel"],
                "excluded_components": [
                    "generic_spectral_solver",
                    "production_defaults",
                    "scientific_benchmark_develop",
                ],
            },
            "runtime_paths": {
                "default": PlaneRuntimePath.LEGACY_PRODUCTION.value,
                "opt_in_candidate": PlaneRuntimePath.SEPARATED_CANARY.value,
                "simultaneous_selection_forbidden": True,
                "selection_authority": "resolved_plane_run_spec",
                "legacy_rollback_retained": True,
            },
            "dependency_direction": [
                "model",
                "physical_boundary_conditions",
                "geometry",
                "spectral_plan",
                "backend",
                "runtime",
                "workflow",
            ],
            "responsibilities": [
                value.to_metadata() for value in self.responsibilities
            ],
            "phases": [value.to_metadata() for value in self.phases],
            "fixed_migration_gates": _fixed_migration_gates(),
            "implementation_file_sha256": dict(
                self.implementation_file_sha256
            ),
            "authorizations": {
                "design_complete": True,
                "implementation_started": False,
                "production_path_changed": False,
                "production_default_change": False,
                "channel_migration": False,
                "generic_solver_migration": False,
                "long_simulation": False,
            },
            "next_authorized_action": (
                "implement_O.1_pure_configuration_extraction"
            ),
        }


def _load_qualified_stage_n41(path: str | Path) -> StageOQualificationEvidence:
    report_path = Path(path).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Stage N.4.1 qualification report is missing: {report_path}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Stage N.4.1 qualification report is invalid") from exc
    if not isinstance(report, Mapping):
        raise TypeError("Stage N.4.1 qualification must be a JSON object")
    comparison = report.get("comparison")
    trajectory = report.get("trajectory")
    if not isinstance(comparison, Mapping) or not isinstance(trajectory, Mapping):
        raise ValueError("Stage N.4.1 qualification lacks gate records")
    required_true = (
        "final_q_identity_gate",
        "transform_count_identity_gate",
        "lifecycle_identity_gate",
        "performance_non_regression_gate",
        "memory_gate",
    )
    if (
        report.get("qualification_stage") != "N.4"
        or report.get("classification") != "A_recommended"
        or report.get("numerical_equivalence_passed") is not True
        or report.get("eligible_for_stage_o_production_migration_design")
        is not True
        or report.get("eligible_for_production_promotion") is not False
        or report.get("production_path_changed") is not False
        or any(comparison.get(key) is not True for key in required_true)
    ):
        raise ValueError("Stage N.4.1 report does not authorize Stage O design")
    try:
        maximum_error = float(trajectory["maximum_gate_relative_l2"])
        tolerance = float(trajectory["relative_l2_tolerance"])
        mean_ratio = float(
            comparison["mean_timestep_ratio_candidate_over_control"]
        )
        mean_limit = float(comparison["maximum_mean_timestep_ratio"])
        allocated_ratio = float(
            comparison["peak_allocated_ratio_candidate_over_control"]
        )
        reserved_ratio = float(
            comparison["peak_reserved_ratio_candidate_over_control"]
        )
        memory_limit = float(comparison["maximum_memory_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Stage N.4.1 report has incomplete numeric gates") from exc
    values = (
        maximum_error,
        tolerance,
        mean_ratio,
        mean_limit,
        allocated_ratio,
        reserved_ratio,
        memory_limit,
    )
    if (
        any(not math.isfinite(value) or value < 0.0 for value in values)
        or tolerance != 1.0e-10
        or maximum_error > tolerance
        or mean_limit != 1.03
        or mean_ratio > mean_limit
        or memory_limit != 1.03
        or allocated_ratio > memory_limit
        or reserved_ratio > memory_limit
    ):
        raise ValueError("Stage N.4.1 numeric gates are not qualified")
    return StageOQualificationEvidence(
        path=str(report_path),
        sha256=file_sha256(report_path),
        maximum_relative_l2=maximum_error,
        mean_timestep_ratio=mean_ratio,
        peak_allocated_ratio=allocated_ratio,
        peak_reserved_ratio=reserved_ratio,
    )


def _responsibilities() -> tuple[MigrationResponsibility, ...]:
    return (
        MigrationResponsibility(
            "model_equations",
            "pssolver.models.active_nematics",
            "pssolver.models.active_nematics",
            MigrationDisposition.PRESERVE,
            "O.1",
            "Beris--Edwards equations and Q convention remain unchanged",
        ),
        MigrationResponsibility(
            "benchmark_parameterization",
            "Plane_beris_edwards_stokes.py",
            "pssolver.presets.active_nematics",
            MigrationDisposition.EXTRACT_SHARED,
            "O.1",
            "A-to-K/zeta/L1 mapping and defaults remain exactly reproducible",
        ),
        MigrationResponsibility(
            "run_configuration",
            "argparse namespace and module globals",
            "immutable PlaneBerisEdwardsRunSpec",
            MigrationDisposition.EXTRACT_SHARED,
            "O.1",
            "CLI and programmatic construction resolve one canonical spec",
        ),
        MigrationResponsibility(
            "boundary_and_geometry",
            "production driver tuples",
            "physical BC specs plus PlaneSlab",
            MigrationDisposition.ADAPT,
            "O.1",
            "physical boundary semantics and component parity are unchanged",
        ),
        MigrationResponsibility(
            "spectral_planning",
            "legacy inline construction",
            "immutable SpectralPlan",
            MigrationDisposition.ADAPT,
            "O.1",
            "resolved bases, masks, storage, dtype, and zero modes match",
        ),
        MigrationResponsibility(
            "runtime_assembly",
            "production script and experimental builder",
            "Plane runtime factory with explicit backend selection",
            MigrationDisposition.DUAL_PATH,
            "O.2",
            "legacy remains default and candidate remains explicitly opt-in",
        ),
        MigrationResponsibility(
            "algebraic_lifecycle",
            "experimental N.4.1 runtime",
            "separated canary execution policy",
            MigrationDisposition.ADAPT,
            "O.2",
            "generation lifetime and transform scheduling remain qualified",
        ),
        MigrationResponsibility(
            "run_workflow",
            "production script loop",
            "shared Plane simulation workflow",
            MigrationDisposition.EXTRACT_SHARED,
            "O.3",
            "step ordering, refresh phase, and final static sync are preserved",
        ),
        MigrationResponsibility(
            "observation_and_metadata",
            "production and shadow writers",
            "canonical observation and additive backend metadata adapter",
            MigrationDisposition.EXTRACT_SHARED,
            "O.3",
            "legacy filenames, arrays, scientific metadata, and COMPLETE order remain",
        ),
        MigrationResponsibility(
            "checkpoint_and_restart",
            "experimental checkpoint coordinator",
            "versioned workflow checkpoint adapter",
            MigrationDisposition.DEFER,
            "O.3",
            "no checkpoint format is cross-loaded without an explicit adapter",
        ),
        MigrationResponsibility(
            "other_geometries",
            "Channel and generic solver production paths",
            "separate future geometry migrations",
            MigrationDisposition.DEFER,
            "after_O.5",
            "Plane qualification grants no Channel or generic authorization",
        ),
    )


def _phases() -> tuple[MigrationPhase, ...]:
    return (
        MigrationPhase(
            "O.1",
            "Extract canonical Plane run configuration and parameter resolution",
            (
                "add pure run-spec and preset modules",
                "make the Plane CLI delegate configuration resolution",
                "add characterization metadata without changing execution",
            ),
            (
                "legacy CLI defaults and validation remain unchanged",
                "legacy dry-run scientific metadata is canonically identical",
                "legacy short trajectory remains byte-identical",
            ),
        ),
        MigrationPhase(
            "O.2",
            "Introduce an opt-in separated canary behind one backend enum",
            (
                "add a Plane runtime protocol and legacy adapter",
                "add a lazily imported separated-canary adapter",
                "record requested and effective runtime path additively",
            ),
            (
                "omitted runtime selection resolves to legacy_production",
                "mixed configuration authorities are rejected",
                "Channel and generic solver imports remain unchanged",
            ),
        ),
        MigrationPhase(
            "O.3",
            "Unify observation, completion, diagnostics, and restart boundaries",
            (
                "add a shared Plane workflow coordinator",
                "adapt both paths to canonical Q/u/p observations",
                "version checkpoint compatibility explicitly",
            ),
            (
                "output names, shapes, dtypes, and COMPLETE ordering match",
                "same-backend split restart is exact",
                "unsupported cross-format restart fails before execution",
            ),
        ),
        MigrationPhase(
            "O.4",
            "Qualify the opt-in production canary without changing defaults",
            (
                "run bounded CPU characterization",
                "run balanced H100 legacy/canary comparisons",
                "archive numerical, lifecycle, memory, and timing evidence",
            ),
            (
                "six-step and 100-step numerical gates pass",
                "transform and lifecycle identities remain qualified",
                "H100 performance and memory non-regression gates pass",
            ),
        ),
        MigrationPhase(
            "O.5",
            "Make a separate evidence-based production promotion decision",
            (
                "review O.1--O.4 evidence",
                "run an explicitly authorized benchmark-shaped stability canary",
                "prepare a reversible default-promotion commit or retain legacy",
            ),
            (
                "longer stability and restart evidence passes",
                "implicit-default smoke passes in a separate commit",
                "explicit user authorization is obtained before promotion",
            ),
            production_default_may_change=False,
        ),
    )


def _fixed_migration_gates() -> dict[str, object]:
    return {
        "configuration": {
            "single_authority": "resolved_plane_run_spec",
            "legacy_default_must_remain": True,
            "mixed_backend_selection_forbidden": True,
            "scientific_defaults_must_remain": True,
        },
        "legacy_characterization": {
            "dry_run_scientific_metadata_canonical_identity": True,
            "short_trajectory_byte_identity": True,
            "initial_q_sha256_identity": True,
        },
        "candidate_numerics": {
            "six_step_relative_l2_tolerance": 1.0e-10,
            "one_hundred_step_relative_l2_tolerance": 1.0e-10,
            "pressure_comparison": "raw_and_demeaned",
            "finite_shapes_and_dtypes_required": True,
        },
        "restart_and_outputs": {
            "same_backend_split_restart_must_be_exact": True,
            "legacy_output_layout_preserved": True,
            "cross_format_restart_requires_explicit_adapter": True,
            "complete_marker_written_last": True,
        },
        "h100_non_regression": {
            "profile_trials": 3,
            "maximum_mean_timestep_ratio": 1.03,
            "maximum_paired_timestep_ratio": 1.05,
            "minimum_paired_fraction": "2/3",
            "maximum_peak_allocated_ratio": 1.03,
            "maximum_peak_reserved_ratio": 1.03,
            "transform_counts_must_match": True,
            "lifecycle_counters_must_match": True,
        },
        "promotion": {
            "short_qualification_cannot_change_default": True,
            "separate_implicit_default_smoke_required": True,
            "explicit_authorization_required": True,
            "rollback_path_retained_until_final_acceptance": True,
        },
    }


def build_stage_o_migration_design(
    *,
    project_root: str | Path,
    stage_n41_qualification: str | Path,
    architecture_source_commit: str,
    production_baseline_commit: str,
) -> StageOMigrationDesign:
    """Build the Stage O design after validating N.4.1 evidence and scope."""

    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"project root is missing: {project}")
    _require_git_sha(architecture_source_commit, "architecture source commit")
    _require_git_sha(production_baseline_commit, "production baseline commit")
    required_files = (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
        "pssolver/execution/policy.py",
        "pssolver/experimental/model_execution.py",
        "pssolver/experimental/projected_scheduler.py",
    )
    identities = []
    for relative in required_files:
        path = project / relative
        if not path.is_file():
            raise FileNotFoundError(f"required migration input is missing: {path}")
        identities.append((relative, file_sha256(path)))
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        text = (project / relative).read_text(encoding="utf-8")
        if "pssolver.experimental" in text:
            raise ValueError(
                f"production module already imports experimental code: {relative}"
            )
    return StageOMigrationDesign(
        project_root=str(project),
        architecture_source_commit=architecture_source_commit,
        production_baseline_commit=production_baseline_commit,
        qualification=_load_qualified_stage_n41(stage_n41_qualification),
        responsibilities=_responsibilities(),
        phases=_phases(),
        implementation_file_sha256=tuple(identities),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the read-only Stage O production migration design."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--stage-n41-qualification",
        type=Path,
        required=True,
    )
    parser.add_argument("--architecture-source-commit", required=True)
    parser.add_argument("--production-baseline-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    design = build_stage_o_migration_design(
        project_root=args.project_root,
        stage_n41_qualification=args.stage_n41_qualification,
        architecture_source_commit=args.architecture_source_commit,
        production_baseline_commit=args.production_baseline_commit,
    )
    print(
        json.dumps(
            design.to_metadata(),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "MigrationDisposition",
    "MigrationPhase",
    "MigrationResponsibility",
    "PlaneRuntimePath",
    "STAGE_O_SCHEMA_VERSION",
    "StageOMigrationDesign",
    "StageOQualificationEvidence",
    "build_stage_o_migration_design",
    "main",
]
