"""Read-only trajectory comparison for production and shadow Plane runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np

from pssolver.run_metadata import prepare_new_run_directory

from ._shadow_support import file_sha256
from .shadow_metadata import (
    ShadowMetadataComparison,
    compare_saved_shadow_to_production_metadata,
)


_SNAPSHOT_PATTERN = re.compile(r"^(Q|u|p)_(\d+)\.npy$")


def _load_metadata(directory: Path, description: str) -> dict[str, object]:
    path = directory / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"{description} metadata is missing: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} metadata is invalid JSON") from exc
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{description} metadata must be a JSON object")
    if metadata.get("status") != "complete":
        raise ValueError(f"{description} metadata is not complete")
    complete = directory / "COMPLETE"
    if not complete.is_file() or complete.read_text(encoding="utf-8") != (
        "complete\n"
    ):
        raise ValueError(f"{description} COMPLETE marker is invalid")
    return dict(metadata)


def _snapshot_steps(directory: Path, prefix: str) -> tuple[int, ...]:
    steps = []
    for path in directory.iterdir():
        match = _SNAPSHOT_PATTERN.fullmatch(path.name)
        if match is not None and match.group(1) == prefix:
            steps.append(int(match.group(2)))
    return tuple(sorted(steps))


def _relative_l2(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> float | None:
    difference_norm = float(np.linalg.norm(candidate - reference))
    reference_norm = float(np.linalg.norm(reference))
    if reference_norm == 0.0:
        return 0.0 if difference_norm == 0.0 else None
    return difference_norm / reference_norm


@dataclass(frozen=True, slots=True)
class ShadowArrayComparison:
    """Numerical and identity comparison for one saved array."""

    step: int
    field: str
    shape: tuple[int, ...]
    dtype: str
    production_sha256: str
    shadow_sha256: str
    raw_relative_l2: float | None
    raw_linf: float
    demeaned_relative_l2: float | None
    demeaned_linf: float | None
    gate_metric: str
    passed: bool

    def to_metadata(self) -> dict[str, object]:
        return {
            "step": self.step,
            "field": self.field,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "production_sha256": self.production_sha256,
            "shadow_sha256": self.shadow_sha256,
            "byte_identical": (
                self.production_sha256 == self.shadow_sha256
            ),
            "raw_relative_l2": self.raw_relative_l2,
            "raw_linf": self.raw_linf,
            "demeaned_relative_l2": self.demeaned_relative_l2,
            "demeaned_linf": self.demeaned_linf,
            "gate_metric": self.gate_metric,
            "passed": self.passed,
        }


def _compare_array(
    production_path: Path,
    shadow_path: Path,
    *,
    step: int,
    field: str,
    tolerance: float,
) -> ShadowArrayComparison:
    production_sha256 = file_sha256(production_path)
    shadow_sha256 = file_sha256(shadow_path)
    production = np.load(production_path, allow_pickle=False)
    shadow = np.load(shadow_path, allow_pickle=False)
    if file_sha256(production_path) != production_sha256:
        raise RuntimeError(f"production {field}_{step} changed while read")
    if file_sha256(shadow_path) != shadow_sha256:
        raise RuntimeError(f"shadow {field}_{step} changed while read")
    if production.shape != shadow.shape:
        raise ValueError(
            f"{field}_{step} production and shadow shapes differ"
        )
    if production.dtype != shadow.dtype:
        raise ValueError(
            f"{field}_{step} production and shadow dtypes differ"
        )
    if not np.issubdtype(production.dtype, np.floating):
        raise TypeError(f"{field}_{step} must have a floating dtype")
    if not np.isfinite(production).all() or not np.isfinite(shadow).all():
        raise ValueError(f"{field}_{step} contains NaN or Inf")

    difference = shadow - production
    raw_relative = _relative_l2(production, shadow)
    raw_linf = float(np.max(np.abs(difference), initial=0.0))
    demeaned_relative = None
    demeaned_linf = None
    if field == "p":
        production_demeaned = production - np.mean(production)
        shadow_demeaned = shadow - np.mean(shadow)
        demeaned_relative = _relative_l2(
            production_demeaned,
            shadow_demeaned,
        )
        demeaned_linf = float(
            np.max(
                np.abs(shadow_demeaned - production_demeaned),
                initial=0.0,
            )
        )
        gate_value = demeaned_relative
        gate_metric = "demeaned_relative_l2"
    else:
        gate_value = raw_relative
        gate_metric = "raw_relative_l2"
    passed = gate_value is not None and gate_value <= tolerance
    return ShadowArrayComparison(
        step=step,
        field=field,
        shape=tuple(production.shape),
        dtype=str(production.dtype),
        production_sha256=production_sha256,
        shadow_sha256=shadow_sha256,
        raw_relative_l2=raw_relative,
        raw_linf=raw_linf,
        demeaned_relative_l2=demeaned_relative,
        demeaned_linf=demeaned_linf,
        gate_metric=gate_metric,
        passed=passed,
    )


@dataclass(frozen=True, slots=True)
class PlaneShadowTrajectoryComparison:
    """Complete side-by-side production and Plane shadow trajectory result."""

    production_directory: Path
    shadow_directory: Path
    relative_l2_tolerance: float
    configuration: ShadowMetadataComparison
    saved_steps: tuple[int, ...]
    arrays: tuple[ShadowArrayComparison, ...]

    @property
    def passed(self) -> bool:
        return self.configuration.compatible and all(
            value.passed for value in self.arrays
        )

    def to_metadata(self) -> dict[str, object]:
        gate_values = [
            (
                value.demeaned_relative_l2
                if value.field == "p"
                else value.raw_relative_l2
            )
            for value in self.arrays
        ]
        finite_gate_values = [
            value for value in gate_values if value is not None
        ]
        return {
            "schema_version": 1,
            "classification": (
                "PASS" if self.passed else "FAIL"
            ),
            "eligible_for_bounded_h100_gate": self.passed,
            "production_directory": str(self.production_directory),
            "shadow_directory": str(self.shadow_directory),
            "relative_l2_tolerance": self.relative_l2_tolerance,
            "configuration": self.configuration.to_metadata(),
            "saved_steps": list(self.saved_steps),
            "array_count": len(self.arrays),
            "maximum_gate_relative_l2": (
                max(finite_gate_values) if finite_gate_values else None
            ),
            "arrays": [value.to_metadata() for value in self.arrays],
        }


def compare_plane_shadow_trajectories(
    production_directory: str | Path,
    shadow_directory: str | Path,
    *,
    relative_l2_tolerance: float = 1.0e-10,
) -> PlaneShadowTrajectoryComparison:
    """Compare complete saved trajectories without modifying either run."""

    if (
        not isinstance(relative_l2_tolerance, (int, float))
        or isinstance(relative_l2_tolerance, bool)
        or not math.isfinite(float(relative_l2_tolerance))
        or float(relative_l2_tolerance) <= 0.0
    ):
        raise ValueError("relative_l2_tolerance must be positive and finite")
    tolerance = float(relative_l2_tolerance)
    production_directory = Path(production_directory).expanduser().resolve()
    shadow_directory = Path(shadow_directory).expanduser().resolve()
    for directory, description in (
        (production_directory, "production"),
        (shadow_directory, "shadow"),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{description} run directory is missing: {directory}"
            )

    production_metadata = _load_metadata(
        production_directory,
        "production",
    )
    shadow_metadata = _load_metadata(shadow_directory, "shadow")
    configuration = compare_saved_shadow_to_production_metadata(
        shadow_metadata,
        production_metadata,
    )

    try:
        source = shadow_metadata["initial_condition"]
        if source["source_metadata_sha256"] != file_sha256(
            production_directory / "metadata.json"
        ):
            raise ValueError("shadow source metadata identity has changed")
        source_q_name = source["source_q_file"]
        if (
            not isinstance(source_q_name, str)
            or Path(source_q_name).name != source_q_name
        ):
            raise ValueError("shadow source initial-Q filename is invalid")
        source_q = production_directory / source_q_name
        if source["source_q_file_sha256"] != file_sha256(source_q):
            raise ValueError("shadow source initial-Q identity has changed")
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "shadow metadata lacks production-source identity"
        ) from exc

    shadow_steps = _snapshot_steps(shadow_directory, "Q")
    if not shadow_steps or shadow_steps[0] != 0:
        raise ValueError("shadow trajectory must contain Q_0")
    completed = shadow_metadata.get("completed_steps")
    if completed != shadow_steps[-1]:
        raise ValueError("shadow completed_steps differs from saved Q frames")
    if production_metadata.get("completed_steps") != completed:
        raise ValueError("production and shadow completed steps differ")
    for prefix in ("u", "p"):
        if _snapshot_steps(shadow_directory, prefix) != shadow_steps:
            raise ValueError(f"shadow {prefix} frame set differs from Q")
    for prefix in ("Q", "u", "p"):
        production_steps = set(_snapshot_steps(production_directory, prefix))
        missing = tuple(step for step in shadow_steps if step not in production_steps)
        if missing:
            raise ValueError(
                f"production {prefix} frames are missing steps {missing!r}"
            )

    arrays = tuple(
        _compare_array(
            production_directory / f"{field}_{step}.npy",
            shadow_directory / f"{field}_{step}.npy",
            step=step,
            field=field,
            tolerance=tolerance,
        )
        for step in shadow_steps
        for field in ("Q", "u", "p")
    )
    return PlaneShadowTrajectoryComparison(
        production_directory=production_directory,
        shadow_directory=shadow_directory,
        relative_l2_tolerance=tolerance,
        configuration=configuration,
        saved_steps=shadow_steps,
        arrays=arrays,
    )


def write_plane_shadow_comparison(
    output_directory: str | Path,
    comparison: PlaneShadowTrajectoryComparison,
) -> Path:
    """Write one immutable comparison bundle outside both input runs."""

    if not isinstance(comparison, PlaneShadowTrajectoryComparison):
        raise TypeError("comparison has the wrong type")
    output = prepare_new_run_directory(output_directory)
    report = output / "comparison.json"
    temporary = output / ".comparison.json.tmp"
    temporary.write_text(
        json.dumps(
            comparison.to_metadata(),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(report)
    marker = "COMPLETE" if comparison.passed else "FAILED"
    (output / marker).write_text(
        ("pass" if comparison.passed else "fail") + "\n",
        encoding="utf-8",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare completed production and Plane shadow trajectories."
        )
    )
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--shadow-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--relative-l2-tolerance",
        type=float,
        default=1.0e-10,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    comparison = compare_plane_shadow_trajectories(
        args.production_dir,
        args.shadow_dir,
        relative_l2_tolerance=args.relative_l2_tolerance,
    )
    report = write_plane_shadow_comparison(
        args.output_dir,
        comparison,
    )
    print(report.read_text(encoding="utf-8"), end="")
    return 0 if comparison.passed else 1


__all__ = [
    "PlaneShadowTrajectoryComparison",
    "ShadowArrayComparison",
    "compare_plane_shadow_trajectories",
    "main",
    "write_plane_shadow_comparison",
]
