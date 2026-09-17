"""Build an exact allow-list policy from one R2R-B benchmark artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from pssolver import (
    BoundedTransformQualificationCell,
    QualifiedBoundedTransformPolicy,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_key(case: dict[str, object]) -> tuple[object, ...]:
    return (
        case["size"],
        case["kind"],
        case["execution_mode"],
        case["value_type"],
        case["retained_count"],
    )


def build_policy_artifact(
    benchmark: dict[str, object],
    *,
    source_path: str,
    source_sha256: str,
    policy_name: str,
    geometry: str,
    local_axis: int | None,
    minimum_speedup: float,
    maximum_peak_allocated_ratio: float,
    minimum_samples: int,
) -> dict[str, object]:
    """Return a JSON-safe exact-cell policy and its acceptance evidence."""

    if benchmark.get("benchmark") != "r2r_b_dense_vs_fft_bounded_axis":
        raise ValueError("input is not an R2R-B dense-versus-FFT artifact")
    if not policy_name or not geometry:
        raise ValueError("policy_name and geometry must be non-empty")
    if local_axis is not None and local_axis < 0:
        raise ValueError("local_axis must be non-negative or omitted")
    if not math.isfinite(minimum_speedup) or minimum_speedup <= 1.0:
        raise ValueError("minimum_speedup must be finite and greater than one")
    if (
        not math.isfinite(maximum_peak_allocated_ratio)
        or maximum_peak_allocated_ratio < 1.0
    ):
        raise ValueError(
            "maximum_peak_allocated_ratio must be finite and at least one"
        )
    if minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")

    config = benchmark["config"]
    environment = benchmark["environment"]
    if config["device"] != "cuda" or not environment["cuda_available"]:
        raise ValueError("R2R-C policy generation requires a CUDA artifact")
    if set(config["algorithms"]) != {"dense", "fft"}:
        raise ValueError("artifact must contain exactly dense and fft cases")

    indexed: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for case in benchmark["cases"]:
        indexed.setdefault(_case_key(case), {})[case["algorithm"]] = case
    if not indexed or any(set(pair) != {"dense", "fft"} for pair in indexed.values()):
        raise ValueError("every benchmark case must have a dense/fft pair")

    cells = []
    reviews = []
    for key, pair in sorted(indexed.items(), key=lambda item: item[0]):
        dense = pair["dense"]
        candidate = pair["fft"]
        correctness = candidate["correctness"]
        tolerance = correctness["tolerance_relative_l2"]
        maximum_error = max(
            correctness["dense_reference_forward_relative_l2"],
            correctness["dense_reference_inverse_relative_l2"],
        )
        dense_peak = dense["cuda_memory"]["peak_allocated_bytes"]
        candidate_peak = candidate["cuda_memory"]["peak_allocated_bytes"]
        if dense_peak is None or candidate_peak is None or dense_peak <= 0:
            raise ValueError("CUDA peak allocated memory is missing")
        memory_ratio = candidate_peak / dense_peak
        common_pass = (
            correctness["all_finite"]
            and maximum_error <= tolerance
            and memory_ratio <= maximum_peak_allocated_ratio
        )

        for direction in ("forward", "inverse"):
            dense_timing = dense["timing"][direction]
            candidate_timing = candidate["timing"][direction]
            samples = min(
                dense_timing["samples"],
                candidate_timing["samples"],
            )
            speedup = (
                dense_timing["median_seconds"]
                / candidate_timing["median_seconds"]
            )
            accepted = (
                common_pass
                and samples >= minimum_samples
                and speedup >= minimum_speedup
            )
            case_id = candidate["case_id"]
            cell_id = f"{case_id}_{direction}"
            evidence_id = f"sha256:{source_sha256}:{case_id}:{direction}"
            review = {
                "cell_id": cell_id,
                "case_key": list(key),
                "direction": direction,
                "accepted": accepted,
                "median_speedup_dense_over_fft": speedup,
                "samples": samples,
                "maximum_dense_reference_relative_l2": maximum_error,
                "tolerance_relative_l2": tolerance,
                "peak_allocated_ratio_fft_over_dense": memory_ratio,
                "evidence_id": evidence_id,
            }
            reviews.append(review)
            if not accepted:
                continue
            cells.append(
                BoundedTransformQualificationCell(
                    cell_id=cell_id,
                    geometry=geometry,
                    local_axis=local_axis,
                    kind=candidate["kind"],
                    direction=direction,
                    execution_mode=candidate["execution_mode"],
                    physical_size=candidate["size"],
                    retained_count=candidate["retained_count"],
                    minimum_line_count=config["line_count"],
                    device_type="cuda",
                    device_name=environment["device_name"],
                    real_dtype=config["dtype"],
                    value_type=candidate["value_type"],
                    selected_algorithm="fft",
                    evidence_id=evidence_id,
                )
            )

    policy = QualifiedBoundedTransformPolicy(
        name=policy_name,
        cells=tuple(cells),
    )
    return {
        "schema_version": 1,
        "artifact": "r2r_c_bounded_transform_policy",
        "source": {
            "path": str(Path(source_path).resolve()),
            "sha256": source_sha256,
            "benchmark": benchmark["benchmark"],
            "config": config,
            "environment": environment,
        },
        "criteria": {
            "minimum_speedup_dense_over_fft": minimum_speedup,
            "maximum_peak_allocated_ratio_fft_over_dense": (
                maximum_peak_allocated_ratio
            ),
            "minimum_timing_samples": minimum_samples,
            "finite_required": True,
            "dense_reference_error_within_case_tolerance_required": True,
            "unmatched_context_fallback": "dense",
        },
        "reviewed_directions": reviews,
        "accepted_cell_count": len(cells),
        "policy": policy.to_metadata(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-name", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--local-axis", type=int)
    parser.add_argument("--minimum-speedup", type=float, default=1.10)
    parser.add_argument(
        "--maximum-peak-allocated-ratio",
        type=float,
        default=3.0,
    )
    parser.add_argument("--minimum-samples", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.benchmark.is_file():
        parser.error(f"benchmark does not exist: {args.benchmark}")
    if args.output.exists() and not args.overwrite:
        parser.error(
            f"output already exists: {args.output}; pass --overwrite"
        )
    return args


def main() -> None:
    args = parse_args()
    source_sha256 = _sha256(args.benchmark)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    artifact = build_policy_artifact(
        benchmark,
        source_path=str(args.benchmark),
        source_sha256=source_sha256,
        policy_name=args.policy_name,
        geometry=args.geometry,
        local_axis=args.local_axis,
        minimum_speedup=args.minimum_speedup,
        maximum_peak_allocated_ratio=args.maximum_peak_allocated_ratio,
        minimum_samples=args.minimum_samples,
    )
    serialized = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
