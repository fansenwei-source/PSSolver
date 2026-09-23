#!/usr/bin/env python3
"""Fail-closed analysis of the frozen P7.6 Channel H100 profile matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
import statistics


RUNTIMES = ("legacy_channel", "compiled_channel_v2")
GRIDS = ((128, 20, 20), (512, 40, 40))
LENGTHS = {(128, 20, 20): (32.0, 5.0, 5.0), (512, 40, 40): (128.0, 10.0, 10.0)}
TRIALS = (1, 2, 3)
MEAN_RATIO_MAX = 1.02
MEDIAN_RATIO_MAX = 1.02
INDIVIDUAL_RATIO_MAX = 1.05
MEMORY_RATIO_MAX = 1.05
PCG_RATIO_MAX = 1.05


def _load(path: str | Path) -> tuple[Path, dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"profile is missing: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("profile must contain a JSON object")
    return source, dict(value)


def _record(path: str | Path, *, expected_commit: str, gpu: str) -> dict[str, object]:
    source, value = _load(path)
    try:
        config = value["config"]
        runtime = value["runtime_identity"]
        environment = value["environment"]
        shape = tuple(config["shape"])
        trial = int(config["trial"])
        name = config["runtime_path"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("profile lacks the P7.6 schema") from exc
    if name not in RUNTIMES or shape not in GRIDS or trial not in TRIALS:
        raise ValueError("profile identity is outside the P7.6 matrix")
    if tuple(config["lengths"]) != LENGTHS[shape]:
        raise ValueError("profile lengths differ from the P7.6 contract")
    frozen = {"device": "cuda", "dt": 0.01, "activity": 5.0, "warmup_steps": 10, "profile_steps": 50, "seed": 24}
    if any(config.get(key) != expected for key, expected in frozen.items()):
        raise ValueError("profile configuration differs from the P7.6 contract")
    if runtime.get("requested") != name or runtime.get("effective") != name or runtime.get("fallback_used") is not False:
        raise ValueError("profile runtime identity or fallback is invalid")
    git = environment.get("git", {})
    if git.get("head") != expected_commit or git.get("dirty") is not False:
        raise ValueError("profile Git identity is invalid")
    if environment.get("cuda_available") is not True or gpu not in str(environment.get("device_name")):
        raise ValueError("profile H100 identity is invalid")
    if environment.get("cuda_matmul_allow_tf32") is not False:
        raise ValueError("profile enabled TF32")
    if value.get("finite") is not True or value.get("completed_steps") != 60:
        raise ValueError("profile is incomplete or non-finite")
    calls = value["transform_calls"]
    pressure = value["pressure_iterations"]
    timing = float(value["throughput"]["mean_timestep_seconds"])
    allocated = int(value["memory"]["peak_allocated_bytes"])
    reserved = int(value["memory"]["peak_reserved_bytes"])
    if not all(math.isfinite(item) and item > 0 for item in (timing, float(pressure["mean"]))):
        raise ValueError("profile timing or PCG evidence is invalid")
    if allocated <= 0 or reserved <= 0 or calls["forward_per_step"] <= 0 or calls["inverse_per_step"] <= 0:
        raise ValueError("profile memory or transform evidence is invalid")
    return {
        "path": str(source), "runtime": name, "shape": shape, "trial": trial,
        "timestep": timing, "allocated": allocated, "reserved": reserved,
        "pcg_mean": float(pressure["mean"]), "pcg_max": int(pressure["maximum"]),
        "forward": float(calls["forward_per_step"]), "inverse": float(calls["inverse_per_step"]),
        "initial": value["initial_q_sha256"], "final": value["final_state_sha256"],
    }


def analyze_profiles(profile_paths: Sequence[str | Path], *, expected_commit: str, gpu: str = "H100") -> dict[str, object]:
    if len(profile_paths) != 12:
        raise ValueError("P7.6 requires exactly 12 profiler JSON files")
    records = [_record(path, expected_commit=expected_commit, gpu=gpu) for path in profile_paths]
    keyed = {(row["shape"], row["trial"], row["runtime"]): row for row in records}
    expected = {(shape, trial, runtime) for shape in GRIDS for trial in TRIALS for runtime in RUNTIMES}
    if set(keyed) != expected:
        raise ValueError("profile matrix is incomplete or duplicated")
    summaries = {}
    for shape in GRIDS:
        legacy = [keyed[(shape, trial, RUNTIMES[0])] for trial in TRIALS]
        compiled = [keyed[(shape, trial, RUNTIMES[1])] for trial in TRIALS]
        for left, right in zip(legacy, compiled, strict=True):
            if left["initial"] != right["initial"] or left["final"] != right["final"]:
                raise ValueError("paired Channel profile state identity failed")
            if left["forward"] != right["forward"] or left["inverse"] != right["inverse"]:
                raise ValueError("paired transform-call contract differs")
            if left["pcg_mean"] != right["pcg_mean"] or left["pcg_max"] != right["pcg_max"]:
                raise ValueError("paired PCG iteration path differs")
        ratios = [right["timestep"] / left["timestep"] for left, right in zip(legacy, compiled, strict=True)]
        mean_ratio = statistics.fmean(row["timestep"] for row in compiled) / statistics.fmean(row["timestep"] for row in legacy)
        median_ratio = statistics.median(row["timestep"] for row in compiled) / statistics.median(row["timestep"] for row in legacy)
        allocated_ratio = max(row["allocated"] for row in compiled) / max(row["allocated"] for row in legacy)
        reserved_ratio = max(row["reserved"] for row in compiled) / max(row["reserved"] for row in legacy)
        pcg_ratio = statistics.fmean(row["pcg_mean"] for row in compiled) / statistics.fmean(row["pcg_mean"] for row in legacy)
        gates = {
            "mean_timestep": mean_ratio <= MEAN_RATIO_MAX,
            "median_timestep": median_ratio <= MEDIAN_RATIO_MAX,
            "individual_timestep": max(ratios) <= INDIVIDUAL_RATIO_MAX,
            "allocated_memory": allocated_ratio <= MEMORY_RATIO_MAX,
            "reserved_memory": reserved_ratio <= MEMORY_RATIO_MAX,
            "pcg_iterations": pcg_ratio <= PCG_RATIO_MAX,
        }
        if not all(gates.values()):
            raise ValueError(f"P7.6 non-regression gate failed for {shape}: {gates}")
        summaries["x".join(map(str, shape))] = {
            "paired_candidate_over_legacy": ratios,
            "mean_timestep_ratio": mean_ratio,
            "median_timestep_ratio": median_ratio,
            "peak_allocated_ratio": allocated_ratio,
            "peak_reserved_ratio": reserved_ratio,
            "pcg_iteration_ratio": pcg_ratio,
            "transform_calls_per_step": {"forward": legacy[0]["forward"], "inverse": legacy[0]["inverse"]},
            "gates": gates,
        }
    return {
        "schema_version": 1,
        "classification": "PASS_P7_6_CHANNEL_H100_PROFILE_MATRIX",
        "expected_commit": expected_commit,
        "gpu": gpu,
        "profile_count": len(records),
        "grids": summaries,
        "eligible_for_trajectory_and_restart_gates": True,
        "production_default_changed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-gpu", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    result = analyze_profiles(args.profile, expected_commit=args.expected_commit, gpu=args.expected_gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
