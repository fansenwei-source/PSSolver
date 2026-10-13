#!/usr/bin/env python3
"""Fail-closed analyzer for the RC4.5.8 single-H100 closure."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from benchmarks.plan_rc458_tf32_h100 import build_plan
from benchmarks.profile_rc458_runtime_family import SCHEMA as PROFILE_SCHEMA


SCHEMA = "pssolver.rc4_5_8.tf32_h100_analysis.v1"
PASS_CLASSIFICATION = "PASS_RC4_5_8_SINGLE_H100_TF32_CUMULATIVE_CLOSURE"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("analysis JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _validate_auxiliary(value: dict[str, object]) -> None:
    _require(
        value.get("schema") == "pssolver.rc4_5_8.tf32_h100_auxiliary.v1",
        "auxiliary schema mismatch",
    )
    required = (
        "installed_wheel_identity_passed",
        "environment_passed",
        "cuda_only_tests_passed",
        "tf32_scope_gate_passed",
        "consumer_installed_wheel_h100_suite_passed",
        "source_shadow_import",
        "automatic_retry",
    )
    for name in required:
        _require(isinstance(value.get(name), bool), f"auxiliary {name} missing")
    for name in required[:5]:
        _require(value[name] is True, f"auxiliary {name} failed")
    _require(value["source_shadow_import"] is False, "source shadow import detected")
    _require(value["automatic_retry"] is False, "automatic retry was used")
    _require(value.get("formal_h100_submission_count") == 1, "submission count differs")
    _require(value.get("cuda_only_passed") == 7, "CUDA-only pass count differs")
    _require(value.get("cuda_only_failed") == 0, "CUDA-only failure recorded")
    _require(value.get("cuda_only_skipped") == 0, "CUDA-only skip recorded")
    _require(value.get("consumer_failed") == 0, "Consumer H100 failure recorded")
    _require(value.get("consumer_xfailed") == 0, "Consumer H100 xfail recorded")
    _require(
        value.get("consumer_skipped") == 1
        and value.get("consumer_skip_node_id")
        == "tests/test_warmup_experiments.py::test_r5_benchmark",
        "Consumer H100 skip identity differs",
    )


def analyze(profile_root: Path, auxiliary: dict[str, object]) -> dict[str, object]:
    plan = build_plan()
    thresholds = plan["thresholds"]
    _require(isinstance(thresholds, dict), "threshold contract must be an object")
    _validate_auxiliary(auxiliary)
    expected = plan["profile_matrix"]
    _require(isinstance(expected, list), "profile matrix must be a list")
    observed: dict[str, dict[str, object]] = {}
    for entry in expected:
        _require(isinstance(entry, dict), "profile matrix entry must be an object")
        key = entry["key"]
        _require(isinstance(key, str), "profile matrix key must be a string")
        path = profile_root / f"{key}.json"
        _require(path.is_file() and not path.is_symlink(), f"profile missing: {key}")
        profile = _json(path)
        _require(profile.get("schema") == PROFILE_SCHEMA, f"profile schema: {key}")
        config = profile.get("config")
        provider = profile.get("provider_identity")
        runtime = profile.get("runtime_identity")
        environment = profile.get("environment")
        timing = profile.get("timing")
        memory = profile.get("memory")
        transforms = profile.get("transform_calls")
        compiler = profile.get("compiler")
        for name, value in (
            ("config", config),
            ("provider_identity", provider),
            ("runtime_identity", runtime),
            ("environment", environment),
            ("timing", timing),
            ("memory", memory),
            ("transform_calls", transforms),
            ("compiler", compiler),
        ):
            _require(isinstance(value, dict), f"{key} {name} must be an object")
        grid_contract = plan["grid_contracts"][entry["grid"]]
        for name in ("family", "trial", "shape", "lengths", "device", "dtype"):
            if name == "device":
                expected_name = "cuda"
            elif name in {"shape", "lengths"}:
                expected_name = grid_contract[name]
            else:
                expected_name = entry[name]
            _require(config.get(name) == expected_name, f"{key} config {name} differs")
        _require(provider.get("role") == entry["provider_role"], f"{key} role differs")
        _require(provider.get("commit") == entry["provider_commit"], f"{key} commit differs")
        _require(runtime.get("requested") == runtime.get("effective"), f"{key} runtime mismatch")
        _require(runtime.get("fallback_used") is False, f"{key} runtime fallback")
        _require(environment.get("allocated_device") == "cuda:0", f"{key} device differs")
        _require("H100" in str(environment.get("device_name")), f"{key} is not H100")
        _require(profile.get("finite") is True, f"{key} is non-finite")
        _require(
            profile.get("completed_profile_steps") == thresholds["completed_steps"],
            f"{key} timed step count differs",
        )
        _require(
            profile.get("runtime_completed_steps")
            == int(entry["warmup_steps"]) + int(entry["timed_steps"]),
            f"{key} runtime step count differs",
        )
        _require(compiler.get("graph_breaks") == 0, f"{key} graph break")
        _require(compiler.get("compile_fallback") is False, f"{key} compile fallback")
        _require(
            transforms.get("forward_per_step") == entry["forward_transforms_per_step"],
            f"{key} forward transform count differs",
        )
        _require(
            transforms.get("inverse_per_step") == entry["inverse_transforms_per_step"],
            f"{key} inverse transform count differs",
        )
        samples = timing.get("samples_seconds")
        _require(
            isinstance(samples, list) and len(samples) == entry["timed_steps"],
            f"{key} timing sample count differs",
        )
        for index, value in enumerate(samples):
            _number(value, f"{key} sample {index}")
        _number(timing.get("mean_timestep_seconds"), f"{key} mean")
        _number(timing.get("median_timestep_seconds"), f"{key} median")
        _integer(memory.get("peak_allocated_bytes"), f"{key} peak allocated")
        _integer(memory.get("peak_reserved_bytes"), f"{key} peak reserved")
        for name in ("initial_state_sha256", "final_state_sha256"):
            value = profile.get(name)
            _require(isinstance(value, str) and len(value) == 64, f"{key} {name}")
        observed[key] = profile

    actual_files = sorted(path.name for path in profile_root.glob("*.json"))
    expected_files = sorted(f"{entry['key']}.json" for entry in expected)
    _require(actual_files == expected_files, "profile file set differs from frozen plan")

    groups: list[dict[str, object]] = []
    for family in plan["tf32_scope_gate"]["runtime_families"]:
        grids = sorted(
            {
                str(entry["grid"])
                for entry in expected
                if entry["family"] == family
            }
        )
        for grid in grids:
            baseline = []
            candidate = []
            allocated_ratios = []
            reserved_ratios = []
            for trial in range(1, 4):
                a = observed[f"baseline_{family}_{grid}_trial{trial}"]
                b = observed[f"candidate_{family}_{grid}_trial{trial}"]
                _require(
                    a["initial_state_sha256"] == b["initial_state_sha256"],
                    f"{family}/{grid}/trial{trial} initial state differs",
                )
                _require(
                    a["final_state_sha256"] == b["final_state_sha256"],
                    f"{family}/{grid}/trial{trial} final state differs",
                )
                at = _number(a["timing"]["mean_timestep_seconds"], "baseline time")
                bt = _number(b["timing"]["mean_timestep_seconds"], "candidate time")
                baseline.append(at)
                candidate.append(bt)
                pair_ratio = bt / at
                _require(
                    pair_ratio <= thresholds["paired_timestep_ratio_max"],
                    f"{family}/{grid}/trial{trial} paired time regression",
                )
                for name, ratios in (
                    ("peak_allocated_bytes", allocated_ratios),
                    ("peak_reserved_bytes", reserved_ratios),
                ):
                    av = _integer(a["memory"][name], f"baseline {name}")
                    bv = _integer(b["memory"][name], f"candidate {name}")
                    _require(av > 0 and bv > 0, f"{family}/{grid} invalid memory")
                    ratios.append(bv / av)
            mean_ratio = statistics.fmean(candidate) / statistics.fmean(baseline)
            median_ratio = statistics.median(candidate) / statistics.median(baseline)
            _require(mean_ratio <= thresholds["mean_timestep_ratio_max"], f"{family}/{grid} mean regression")
            _require(median_ratio <= thresholds["median_timestep_ratio_max"], f"{family}/{grid} median regression")
            _require(max(allocated_ratios) <= thresholds["peak_allocated_ratio_max"], f"{family}/{grid} allocated regression")
            _require(max(reserved_ratios) <= thresholds["peak_reserved_ratio_max"], f"{family}/{grid} reserved regression")
            groups.append(
                {
                    "family": family,
                    "grid": grid,
                    "mean_timestep_ratio": mean_ratio,
                    "median_timestep_ratio": median_ratio,
                    "paired_timestep_ratios": [b / a for a, b in zip(baseline, candidate, strict=True)],
                    "peak_allocated_ratios": allocated_ratios,
                    "peak_reserved_ratios": reserved_ratios,
                    "passed": True,
                }
            )
    return {
        "schema": SCHEMA,
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
        "profile_count": len(observed),
        "group_count": len(groups),
        "groups": groups,
        "auxiliary": auxiliary,
        "eligible_for_rc4_release_review": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, analyze(args.profile_root, _json(args.auxiliary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
