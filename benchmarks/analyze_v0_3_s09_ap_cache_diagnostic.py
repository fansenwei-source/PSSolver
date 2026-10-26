#!/usr/bin/env python3
"""Summarize S-09 A-p cache diagnostic data without a qualification verdict."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics


SCHEMA = "pssolver.v0_3.s09.ap_cache_diagnostic_analysis.v1"
ROLES = ("baseline", "candidate_cache_on", "candidate_cache_off")


def _atomic_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(payload):
            raise RuntimeError("diagnostic analysis JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _positive_number(value: object, description: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{description} must be positive and finite")
    return result


def _ratio_summary(
    reference: list[dict[str, object]],
    observed: list[dict[str, object]],
) -> dict[str, object]:
    by_trial_reference = {int(item["trial"]): item for item in reference}
    by_trial_observed = {int(item["trial"]): item for item in observed}
    if set(by_trial_reference) != set(range(1, 7)) or set(by_trial_observed) != set(range(1, 7)):
        raise ValueError("each role must contain trials 1..6")
    trial_ratios: list[float] = []
    baseline_then_observed: list[float] = []
    observed_then_baseline: list[float] = []
    for trial in range(1, 7):
        baseline = by_trial_reference[trial]
        candidate = by_trial_observed[trial]
        ratio = _positive_number(candidate["mean"], "candidate mean") / _positive_number(
            baseline["mean"], "baseline mean"
        )
        trial_ratios.append(ratio)
        if int(baseline["position"]) < int(candidate["position"]):
            baseline_then_observed.append(ratio)
        else:
            observed_then_baseline.append(ratio)
    ref_samples = [value for item in reference for value in item["samples"]]
    obs_samples = [value for item in observed for value in item["samples"]]
    forward_order_median = statistics.median(baseline_then_observed)
    reverse_order_median = statistics.median(observed_then_baseline)
    return {
        "aggregate_all_sample_mean_ratio": statistics.fmean(obs_samples)
        / statistics.fmean(ref_samples),
        "median_of_profile_means_ratio": statistics.median(
            [float(item["mean"]) for item in observed]
        )
        / statistics.median([float(item["mean"]) for item in reference]),
        "median_of_six_trial_ratios": statistics.median(trial_ratios),
        "six_trial_ratios": trial_ratios,
        "trials_at_or_below_frozen_1_05_reference": sum(
            ratio <= 1.05 for ratio in trial_ratios
        ),
        "baseline_then_observed_median_ratio": forward_order_median,
        "observed_then_baseline_median_ratio": reverse_order_median,
        "order_stratum_imbalance_ratio": max(
            forward_order_median / reverse_order_median,
            reverse_order_median / forward_order_median,
        ),
        "peak_allocated_group_ratio": max(
            int(item["peak_allocated"]) for item in observed
        )
        / max(int(item["peak_allocated"]) for item in reference),
        "peak_reserved_group_ratio": max(
            int(item["peak_reserved"]) for item in observed
        )
        / max(int(item["peak_reserved"]) for item in reference),
        "frozen_reference_only": {
            "aggregate_metrics_at_or_below_1_03": None,
            "order_metrics_at_or_below_1_05": None,
            "peak_memory_at_or_below_1_03": None,
            "not_a_qualification_verdict": True,
        },
    }


def analyze(manifest_path: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("profiles")
    if not isinstance(entries, list) or len(entries) != 72:
        raise ValueError("complete 72-profile matrix manifest required")
    rows: list[dict[str, object]] = []
    for entry in entries:
        result_path = Path(str(entry["result"]))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        config = result["config"]
        samples = [
            _positive_number(value, "timing sample")
            for value in result["timing"]["samples_seconds"]
        ]
        rows.append(
            {
                "key": entry["key"],
                "role": result["diagnostic_role"],
                "family": config["family"],
                "grid": str(entry["key"]).split(f"{config['family']}_", 1)[1].rsplit("_trial", 1)[0],
                "trial": int(config["trial"]),
                "position": int(
                    next(
                        item["order_position"]
                        for item in json.loads(Path(manifest["plan"]).read_text(encoding="utf-8"))["profiles"]
                        if item["key"] == entry["key"]
                    )
                ),
                "samples": samples,
                "mean": statistics.fmean(samples),
                "peak_allocated": int(result["memory"]["peak_allocated_bytes"]),
                "peak_reserved": int(result["memory"]["peak_reserved_bytes"]),
                "initial_state_sha256": result["initial_state_sha256"],
                "final_state_sha256": result["final_state_sha256"],
                "cache_proof": result["cache_control"],
                "finite": result["finite"],
            }
        )
    plan = json.loads(Path(manifest["plan"]).read_text(encoding="utf-8"))
    groups: list[dict[str, object]] = []
    for family in ("channel_legacy", "channel_compiled_v2"):
        for grid in ("channel_R128x20x20", "channel_R512x40x40"):
            group = [row for row in rows if row["family"] == family and row["grid"] == grid]
            roles = {
                role: [row for row in group if row["role"] == role]
                for role in ROLES
            }
            if any(len(values) != 6 for values in roles.values()):
                raise ValueError(f"incomplete diagnostic group {family}/{grid}")
            state_pairs = []
            for trial in range(1, 7):
                on = next(row for row in roles["candidate_cache_on"] if row["trial"] == trial)
                off = next(row for row in roles["candidate_cache_off"] if row["trial"] == trial)
                state_pairs.append(
                    {
                        "trial": trial,
                        "initial_equal": on["initial_state_sha256"] == off["initial_state_sha256"],
                        "final_equal": on["final_state_sha256"] == off["final_state_sha256"],
                    }
                )
            groups.append(
                {
                    "family": family,
                    "grid": grid,
                    "cache_on_vs_baseline": _ratio_summary(
                        roles["baseline"], roles["candidate_cache_on"]
                    ),
                    "cache_off_vs_baseline": _ratio_summary(
                        roles["baseline"], roles["candidate_cache_off"]
                    ),
                    "cache_off_vs_cache_on": _ratio_summary(
                        roles["candidate_cache_on"], roles["candidate_cache_off"]
                    ),
                    "candidate_on_off_state_identity": state_pairs,
                }
            )
    all_off = [row for row in rows if row["role"] == "candidate_cache_off"]
    state_identity = all(
        pair["initial_equal"] and pair["final_equal"]
        for group in groups
        for pair in group["candidate_on_off_state_identity"]
    )
    proofs = all(row["cache_proof"]["proved_disabled"] is True for row in all_off)
    return {
        "schema": SCHEMA,
        "classification": "S09_AP_CACHE_DIAGNOSTIC_DATA_COMPLETE",
        "diagnostic_only": True,
        "qualification_complete": False,
        "s09_complete": False,
        "decision_made": False,
        "requires_user_decision_after_independent_review": True,
        "mode": plan["mode"],
        "profile_count": len(rows),
        "all_profiles_finite": all(row["finite"] is True for row in rows),
        "all_cache_off_proofs_passed": proofs,
        "candidate_cache_on_off_state_byte_identity": state_identity,
        "frozen_thresholds_unchanged_and_not_used_for_verdict": True,
        "groups": groups,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, analyze(args.manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
