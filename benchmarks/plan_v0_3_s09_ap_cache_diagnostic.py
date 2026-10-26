#!/usr/bin/env python3
"""Plan the non-qualifying S-09 Channel A-p cache diagnostic.

The diagnostic keeps the frozen S-09 profile measurement contract but narrows
the matrix to the two Channel runtime families and two Channel grids implicated
by V-41.  It deliberately has no qualification verdict or promotion authority.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


SCHEMA = "pssolver.v0_3.s09.ap_cache_diagnostic_plan.v1"
CLASSIFICATION = "READY_S09_AP_CACHE_DIAGNOSTIC_FOR_R0028_REVIEW"

BASELINE_COMMIT = "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
BASELINE_WHEEL_SHA256 = (
    "a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e"
)
CANDIDATE_COMMIT = "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
CANDIDATE_WHEEL_SHA256 = (
    "4003645025da6d81317eeb7261ccf606aef7f60f20d072518f8baacf56cd4aa8"
)

ROLES = ("baseline", "candidate_cache_on", "candidate_cache_off")
FAMILIES = ("channel_legacy", "channel_compiled_v2")
H100_GRIDS = {
    "channel_R128x20x20": {
        "shape": [128, 20, 20],
        "lengths": [40.0, 6.0, 6.0],
    },
    "channel_R512x40x40": {
        "shape": [512, 40, 40],
        "lengths": [160.0, 12.0, 12.0],
    },
}
CPU_REHEARSAL_GRIDS = {
    "channel_R128x20x20": {
        "shape": [8, 6, 4],
        "lengths": [40.0, 6.0, 6.0],
    },
    "channel_R512x40x40": {
        "shape": [12, 8, 6],
        "lengths": [160.0, 12.0, 12.0],
    },
}


def _role_order(trial: int) -> tuple[str, str, str]:
    offset = (trial - 1) % len(ROLES)
    return ROLES[offset:] + ROLES[:offset]


def build_profile_matrix(*, mode: str = "h100") -> list[dict[str, object]]:
    """Return 72 independently executed profiles in balanced three-role order."""

    if mode not in {"h100", "cpu_rehearsal"}:
        raise ValueError("mode must be h100 or cpu_rehearsal")
    grids = H100_GRIDS if mode == "h100" else CPU_REHEARSAL_GRIDS
    warmup_steps = 10 if mode == "h100" else 1
    timed_steps = 60 if mode == "h100" else 2
    rows: list[dict[str, object]] = []
    sequence = 0
    for family in FAMILIES:
        for grid, geometry in grids.items():
            for trial in range(1, 7):
                for position, role in enumerate(_role_order(trial), start=1):
                    sequence += 1
                    baseline = role == "baseline"
                    rows.append(
                        {
                            "sequence": sequence,
                            "key": f"{role}_{family}_{grid}_trial{trial}",
                            "family": family,
                            "grid": grid,
                            "trial": trial,
                            "order_position": position,
                            "diagnostic_role": role,
                            "provider_role": "baseline" if baseline else "candidate",
                            "provider_commit": (
                                BASELINE_COMMIT if baseline else CANDIDATE_COMMIT
                            ),
                            "provider_wheel_sha256": (
                                BASELINE_WHEEL_SHA256
                                if baseline
                                else CANDIDATE_WHEEL_SHA256
                            ),
                            "shape": list(geometry["shape"]),
                            "lengths": list(geometry["lengths"]),
                            "device": "cuda" if mode == "h100" else "cpu",
                            "dtype": "float32",
                            "warmup_steps": warmup_steps,
                            "timed_steps": timed_steps,
                            "samples_per_profile": timed_steps,
                            "fixed_block_size": 10 if mode == "h100" else None,
                            "fixed_block_count": 6 if mode == "h100" else None,
                            "steady_forward_transforms_per_step": 11,
                            "steady_inverse_transforms_per_step": 36,
                            "scheduled_refresh_interval": 20,
                            "isolated_subprocess": True,
                            "isolated_empty_triton_cache": True,
                            "isolated_empty_torchinductor_cache": True,
                            "cache_mode": (
                                "diagnostic_harness_disabled"
                                if role == "candidate_cache_off"
                                else "runtime_default"
                            ),
                        }
                    )
    if len(rows) != 72:
        raise AssertionError("diagnostic matrix must contain exactly 72 profiles")
    return rows


def build_plan(*, mode: str = "h100") -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "plan_step": "S-09",
        "decision": "D-0012",
        "review_request": "R-0028",
        "mode": mode,
        "diagnostic_only": True,
        "qualification_complete": False,
        "s09_complete": False,
        "changes_runtime_or_wheel": False,
        "changes_frozen_plan_or_thresholds": False,
        "purpose": (
            "Measure timing and peak CUDA memory with the candidate Channel "
            "A-p warm-start cache enabled and disabled before deciding V-41."
        ),
        "identities": {
            "baseline_commit": BASELINE_COMMIT,
            "baseline_wheel_sha256": BASELINE_WHEEL_SHA256,
            "candidate_commit": CANDIDATE_COMMIT,
            "candidate_wheel_sha256": CANDIDATE_WHEEL_SHA256,
        },
        "measurement_contract": {
            "families": list(FAMILIES),
            "grids": list(H100_GRIDS),
            "trials_per_cell": 6,
            "roles": list(ROLES),
            "profiles": 72,
            "formal_samples_per_profile": 60,
            "all_samples_retained": True,
            "balanced_cyclic_role_order": True,
            "independent_profile_subprocesses": True,
            "empty_compile_cache_per_profile": True,
            "frozen_reference_thresholds_reported_not_adjudicated": {
                "aggregate_ratio": 1.03,
                "paired_or_order_ratio": 1.05,
                "peak_memory_ratio": 1.03,
            },
        },
        "cache_off_contract": {
            "implemented_only_in_diagnostic_harness": True,
            "runtime_and_wheel_unchanged": True,
            "cached_operator_action_lookup_always_misses": True,
            "operator_action_is_never_stored": True,
            "pressure_warm_start_remains_enabled": True,
            "proof_requires_suppressed_store_and_lookup_counts": True,
            "proof_requires_cache_fields_none_after_every_store": True,
        },
        "profiles": build_profile_matrix(mode=mode),
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
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
            raise RuntimeError("diagnostic plan JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("h100", "cpu_rehearsal"), default="h100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, build_plan(mode=args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
