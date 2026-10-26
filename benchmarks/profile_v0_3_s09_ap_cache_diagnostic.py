#!/usr/bin/env python3
"""Run one non-qualifying S-09 Channel A-p cache diagnostic profile."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import tempfile
import time
from typing import Iterator

import torch

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import profile_rc458_runtime_family as frozen
from benchmarks.plan_v0_3_s09_ap_cache_diagnostic import ROLES
from benchmarks.v0_3_s09_revert_pressure_proxy import use_revert_pressure_proxy
from pssolver.linear_solvers.stokes.channel_no_slip import (
    ChannelNoSlipModalStokesSolver,
)


SCHEMA = "pssolver.v0_3.s09.ap_cache_diagnostic_profile.v1"


def _empty_cache_proof(role: str) -> dict[str, object]:
    return {
        "diagnostic_role": role,
        "patch_applied": False,
        "lookup_calls": 0,
        "store_calls": 0,
        "suppressed_operator_action_stores": 0,
        "cache_fields_nonempty_after_store": 0,
        "all_cache_lookups_forced_to_miss": False,
        "all_operator_action_stores_suppressed": False,
        "pressure_warm_start_preserved": True,
        "proved_disabled": False,
    }


@contextmanager
def disable_operator_action_cache(role: str) -> Iterator[dict[str, object]]:
    """Disable only the A-p cache while retaining ``pressure_guess``.

    This is a diagnostic-harness monkeypatch.  The installed Provider wheel is
    neither edited nor rebuilt.  Restoring both methods in ``finally`` keeps
    the override scoped to one independent profile process.
    """

    proof = _empty_cache_proof(role)
    if role != "candidate_cache_off":
        yield proof
        return

    cls = ChannelNoSlipModalStokesSolver
    original_store = cls._store_pressure_guess
    original_cached = cls._cached_pressure_operator_action

    def store_without_operator_action(
        self,
        pressure_hat: torch.Tensor,
        *,
        operator_action: torch.Tensor | None = None,
        working_exponent: int | None = None,
    ) -> None:
        proof["store_calls"] = int(proof["store_calls"]) + 1
        if operator_action is not None or working_exponent is not None:
            proof["suppressed_operator_action_stores"] = (
                int(proof["suppressed_operator_action_stores"]) + 1
            )
        original_store(
            self,
            pressure_hat,
            operator_action=None,
            working_exponent=None,
        )
        names = (
            "_pressure_guess_operator_action",
            "_pressure_guess_operator_action_guess",
            "_pressure_guess_operator_action_guess_version",
            "_pressure_guess_operator_action_working_exponent",
        )
        if any(getattr(self, name, None) is not None for name in names):
            proof["cache_fields_nonempty_after_store"] = (
                int(proof["cache_fields_nonempty_after_store"]) + 1
            )

    def cached_operator_action_always_misses(
        self,
        pressure_guess: torch.Tensor,
        pressure_work: torch.Tensor,
        working_exponent: int,
    ) -> None:
        del pressure_guess, pressure_work, working_exponent
        proof["lookup_calls"] = int(proof["lookup_calls"]) + 1
        self._clear_pressure_guess_operator_action()
        return None

    cls._store_pressure_guess = store_without_operator_action
    cls._cached_pressure_operator_action = cached_operator_action_always_misses
    proof["patch_applied"] = True
    try:
        yield proof
    finally:
        cls._store_pressure_guess = original_store
        cls._cached_pressure_operator_action = original_cached
        proof["all_cache_lookups_forced_to_miss"] = int(proof["lookup_calls"]) > 0
        proof["all_operator_action_stores_suppressed"] = (
            int(proof["suppressed_operator_action_stores"]) > 0
            and int(proof["cache_fields_nonempty_after_store"]) == 0
        )
        proof["proved_disabled"] = bool(
            proof["patch_applied"]
            and proof["all_cache_lookups_forced_to_miss"]
            and proof["all_operator_action_stores_suppressed"]
        )


def _config(args: argparse.Namespace) -> frozen.RC458ProfileConfig:
    return frozen.RC458ProfileConfig(
        family=args.family,
        provider_role=args.provider_role,
        provider_commit=args.provider_commit,
        provider_wheel_sha256=args.provider_wheel_sha256,
        trial=args.trial,
        shape=args.shape,
        lengths=args.lengths,
        device=args.execution_device,
        dtype=args.dtype,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
        post_flush_rewarm_steps=0,
        seed=args.seed,
    )


def _run_cpu_rehearsal(
    config: frozen.RC458ProfileConfig,
) -> dict[str, object]:
    """Exercise the complete child flow without claiming CPU performance data."""

    torch.manual_seed(config.seed)
    with tempfile.TemporaryDirectory(prefix="pssolver-s09-cache-cpu-") as temporary:
        with torch.no_grad():
            case = frozen._build_case(Path(temporary), config)
            for _ in range(config.warmup_steps):
                case.step()
            samples: list[float] = []
            for _ in range(config.profile_steps):
                started = time.perf_counter()
                case.step()
                samples.append(time.perf_counter() - started)
            state = case.state()
            final_sha = frozen._tensor_sha256(state)
            finite = frozen._finite(state)
            completed = case.completed_steps()
            runtime_identity = dict(case.runtime_identity)
            configuration_identity = dict(case.configuration_identity)
            initial_sha = case.initial_state_sha256
    maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return {
        "schema": frozen.SCHEMA,
        "config": {
            "family": config.family,
            "provider_role": config.provider_role,
            "provider_commit": config.provider_commit,
            "provider_wheel_sha256": config.provider_wheel_sha256,
            "trial": config.trial,
            "shape": list(config.shape),
            "lengths": list(config.lengths),
            "device": "cpu",
            "dtype": config.dtype,
            "warmup_steps": config.warmup_steps,
            "profile_steps": config.profile_steps,
            "post_flush_rewarm_steps": 0,
            "seed": config.seed,
        },
        "provider_identity": {
            "role": config.provider_role,
            "commit": config.provider_commit,
            "wheel_sha256": config.provider_wheel_sha256,
        },
        "configuration_identity": configuration_identity,
        "runtime_identity": runtime_identity,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "requested_device": "cpu",
            "allocated_device": "cpu",
            "pssolver_import": str(Path(__import__("pssolver").__file__).resolve()),
        },
        "timing": {
            "samples_seconds": samples,
            "mean_timestep_seconds": statistics.fmean(samples),
            "median_timestep_seconds": statistics.median(samples),
            "cpu_rehearsal_only": True,
        },
        "memory": {
            "peak_allocated_bytes": maximum_rss,
            "peak_reserved_bytes": maximum_rss,
            "cpu_ru_maxrss_proxy_only": True,
        },
        "initial_state_sha256": initial_sha,
        "final_state_sha256": final_sha,
        "completed_profile_steps": config.profile_steps,
        "runtime_completed_steps": completed,
        "finite": finite,
    }


def run_diagnostic(
    config: frozen.RC458ProfileConfig,
    *,
    diagnostic_role: str,
) -> dict[str, object]:
    if diagnostic_role not in ROLES:
        raise ValueError(f"diagnostic_role must be one of {ROLES}")
    expected_provider_role = "baseline" if diagnostic_role == "baseline" else "candidate"
    if config.provider_role != expected_provider_role:
        raise ValueError("diagnostic role and provider role disagree")
    with use_revert_pressure_proxy(diagnostic_role) as proof:
        result = (
            frozen.run_profile(config)
            if config.device == "cuda"
            else _run_cpu_rehearsal(config)
        )
    if (
        diagnostic_role == "candidate_revert_proxy"
        and not proof["proved_revert_proxy"]
    ):
        raise RuntimeError(f"f94b9be revert-proxy proof failed: {proof!r}")
    result = dict(result)
    result.update(
        {
            "schema": SCHEMA,
            "diagnostic_role": diagnostic_role,
            "diagnostic_only": True,
            "qualification_complete": False,
            "s09_complete": False,
            "cache_control": proof,
        }
    )
    return result


def _triplet(value: str, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("channel_legacy", "channel_compiled_v2"), required=True)
    parser.add_argument("--diagnostic-role", choices=ROLES, required=True)
    parser.add_argument("--provider-role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--provider-commit", required=True)
    parser.add_argument("--provider-wheel-sha256", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--shape", type=lambda value: _triplet(value, int), required=True)
    parser.add_argument("--lengths", type=lambda value: _triplet(value, float), required=True)
    parser.add_argument("--execution-device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--profile-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = _config(args)
    result = run_diagnostic(config, diagnostic_role=args.diagnostic_role)
    frozen._atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
