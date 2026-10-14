#!/usr/bin/env python3
"""Exercise RC4.5.8 scoped TF32 semantics on the allocated H100."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from benchmarks.plan_rc458_tf32_h100 import CALLER_STATES, FAMILIES
from benchmarks.profile_rc458_runtime_family import (
    RC458ProfileConfig,
    _build_case,
    _finite,
)
from pssolver.configuration.execution_device import (
    scoped_tf32_execution,
    snapshot_tf32_execution_state,
)


SCHEMA = "pssolver.rc4_5_8.tf32_scope_h100.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _set_state(value: dict[str, object]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(value["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(value["cudnn_allow_tf32"])
    torch.set_float32_matmul_precision(str(value["float32_matmul_precision"]))


def _metadata() -> dict[str, object]:
    return snapshot_tf32_execution_state().to_metadata()


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
            raise RuntimeError("TF32 report JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _primitive_gates(caller: dict[str, object]) -> dict[str, object]:
    _set_state(caller)
    before = _metadata()
    exceptional = False
    try:
        with scoped_tf32_execution(
            "off", real_dtype=torch.float32, device="cuda"
        ):
            raise LookupError("rc458 intentional scope exception")
    except LookupError:
        exceptional = True
    _require(_metadata() == before, "exceptional scope did not restore caller")

    with scoped_tf32_execution(
        "off", real_dtype=torch.float32, device="cuda"
    ):
        outer = _metadata()
        with scoped_tf32_execution(
            "on", real_dtype=torch.float32, device="cuda"
        ):
            inner = _metadata()
            _require(inner["cuda_matmul_allow_tf32"] is True, "nested on failed")
            _require(inner["cudnn_allow_tf32"] is True, "nested cuDNN on failed")
        _require(_metadata() == outer, "nested inner scope did not restore LIFO")
    _require(_metadata() == before, "nested outer scope did not restore caller")

    invalid_rejected = False
    try:
        with scoped_tf32_execution(
            "invalid", real_dtype=torch.float32, device="cuda"
        ):
            raise AssertionError("invalid policy entered its body")
    except ValueError:
        invalid_rejected = True
    _require(_metadata() == before, "invalid policy mutated caller state")
    return {
        "caller": before,
        "exceptional_restoration": exceptional,
        "nested_lifo": True,
        "invalid_policy_rejected_without_mutation": invalid_rejected,
    }


def run_gate(
    *, provider_commit: str, provider_wheel_sha256: str
) -> dict[str, object]:
    if len(provider_commit) != 40 or len(provider_wheel_sha256) != 64:
        raise ValueError("full Provider commit and wheel SHA-256 are required")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one visible CUDA device required")
    device = torch.empty((), device="cuda").device
    _require(str(device) == "cuda:0", "allocated device must be cuda:0")
    original = _metadata()
    primitive = []
    families = []
    try:
        for caller in CALLER_STATES:
            primitive.append(_primitive_gates(caller))
            for family in FAMILIES:
                _set_state(caller)
                before = _metadata()
                channel = family.key.startswith("channel_")
                shape = (8, 6, 4) if channel else (8, 8, 4)
                lengths = (8.0, 6.0, 4.0) if channel else (8.0, 8.0, 4.0)
                with tempfile.TemporaryDirectory(prefix="pssolver-rc458-scope-") as temporary:
                    config = RC458ProfileConfig(
                        family=family.key,
                        provider_role="candidate",
                        provider_commit=provider_commit,
                        provider_wheel_sha256=provider_wheel_sha256,
                        trial=1,
                        shape=shape,
                        lengths=lengths,
                        device="cuda",
                        dtype=family.dtype,
                        warmup_steps=0,
                        profile_steps=1,
                    )
                    with torch.no_grad():
                        case = _build_case(Path(temporary), config)
                        after_construction = _metadata()
                        _require(
                            after_construction == before,
                            f"{family.key} construction mutated caller state",
                        )
                        case.step()
                        torch.cuda.synchronize(device)
                        after_step = _metadata()
                        _require(
                            after_step == before,
                            f"{family.key} step did not restore caller state",
                        )
                        finite = _finite(case.state())
                        _require(finite, f"{family.key} produced non-finite state")
                families.append(
                    {
                        "caller": caller["key"],
                        "family": family.key,
                        "construction_preserved": True,
                        "step_restored": True,
                        "finite": finite,
                        "runtime_fallback": case.runtime_identity["fallback_used"],
                        "passed": case.runtime_identity["fallback_used"] is False,
                    }
                )
    finally:
        _set_state(
            {
                "float32_matmul_precision": original[
                    "torch_float32_matmul_precision"
                ],
                "cuda_matmul_allow_tf32": original["cuda_matmul_allow_tf32"],
                "cudnn_allow_tf32": original["cudnn_allow_tf32"],
            }
        )
    _require(all(row["passed"] for row in families), "runtime-family scope gate failed")
    return {
        "schema": SCHEMA,
        "provider_commit": provider_commit,
        "provider_wheel_sha256": provider_wheel_sha256,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "caller_state_count": len(primitive),
        "runtime_family_case_count": len(families),
        "primitive_gates": primitive,
        "runtime_family_gates": families,
        "original_state_restored": _metadata() == original,
        "passed": len(families) == 48 and all(row["passed"] for row in families),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-commit", required=True)
    parser.add_argument("--provider-wheel-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(
        args.output,
        run_gate(
            provider_commit=args.provider_commit,
            provider_wheel_sha256=args.provider_wheel_sha256,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
