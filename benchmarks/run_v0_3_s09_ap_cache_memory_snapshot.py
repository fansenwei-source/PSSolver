#!/usr/bin/env python3
"""Run one untimed R512 CUDA allocator snapshot for V-41 attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import profile_rc458_runtime_family as frozen
from benchmarks.profile_v0_3_s09_ap_cache_diagnostic import _config, _triplet
from benchmarks.v0_3_s09_revert_pressure_proxy import use_revert_pressure_proxy


SCHEMA = "pssolver.v0_3.s09.ap_cache_memory_snapshot.v1"
ROLES = ("candidate_cache_on", "candidate_revert_proxy")
TF32_OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _tf32() -> dict[str, object]:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            raise RuntimeError("memory-snapshot JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def run_snapshot(
    config: frozen.RC458ProfileConfig,
    *,
    diagnostic_role: str,
    snapshot_path: Path,
) -> dict[str, object]:
    if diagnostic_role not in ROLES:
        raise ValueError(f"snapshot role must be one of {ROLES}")
    if config.family != "channel_legacy":
        raise ValueError("V-41 snapshot uses representative channel_legacy only")
    snapshot_path = snapshot_path.absolute()
    if snapshot_path.exists() or snapshot_path.is_symlink():
        raise FileExistsError(f"snapshot output already exists: {snapshot_path}")

    tf32_before = _tf32()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tf32_after = _tf32()
    if tf32_after != TF32_OFF:
        raise RuntimeError(f"TF32-off enforcement failed: {tf32_after!r}")
    torch.manual_seed(config.seed)
    snapshot_taken = False
    with use_revert_pressure_proxy(diagnostic_role) as proof:
        with tempfile.TemporaryDirectory(prefix="pssolver-s09-snapshot-") as temporary:
            with torch.no_grad():
                case = frozen._build_case(Path(temporary), config)
                for _ in range(config.warmup_steps):
                    case.step()
                if config.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.memory._record_memory_history(max_entries=200_000)
                try:
                    for _ in range(config.profile_steps):
                        case.step()
                    if config.device == "cuda":
                        torch.cuda.synchronize()
                        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.cuda.memory._dump_snapshot(str(snapshot_path))
                        snapshot_taken = True
                    state = case.state()
                    final_sha = frozen._tensor_sha256(state)
                    finite = frozen._finite(state)
                    completed = case.completed_steps()
                finally:
                    if config.device == "cuda":
                        torch.cuda.memory._record_memory_history(enabled=None)
    if diagnostic_role == "candidate_revert_proxy" and not proof["proved_revert_proxy"]:
        raise RuntimeError(f"revert-proxy proof failed during snapshot: {proof!r}")
    if config.device == "cuda" and not snapshot_taken:
        raise RuntimeError("CUDA memory snapshot was not persisted")
    return {
        "schema": SCHEMA,
        "diagnostic_only": True,
        "included_in_timing_statistics": False,
        "diagnostic_role": diagnostic_role,
        "family": config.family,
        "shape": list(config.shape),
        "lengths": list(config.lengths),
        "device": config.device,
        "warmup_steps": config.warmup_steps,
        "profile_steps": config.profile_steps,
        "runtime_completed_steps": completed,
        "finite": finite,
        "final_state_sha256": final_sha,
        "revert_proxy": proof,
        "snapshot": {
            "requested": config.device == "cuda",
            "taken": snapshot_taken,
            "cpu_rehearsal_skip": config.device == "cpu",
            "path": str(snapshot_path) if snapshot_taken else None,
            "sha256": _sha256(snapshot_path) if snapshot_taken else None,
        },
        "tf32": {"before": tf32_before, "after": tf32_after},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-role", choices=ROLES, required=True)
    parser.add_argument("--family", default="channel_legacy")
    parser.add_argument("--provider-role", default="candidate")
    parser.add_argument("--provider-commit", required=True)
    parser.add_argument("--provider-wheel-sha256", required=True)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--shape", type=lambda value: _triplet(value, int), required=True)
    parser.add_argument("--lengths", type=lambda value: _triplet(value, float), required=True)
    parser.add_argument("--execution-device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config = _config(args)
    report = run_snapshot(
        config,
        diagnostic_role=args.diagnostic_role,
        snapshot_path=args.snapshot,
    )
    _atomic_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
