#!/usr/bin/env python3
"""Enforce and persist the RC6 TF32 policy before one profile child runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from benchmarks import profile_rc458_runtime_family as profile


SCHEMA = "pssolver.rc6.profile_child_tf32_bootstrap.v1"
TF32_OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
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
            raise RuntimeError("TF32-bootstrap JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _capture_tf32() -> dict[str, object]:
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def _enforce_tf32_off() -> tuple[dict[str, object], dict[str, object], list[dict[str, str]]]:
    before = _capture_tf32()
    errors: list[dict[str, str]] = []
    assignments = (
        (
            "float32_matmul_precision",
            lambda: torch.set_float32_matmul_precision("highest"),
        ),
        (
            "cuda_matmul_allow_tf32",
            lambda: setattr(torch.backends.cuda.matmul, "allow_tf32", False),
        ),
        (
            "cudnn_allow_tf32",
            lambda: setattr(torch.backends.cudnn, "allow_tf32", False),
        ),
    )
    for name, assignment in assignments:
        try:
            assignment()
        except Exception as exc:  # pragma: no cover - backend-specific failure path
            errors.append(
                {
                    "field": name,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
    return before, _capture_tf32(), errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--tf32-bootstrap-output", type=Path, required=True)
    parser.add_argument("--profile-script", type=Path, required=True)
    args, profile_argv = parser.parse_known_args(argv)

    expected_script = Path(profile.__file__).resolve()
    requested_script = args.profile_script.resolve()
    if requested_script != expected_script:
        raise ValueError(
            f"profile script identity differs: {requested_script} != {expected_script}"
        )
    if not requested_script.is_file() or requested_script.is_symlink():
        raise FileNotFoundError(
            f"regular profile support script required: {requested_script}"
        )

    before, after, assignment_errors = _enforce_tf32_off()
    passed = not assignment_errors and after == TF32_OFF
    report = {
        "schema": SCHEMA,
        "profile_script": str(requested_script),
        "profile_script_sha256": _sha256(requested_script),
        "policy": TF32_OFF,
        "before": before,
        "after": after,
        "assignment_errors": assignment_errors,
        "passed": passed,
        "profile_start_authorized_after_persisted_enforcement": True,
    }
    _atomic_json(args.tf32_bootstrap_output, report)
    if not passed:
        raise RuntimeError(
            f"TF32 child bootstrap failed: observed={after!r}, errors={assignment_errors!r}"
        )
    return profile.main(profile_argv)


if __name__ == "__main__":
    raise SystemExit(main())
