#!/usr/bin/env python3
"""Compare two completed Channel workflows with byte-identity gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_channel_outputs(left: str | Path, right: str | Path, *, step: int, role: str) -> dict[str, object]:
    directories = tuple(Path(value).expanduser().resolve() for value in (left, right))
    for directory in directories:
        if not (directory / "COMPLETE").is_file():
            raise FileNotFoundError(f"completed Channel workflow is missing: {directory}")
    arrays = []
    for prefix in ("Q", "u", "p"):
        paths = tuple(directory / f"{prefix}_{step}.npy" for directory in directories)
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"paired {prefix} output is missing")
        values = tuple(np.load(path, allow_pickle=False) for path in paths)
        if values[0].shape != values[1].shape or values[0].dtype != values[1].dtype:
            raise ValueError(f"paired {prefix} shape or dtype differs")
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError(f"paired {prefix} contains NaN or Inf")
        identical = _sha256(paths[0]) == _sha256(paths[1]) and np.array_equal(*values)
        if not identical:
            raise ValueError(f"paired {prefix} is not byte-identical")
        arrays.append({"name": prefix, "shape": list(values[0].shape), "dtype": str(values[0].dtype), "sha256": _sha256(paths[0]), "byte_identical": True})
    return {
        "schema_version": 1,
        "classification": "PASS",
        "comparison_role": role,
        "step": step,
        "left": str(directories[0]),
        "right": str(directories[1]),
        "array_count": len(arrays),
        "arrays": arrays,
        "byte_identity_gate": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    result = compare_channel_outputs(args.left, args.right, step=args.step, role=args.role)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
