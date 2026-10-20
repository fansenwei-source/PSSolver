#!/usr/bin/env python3
"""Bind one qualified target-parameter Q2D checkpoint for V3 initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def read_json_object(path: Path, *, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def create_manifest(
    *,
    checkpoint_path: Path,
    qualification_report_path: Path,
    frank_k: float,
    zeta: float,
) -> dict:
    """Validate immutable inputs and return the canonical V3 manifest object."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    qualification_report_path = qualification_report_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Q2D checkpoint does not exist: {checkpoint_path}")
    if not qualification_report_path.is_file():
        raise FileNotFoundError(
            f"qualification report does not exist: {qualification_report_path}"
        )
    before = {
        "checkpoint": sha256_file(checkpoint_path),
        "qualification_report": sha256_file(qualification_report_path),
    }
    report = read_json_object(
        qualification_report_path,
        label="qualification report",
    )
    if report.get("candidate_2d_statistical_steady_state") is not True:
        raise ValueError(
            "qualification report must set "
            "candidate_2d_statistical_steady_state=true"
        )
    report_parameters = report.get("parameters")
    if not isinstance(report_parameters, dict):
        raise ValueError("qualification report must contain a parameters object")
    for name, expected in (("frank_k", frank_k), ("zeta", zeta)):
        try:
            declared = float(report_parameters[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"qualification report parameter {name!r} is missing or invalid"
            ) from error
        if not math.isclose(declared, expected, rel_tol=1.0e-12, abs_tol=1.0e-14):
            raise ValueError(
                f"qualification report {name}={declared:.17g} does not match "
                f"requested {expected:.17g}"
            )

    recommended = report.get("recommended_checkpoint")
    if not isinstance(recommended, dict):
        raise ValueError("qualification report must recommend one checkpoint object")
    recommended_path = Path(recommended.get("path", "")).expanduser().resolve()
    if recommended_path != checkpoint_path:
        raise ValueError(
            f"recommended checkpoint is {recommended_path}, not {checkpoint_path}"
        )
    if recommended.get("sha256") != before["checkpoint"]:
        raise ValueError("recommended checkpoint SHA-256 is missing or incorrect")

    q2d = np.load(checkpoint_path, allow_pickle=False)
    if q2d.ndim == 4 and q2d.shape[2] == 1:
        q2d = q2d[:, :, 0, :]
    if q2d.ndim != 3 or q2d.shape[-1] != 5:
        raise ValueError(
            f"Q2D checkpoint must have shape (Nx,Ny,5), got {q2d.shape}"
        )
    if not np.issubdtype(q2d.dtype, np.floating):
        raise ValueError(f"Q2D checkpoint must be floating point, got {q2d.dtype}")
    if not np.all(np.isfinite(q2d)):
        raise ValueError("Q2D checkpoint contains NaN or Inf")

    after = {
        "checkpoint": sha256_file(checkpoint_path),
        "qualification_report": sha256_file(qualification_report_path),
    }
    if after != before:
        raise RuntimeError("a V3 manifest input changed while it was being inspected")
    return {
        "schema_version": 1,
        "protocol": "V3",
        "qualified": True,
        "qualification_meaning": (
            "target-parameter two-dimensional statistical steady state"
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": before["checkpoint"],
            "size_bytes": checkpoint_path.stat().st_size,
            "shape": list(q2d.shape),
            "dtype": str(q2d.dtype),
        },
        "parameters": {
            "frank_k": float(frank_k),
            "zeta": float(zeta),
            "zeta_over_k": float(zeta / frank_k),
        },
        "qualification_report": {
            "path": str(qualification_report_path),
            "sha256": before["qualification_report"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--frank-k", type=finite_positive, required=True)
    parser.add_argument("--zeta", type=finite_positive, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    manifest = create_manifest(
        checkpoint_path=args.checkpoint,
        qualification_report_path=args.qualification_report,
        frank_k=args.frank_k,
        zeta=args.zeta,
    )
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
