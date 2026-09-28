#!/usr/bin/env python3
"""Prepare the immutable analytic Q inputs for the P9.5 H100 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


INPUT_SCHEMA_VERSION = 1
FROZEN_GRIDS = (
    ("R128", (128, 128, 32), (100.0, 100.0, 20.0)),
    ("R320", (320, 320, 80), (100.0, 100.0, 20.0)),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_analytic_q(path: Path, shape: tuple[int, int, int]) -> None:
    """Write a deterministic smooth, finite, traceless-Q representation."""
    if len(shape) != 3 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 1
        for value in shape
    ):
        raise ValueError("shape must contain three integer values greater than one")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(*shape, 5),
    )
    x = np.arange(shape[0], dtype=np.float64)[:, None, None]
    y = np.arange(shape[1], dtype=np.float64)[None, :, None]
    z = np.arange(shape[2], dtype=np.float64)[None, None, :]
    x *= (2.0 * np.pi) / shape[0]
    y *= (2.0 * np.pi) / shape[1]
    z *= (2.0 * np.pi) / shape[2]
    values[..., 0] = 0.18 + 0.012 * np.sin(x + 0.5 * z)
    values[..., 1] = 0.009 * np.cos(x - y)
    values[..., 2] = 0.007 * np.sin(y + z)
    values[..., 3] = -0.09 + 0.011 * np.cos(y - 0.25 * z)
    values[..., 4] = 0.006 * np.cos(x + y + z)
    values.flush()
    del values


def prepare_inputs(output_root: Path) -> dict[str, object]:
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse existing output root {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        records = []
        for grid_id, shape, lengths in FROZEN_GRIDS:
            path = temporary / grid_id / "Q_0.npy"
            write_analytic_q(path, shape)
            records.append(
                {
                    "id": grid_id,
                    "shape": list(shape),
                    "lengths": list(lengths),
                    "path": f"{grid_id}/Q_0.npy",
                    "dtype": "float64",
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                    "finite": bool(np.isfinite(np.load(path, mmap_mode="r")).all()),
                }
            )
        metadata = {
            "schema_version": INPUT_SCHEMA_VERSION,
            "phase": "P9.5",
            "kind": "p95_analytic_initial_q_bundle",
            "construction": "analytic_periodic_float64_batch_one",
            "grids": records,
        }
        (temporary / "input_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
        return metadata
    except BaseException:
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if temporary.exists():
            temporary.rmdir()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = prepare_inputs(args.output_root)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
