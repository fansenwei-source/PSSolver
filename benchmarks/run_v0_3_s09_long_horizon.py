#!/usr/bin/env python3
"""Run the three frozen S-09 CUDA Hermitian long-horizon cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.plan_v0_3_s09_h100 import CANDIDATE_COMMIT, CANDIDATE_WHEEL_SHA256


SCHEMA = "pssolver.v0_3.s09.long_horizon.v1"
CASES = (
    ("loop3d", "hermitian_half"),
    ("loop3d", "full_complex"),
    ("r1", "hermitian_half"),
)


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
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
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def summarize(reports: list[dict[str, object]]) -> dict[str, object]:
    if len(reports) != len(CASES):
        raise ValueError("long-horizon report count differs")
    frozen: list[dict[str, object]] = []
    all_finite = True
    all_within = True
    for (case, storage), report in zip(CASES, reports, strict=True):
        config = report.get("config")
        if not isinstance(config, dict):
            raise TypeError("long-horizon config missing")
        if config.get("case") != case or config.get("spectral_storage") != storage:
            raise ValueError("long-horizon case identity differs")
        if config.get("horizon") != 200.0 or config.get("device") != "cuda":
            raise ValueError("long-horizon execution contract differs")
        if report.get("completed_time") != 200.0 or report.get("passed") is not True:
            raise ValueError(f"long-horizon case failed: {case}/{storage}")
        samples = report.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("long-horizon samples missing")
        finite = all(row.get("finite") is True for row in samples if isinstance(row, dict))
        within = all(
            float(row["hermitian_violation"]) <= float(row["violation_bound"])
            for row in samples
            if isinstance(row, dict)
        )
        all_finite = all_finite and finite
        all_within = all_within and within
        frozen.append({"case": case, "storage": storage, "horizon": 200.0})
    if not all_finite or not all_within:
        raise ValueError("long-horizon finite/Hermitian gate failed")
    return {
        "schema": SCHEMA,
        "name": "cuda_long_horizon_hermitian",
        "provider_candidate_commit": CANDIDATE_COMMIT,
        "candidate_wheel_sha256": CANDIDATE_WHEEL_SHA256,
        "cases": frozen,
        "all_finite": all_finite,
        "all_within_bound": all_within,
        "passed": True,
    }


def _child(args: argparse.Namespace) -> int:
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from benchmarks.check_periodic_hermitian_stability import main as run_case

    return run_case(
        [
            "--case", args.case,
            "--spectral-storage", args.storage,
            "--device", "cuda",
            "--horizon", "200.0",
            "--tolerance-factor", "64.0",
            "--output", str(args.output),
        ]
    )


def _parent(args: argparse.Namespace) -> int:
    root = args.case_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    reports: list[dict[str, object]] = []
    for case, storage in CASES:
        output = root / f"{case}_{storage}.json"
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
            environment.pop(name, None)
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "NVIDIA_TF32_OVERRIDE": "0",
                "PYTHONPATH": str(args.support_root.expanduser().resolve()),
            }
        )
        completed = subprocess.run(
            [
                str(args.python.expanduser().absolute()), str(script),
                "--internal-child", "--case", case, "--storage", storage,
                "--output", str(output),
            ],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        (root / f"{case}_{storage}.stdout").write_text(completed.stdout, encoding="utf-8")
        (root / f"{case}_{storage}.stderr").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"long-horizon child failed: {case}/{storage}")
        reports.append(json.loads(output.read_text(encoding="utf-8")))
    _atomic_json(args.output, summarize(reports))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-child", action="store_true")
    parser.add_argument("--case", choices=("loop3d", "r1"))
    parser.add_argument("--storage", choices=("hermitian_half", "full_complex"))
    parser.add_argument("--python", type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--case-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.internal_child:
        if args.case is None or args.storage is None:
            raise ValueError("long-horizon child identity is incomplete")
        return _child(args)
    if args.python is None or args.support_root is None or args.case_root is None:
        raise ValueError("long-horizon parent arguments are incomplete")
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
