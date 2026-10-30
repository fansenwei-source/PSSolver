#!/usr/bin/env python3
"""Verify that the S-16 runtime families remain bitwise equal to v0.3.0.

The parent process does not import PSSolver.  Each provider is loaded in a
separate child with an explicit source root, while the frozen S-09 runtime
family constructors provide the standard Plane, Periodic, and Channel cases.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

try:
    from benchmarks._artifact_io import atomic_write_json, atomic_write_text
except ModuleNotFoundError:  # direct execution sets sys.path[0] to benchmarks/
    from _artifact_io import atomic_write_json, atomic_write_text


SCHEMA = "pssolver.v0_4.s16.numerical_equivalence.v1"
CHILD_SCHEMA = "pssolver.v0_4.s16.numerical_equivalence_child.v1"
FAMILIES = (
    "plane_legacy_production",
    "plane_compiled_v2",
    "periodic_complete_stress",
    "channel_legacy",
    "channel_compiled_v2",
    "channel_complete_stress",
    "periodic_functional",
    "channel_functional",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_record(value: object) -> dict[str, object]:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("runtime state must contain only tensors")
    tensor = value.detach().contiguous().cpu()
    array = tensor.numpy()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": _sha256_bytes(array.tobytes(order="C")),
        "finite": bool(tensor.isfinite().all().item()),
    }

def _child(args: argparse.Namespace) -> None:
    import pssolver
    import torch
    from benchmarks.profile_rc458_runtime_family import _build_case
    from benchmarks.run_rc5_g1_numerical import _config, _tf32_off

    _tf32_off()
    config = replace(
        _config(args.family, args.role, args.commit, "0" * 64),
        device="cpu",
        warmup_steps=0,
        profile_steps=2,
        post_flush_rewarm_steps=0,
    )
    work = args.output.parent / f"{args.role}_{args.family}_work"
    work.mkdir(parents=True, exist_ok=False)
    case = _build_case(work, config)
    initial = [_tensor_record(value) for value in case.state()]
    for _ in range(2):
        case.step()
    final = [_tensor_record(value) for value in case.state()]
    if not all(row["finite"] for row in initial + final):
        raise RuntimeError("runtime state is non-finite")
    imported = Path(pssolver.__file__).resolve()
    source_root = args.source_root.resolve()
    if not imported.is_relative_to(source_root):
        raise RuntimeError(f"pssolver import {imported} is outside {source_root}")
    atomic_write_json(
        args.output,
        {
            "schema": CHILD_SCHEMA,
            "role": args.role,
            "family": args.family,
            "commit": args.commit,
            "source_root": str(source_root),
            "pssolver_import": str(imported),
            "torch_version": torch.__version__,
            "completed_steps": int(case.completed_steps()),
            "initial": initial,
            "final": final,
        },
    )


def _run_child(
    *,
    python: Path,
    script: Path,
    source_root: Path,
    output_root: Path,
    role: str,
    commit: str,
    family: str,
) -> dict[str, object]:
    output = output_root / "children" / f"{role}_{family}.json"
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "LD_LIBRARY_PATH"):
        environment.pop(name, None)
    environment.update(
        {
            "PYTHONPATH": str(source_root),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    command = [
        str(python),
        str(script),
        "--internal-child",
        "--source-root",
        str(source_root),
        "--role",
        role,
        "--commit",
        commit,
        "--family",
        family,
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(log_root / f"{role}_{family}.stdout", completed.stdout)
    atomic_write_text(log_root / f"{role}_{family}.stderr", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{role}/{family} child failed with {completed.returncode}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _parent(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    roots = {
        "baseline": (args.baseline_root.resolve(), args.baseline_commit),
        "candidate": (args.candidate_root.resolve(), args.candidate_commit),
    }
    children: dict[tuple[str, str], dict[str, object]] = {}
    for family in FAMILIES:
        for role in ("baseline", "candidate"):
            source_root, commit = roots[role]
            children[(role, family)] = _run_child(
                python=args.python.resolve(),
                script=script,
                source_root=source_root,
                output_root=output_root,
                role=role,
                commit=commit,
                family=family,
            )

    comparisons: list[dict[str, object]] = []
    for family in FAMILIES:
        baseline = children[("baseline", family)]
        candidate = children[("candidate", family)]
        initial_equal = baseline["initial"] == candidate["initial"]
        final_equal = baseline["final"] == candidate["final"]
        if not initial_equal or not final_equal:
            raise RuntimeError(f"{family} is not bitwise equal")
        comparisons.append(
            {
                "family": family,
                "initial_bitwise_equal": True,
                "final_bitwise_equal": True,
                "completed_steps": 2,
            }
        )
    atomic_write_json(
        output_root / "numerical_equivalence.json",
        {
            "schema": SCHEMA,
            "baseline_commit": args.baseline_commit,
            "candidate_commit": args.candidate_commit,
            "python": str(args.python.resolve()),
            "family_count": len(comparisons),
            "application_families": ["Plane", "Periodic", "Channel"],
            "comparisons": comparisons,
            "all_initial_states_bitwise_equal": True,
            "all_final_states_bitwise_equal": True,
            "passed": True,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--internal-child", action="store_true")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--candidate-commit")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--role", choices=("baseline", "candidate"))
    parser.add_argument("--commit")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    required = (
        ("source_root", "role", "commit", "family", "output")
        if args.internal_child
        else (
            "python",
            "baseline_root",
            "candidate_root",
            "baseline_commit",
            "candidate_commit",
            "output_root",
        )
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")
    return args


def main() -> None:
    args = _parse_args()
    if args.internal_child:
        _child(args)
    else:
        _parent(args)


if __name__ == "__main__":
    main()
