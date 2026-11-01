#!/usr/bin/env python3
"""Compare every R4 CPU runtime step and refresh clock bit for bit."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

try:
    from benchmarks._artifact_io import (
        atomic_write_json,
        atomic_write_text,
        git_identity,
        runtime_environment,
    )
except ModuleNotFoundError:  # direct execution sets sys.path[0] to benchmarks/
    from _artifact_io import (  # type: ignore[no-redef]
        atomic_write_json,
        atomic_write_text,
        git_identity,
        runtime_environment,
    )


SCHEMA = "pssolver.v0_4.s18.r4.step_equivalence.v1"
CHILD_SCHEMA = "pssolver.v0_4.s18.r4.step_equivalence_child.v1"
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
REFRESH_INTERVALS = (None, 1, 2, 7)
STEPS = 20


def _tensor_record(value: object) -> dict[str, object]:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("runtime state must contain only tensors")
    tensor = value.detach().contiguous().cpu()
    payload = tensor.numpy().tobytes(order="C")
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "finite": bool(tensor.isfinite().all().item()),
    }


def _progress_record(case: object) -> dict[str, object]:
    integrator = case.solver.integrator
    state = getattr(integrator, "runtime_state", None)
    progress = getattr(state, "progress", None)
    representations = getattr(state, "representations", None)
    return {
        "completed_steps": int(case.completed_steps()),
        "spectral_refresh_interval": integrator.spectral_refresh_interval,
        "step_count": int(integrator.step_count),
        "refresh_count": int(integrator.refresh_count),
        "runtime_state_progress": (
            None if progress is None else progress.to_metadata()
        ),
        "representation_ledger": (
            None if representations is None else representations.to_metadata()
        ),
    }


def _state_record(case: object) -> dict[str, object]:
    tensors = [_tensor_record(value) for value in case.state()]
    if not all(row["finite"] for row in tensors):
        raise RuntimeError("runtime state is non-finite")
    return {"tensors": tensors, "progress": _progress_record(case)}


def _child(args: argparse.Namespace) -> None:
    import pssolver
    import torch
    from benchmarks.profile_rc458_runtime_family import _build_case
    from benchmarks.run_rc5_g1_numerical import _config, _tf32_off

    _tf32_off()
    traces: list[dict[str, object]] = []
    for interval in REFRESH_INTERVALS:
        label = "none" if interval is None else str(interval)
        config = replace(
            _config(args.family, args.role, args.commit, "0" * 64),
            device="cpu",
            warmup_steps=0,
            profile_steps=STEPS,
            post_flush_rewarm_steps=0,
        )
        work = args.output.parent / f"{args.role}_{args.family}_{label}_work"
        work.mkdir(parents=True, exist_ok=False)
        case = _build_case(work, config)
        case.solver.integrator.set_spectral_refresh_interval(interval)
        steps = [_state_record(case)]
        for _ in range(STEPS):
            case.step()
            steps.append(_state_record(case))
        if int(case.completed_steps()) != STEPS:
            raise RuntimeError("runtime completed-step count differs")
        traces.append(
            {
                "refresh_interval": interval,
                "runtime_identity": case.runtime_identity,
                "configuration_identity": case.configuration_identity,
                "steps": steps,
            }
        )

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
            "git": git_identity(source_root),
            "environment": runtime_environment(),
            "refresh_intervals": list(REFRESH_INTERVALS),
            "formal_steps_per_interval": STEPS,
            "traces": traces,
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
            "CUDA_VISIBLE_DEVICES": "",
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
    logs = output_root / "logs"
    atomic_write_text(logs / f"{role}_{family}.stdout", completed.stdout)
    atomic_write_text(logs / f"{role}_{family}.stderr", completed.stderr)
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
        if baseline["traces"] != candidate["traces"]:
            raise RuntimeError(f"{family} step trace is not bitwise equal")
        comparisons.append(
            {
                "family": family,
                "refresh_interval_count": len(REFRESH_INTERVALS),
                "formal_steps_per_interval": STEPS,
                "state_and_progress_trace_bitwise_equal": True,
                "runtime_identity_equal": True,
                "configuration_identity_equal": True,
            }
        )
    atomic_write_json(
        output_root / "step_equivalence.json",
        {
            "schema": SCHEMA,
            "baseline_commit": args.baseline_commit,
            "candidate_commit": args.candidate_commit,
            "python": str(args.python.resolve()),
            "families": list(FAMILIES),
            "refresh_intervals": list(REFRESH_INTERVALS),
            "formal_steps_per_interval": STEPS,
            "total_compared_transitions": (
                len(FAMILIES) * len(REFRESH_INTERVALS) * STEPS
            ),
            "comparisons": comparisons,
            "all_state_and_progress_traces_bitwise_equal": True,
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
