#!/usr/bin/env python3
"""Run the eight-family RC5/RC6 installed-wheel numerical comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc6_single_h100_qualification import (
    PROVIDER_BASELINE,
    PROVIDER_BASELINE_WHEEL_SHA256,
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
)
from benchmarks import run_rc5_g1_numerical as rc5


SCHEMA = "pssolver.rc6.numerical_h100.v1"


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
            raise RuntimeError("numerical JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def summarize(children: dict[tuple[str, str], dict[str, object]]) -> dict[str, object]:
    expected = {(role, family) for role in ("baseline", "candidate") for family in rc5.FAMILIES}
    if set(children) != expected:
        raise ValueError("numerical child set differs")
    families: list[dict[str, object]] = []
    for family in rc5.FAMILIES:
        baseline = children[("baseline", family)]
        candidate = children[("candidate", family)]
        for role, report in (("baseline", baseline), ("candidate", candidate)):
            if report.get("schema") != rc5.CHILD_SCHEMA or report.get("provider_role") != role:
                raise ValueError(f"{family}/{role} child identity differs")
            if report.get("completed_steps") != 2 or report.get("finite") is not True:
                raise ValueError(f"{family}/{role} execution failed")
            if report.get("fallback_used") is not False:
                raise ValueError(f"{family}/{role} used fallback")
            expected_version = "0.2.0rc5" if role == "baseline" else "0.2.0rc6"
            if report.get("pssolver_version") != expected_version:
                raise ValueError(f"{family}/{role} package version differs")
            if "site-packages" not in str(report.get("pssolver_import", "")):
                raise ValueError(f"{family}/{role} is not an installed-wheel import")
        if baseline.get("initial_state_sha256") != candidate.get("initial_state_sha256"):
            raise ValueError(f"{family} initial state differs")
        if baseline.get("final_state_sha256") != candidate.get("final_state_sha256"):
            raise ValueError(f"{family} final state differs")
        entry: dict[str, object] = {
            "family": family,
            "completed_steps": 2,
            "finite": True,
            "fallback_used": False,
            "byte_identical": True,
            "relative_l2": 0.0,
            "linf": 0.0,
            "candidate_hermitian_violation": 0.0,
            "initial_state_sha256": baseline["initial_state_sha256"],
            "final_state_sha256": baseline["final_state_sha256"],
        }
        if family in rc5.REPAIRED:
            violation = candidate.get("hermitian_violation")
            if not isinstance(violation, (int, float)) or float(violation) != 0.0:
                raise ValueError(f"{family} Hermitian violation differs")
            entry["candidate_hermitian_violation"] = float(violation)
        families.append(entry)
    return {
        "schema": SCHEMA,
        "provider_baseline_commit": PROVIDER_BASELINE,
        "provider_candidate_commit": PROVIDER_CANDIDATE,
        "passed": True,
        "family_count": 8,
        "families": families,
    }


def _child(args: argparse.Namespace) -> None:
    rc5._run_child(
        family=args.family,
        role=args.role,
        commit=args.commit,
        wheel=args.wheel,
        output=args.output,
    )


def _parent(args: argparse.Namespace) -> dict[str, object]:
    support = args.support_root.expanduser().resolve()
    if not support.is_dir() or support.is_symlink() or (support / "pssolver").exists():
        raise RuntimeError("ordinary non-shadowing support root required")
    child_root = args.child_root.expanduser().resolve()
    child_root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    roles = {
        "baseline": (args.baseline_python, PROVIDER_BASELINE, PROVIDER_BASELINE_WHEEL_SHA256),
        "candidate": (args.candidate_python, PROVIDER_CANDIDATE, PROVIDER_CANDIDATE_WHEEL_SHA256),
    }
    children: dict[tuple[str, str], dict[str, object]] = {}
    for family in rc5.FAMILIES:
        for role in ("baseline", "candidate"):
            python, commit, wheel = roles[role]
            child_output = child_root / f"{role}_{family}.json"
            environment = dict(os.environ)
            for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
                environment.pop(name, None)
            environment.update({
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "NVIDIA_TF32_OVERRIDE": "0",
            })
            completed = subprocess.run(
                [
                    str(Path(python).expanduser().absolute()), str(script),
                    "--internal-child", "--family", family, "--role", role,
                    "--commit", commit, "--wheel", wheel,
                    "--output", str(child_output),
                ],
                cwd=support, env=environment, check=False, capture_output=True, text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"{family}/{role} child failed: {completed.stderr}")
            children[(role, family)] = json.loads(child_output.read_text(encoding="utf-8"))
    result = summarize(children)
    _atomic_json(args.output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-child", action="store_true")
    parser.add_argument("--family", choices=rc5.FAMILIES)
    parser.add_argument("--role", choices=("baseline", "candidate"))
    parser.add_argument("--commit")
    parser.add_argument("--wheel")
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--child-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.internal_child:
        if None in (args.family, args.role, args.commit, args.wheel):
            raise ValueError("internal child identity is incomplete")
        _child(args)
        return 0
    if None in (args.baseline_python, args.candidate_python, args.support_root, args.child_root):
        raise ValueError("parent numerical arguments are incomplete")
    _parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
