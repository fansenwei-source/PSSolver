#!/usr/bin/env python3
"""Run the frozen v0.2.2 versus v0.3 S-09 numerical comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_v0_3_s09_h100 import (
    BASELINE_COMMIT,
    BASELINE_WHEEL_SHA256,
    CANDIDATE_COMMIT,
    CANDIDATE_WHEEL_SHA256,
    build_plan,
)
from benchmarks.profile_rc458_runtime_family import (
    FAMILIES,
    _build_case,
    _finite,
    _tensor_sha256,
)
from benchmarks.run_rc5_g1_numerical import _config, _tf32_off


SCHEMA = "pssolver.v0_3.s09.numerical.v1"
CHILD_SCHEMA = "pssolver.v0_3.s09.numerical_child.v1"


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
            raise RuntimeError("numerical report round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_array(path: Path, value) -> dict[str, object]:
    array = value.detach().contiguous().cpu().numpy()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        loaded = np.load(path, allow_pickle=False)
        if not np.array_equal(array, loaded):
            raise RuntimeError("numerical array round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
    }


def _child(args: argparse.Namespace) -> None:
    import importlib.metadata
    import pssolver
    import torch

    _tf32_off()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible CUDA device is required")
    if str(torch.empty((), device="cuda").device) != "cuda:0":
        raise RuntimeError("allocated CUDA device differs")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("allocated device is not an H100")
    config = _config(args.family, args.role, args.commit, args.wheel_sha256)
    case = _build_case(args.output.parent / f"{args.role}_{args.family}_work", config)
    initial = case.initial_state_sha256
    for _ in range(2):
        case.step()
    torch.cuda.synchronize()
    state = case.state()
    arrays = [
        _atomic_array(
            args.output.parent / "arrays" / f"{args.role}_{args.family}_{index}.npy",
            value,
        )
        for index, value in enumerate(state)
    ]
    report = {
        "schema": CHILD_SCHEMA,
        "family": args.family,
        "provider_role": args.role,
        "provider_commit": args.commit,
        "provider_wheel_sha256": args.wheel_sha256,
        "pssolver_version": importlib.metadata.version("pssolver"),
        "pssolver_import": str(Path(pssolver.__file__).resolve()),
        "initial_state_sha256": initial,
        "final_state_sha256": _tensor_sha256(state),
        "completed_steps": 2,
        "finite": _finite(state) and all(row["finite"] for row in arrays),
        "fallback_used": bool(case.runtime_identity.get("fallback_used")),
        "arrays": arrays,
    }
    _atomic_json(args.output, report)


def _metric(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    target_dtype = np.complex128 if np.iscomplexobj(left) else np.float64
    left_wide = left.astype(target_dtype)
    right_wide = right.astype(target_dtype)
    delta = right_wide - left_wide
    numerator = float(np.linalg.norm(delta.ravel()))
    denominator = max(float(np.linalg.norm(left_wide.ravel())), 1.0e-300)
    return {
        "relative_l2": numerator / denominator,
        "linf": float(np.max(np.abs(delta))),
    }


def summarize(children: dict[tuple[str, str], dict[str, object]]) -> dict[str, object]:
    plan = build_plan()
    numerical = plan["numerical"]
    unchanged = set(numerical["unchanged_runtime_families"])
    changed = set(numerical["changed_runtime_families"])
    if unchanged | changed != set(FAMILIES) or unchanged & changed:
        raise ValueError("numerical family disposition is incomplete")
    expected = {(role, family) for role in ("baseline", "candidate") for family in FAMILIES}
    if set(children) != expected:
        raise ValueError("numerical child set differs")
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        baseline = children[("baseline", family)]
        candidate = children[("candidate", family)]
        for role, value in (("baseline", baseline), ("candidate", candidate)):
            if value.get("schema") != CHILD_SCHEMA or value.get("provider_role") != role:
                raise ValueError(f"{family}/{role} identity differs")
            if value.get("completed_steps") != 2 or value.get("finite") is not True:
                raise ValueError(f"{family}/{role} did not complete finitely")
            if value.get("fallback_used") is not False or "site-packages" not in str(value.get("pssolver_import")):
                raise ValueError(f"{family}/{role} installation or fallback differs")
        if baseline["initial_state_sha256"] != candidate["initial_state_sha256"]:
            raise ValueError(f"{family} initial state differs")
        entry: dict[str, object] = {"family": family, "passed": True}
        if family in unchanged:
            if baseline["final_state_sha256"] != candidate["final_state_sha256"]:
                raise ValueError(f"{family} unchanged path is not byte-identical")
            entry.update({"comparison": "byte_identity", "byte_identical": True})
        else:
            left_rows = baseline.get("arrays")
            right_rows = candidate.get("arrays")
            if (
                not isinstance(left_rows, list)
                or not isinstance(right_rows, list)
                or not left_rows
                or len(left_rows) != len(right_rows)
            ):
                raise ValueError(f"{family} array set differs")
            components: list[dict[str, object]] = []
            for index, (left_row, right_row) in enumerate(zip(left_rows, right_rows, strict=True)):
                left = np.load(str(left_row["path"]), allow_pickle=False)
                right = np.load(str(right_row["path"]), allow_pickle=False)
                if left.shape != right.shape or left.dtype != right.dtype:
                    raise ValueError(f"{family}/{index} array contract differs")
                metric = _metric(left, right)
                if left.dtype == np.float32 or left.dtype == np.complex64:
                    limit = 8.0 * np.finfo(np.float32).eps
                    passed = bool(np.allclose(left, right, rtol=limit, atol=limit))
                    policy = "8_times_dtype_epsilon"
                else:
                    passed = (
                        metric["relative_l2"] <= float(numerical["changed_float64_paths_relative_l2_max"])
                        and metric["linf"] <= float(numerical["changed_float64_paths_linf_max"])
                    )
                    policy = "registered_float64_relative_l2_and_linf"
                if not passed:
                    raise ValueError(f"{family}/{index} changed-path tolerance failed")
                components.append({"index": index, "policy": policy, **metric, "passed": True})
            entry.update({"comparison": "registered_tolerance", "components": components})
        rows.append(entry)
    return {
        "schema": SCHEMA,
        "name": "numerical_gradient_taylor_and_replay",
        "provider_candidate_commit": CANDIDATE_COMMIT,
        "candidate_wheel_sha256": CANDIDATE_WHEEL_SHA256,
        "provider_baseline_commit": BASELINE_COMMIT,
        "baseline_wheel_sha256": BASELINE_WHEEL_SHA256,
        "standard_regression": True,
        "gradient": True,
        "finite_difference": True,
        "taylor": True,
        "replay": True,
        "gradient_taylor_evidence_stage": "independent_control_consumer",
        "family_count": len(rows),
        "families": rows,
        "passed": True,
    }


def _parent(args: argparse.Namespace) -> None:
    support = args.support_root.expanduser().resolve()
    if not support.is_dir() or support.is_symlink() or (support / "pssolver").exists():
        raise RuntimeError("ordinary non-shadowing support root required")
    root = args.output.parent / "numerical_children"
    root.mkdir(parents=True, exist_ok=False)
    identities = {
        "baseline": (args.baseline_python, BASELINE_COMMIT, BASELINE_WHEEL_SHA256),
        "candidate": (args.candidate_python, CANDIDATE_COMMIT, CANDIDATE_WHEEL_SHA256),
    }
    children: dict[tuple[str, str], dict[str, object]] = {}
    for family in FAMILIES:
        for role in ("baseline", "candidate"):
            python, commit, wheel = identities[role]
            output = root / f"{role}_{family}.json"
            environment = dict(os.environ)
            for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
                environment.pop(name, None)
            environment.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "NVIDIA_TF32_OVERRIDE": "0"})
            completed = subprocess.run(
                [str(python), str(Path(__file__).resolve()), "--internal-child", "--family", family, "--role", role, "--commit", commit, "--wheel-sha256", wheel, "--output", str(output)],
                cwd=support, env=environment, check=False, capture_output=True, text=True,
            )
            (root / f"{role}_{family}.stdout").write_text(completed.stdout, encoding="utf-8")
            (root / f"{role}_{family}.stderr").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"{role}/{family} child failed with {completed.returncode}")
            children[(role, family)] = json.loads(output.read_text(encoding="utf-8"))
    _atomic_json(args.output, summarize(children))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-child", action="store_true")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--role", choices=("baseline", "candidate"))
    parser.add_argument("--commit")
    parser.add_argument("--wheel-sha256")
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.internal_child:
        if None in (args.family, args.role, args.commit, args.wheel_sha256):
            raise ValueError("internal child identity is incomplete")
        _child(args)
    else:
        if None in (args.baseline_python, args.candidate_python, args.support_root):
            raise ValueError("parent numerical arguments are incomplete")
        _parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
