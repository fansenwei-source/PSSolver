#!/usr/bin/env python3
"""Run the rc5 G-1 A/B numerical comparison outside source checkouts.

The parent launches one installed-wheel child for each Provider role and
runtime family.  Children persist hashes for every family and physical
Q/velocity/pressure arrays for repaired Periodic families.  The parent never
imports either Provider wheel and performs the frozen comparison itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SCHEMA = "pssolver.rc5.g1.numerical_h100.v1"
CHILD_SCHEMA = "pssolver.rc5.g1.numerical_child.v1"
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
REPAIRED = frozenset(("periodic_complete_stress", "periodic_functional"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
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


def _atomic_npy(path: Path, value: object) -> dict[str, object]:
    import torch

    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    if not isinstance(value, torch.Tensor):
        raise TypeError("physical artifact must be a tensor")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    array = value.detach().contiguous().cpu().numpy()
    try:
        with temporary.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if not np.array_equal(np.load(path, allow_pickle=False), array):
            raise RuntimeError("physical artifact round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "finite": bool(np.isfinite(array).all()),
    }


def _tf32_off() -> None:
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    _require(
        (
            torch.get_float32_matmul_precision(),
            bool(torch.backends.cuda.matmul.allow_tf32),
            bool(torch.backends.cudnn.allow_tf32),
        )
        == ("highest", False, False),
        "cannot enforce the frozen TF32-off numerical policy",
    )


def _config(family: str, role: str, commit: str, wheel: str):
    from benchmarks.profile_rc458_runtime_family import RC458ProfileConfig

    channel = family.startswith("channel_")
    plane = family.startswith("plane_")
    shape = (8, 8, 6) if plane else ((8, 6, 4) if channel else (8, 8, 4))
    dtype = "float32" if family in {"channel_legacy", "channel_compiled_v2"} else "float64"
    return RC458ProfileConfig(
        family=family,
        provider_role=role,
        provider_commit=commit,
        provider_wheel_sha256=wheel,
        trial=1,
        shape=shape,
        lengths=tuple(float(value) for value in shape),
        device="cuda",
        dtype=dtype,
        warmup_steps=0,
        profile_steps=2,
        seed=20261015,
    )


def _controls(runtime):
    import torch

    spec = runtime.control_specs[0].tensor
    dtype = getattr(torch, spec.dtype) if isinstance(spec.dtype, str) else spec.dtype
    return {"activity": torch.full(spec.shape, 0.01, dtype=dtype, device=spec.device)}


def _periodic_case(root: Path, config, *, functional: bool):
    from benchmarks.profile_rc458_runtime_family import _complete_simulation, _tensor_sha256
    from pssolver import compile_simulation
    from pssolver.applications.periodic_beris_edwards import load_periodic_initial_q
    from pssolver.functional import build_functional_runtime, periodic_activity_functional_request
    from pssolver.runtime.periodic_beris_edwards import (
        PeriodicRuntimeBuildRequest,
        build_periodic_beris_edwards_runtime,
    )

    simulation = _complete_simulation(root, config, channel=False)
    if functional:
        runtime = build_functional_runtime(
            periodic_activity_functional_request(simulation.specification)
        )
        state = runtime.initial_state()
        initial_sha = _tensor_sha256(state)
        controls = _controls(runtime)
        for step in range(2):
            state = runtime.step(state, controls, step)
        observations = runtime.observe(state, controls)
        identity = runtime.identity().to_metadata()["execution"]["functional_runtime"]
        return state, observations, initial_sha, bool(identity.get("fallback_used"))

    spec = compile_simulation(simulation).application_request
    initial, _, _ = load_periodic_initial_q(spec)
    adapter = build_periodic_beris_edwards_runtime(
        PeriodicRuntimeBuildRequest(spec, initial, config.device)
    )
    state = lambda: (adapter.fields.spatial, adapter.fields.spectral)
    initial_sha = _tensor_sha256(state())
    adapter.advance(2)
    adapter.synchronize_for_observation()
    observations = {
        "Q": adapter.fields.spatial[:5],
        "velocity": adapter.fields.spatial[5:8],
        "pressure": adapter.fields.spatial[8],
    }
    return state(), observations, initial_sha, bool(adapter.to_metadata().get("fallback_used"))


def _run_child(*, family: str, role: str, commit: str, wheel: str, output: Path) -> dict[str, object]:
    import importlib.metadata
    import torch
    import pssolver
    from benchmarks.check_periodic_hermitian_stability import _hermitian_violation
    from benchmarks.profile_rc458_runtime_family import _build_case, _finite, _tensor_sha256

    _tf32_off()
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    _require(torch.cuda.device_count() == 1, "exactly one CUDA device is required")
    allocated = torch.empty((), device="cuda").device
    _require(str(allocated) == "cuda:0", "allocated CUDA device differs")
    _require("H100" in torch.cuda.get_device_name(0), "allocated GPU is not H100")
    config = _config(family, role, commit, wheel)
    working = output.parent / f"{role}_{family}_working"
    working.mkdir(parents=True, exist_ok=False)
    if family in REPAIRED:
        state, observations, initial_sha, fallback = _periodic_case(
            working, config, functional=family == "periodic_functional"
        )
        artifacts = {
            name: _atomic_npy(output.parent / f"{role}_{family}_{name}.npy", value)
            for name, value in observations.items()
        }
        violation = float(
            _hermitian_violation(state[1], config.shape, "hermitian_half").detach().cpu().item()
        )
    else:
        case = _build_case(working, config)
        initial_sha = case.initial_state_sha256
        for _ in range(2):
            case.step()
        state = case.state()
        fallback = bool(case.runtime_identity.get("fallback_used"))
        artifacts = {}
        violation = None
    torch.cuda.synchronize()
    result = {
        "schema": CHILD_SCHEMA,
        "family": family,
        "provider_role": role,
        "provider_commit": commit,
        "provider_wheel_sha256": wheel,
        "pssolver_version": importlib.metadata.version("pssolver"),
        "pssolver_import": str(Path(pssolver.__file__).resolve()),
        "allocated_device": str(allocated),
        "device_name": torch.cuda.get_device_name(0),
        "tf32": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "initial_state_sha256": initial_sha,
        "final_state_sha256": _tensor_sha256(state),
        "finite": _finite(state) and all(value["finite"] for value in artifacts.values()),
        "fallback_used": fallback,
        "completed_steps": 2,
        "physical_artifacts": artifacts,
        "hermitian_violation": violation,
    }
    _atomic_json(output, result)
    return result


def _metric(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = right - left
    denominator = float(np.linalg.norm(left.ravel()))
    numerator = float(np.linalg.norm(delta.ravel()))
    return {
        "relative_l2": numerator / denominator if denominator else numerator,
        "linf": float(np.max(np.abs(delta))),
    }


def summarize(
    children: dict[tuple[str, str], dict[str, object]],
    *,
    baseline_commit: str,
    candidate_commit: str,
    relative_l2_max: float = 1.0e-12,
    linf_max: float = 1.0e-12,
) -> dict[str, object]:
    expected = {(role, family) for role in ("baseline", "candidate") for family in FAMILIES}
    _require(set(children) == expected, "numerical child set differs")
    entries: list[dict[str, object]] = []
    for family in FAMILIES:
        baseline = children[("baseline", family)]
        candidate = children[("candidate", family)]
        for role, report in (("baseline", baseline), ("candidate", candidate)):
            _require(report.get("schema") == CHILD_SCHEMA, f"{family}/{role} schema differs")
            _require(report.get("provider_role") == role, f"{family}/{role} role differs")
            _require(report.get("family") == family, f"{family}/{role} family differs")
            _require(report.get("finite") is True, f"{family}/{role} is non-finite")
            _require(report.get("fallback_used") is False, f"{family}/{role} used fallback")
            _require(report.get("completed_steps") == 2, f"{family}/{role} step count differs")
        _require(
            baseline.get("initial_state_sha256") == candidate.get("initial_state_sha256"),
            f"{family} initial state differs",
        )
        entry: dict[str, object] = {
            "family": family,
            "finite": True,
            "fallback_used": False,
            "initial_state_sha256": baseline["initial_state_sha256"],
            "baseline_final_state_sha256": baseline["final_state_sha256"],
            "candidate_final_state_sha256": candidate["final_state_sha256"],
        }
        if family not in REPAIRED:
            identical = baseline.get("final_state_sha256") == candidate.get("final_state_sha256")
            _require(identical, f"{family} final state is not byte-identical")
            entry.update({"comparison": "byte_identity", "byte_identical": True})
        else:
            components: dict[str, object] = {}
            for name in ("Q", "velocity", "pressure"):
                left_path = Path(str(baseline["physical_artifacts"][name]["path"]))
                right_path = Path(str(candidate["physical_artifacts"][name]["path"]))
                metrics = _metric(np.load(left_path, allow_pickle=False), np.load(right_path, allow_pickle=False))
                _require(metrics["relative_l2"] <= relative_l2_max, f"{family}/{name} relative L2 failed")
                _require(metrics["linf"] <= linf_max, f"{family}/{name} Linf failed")
                components[name] = metrics
            violation = candidate.get("hermitian_violation")
            _require(
                isinstance(violation, (int, float))
                and not isinstance(violation, bool)
                and math.isfinite(float(violation))
                and float(violation) == 0.0,
                f"{family} candidate Hermitian violation is not zero",
            )
            entry.update(
                {
                    "comparison": "physical_tolerance",
                    "components": components,
                    "candidate_hermitian_violation": float(violation),
                }
            )
        entries.append(entry)
    return {
        "schema": SCHEMA,
        "provider_baseline_commit": baseline_commit,
        "provider_candidate_commit": candidate_commit,
        "passed": True,
        "family_count": len(entries),
        "families": entries,
    }


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def _parent(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    child_root = output.parent / "numerical_children"
    child_root.mkdir(parents=True, exist_ok=False)
    support = args.support_root.expanduser().resolve()
    _require(not (support / "pssolver").exists(), "support root contains pssolver source")
    script = Path(__file__).resolve()
    children: dict[tuple[str, str], dict[str, object]] = {}
    identities = {
        "baseline": (args.baseline_python, args.baseline_commit, args.baseline_wheel_sha256),
        "candidate": (args.candidate_python, args.candidate_commit, args.candidate_wheel_sha256),
    }
    for family in FAMILIES:
        for role in ("baseline", "candidate"):
            python, commit, wheel = identities[role]
            child_output = child_root / f"{role}_{family}.json"
            environment = dict(os.environ)
            for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
                environment.pop(name, None)
            environment.update(
                {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "NVIDIA_TF32_OVERRIDE": "0"}
            )
            command = [
                str(Path(python).expanduser().absolute()),
                str(script),
                "--internal-child",
                "--family", family,
                "--role", role,
                "--commit", commit,
                "--wheel-sha256", wheel,
                "--output", str(child_output),
            ]
            result = subprocess.run(
                command, cwd=support, env=environment, check=False, capture_output=True, text=True
            )
            (child_root / f"{role}_{family}.stdout").write_text(result.stdout, encoding="utf-8")
            (child_root / f"{role}_{family}.stderr").write_text(result.stderr, encoding="utf-8")
            _require(result.returncode == 0, f"{role}/{family} child failed with {result.returncode}")
            children[(role, family)] = _load_json(child_output)
    result = summarize(
        children,
        baseline_commit=args.baseline_commit,
        candidate_commit=args.candidate_commit,
    )
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-root", type=Path)
    parser.add_argument("--baseline-python")
    parser.add_argument("--candidate-python")
    parser.add_argument("--baseline-commit")
    parser.add_argument("--candidate-commit")
    parser.add_argument("--baseline-wheel-sha256")
    parser.add_argument("--candidate-wheel-sha256")
    parser.add_argument("--internal-child", action="store_true")
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--role", choices=("baseline", "candidate"))
    parser.add_argument("--commit")
    parser.add_argument("--wheel-sha256")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.internal_child:
        for name in ("family", "role", "commit", "wheel_sha256"):
            if getattr(args, name) is None:
                raise ValueError(f"--{name.replace('_', '-')} is required for child mode")
        _run_child(
            family=args.family,
            role=args.role,
            commit=args.commit,
            wheel=args.wheel_sha256,
            output=args.output,
        )
    else:
        for name in (
            "support_root", "baseline_python", "candidate_python", "baseline_commit",
            "candidate_commit", "baseline_wheel_sha256", "candidate_wheel_sha256",
        ):
            if getattr(args, name) is None:
                raise ValueError(f"--{name.replace('_', '-')} is required")
        _parent(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
