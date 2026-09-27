#!/usr/bin/env python3
"""Analysis-only diagnostics for the P8.4.5 static-lifting recovery.

The commands in this module deliberately do not decide a qualification gate.
They preserve the frozen qualification result while collecting the evidence
needed to distinguish numerical roundoff from CUDA allocator reservation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

if __package__:
    from benchmarks.profile_beris_edwards_timestep import _git_provenance
    from benchmarks.profile_plane_runtime_timestep import (
        _initial_q,
        _tensor_mapping_sha256,
    )
    from benchmarks.qualify_plane_static_lifting import (
        LiftingProfileConfig,
        PROFILE_VARIANTS,
        _base_profile_config,
        _build_runtime,
    )
else:
    from profile_beris_edwards_timestep import _git_provenance
    from profile_plane_runtime_timestep import (
        _initial_q,
        _tensor_mapping_sha256,
    )
    from qualify_plane_static_lifting import (
        LiftingProfileConfig,
        PROFILE_VARIANTS,
        _base_profile_config,
        _build_runtime,
    )
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.plane_beris_edwards import plane_physical_component


DIAGNOSTIC_SCHEMA_VERSION = 1
STATE_FIELDS = (*Q_COMPONENTS, "ux", "uy", "uz", "p")
_MEMORY_STAT_KEYS = (
    "allocated_bytes.all.current",
    "allocated_bytes.all.peak",
    "reserved_bytes.all.current",
    "reserved_bytes.all.peak",
    "active_bytes.all.current",
    "active_bytes.all.peak",
    "inactive_split_bytes.all.current",
    "inactive_split_bytes.all.peak",
    "requested_bytes.all.current",
    "requested_bytes.all.peak",
    "num_alloc_retries",
    "num_ooms",
)


def _shape(value: str) -> tuple[int, int, int]:
    parsed = tuple(int(item) for item in value.split(","))
    if len(parsed) != 3 or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(
            "shape requires three comma-separated positive integers"
        )
    return parsed


def _lengths(value: str) -> tuple[float, float, float]:
    parsed = tuple(float(item) for item in value.split(","))
    if len(parsed) != 3 or any(
        not math.isfinite(item) or item <= 0.0 for item in parsed
    ):
        raise argparse.ArgumentTypeError(
            "lengths require three comma-separated positive finite values"
        )
    return parsed


def _atomic_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_array(value: torch.Tensor) -> np.ndarray:
    # Some historical lifting runtimes return successive components through a
    # shared reconstruction workspace.  Own the host bytes immediately so a
    # later component request cannot mutate an earlier diagnostic array.
    return value.detach().cpu().contiguous().numpy().copy()


def capture_state(
    *,
    variant: str,
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
    device: str,
    steps: int,
    seed: int,
    pointwise_execution: str,
    arrays_output: Path,
) -> dict[str, object]:
    """Run a short trajectory and save internal and physical state arrays."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    config = LiftingProfileConfig(
        variant=variant,
        shape=shape,
        lengths=lengths,
        device=device,
        warmup_steps=0,
        profile_steps=1,
        pointwise_execution=pointwise_execution,
        seed=seed,
    )
    allocated = torch.device(device)
    if allocated.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(seed)
    if allocated.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    initial_values = _initial_q(_base_profile_config(config))
    initial_sha256 = _tensor_mapping_sha256(initial_values)
    with torch.no_grad():
        run_spec, simulation, adapter = _build_runtime(config, initial_values)
        del initial_values
        adapter.advance(steps)
        adapter.synchronize_for_observation()
        if allocated.type == "cuda":
            torch.cuda.synchronize(allocated)
        arrays: dict[str, np.ndarray] = {}
        for name in STATE_FIELDS:
            arrays[f"internal__{name}"] = _tensor_array(adapter.fields[name])
        for name in Q_COMPONENTS:
            arrays[f"physical__{name}"] = _tensor_array(
                plane_physical_component(adapter, name)
            )
        for name in ("ux", "uy", "uz", "p"):
            arrays[f"physical__{name}"] = arrays[f"internal__{name}"]
        finite = all(bool(np.isfinite(value).all()) for value in arrays.values())
        runtime = adapter.to_metadata()
        completed_steps = adapter.completed_steps
    _atomic_npz(arrays_output, arrays)
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_state_capture",
        "analysis_only": True,
        "config": {
            "variant": variant,
            "shape": list(shape),
            "lengths": list(lengths),
            "device": device,
            "steps": steps,
            "seed": seed,
            "pointwise_execution": pointwise_execution,
        },
        "configuration_identity": run_spec.identity_metadata(),
        "simulation_identity": simulation.canonical_sha256(),
        "runtime_identity": runtime,
        "git": _git_provenance(),
        "pssolver_import": __import__("pssolver").__file__,
        "initial_q_sha256": initial_sha256,
        "completed_steps": completed_steps,
        "finite": finite,
        "arrays": {
            "path": str(arrays_output.resolve()),
            "sha256": _file_sha256(arrays_output),
            "keys": sorted(arrays),
        },
    }


def _ordered_float64(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.dtype("float64"):
        raise TypeError("ULP diagnostics require float64 arrays")
    bits = values.view(np.uint64)
    sign = np.uint64(1 << 63)
    return np.where((bits & sign) != 0, ~bits, bits | sign)


def _array_error(actual: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    if actual.shape != reference.shape:
        raise ValueError("array shapes differ")
    if actual.dtype != reference.dtype:
        raise ValueError("array dtypes differ")
    if actual.dtype != np.dtype("float64"):
        raise TypeError("comparison currently requires float64 arrays")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("comparison arrays must be finite")
    difference = actual - reference
    absolute = np.abs(difference)
    numerator_squared = float(np.vdot(difference.ravel(), difference.ravel()))
    denominator_squared = float(np.vdot(reference.ravel(), reference.ravel()))
    if denominator_squared == 0.0:
        relative_l2 = 0.0 if numerator_squared == 0.0 else None
    else:
        relative_l2 = math.sqrt(numerator_squared / denominator_squared)
    actual_ordered = _ordered_float64(actual)
    reference_ordered = _ordered_float64(reference)
    upper = np.maximum(actual_ordered, reference_ordered)
    lower = np.minimum(actual_ordered, reference_ordered)
    ulp = upper - lower
    ulp = np.where(difference == 0.0, np.uint64(0), ulp)
    return {
        "byte_identical": bool(actual.tobytes() == reference.tobytes()),
        "mismatch_count": int(np.count_nonzero(difference)),
        "element_count": int(actual.size),
        "relative_l2": relative_l2,
        "rms": math.sqrt(numerator_squared / actual.size),
        "linf": float(absolute.max(initial=0.0)),
        "maximum_ulp": int(ulp.max(initial=np.uint64(0))),
        "numerator_squared": numerator_squared,
        "denominator_squared": denominator_squared,
    }


def compare_states(
    *,
    reference_path: Path,
    actual_path: Path,
) -> dict[str, object]:
    """Compare two state archives without assigning a scientific tolerance."""

    with np.load(reference_path, allow_pickle=False) as reference_archive:
        with np.load(actual_path, allow_pickle=False) as actual_archive:
            reference_keys = set(reference_archive.files)
            actual_keys = set(actual_archive.files)
            if reference_keys != actual_keys:
                raise ValueError("state archive keys differ")
            fields: dict[str, dict[str, object]] = {}
            aggregate = {
                "internal": {
                    "numerator_squared": 0.0,
                    "denominator_squared": 0.0,
                    "linf": 0.0,
                    "maximum_ulp": 0,
                    "mismatch_count": 0,
                    "element_count": 0,
                },
                "physical": {
                    "numerator_squared": 0.0,
                    "denominator_squared": 0.0,
                    "linf": 0.0,
                    "maximum_ulp": 0,
                    "mismatch_count": 0,
                    "element_count": 0,
                },
            }
            for key in sorted(reference_keys):
                error = _array_error(actual_archive[key], reference_archive[key])
                fields[key] = error
                group = key.split("__", 1)[0]
                target = aggregate[group]
                target["numerator_squared"] += error["numerator_squared"]
                target["denominator_squared"] += error["denominator_squared"]
                target["linf"] = max(target["linf"], error["linf"])
                target["maximum_ulp"] = max(
                    target["maximum_ulp"], error["maximum_ulp"]
                )
                target["mismatch_count"] += error["mismatch_count"]
                target["element_count"] += error["element_count"]
    aggregates: dict[str, dict[str, object]] = {}
    for group, raw in aggregate.items():
        numerator_squared = float(raw["numerator_squared"])
        denominator_squared = float(raw["denominator_squared"])
        aggregates[group] = {
            "byte_identical": raw["mismatch_count"] == 0,
            "mismatch_count": raw["mismatch_count"],
            "element_count": raw["element_count"],
            "relative_l2": (
                math.sqrt(numerator_squared / denominator_squared)
                if denominator_squared > 0.0
                else (0.0 if numerator_squared == 0.0 else None)
            ),
            "rms": math.sqrt(numerator_squared / int(raw["element_count"])),
            "linf": raw["linf"],
            "maximum_ulp": raw["maximum_ulp"],
        }
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_state_comparison",
        "analysis_only": True,
        "reference": {
            "path": str(reference_path.resolve()),
            "sha256": _file_sha256(reference_path),
        },
        "actual": {
            "path": str(actual_path.resolve()),
            "sha256": _file_sha256(actual_path),
        },
        "aggregates": aggregates,
        "fields": fields,
        "tolerance_applied": False,
        "qualification_decision": None,
    }


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def adjudicate_equivalence(
    *,
    reference_capture_path: Path,
    actual_capture_path: Path,
    comparison_path: Path,
    expected_reference_commit: str,
    expected_actual_commit: str,
    expected_reference_package_root: Path,
    expected_actual_package_root: Path,
    minimum_steps: int,
    relative_l2_tolerance: float,
    linf_tolerance: float,
) -> dict[str, object]:
    """Bind captures to one preregistered cross-version numerical gate."""

    if minimum_steps <= 0:
        raise ValueError("minimum_steps must be positive")
    if relative_l2_tolerance < 0.0 or linf_tolerance < 0.0:
        raise ValueError("equivalence tolerances must be nonnegative")
    reference = _json_object(reference_capture_path)
    actual = _json_object(actual_capture_path)
    comparison = _json_object(comparison_path)
    if reference.get("kind") != "plane_static_lifting_state_capture":
        raise ValueError("reference capture kind is invalid")
    if actual.get("kind") != "plane_static_lifting_state_capture":
        raise ValueError("actual capture kind is invalid")
    if comparison.get("kind") != "plane_static_lifting_state_comparison":
        raise ValueError("comparison kind is invalid")
    reference_config = reference.get("config")
    actual_config = actual.get("config")
    if not isinstance(reference_config, Mapping) or not isinstance(
        actual_config, Mapping
    ):
        raise TypeError("capture config is missing")
    if dict(reference_config) != dict(actual_config):
        raise ValueError("capture configurations differ")
    if reference_config.get("variant") != "strong_planar_lifting":
        raise ValueError("cross-version capture must use strong planar lifting")
    if reference_config.get("pointwise_execution") != "compile":
        raise ValueError("cross-version capture must use compiled pointwise execution")
    steps = int(reference_config.get("steps", -1))
    if steps < minimum_steps:
        raise ValueError("cross-version capture is shorter than the frozen minimum")
    for capture, commit, package_root, label in (
        (
            reference,
            expected_reference_commit,
            expected_reference_package_root,
            "reference",
        ),
        (actual, expected_actual_commit, expected_actual_package_root, "actual"),
    ):
        git = capture.get("git")
        if not isinstance(git, Mapping) or git.get("head") != commit:
            raise ValueError(f"{label} Git identity mismatch")
        if git.get("dirty") is not False:
            raise ValueError(f"{label} Git worktree is dirty")
        imported = Path(str(capture.get("pssolver_import"))).resolve()
        if not imported.is_relative_to(package_root.resolve()):
            raise ValueError(f"{label} package import mismatch")
        if capture.get("finite") is not True:
            raise ValueError(f"{label} capture is non-finite")
        runtime = capture.get("runtime_identity")
        if not isinstance(runtime, Mapping):
            raise TypeError(f"{label} runtime identity is missing")
        if runtime.get("requested") != runtime.get("effective"):
            raise ValueError(f"{label} runtime selection changed")
        if runtime.get("fallback_used") is not False:
            raise ValueError(f"{label} runtime fallback was used")
    initial_q_identical = (
        reference.get("initial_q_sha256") == actual.get("initial_q_sha256")
    )
    if not initial_q_identical:
        raise ValueError("cross-version initial Q differs")
    reference_arrays = reference.get("arrays")
    actual_arrays = actual.get("arrays")
    comparison_reference = comparison.get("reference")
    comparison_actual = comparison.get("actual")
    if not all(
        isinstance(value, Mapping)
        for value in (
            reference_arrays,
            actual_arrays,
            comparison_reference,
            comparison_actual,
        )
    ):
        raise TypeError("array identity metadata is missing")
    if reference_arrays.get("sha256") != comparison_reference.get("sha256"):
        raise ValueError("reference array identity mismatch")
    if actual_arrays.get("sha256") != comparison_actual.get("sha256"):
        raise ValueError("actual array identity mismatch")
    fields = comparison.get("fields")
    if not isinstance(fields, Mapping) or not fields:
        raise TypeError("comparison fields are missing")
    maximum_relative_l2 = 0.0
    maximum_linf = 0.0
    all_fields_within_tolerance = True
    field_gates: dict[str, dict[str, object]] = {}
    for name, raw in sorted(fields.items()):
        if not isinstance(raw, Mapping):
            raise TypeError(f"comparison field is invalid: {name}")
        relative_l2_raw = raw.get("relative_l2")
        if relative_l2_raw is None:
            relative_l2 = math.inf
        else:
            relative_l2 = float(relative_l2_raw)
        linf = float(raw["linf"])
        if not math.isfinite(relative_l2) or not math.isfinite(linf):
            passed = False
        else:
            passed = (
                relative_l2 <= relative_l2_tolerance
                and linf <= linf_tolerance
            )
        maximum_relative_l2 = max(maximum_relative_l2, relative_l2)
        maximum_linf = max(maximum_linf, linf)
        all_fields_within_tolerance &= passed
        field_gates[str(name)] = {
            "relative_l2": relative_l2_raw,
            "linf": linf,
            "maximum_ulp": int(raw["maximum_ulp"]),
            "mismatch_count": int(raw["mismatch_count"]),
            "within_tolerance": passed,
        }
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_cross_version_equivalence",
        "reference_commit": expected_reference_commit,
        "actual_commit": expected_actual_commit,
        "shape": list(reference_config["shape"]),
        "steps": steps,
        "pointwise_execution": reference_config["pointwise_execution"],
        "initial_q_identical": initial_q_identical,
        "finite": reference.get("finite") is True and actual.get("finite") is True,
        "relative_l2_tolerance": relative_l2_tolerance,
        "linf_tolerance": linf_tolerance,
        "maximum_field_relative_l2": maximum_relative_l2,
        "maximum_field_linf": maximum_linf,
        "all_fields_within_tolerance": all_fields_within_tolerance,
        "same_runtime_restart_byte_identity_required_separately": True,
        "fields": field_gates,
    }


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {key: int(stats.get(key, 0)) for key in _MEMORY_STAT_KEYS}


def diagnose_memory(
    *,
    variant: str,
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
    warmup_steps: int,
    profile_steps: int,
    seed: int,
) -> dict[str, object]:
    """Record allocator state at each stage of one fresh CUDA process."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for allocator diagnostics")
    device = torch.device("cuda")
    config = LiftingProfileConfig(
        variant=variant,
        shape=shape,
        lengths=lengths,
        device="cuda",
        warmup_steps=warmup_steps,
        profile_steps=profile_steps,
        pointwise_execution="compile",
        seed=seed,
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    snapshots = {"empty_process": _memory_snapshot(device)}
    initial_values = _initial_q(_base_profile_config(config))
    initial_sha256 = _tensor_mapping_sha256(initial_values)
    with torch.no_grad():
        run_spec, simulation, adapter = _build_runtime(config, initial_values)
        torch.cuda.synchronize(device)
        snapshots["after_build"] = _memory_snapshot(device)
        del initial_values
        gc.collect()
        torch.cuda.empty_cache()
        snapshots["after_construction_cleanup"] = _memory_snapshot(device)
        adapter.advance(warmup_steps)
        torch.cuda.synchronize(device)
        snapshots["after_warmup"] = _memory_snapshot(device)
        torch.cuda.reset_peak_memory_stats(device)
        snapshots["profile_start"] = _memory_snapshot(device)
        adapter.advance(profile_steps)
        torch.cuda.synchronize(device)
        snapshots["after_profile"] = _memory_snapshot(device)
        adapter.synchronize_for_observation()
        torch.cuda.synchronize(device)
        snapshots["after_observation"] = _memory_snapshot(device)
        finite = all(
            bool(torch.isfinite(adapter.fields[name]).all().item())
            for name in STATE_FIELDS
        )
        runtime = adapter.to_metadata()
        completed_steps = adapter.completed_steps
        lifting = getattr(adapter.solver.model, "static_lifting_runtime", None)
        lifting_storage = None if lifting is None else lifting.storage_metadata()
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_cuda_allocator_diagnostic",
        "analysis_only": True,
        "config": {
            "variant": variant,
            "shape": list(shape),
            "lengths": list(lengths),
            "warmup_steps": warmup_steps,
            "profile_steps": profile_steps,
            "seed": seed,
            "pointwise_execution": "compile",
        },
        "configuration_identity": run_spec.identity_metadata(),
        "simulation_identity": simulation.canonical_sha256(),
        "runtime_identity": runtime,
        "git": _git_provenance(),
        "pssolver_import": __import__("pssolver").__file__,
        "initial_q_sha256": initial_sha256,
        "completed_steps": completed_steps,
        "finite": finite,
        "lifting_storage": lifting_storage,
        "memory_stat_keys": list(_MEMORY_STAT_KEYS),
        "snapshots": snapshots,
        "qualification_decision": None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture-state")
    capture.add_argument("--variant", choices=PROFILE_VARIANTS, required=True)
    capture.add_argument("--shape", type=_shape, required=True)
    capture.add_argument(
        "--lengths", type=_lengths, default=(100.0, 100.0, 20.0)
    )
    capture.add_argument("--device", choices=("cpu", "cuda"), required=True)
    capture.add_argument("--steps", type=int, default=3)
    capture.add_argument("--seed", type=int, default=20260926)
    capture.add_argument(
        "--pointwise-execution", choices=("eager", "compile"), default="compile"
    )
    capture.add_argument("--arrays-output", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare-states")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    adjudicate = subparsers.add_parser("adjudicate-equivalence")
    adjudicate.add_argument("--reference-capture", type=Path, required=True)
    adjudicate.add_argument("--actual-capture", type=Path, required=True)
    adjudicate.add_argument("--comparison", type=Path, required=True)
    adjudicate.add_argument("--expected-reference-commit", required=True)
    adjudicate.add_argument("--expected-actual-commit", required=True)
    adjudicate.add_argument(
        "--expected-reference-package-root", type=Path, required=True
    )
    adjudicate.add_argument(
        "--expected-actual-package-root", type=Path, required=True
    )
    adjudicate.add_argument("--minimum-steps", type=int, required=True)
    adjudicate.add_argument("--relative-l2-tolerance", type=float, required=True)
    adjudicate.add_argument("--linf-tolerance", type=float, required=True)
    adjudicate.add_argument("--output", type=Path, required=True)

    memory = subparsers.add_parser("diagnose-memory")
    memory.add_argument("--variant", choices=PROFILE_VARIANTS, required=True)
    memory.add_argument("--shape", type=_shape, required=True)
    memory.add_argument(
        "--lengths", type=_lengths, default=(100.0, 100.0, 20.0)
    )
    memory.add_argument("--warmup-steps", type=int, default=10)
    memory.add_argument("--profile-steps", type=int, default=50)
    memory.add_argument("--seed", type=int, default=20260926)
    memory.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "capture-state":
        result = capture_state(
            variant=args.variant,
            shape=args.shape,
            lengths=args.lengths,
            device=args.device,
            steps=args.steps,
            seed=args.seed,
            pointwise_execution=args.pointwise_execution,
            arrays_output=args.arrays_output,
        )
    elif args.command == "compare-states":
        result = compare_states(
            reference_path=args.reference,
            actual_path=args.actual,
        )
    elif args.command == "adjudicate-equivalence":
        result = adjudicate_equivalence(
            reference_capture_path=args.reference_capture,
            actual_capture_path=args.actual_capture,
            comparison_path=args.comparison,
            expected_reference_commit=args.expected_reference_commit,
            expected_actual_commit=args.expected_actual_commit,
            expected_reference_package_root=args.expected_reference_package_root,
            expected_actual_package_root=args.expected_actual_package_root,
            minimum_steps=args.minimum_steps,
            relative_l2_tolerance=args.relative_l2_tolerance,
            linf_tolerance=args.linf_tolerance,
        )
    else:
        result = diagnose_memory(
            variant=args.variant,
            shape=args.shape,
            lengths=args.lengths,
            warmup_steps=args.warmup_steps,
            profile_steps=args.profile_steps,
            seed=args.seed,
        )
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
