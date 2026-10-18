#!/usr/bin/env python3
"""Persist the component-resolved RC6 Channel pressure H100 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


SUMMARY_SCHEMA = "pssolver.rc6.pressure_floor_component_diagnostic.v1"
COMPONENT_SCHEMA = "pssolver.rc6.pressure_floor_component.v1"
ENVIRONMENT_SCHEMA = "pssolver.rc6.pressure_floor_environment.v1"
DTYPES = ("float32", "float64")
MODES = ("tolerance", "fixed")


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
            raise RuntimeError("pressure diagnostic JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def adjudicate_tolerance(observation: dict[str, Any]) -> dict[str, Any]:
    floor = observation.get("floor")
    recorded = observation.get("recorded_relative_residual")
    recomputed = observation.get("recomputed_relative_residual")
    valid_floor = _finite_number(floor) and float(floor) > 0.0
    predicates = {
        "termination_reason_is_relative_tolerance_met": (
            observation.get("termination_reason") == "relative_tolerance_met"
        ),
        "recorded_relative_residual_is_finite": _finite_number(recorded),
        "recorded_relative_residual_at_or_below_floor": bool(
            valid_floor
            and _finite_number(recorded)
            and float(recorded) <= float(floor)
        ),
        "recomputed_relative_residual_is_finite": _finite_number(recomputed),
        "recomputed_relative_residual_at_or_below_floor": bool(
            valid_floor
            and _finite_number(recomputed)
            and float(recomputed) <= float(floor)
        ),
    }
    return {**observation, "predicates": predicates, "passed": all(predicates.values())}


def adjudicate_fixed(observation: dict[str, Any]) -> dict[str, Any]:
    predicates = {
        "termination_reason_is_fixed_iterations_completed": (
            observation.get("termination_reason") == "fixed_iterations_completed"
        ),
        "pressure_is_finite": observation.get("pressure_finite") is True,
    }
    return {**observation, "predicates": predicates, "passed": all(predicates.values())}


def _tensor_sha256(value) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _capture_environment(device: str) -> dict[str, Any]:
    import pssolver
    import torch

    before = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    after = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }
    requested = torch.device(device)
    allocated = torch.empty((), device=requested).device
    module_path = Path(pssolver.__file__).resolve()
    purelib = Path(sys.prefix).resolve() / "lib"
    return {
        "schema": ENVIRONMENT_SCHEMA,
        "pssolver_version": importlib.metadata.version("pssolver"),
        "pssolver_import": str(module_path),
        "sys_prefix": str(Path(sys.prefix).resolve()),
        "installed_import_under_prefix": module_path.is_relative_to(purelib),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_device_count": int(torch.cuda.device_count()),
        "requested_device": device,
        "allocated_device": str(allocated),
        "device_name": torch.cuda.get_device_name(allocated),
        "tf32_before": before,
        "tf32_after": after,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "passed": bool(
            importlib.metadata.version("pssolver") == "0.2.0rc6"
            and module_path.is_relative_to(purelib)
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1
            and str(allocated) == "cuda:0"
            and "H100" in torch.cuda.get_device_name(allocated)
            and not after["cuda_matmul_allow_tf32"]
            and not after["cudnn_allow_tf32"]
        ),
    }


def _build_case(dtype_name: str, device: str) -> dict[str, Any]:
    import torch
    from pssolver import SpectralSolver
    from pssolver.functional import ChannelPressureTransposeOperator
    from pssolver.linear_solvers.stokes import (
        ChannelNoSlipModalStokesSolver,
        channel_pressure_minimum_relative_tolerance,
    )

    dtype = getattr(torch, dtype_name)
    shape = (8, 6, 4)
    floor = channel_pressure_minimum_relative_tolerance(dtype)
    spectral = SpectralSolver(
        shape, L=(8.0, 6.0, 4.0), device=device, dtype=dtype,
    )
    solver = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=floor,
        pressure_max_iterations=2000,
        pressure_fixed_iterations=None,
    )
    operator = ChannelPressureTransposeOperator(solver)
    complex_dtype = torch.complex64 if dtype is torch.float32 else torch.complex128
    generator = torch.Generator(device=device).manual_seed(9)
    rhs = operator._project_gauge(torch.complex(
        torch.randn((1, *shape), generator=generator, dtype=dtype, device=device),
        torch.randn((1, *shape), generator=generator, dtype=dtype, device=device),
    ).to(complex_dtype))
    return {
        "dtype_name": dtype_name,
        "dtype": dtype,
        "shape": shape,
        "floor": float(floor),
        "spectral": spectral,
        "solver": solver,
        "operator": operator,
        "rhs": rhs,
        "rhs_sha256": _tensor_sha256(rhs),
    }


def _observe_tolerance(case: dict[str, Any]) -> dict[str, Any]:
    import torch
    from pssolver.linear_solvers.stokes.channel_no_slip import (
        channel_pressure_active_rms,
    )

    solver = case["solver"]
    operator = case["operator"]
    rhs = case["rhs"]
    pressure = solver._solve_pressure(rhs.clone())
    residual = operator._project_gauge(
        rhs - operator.apply_pressure_operator(pressure)
    )
    numerator = channel_pressure_active_rms(
        residual, solver.pressure_inactive_mask,
    )
    denominator = channel_pressure_active_rms(
        rhs, solver.pressure_inactive_mask,
    )
    recomputed = numerator.float64_value / denominator.float64_value
    observation = {
        "schema": COMPONENT_SCHEMA,
        "dtype": case["dtype_name"],
        "mode": "tolerance",
        "shape": list(case["shape"]),
        "seed": 9,
        "friction": 0.2,
        "viscosity": 0.73,
        "floor": case["floor"],
        "requested_relative_tolerance": case["floor"],
        "pressure_max_iterations": 2000,
        "pressure_fixed_iterations": None,
        "rhs_sha256": case["rhs_sha256"],
        "termination_reason": solver.last_pressure_termination_reason,
        "iterations": int(solver.last_pressure_iterations),
        "recorded_relative_residual": solver.last_pressure_relative_residual,
        "recomputed_relative_residual": float(recomputed),
        "recomputed_residual_magnitude": numerator.to_metadata(),
        "rhs_magnitude": denominator.to_metadata(),
        "pressure_finite": bool(torch.isfinite(pressure).all().item()),
        "pressure_sha256": _tensor_sha256(pressure),
        "public_diagnostics": solver.last_pressure_diagnostics,
        "exception": None,
    }
    return adjudicate_tolerance(observation)


def _observe_fixed(case: dict[str, Any]) -> dict[str, Any]:
    import torch
    from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver

    fixed = ChannelNoSlipModalStokesSolver(
        case["spectral"].transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=case["floor"] / 8.0,
        pressure_max_iterations=4,
        pressure_fixed_iterations=4,
    )
    pressure = fixed._solve_pressure(case["rhs"].clone())
    observation = {
        "schema": COMPONENT_SCHEMA,
        "dtype": case["dtype_name"],
        "mode": "fixed",
        "shape": list(case["shape"]),
        "seed": 9,
        "friction": 0.2,
        "viscosity": 0.73,
        "floor": case["floor"],
        "requested_relative_tolerance": case["floor"] / 8.0,
        "pressure_max_iterations": 4,
        "pressure_fixed_iterations": 4,
        "rhs_sha256": case["rhs_sha256"],
        "termination_reason": fixed.last_pressure_termination_reason,
        "iterations": int(fixed.last_pressure_iterations),
        "recorded_relative_residual": fixed.last_pressure_relative_residual,
        "pressure_finite": bool(torch.isfinite(pressure).all().item()),
        "pressure_sha256": _tensor_sha256(pressure),
        "public_diagnostics": fixed.last_pressure_diagnostics,
        "exception": None,
    }
    return adjudicate_fixed(observation)


def _exception_component(dtype_name: str, mode: str, error: Exception) -> dict[str, Any]:
    return {
        "schema": COMPONENT_SCHEMA,
        "dtype": dtype_name,
        "mode": mode,
        "exception": {"type": type(error).__name__, "message": str(error)},
        "predicates": {},
        "passed": False,
    }


def execute(output_root: Path, *, device: str = "cuda") -> dict[str, Any]:
    root = output_root.expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    environment_path = root / "environment.json"
    try:
        environment = _capture_environment(device)
    except Exception as error:
        environment = {
            "schema": ENVIRONMENT_SCHEMA,
            "exception": {"type": type(error).__name__, "message": str(error)},
            "passed": False,
        }
    _atomic_json(environment_path, environment)

    components: list[dict[str, Any]] = []
    if environment.get("passed") is True:
        for dtype_name in DTYPES:
            try:
                case = _build_case(dtype_name, device)
            except Exception as error:
                for mode in MODES:
                    component = _exception_component(dtype_name, mode, error)
                    path = root / f"{dtype_name}_{mode}.json"
                    _atomic_json(path, component)
                    components.append({**component, "path": str(path), "sha256": _sha256(path)})
                continue
            for mode, observer in (
                ("tolerance", _observe_tolerance),
                ("fixed", _observe_fixed),
            ):
                try:
                    component = observer(case)
                except Exception as error:
                    component = _exception_component(dtype_name, mode, error)
                path = root / f"{dtype_name}_{mode}.json"
                _atomic_json(path, component)
                components.append({**component, "path": str(path), "sha256": _sha256(path)})

    expected = {(dtype, mode) for dtype in DTYPES for mode in MODES}
    observed = {(item.get("dtype"), item.get("mode")) for item in components}
    complete = environment.get("passed") is True and observed == expected and all(
        item.get("exception") is None for item in components
    )
    passed = complete and all(item.get("passed") is True for item in components)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "classification": (
            "PASS_RC6_PRESSURE_FLOOR_COMPONENT_DIAGNOSTIC"
            if passed else
            "FAIL_RC6_PRESSURE_FLOOR_COMPONENT_DIAGNOSTIC"
        ),
        "analysis_only": True,
        "qualification_complete": False,
        "environment": {
            "path": str(environment_path),
            "sha256": _sha256(environment_path),
            "passed": environment.get("passed") is True,
        },
        "expected_component_count": 4,
        "component_count": len(components),
        "complete": complete,
        "passed": passed,
        "components": components,
    }
    _atomic_json(root / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args(argv)
    result = execute(args.output_root, device=args.device)
    # A completed diagnostic may legitimately classify the frozen gate as
    # FAIL.  Only an incomplete observation makes the diagnostic process fail.
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
