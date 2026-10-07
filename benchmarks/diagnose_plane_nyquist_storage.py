#!/usr/bin/env python3
"""Analysis-only Plane free-slip Nyquist/storage diagnostic for RC4.1.

The frozen RC4.1.0 record captures the released rc3 mismatch.  This program
compares full-complex and Hermitian-half storage for identical real forces and
can therefore adjudicate either the original defect or a candidate repair.
The filtered control still removes only the formerly causal non-reduced-axis
Nyquist input plane.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import torch

from pssolver.transforms import (
    FreeSlipModalStokesSolver,
    TensorProductTransformBackend,
)


SCHEMA = "pssolver.rc4_1_1.plane_nyquist_storage_diagnostic.v2"
BASELINE_COMMIT = "5071d73a00e0d19be62ebd39edf9918818d1267a"
STORAGE_EQUIVALENCE_TOLERANCE = 1.0e-12
TANGENTIAL_BC = ("periodic", "periodic", "neumann")
NORMAL_BC = ("periodic", "periodic", "dirichlet")
PRESSURE_BC = TANGENTIAL_BC
FIELD_NAMES = ("ux", "uy", "uz", "pressure")
FORCE_NAMES = ("fx", "fy", "fz")


def _backend(
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
    *,
    spectral_storage: str,
    hermitian_axis: int | None,
) -> TensorProductTransformBackend:
    return TensorProductTransformBackend(
        shape,
        lengths,
        device="cpu",
        dtype=torch.float64,
        execution_order="real_first",
        spectral_storage=spectral_storage,
        hermitian_axis=hermitian_axis,
    )


def _solver(backend: TensorProductTransformBackend) -> FreeSlipModalStokesSolver:
    return FreeSlipModalStokesSolver(
        backend,
        tangential_boundary_conditions=TANGENTIAL_BC,
        normal_boundary_conditions=NORMAL_BC,
        pressure_boundary_conditions=PRESSURE_BC,
        friction=0.23,
        viscosity=0.73,
        zero_mode_policy="friction",
    )


def _norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.reshape(-1)).item())


def _relative_l2(observed: torch.Tensor, reference: torch.Tensor) -> float:
    scale = max(_norm(reference), torch.finfo(reference.dtype).tiny)
    return _norm(observed - reference) / scale


def _spectral_axis(tensor: torch.Tensor, axis: int) -> int:
    return tensor.ndim - 3 + axis


def _nyquist_fraction(
    backend: TensorProductTransformBackend,
    tensor: torch.Tensor,
    boundary_conditions: tuple[str, str, str],
    axis: int,
) -> float:
    if backend.shape[axis] % 2:
        return 0.0
    spectral = backend.forward(tensor, boundary_conditions)
    plane = spectral.select(_spectral_axis(spectral, axis), backend.shape[axis] // 2)
    return _norm(plane) / max(_norm(spectral), torch.finfo(backend.real_dtype).tiny)


def _remove_periodic_nyquist(
    backend: TensorProductTransformBackend,
    tensor: torch.Tensor,
    boundary_conditions: tuple[str, str, str],
    axis: int,
) -> torch.Tensor:
    if backend.shape[axis] % 2:
        return tensor.clone()
    spectral = backend.forward(tensor, boundary_conditions)
    cleaned = spectral.clone()
    index = [slice(None)] * spectral.ndim
    index[_spectral_axis(spectral, axis)] = backend.shape[axis] // 2
    cleaned[tuple(index)] = 0
    return backend.inverse(cleaned, boundary_conditions)


def _collocation_periodic_derivative(
    backend: TensorProductTransformBackend,
    tensor: torch.Tensor,
    boundary_conditions: tuple[str, str, str],
    axis: int,
) -> torch.Tensor:
    """Differentiate with the real-grid convention that zeros even Nyquist."""
    spectral = backend.forward(tensor, boundary_conditions)
    modes = backend.get_metadata(boundary_conditions).axis_modes[axis].clone()
    if backend.shape[axis] % 2 == 0:
        modes[backend.shape[axis] // 2] = 0
    view = [1] * spectral.ndim
    view[_spectral_axis(spectral, axis)] = modes.numel()
    derivative = spectral * (1j * modes).view(view).to(dtype=spectral.dtype)
    return backend.inverse(derivative, boundary_conditions)


def _collocation_divergence(
    backend: TensorProductTransformBackend,
    velocity: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    dux_dx = _collocation_periodic_derivative(
        backend, velocity[0], TANGENTIAL_BC, 0
    )
    duy_dy = _collocation_periodic_derivative(
        backend, velocity[1], TANGENTIAL_BC, 1
    )
    uz_hat = backend.forward(velocity[2], NORMAL_BC)
    duz_hat, derivative_bcs = backend.gradient_hat(uz_hat, NORMAL_BC, axis=2)
    duz_dz = backend.inverse(duz_hat, derivative_bcs)
    return dux_dx + duy_dy + duz_dz


def _solve(
    forces: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
    *,
    spectral_storage: str,
    hermitian_axis: int | None,
    diagnostic_backend: TensorProductTransformBackend,
) -> dict[str, Any]:
    backend = _backend(
        shape,
        lengths,
        spectral_storage=spectral_storage,
        hermitian_axis=hermitian_axis,
    )
    solver = _solver(backend)
    force_hats = (
        backend.forward(forces[0], TANGENTIAL_BC),
        backend.forward(forces[1], TANGENTIAL_BC),
        backend.forward(forces[2], NORMAL_BC),
    )
    solution_hats = solver.solve_force_hats(*force_hats)
    physical = (
        backend.inverse(solution_hats[0], TANGENTIAL_BC),
        backend.inverse(solution_hats[1], TANGENTIAL_BC),
        backend.inverse(solution_hats[2], NORMAL_BC),
        backend.inverse(solution_hats[3], PRESSURE_BC),
    )
    native_divergence_hat = solver.divergence_hat(*solution_hats[:3])
    velocity_scale_hat = max(
        math.sqrt(sum(_norm(value) ** 2 for value in solution_hats[:3])),
        torch.finfo(backend.real_dtype).tiny,
    )
    velocity_scale = max(
        math.sqrt(sum(_norm(value) ** 2 for value in physical[:3])),
        torch.finfo(backend.real_dtype).tiny,
    )
    collocation_divergence = _collocation_divergence(
        diagnostic_backend, physical[:3]
    )
    output_nyquist = {}
    for name, value, bcs in zip(
        FIELD_NAMES,
        physical,
        (TANGENTIAL_BC, TANGENTIAL_BC, NORMAL_BC, PRESSURE_BC),
        strict=True,
    ):
        output_nyquist[name] = {
            "x_fraction": _nyquist_fraction(
                diagnostic_backend, value, bcs, 0
            ),
            "y_fraction": _nyquist_fraction(
                diagnostic_backend, value, bcs, 1
            ),
        }
    finite = all(bool(torch.isfinite(value).all().item()) for value in physical)
    return {
        "physical": physical,
        "finite": finite,
        "native_divergence_relative_l2": _norm(native_divergence_hat)
        / velocity_scale_hat,
        "collocation_divergence_relative_l2": _norm(collocation_divergence)
        / velocity_scale,
        "collocation_divergence_linf": float(
            collocation_divergence.abs().max().item()
        ),
        "pressure_iterations": int(solver.last_pressure_iterations),
        "pressure_relative_residual": float(
            solver.last_pressure_relative_residual
        ),
        "output_nyquist": output_nyquist,
    }


def _compare(
    full: dict[str, Any], half: dict[str, Any]
) -> dict[str, dict[str, float]]:
    comparisons = {}
    for name, full_value, half_value in zip(
        FIELD_NAMES, full["physical"], half["physical"], strict=True
    ):
        comparisons[name] = {
            "relative_l2": _relative_l2(half_value, full_value),
            "linf": float((half_value - full_value).abs().max().item()),
            "full_complex_l2": _norm(full_value),
        }
    return comparisons


def _public_storage_record(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "physical"}


def diagnose_case(
    name: str,
    shape: tuple[int, int, int],
    *,
    hermitian_axis: int,
    seed: int,
) -> dict[str, Any]:
    lengths = (2.0 * math.pi, 3.0 * math.pi, 2.5)
    generator = torch.Generator().manual_seed(seed)
    forces = tuple(
        torch.randn((1, *shape), generator=generator, dtype=torch.float64)
        for _ in range(3)
    )
    diagnostic_backend = _backend(
        shape,
        lengths,
        spectral_storage="full_complex",
        hermitian_axis=None,
    )
    non_reduced_axis = 1 - hermitian_axis
    force_bcs = (TANGENTIAL_BC, TANGENTIAL_BC, NORMAL_BC)
    input_nyquist = {
        force_name: {
            "x_fraction": _nyquist_fraction(
                diagnostic_backend, force, bcs, 0
            ),
            "y_fraction": _nyquist_fraction(
                diagnostic_backend, force, bcs, 1
            ),
        }
        for force_name, force, bcs in zip(
            FORCE_NAMES, forces, force_bcs, strict=True
        )
    }

    def run_pair(input_forces: tuple[torch.Tensor, ...]) -> dict[str, Any]:
        full = _solve(
            input_forces,
            shape,
            lengths,
            spectral_storage="full_complex",
            hermitian_axis=None,
            diagnostic_backend=diagnostic_backend,
        )
        half = _solve(
            input_forces,
            shape,
            lengths,
            spectral_storage="hermitian_half",
            hermitian_axis=hermitian_axis,
            diagnostic_backend=diagnostic_backend,
        )
        return {
            "comparison": _compare(full, half),
            "full_complex": _public_storage_record(full),
            "hermitian_half": _public_storage_record(half),
        }

    raw = run_pair(forces)
    filtered_forces = tuple(
        _remove_periodic_nyquist(
            diagnostic_backend, force, bcs, non_reduced_axis
        )
        for force, bcs in zip(forces, force_bcs, strict=True)
    )
    filtered = run_pair(filtered_forces)
    velocity_names = FIELD_NAMES[:3]
    raw_velocity_max = max(
        raw["comparison"][field]["relative_l2"] for field in velocity_names
    )
    filtered_velocity_max = max(
        filtered["comparison"][field]["relative_l2"]
        for field in velocity_names
    )
    return {
        "name": name,
        "shape": list(shape),
        "lengths": list(lengths),
        "hermitian_axis": hermitian_axis,
        "non_reduced_periodic_axis": non_reduced_axis,
        "non_reduced_axis_is_even": shape[non_reduced_axis] % 2 == 0,
        "input_nyquist": input_nyquist,
        "raw": raw,
        "non_reduced_nyquist_removed": filtered,
        "summary": {
            "raw_max_velocity_relative_l2": raw_velocity_max,
            "raw_pressure_relative_l2": raw["comparison"]["pressure"][
                "relative_l2"
            ],
            "filtered_max_velocity_relative_l2": filtered_velocity_max,
            "filtered_pressure_relative_l2": filtered["comparison"][
                "pressure"
            ]["relative_l2"],
        },
    }


def diagnose(*, seed: int = 24680) -> dict[str, Any]:
    cases = (
        diagnose_case(
            "even_xy_default_axis1", (16, 16, 8), hermitian_axis=1, seed=seed
        ),
        diagnose_case(
            "odd_xy_default_axis1", (15, 15, 8), hermitian_axis=1, seed=seed
        ),
        diagnose_case(
            "even_x_only_default_axis1",
            (16, 15, 8),
            hermitian_axis=1,
            seed=seed,
        ),
        diagnose_case(
            "even_y_only_default_axis1",
            (15, 16, 8),
            hermitian_axis=1,
            seed=seed,
        ),
        diagnose_case(
            "even_xy_rotated_axis0", (16, 16, 8), hermitian_axis=0, seed=seed
        ),
    )
    repaired = all(
        case["summary"][metric] <= STORAGE_EQUIVALENCE_TOLERANCE
        for case in cases
        for metric in (
            "raw_max_velocity_relative_l2",
            "raw_pressure_relative_l2",
        )
    )
    classification = (
        "PASS_PLANE_PERIODIC_NYQUIST_STORAGE_EQUIVALENCE"
        if repaired
        else "REPRODUCED_PLANE_NON_REDUCED_PERIODIC_NYQUIST_STORAGE_MISMATCH"
    )
    return {
        "schema": SCHEMA,
        "classification": classification,
        "analysis_only": True,
        "diagnostic_modifies_solver": False,
        "baseline": {
            "release": "v0.2.0rc3",
            "commit": BASELINE_COMMIT,
        },
        "configuration": {
            "dtype": "float64",
            "device": "cpu",
            "execution_order": "real_first",
            "dealias_rule": "none",
            "friction": 0.23,
            "viscosity": 0.73,
            "seed": seed,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "causal_finding": {
            "trigger": "an even non-reduced periodic axis",
            "control": "removing only that input Nyquist plane restores storage equivalence",
            "released_rc3_mismatch_frozen_separately": True,
            "current_storage_equivalence_passed": repaired,
        },
        "cases": list(cases),
    }


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": report["schema"],
        "classification": report["classification"],
        "baseline": report["baseline"],
        "configuration": report["configuration"],
        "cases": {
            case["name"]: case["summary"] for case in report["cases"]
        },
        "causal_finding": report["causal_finding"],
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--seed", type=int, default=24680)
    args = parser.parse_args()
    report = diagnose(seed=args.seed)
    payload = summary(report) if args.summary_only else report
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        write_json_atomic(args.output, payload)
        print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
