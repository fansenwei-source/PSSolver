#!/usr/bin/env python3
"""Compare production and DAL-functional channel forward trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Channel_dal import (  # noqa: E402
    DEFAULT_INITIAL,
    DEFAULT_TARGET,
    SOURCE_LENGTHS,
    automatic_path_mask,
    load_q,
    q_input_summary,
    save_q,
)
from pssolver import write_run_metadata  # noqa: E402
from pssolver.models.active_nematics import (  # noqa: E402
    Q_convention_metadata,
    Q_magnitude,
    S_from_Q,
)
from pssolver.channel import (  # noqa: E402
    PRESSURE_MODAL_BC,
    Q_BC,
    Q_COMPONENTS,
    U_BC,
    build_active_nematic_channel,
)
from pssolver.control import (  # noqa: E402
    FunctionalSemiImplicitStep,
    conservative_block_average,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=pathlib.Path, default=DEFAULT_INITIAL)
    parser.add_argument("--target", type=pathlib.Path, default=DEFAULT_TARGET)
    parser.add_argument("--mask", type=pathlib.Path, default=None)
    parser.add_argument("--stride", type=int, nargs=3, default=(8, 2, 2))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--alpha-mean", type=float, default=2.5)
    parser.add_argument("--alpha-amplitude", type=float, default=1.5)
    parser.add_argument("--auto-mask-margin", type=int, nargs=3, default=(16, 4, 4))
    parser.add_argument("--auto-mask-relative-threshold", type=float, default=0.2)
    parser.add_argument("--mask-transition-width", type=float, default=1.5)
    parser.add_argument("--pressure-rel-tol", type=float, default=1e-7)
    parser.add_argument("--pressure-max-iter", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--report-steps",
        type=int,
        nargs="*",
        default=(1, 2, 5, 10, 20),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data_dal_forward_validation"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.initial.is_file():
        raise FileNotFoundError(f"Initial Q tensor does not exist: {args.initial}")
    if not args.target.is_file():
        raise FileNotFoundError(f"Target Q tensor does not exist: {args.target}")
    if args.mask is not None and not args.mask.is_file():
        raise FileNotFoundError(f"Mask does not exist: {args.mask}")
    if any(value <= 0 for value in args.stride):
        raise ValueError("--stride entries must be positive.")
    if args.steps <= 0 or args.dt <= 0:
        raise ValueError("--steps and --dt must be positive.")
    if args.pressure_rel_tol <= 0 or args.pressure_max_iter <= 0:
        raise ValueError("Pressure solver tolerances and iteration count must be positive.")
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("--rtol and --atol must be non-negative.")
    if args.rtol == 0 and args.atol == 0:
        raise ValueError("At least one of --rtol and --atol must be positive.")
    if any(value < 0 for value in args.auto_mask_margin):
        raise ValueError("--auto-mask-margin entries must be non-negative.")
    if not 0 < args.auto_mask_relative_threshold <= 1:
        raise ValueError("--auto-mask-relative-threshold must lie in (0, 1].")


def _declared_canonical_order_parameters(
    q_path: pathlib.Path,
) -> tuple[float, float]:
    """Read positive, distinct ``S_initial`` and ``S_bulk`` declarations."""

    metadata_path = q_path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Canonical Q metadata does not exist: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    if metadata.get("schema_version") != 1:
        raise ValueError(f"{metadata_path} must declare schema_version=1")
    model = metadata.get("model")
    if not isinstance(model, dict) or model.get("name") != "active_nematics":
        raise ValueError(
            f"{metadata_path} must declare model.name='active_nematics'"
        )
    expected_convention = Q_convention_metadata()
    if model.get("Q_convention") != expected_convention:
        raise ValueError(
            f"{metadata_path} must declare the complete canonical Q convention "
            f"{expected_convention!r}"
        )
    parameters = model.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{metadata_path} is missing model.parameters")
    missing = [name for name in ("S_initial", "S_bulk") if name not in parameters]
    if missing:
        raise ValueError(
            f"{metadata_path} is missing model.parameters fields {missing}"
        )
    values = {}
    for name in ("S_initial", "S_bulk"):
        raw_value = parameters[name]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(
                f"{metadata_path} has non-numeric model.parameters.{name}="
                f"{raw_value!r}"
            )
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{metadata_path} has invalid model.parameters.{name}={value!r}"
            )
        values[name] = value
    return values["S_initial"], values["S_bulk"]


def validate_canonical_q_inputs(
    initial_path: pathlib.Path,
    target_path: pathlib.Path,
) -> tuple[dict, dict, float]:
    """Validate source amplitudes and one thermodynamic ``S_bulk`` separately."""

    _, S_bulk = _declared_canonical_order_parameters(initial_path)
    initial_summary = q_input_summary(
        initial_path,
        expected_S_bulk=S_bulk,
    )
    target_summary = q_input_summary(
        target_path,
        expected_S_bulk=S_bulk,
    )
    return initial_summary, target_summary, S_bulk


def dynamic_q_output_summary(
    q_path: pathlib.Path,
    *,
    expected_S_initial: float,
    expected_S_bulk: float,
) -> dict:
    """Validate a dynamic Q artifact without equating final and initial order."""

    q_values = load_q(q_path)
    declared_S_initial, declared_S_bulk = _declared_canonical_order_parameters(
        q_path
    )
    expected = {
        "S_initial": float(expected_S_initial),
        "S_bulk": float(expected_S_bulk),
    }
    declared = {
        "S_initial": declared_S_initial,
        "S_bulk": declared_S_bulk,
    }
    for name in ("S_initial", "S_bulk"):
        if not np.isclose(declared[name], expected[name], rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(
                f"{q_path.parent / 'metadata.json'} declares {name}="
                f"{declared[name]:.8g}, expected {expected[name]:.8g} for this run"
            )

    S_values = np.asarray(S_from_Q(q_values))
    magnitudes = np.asarray(Q_magnitude(q_values))
    ordered_cutoff = float(np.quantile(magnitudes, 0.75))
    observed_final_S = float(np.median(S_values[magnitudes >= ordered_cutoff]))
    return {
        "artifact_role": "dynamic_Q_output",
        "shape": list(q_values.shape),
        "finite": True,
        "metadata": str(q_path.parent / "metadata.json"),
        "Q_convention": Q_convention_metadata(),
        "S_initial": declared_S_initial,
        "S_bulk": declared_S_bulk,
        "observed_final_ordered_S": observed_final_S,
    }


def load_problem(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int], dict]:
    initial = load_q(args.initial)
    target = load_q(args.target)
    if initial.shape != target.shape:
        raise ValueError(
            f"Initial shape {initial.shape} does not match target shape {target.shape}."
        )

    stride = tuple(args.stride)
    if args.mask is None:
        mask, mask_metadata = automatic_path_mask(
            initial,
            target,
            stride,
            tuple(args.auto_mask_margin),
            args.auto_mask_relative_threshold,
            args.mask_transition_width,
            device,
        )
    else:
        mask_values = np.load(args.mask)
        if mask_values.shape != initial.shape[:3]:
            raise ValueError(
                f"Mask shape {mask_values.shape} does not match Q shape {initial.shape[:3]}."
            )
        mask = torch.from_numpy(
            conservative_block_average(mask_values, stride).copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )
        mask_metadata = {"mode": "file", "path": str(args.mask)}

    initial_values = conservative_block_average(initial, stride).copy()
    initial_q = (
        torch.from_numpy(initial_values)
        .movedim(-1, 0)
        .unsqueeze(1)
        .to(device=device, dtype=torch.float32)
    )
    return initial_q, mask, tuple(initial_values.shape[:3]), mask_metadata


def initial_fields(initial_q: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        name: initial_q[index, 0].clone()
        for index, name in enumerate(Q_COMPONENTS)
    }


def build_solver(
    initial_q: torch.Tensor,
    shape: tuple[int, int, int],
    args: argparse.Namespace,
    device: torch.device,
):
    return build_active_nematic_channel(
        shape,
        SOURCE_LENGTHS,
        args.dt,
        initial_fields(initial_q),
        device=device,
        batchsize=1,
        pressure_rel_tol=args.pressure_rel_tol,
        pressure_max_iter=args.pressure_max_iter,
    )


def error_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float | bool]:
    difference = candidate - reference
    max_abs = float(difference.abs().max().item())
    l2_reference = float(torch.linalg.vector_norm(reference).item())
    l2_difference = float(torch.linalg.vector_norm(difference).item())
    relative_l2 = l2_difference / max(l2_reference, torch.finfo(reference.dtype).eps)
    reference_scale = float(reference.abs().max().item())
    allowed = atol + rtol * reference_scale
    close_ratio = max_abs / max(allowed, torch.finfo(reference.dtype).eps)
    finite = bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all())
    return {
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "reference_max_abs": reference_scale,
        "close_ratio": close_ratio,
        "passed": finite and close_ratio <= 1.0,
    }


def activity_amplitude(step: int, steps: int, mean: float, amplitude: float) -> float:
    phase = 2.0 * math.pi * step / max(steps, 1)
    return mean + amplitude * math.sin(phase)


def flatten_row(step: int, amplitude: float, metrics: dict) -> dict[str, float | int | bool]:
    row: dict[str, float | int | bool] = {
        "step": step,
        "alpha_amplitude": amplitude,
    }
    for group, values in metrics.items():
        for name, value in values.items():
            row[f"{group}_{name}"] = value
    return row


def validation_run_metadata(
    *,
    args: argparse.Namespace,
    shape: tuple[int, int, int],
    S_bulk: float,
    initial_input: dict,
    target_input: dict,
    mask_metadata: dict,
) -> dict:
    """Build canonical metadata for the production/functional PDE comparison."""

    return {
        "schema_version": 1,
        "script": pathlib.Path(__file__).name,
        "solver": {
            "shape": list(shape),
            "lengths": list(SOURCE_LENGTHS),
            "dt": args.dt,
            "steps": args.steps,
            "save_interval": None,
        },
        "model": {
            "name": "active_nematics",
            "Q_convention": Q_convention_metadata(),
            "parameters": {
                "A": -1.0,
                "B": -6.0,
                "C": 6.0,
                "L1": 1.0,
                "S_initial": initial_input["S_initial"],
                "S_bulk": S_bulk,
                "flow_alignment": 1.0,
                "active_stress_sign": -1.0,
                "alpha": {
                    "type": "sinusoidal",
                    "mean": args.alpha_mean,
                    "amplitude": args.alpha_amplitude,
                },
                "fric": 0.0,
                "eta": 1.0,
            },
        },
        "boundary_conditions": {
            "Q": list(Q_BC),
            "velocity": list(U_BC),
            "pressure_modal": list(PRESSURE_MODAL_BC),
        },
        "numerics": {
            "integrator": "semi_implicit_euler",
            "dealiasing": "none",
            "velocity_zero_mode": "not_applicable_with_dirichlet_walls",
            "grid_transfer": {
                "method": "conservative_block_average",
                "stride": list(args.stride),
            },
            "pressure_solver": {
                "relative_tolerance": args.pressure_rel_tol,
                "maximum_iterations": args.pressure_max_iter,
                "fixed_iterations": None,
                "warm_start_cleared_each_step": True,
            },
        },
        "initial_condition": {
            "name": "external_Q",
            "path": str(args.initial.resolve()),
            "input": initial_input,
        },
        "target": {
            "name": "external_Q",
            "path": str(args.target.resolve()),
            "input": target_input,
        },
        "validation": {
            "comparison": "production_vs_functional_semi_implicit_step",
            "rtol": args.rtol,
            "atol": args.atol,
            "report_steps": list(args.report_steps),
            "mask": None if args.mask is None else str(args.mask.resolve()),
            "mask_metadata": mask_metadata,
        },
    }


def run_validation(args: argparse.Namespace) -> dict:
    validate_args(args)
    device = torch.device(args.device)
    initial_input, target_input, S_bulk = validate_canonical_q_inputs(
        args.initial,
        args.target,
    )
    initial_q, mask, shape, mask_metadata = load_problem(args, device)
    production_solver = build_solver(initial_q, shape, args, device)
    functional_solver = build_solver(initial_q, shape, args, device)
    stepper = FunctionalSemiImplicitStep(functional_solver, deterministic=True)

    args.output.mkdir(parents=True, exist_ok=True)
    run_metadata = validation_run_metadata(
        args=args,
        shape=shape,
        S_bulk=S_bulk,
        initial_input=initial_input,
        target_input=target_input,
        mask_metadata=mask_metadata,
    )
    write_run_metadata(args.output, run_metadata, status="running")

    functional_q = torch.stack(
        [functional_solver.fields[name] for name in Q_COMPONENTS]
    ).detach()
    report_steps = set(args.report_steps) | {1, args.steps}
    rows = []
    amplitudes = []

    for step in range(args.steps):
        amplitude = activity_amplitude(
            step,
            args.steps,
            args.alpha_mean,
            args.alpha_amplitude,
        )
        amplitudes.append(amplitude)
        alpha = (amplitude * mask).unsqueeze(0)

        functional_q = stepper(functional_q, alpha).detach()
        functional_static = functional_solver.fields.spatial[
            functional_solver.fields.dyn_count :
        ].detach().clone()

        production_solver.parameters["alpha"] = alpha
        production_solver.model.static_model.pressure_guess = None
        production_solver.integrator.step()
        production_q = production_solver.fields.spatial[
            : production_solver.fields.dyn_count
        ].detach().clone()
        production_static = production_solver.fields.spatial[
            production_solver.fields.dyn_count :
        ].detach().clone()

        metrics = {
            "q": error_metrics(
                functional_q,
                production_q,
                rtol=args.rtol,
                atol=args.atol,
            ),
            "velocity": error_metrics(
                functional_static[:3],
                production_static[:3],
                rtol=args.rtol,
                atol=args.atol,
            ),
            "pressure": error_metrics(
                functional_static[3:4],
                production_static[3:4],
                rtol=args.rtol,
                atol=args.atol,
            ),
        }
        row = flatten_row(step + 1, amplitude, metrics)
        rows.append(row)
        if step + 1 in report_steps:
            print(
                f"{step + 1:5d}  {amplitude:9.5f}  "
                f"{metrics['q']['relative_l2']:.3e}  {metrics['q']['max_abs']:.3e}  "
                f"{metrics['velocity']['relative_l2']:.3e}  "
                f"{metrics['pressure']['relative_l2']:.3e}",
                flush=True,
            )

    passed = all(
        row[f"{group}_passed"]
        for row in rows
        for group in ("q", "velocity", "pressure")
    )
    worst = {
        group: {
            "max_abs": max(float(row[f"{group}_max_abs"]) for row in rows),
            "relative_l2": max(float(row[f"{group}_relative_l2"]) for row in rows),
            "close_ratio": max(float(row[f"{group}_close_ratio"]) for row in rows),
        }
        for group in ("q", "velocity", "pressure")
    }

    with (args.output / "forward_consistency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_q(args.output / "Q_functional_final.npy", functional_q)
    save_q(args.output / "Q_production_final.npy", production_q)
    np.save(args.output / "alpha_amplitudes.npy", np.asarray(amplitudes))
    np.save(args.output / "control_mask.npy", mask.detach().cpu().numpy())

    summary = {
        "passed": passed,
        "initial": str(args.initial),
        "target": str(args.target),
        "mask": str(args.mask) if args.mask is not None else None,
        "mask_metadata": mask_metadata,
        "shape": list(shape),
        "source_lengths": list(SOURCE_LENGTHS),
        "stride": list(args.stride),
        "steps": args.steps,
        "dt": args.dt,
        "final_time": args.steps * args.dt,
        "activity": {
            "mean": args.alpha_mean,
            "amplitude": args.alpha_amplitude,
            "minimum": min(amplitudes),
            "maximum": max(amplitudes),
        },
        "tolerances": {"rtol": args.rtol, "atol": args.atol},
        "pressure": {
            "relative_tolerance": args.pressure_rel_tol,
            "maximum_iterations": args.pressure_max_iter,
            "warm_start_cleared_each_step": True,
        },
        "worst_errors": worst,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    functional_output = dynamic_q_output_summary(
        args.output / "Q_functional_final.npy",
        expected_S_initial=initial_input["S_initial"],
        expected_S_bulk=S_bulk,
    )
    production_output = dynamic_q_output_summary(
        args.output / "Q_production_final.npy",
        expected_S_initial=initial_input["S_initial"],
        expected_S_bulk=S_bulk,
    )
    run_metadata["results"] = {
        "passed": passed,
        "Q_functional_final": functional_output,
        "Q_production_final": production_output,
        "worst_errors": worst,
        "summary": str((args.output / "summary.json").resolve()),
    }
    write_run_metadata(args.output, run_metadata, status="complete")
    return summary


def main() -> None:
    args = parse_args()
    print(
        " step      alpha       Q rel L2      Q max abs      u rel L2      p rel L2",
        flush=True,
    )
    summary = run_validation(args)
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
