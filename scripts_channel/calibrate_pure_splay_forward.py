#!/usr/bin/env python3
"""Calibrate pure-splay-loop stability and motion under constant activity."""

from __future__ import annotations

import argparse
import csv
import json
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
    conservative_block_average,
    loop_core_metrics,
    periodic_profile_shift,
    x_disturbance_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=pathlib.Path, default=DEFAULT_INITIAL)
    parser.add_argument("--target", type=pathlib.Path, default=DEFAULT_TARGET)
    parser.add_argument("--mask", type=pathlib.Path, default=None)
    parser.add_argument("--stride", type=int, nargs=3, default=(8, 2, 2))
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.0, 2.5, 5.0))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--auto-mask-margin", type=int, nargs=3, default=(16, 4, 4))
    parser.add_argument("--auto-mask-relative-threshold", type=float, default=0.2)
    parser.add_argument("--mask-transition-width", type=float, default=1.5)
    parser.add_argument("--core-deficit-threshold", type=float, default=0.2)
    parser.add_argument("--pressure-rel-tol", type=float, default=1e-7)
    parser.add_argument("--pressure-max-iter", type=int, default=100)
    parser.add_argument("--motion-speed-threshold", type=float, default=1e-3)
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data_pure_splay_forward_calibration"),
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
    if args.steps <= 0 or args.sample_every <= 0 or args.dt <= 0:
        raise ValueError("--steps, --sample-every, and --dt must be positive.")
    if not args.alphas or any(value < 0 for value in args.alphas):
        raise ValueError("--alphas must contain non-negative values.")
    if any(value < 0 for value in args.auto_mask_margin):
        raise ValueError("--auto-mask-margin entries must be non-negative.")
    if not 0 < args.auto_mask_relative_threshold <= 1:
        raise ValueError("--auto-mask-relative-threshold must lie in (0, 1].")
    if not 0 <= args.core_deficit_threshold < 1:
        raise ValueError("--core-deficit-threshold must lie in [0, 1).")
    if args.motion_speed_threshold < 0:
        raise ValueError("--motion-speed-threshold must be non-negative.")


def q_tensor_from_solver(solver) -> torch.Tensor:
    return torch.stack([solver.fields[name] for name in Q_COMPONENTS]).detach()


def q_numpy(q_tensor: torch.Tensor) -> np.ndarray:
    return q_tensor.detach().cpu().squeeze(1).movedim(0, -1).numpy()


def _declared_canonical_order_parameters(
    q_path: pathlib.Path,
) -> tuple[float, float]:
    """Read positive, distinct ``S_initial`` and ``S_bulk`` declarations."""

    metadata_path = q_path.parent / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Canonical Q metadata does not exist: {metadata_path}"
        )
    with metadata_path.open() as handle:
        metadata = json.load(handle)
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


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_problem(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int], dict]:
    initial_values = load_q(args.initial)
    target_values = load_q(args.target)
    if initial_values.shape != target_values.shape:
        raise ValueError(
            f"Initial shape {initial_values.shape} does not match "
            f"target shape {target_values.shape}."
        )
    stride = tuple(args.stride)
    if args.mask is None:
        mask, mask_metadata = automatic_path_mask(
            initial_values,
            target_values,
            stride,
            tuple(args.auto_mask_margin),
            args.auto_mask_relative_threshold,
            args.mask_transition_width,
            device,
        )
    else:
        mask_values = np.load(args.mask)
        if mask_values.shape != initial_values.shape[:3]:
            raise ValueError(
                f"Mask shape {mask_values.shape} does not match "
                f"Q shape {initial_values.shape[:3]}."
            )
        mask = torch.from_numpy(
            conservative_block_average(mask_values, stride).copy()
        ).to(
            device=device,
            dtype=torch.float32,
        )
        mask_metadata = {"mode": "file", "path": str(args.mask)}

    initial_values = conservative_block_average(initial_values, stride).copy()
    target_values = conservative_block_average(target_values, stride).copy()
    initial_q = (
        torch.from_numpy(initial_values)
        .movedim(-1, 0)
        .unsqueeze(1)
        .to(device=device, dtype=torch.float32)
    )
    target_q = (
        torch.from_numpy(target_values)
        .movedim(-1, 0)
        .unsqueeze(1)
        .to(device=device, dtype=torch.float32)
    )
    return initial_q, target_q, mask, tuple(initial_values.shape[:3]), mask_metadata


def build_solver(
    initial_q: torch.Tensor,
    shape: tuple[int, int, int],
    args: argparse.Namespace,
    device: torch.device,
):
    initial_fields = {
        name: initial_q[index, 0].clone()
        for index, name in enumerate(Q_COMPONENTS)
    }
    return build_active_nematic_channel(
        shape,
        SOURCE_LENGTHS,
        args.dt,
        initial_fields,
        device=device,
        batchsize=1,
        pressure_rel_tol=args.pressure_rel_tol,
        pressure_max_iter=args.pressure_max_iter,
    )


def static_velocity_metrics(solver) -> dict[str, float | int]:
    static_hat = solver.model.compute_static()
    start = solver.fields.dyn_count
    static_spatial = torch.stack(
        [
            solver.fields.inverse_transform(start + index, spectral=static_hat[index])
            for index in range(solver.fields.stat_count)
        ]
    )
    velocity = static_spatial[:3]
    model = solver.model.static_model
    return {
        "velocity_rms": float(torch.sqrt(torch.mean(velocity.square())).item()),
        "velocity_max": float(velocity.abs().max().item()),
        "pressure_iterations": int(model.last_pressure_iterations),
        "pressure_relative_residual": float(model.last_pressure_relative_residual),
    }


def normalized_masked_mismatch(
    q: torch.Tensor,
    target_q: torch.Tensor,
    mask: torch.Tensor,
    normalization: float,
) -> float:
    weighted = (q - target_q) * torch.sqrt(mask).unsqueeze(0).unsqueeze(0)
    return float(torch.linalg.vector_norm(weighted).item() / max(normalization, 1e-30))


def diagnostic_row(
    *,
    step: int,
    q: torch.Tensor,
    target_q: torch.Tensor,
    mask: torch.Tensor,
    mismatch_normalization: float,
    reference_profile: np.ndarray,
    S_bulk: float,
    core_deficit_threshold: float,
    solver,
) -> dict[str, float | int | bool]:
    q_values = q_numpy(q)
    finite = bool(np.isfinite(q_values).all())
    if not finite:
        return {
            "step": step,
            "time": step * solver.dt,
            "finite": False,
            "loop_detected": False,
            "yz_plane_topology": False,
        }

    core = loop_core_metrics(
        q_values,
        SOURCE_LENGTHS,
        S_bulk=S_bulk,
        deficit_threshold=core_deficit_threshold,
    )
    profile = x_disturbance_profile(
        q_values,
        S_bulk=S_bulk,
    )
    shift = periodic_profile_shift(reference_profile, profile, SOURCE_LENGTHS[0])
    velocity = static_velocity_metrics(solver)
    counts = core["defect_plaquettes"]
    return {
        "step": step,
        "time": step * solver.dt,
        "finite": finite,
        "loop_detected": bool(core["detected"]),
        "yz_plane_topology": bool(core["yz_plane_topology"]),
        "center_x": float(core["center_x"]),
        "center_y": float(core["center_y"]),
        "center_z": float(core["center_z"]),
        "radius_rms": float(core["radius_rms"]),
        "radius_y_rms": float(core["radius_y_rms"]),
        "radius_z_rms": float(core["radius_z_rms"]),
        "x_thickness_rms": float(core["x_thickness_rms"]),
        "core_mass": float(core["core_mass"]),
        "S_min": float(core["S_min"]),
        "S_median": float(core["S_median"]),
        "defect_xy": int(counts["xy"]),
        "defect_xz": int(counts["xz"]),
        "defect_yz": int(counts["yz"]),
        "template_shift_x": float(shift["shift_physical"]),
        "template_correlation": float(shift["correlation"]),
        "normalized_target_mismatch": normalized_masked_mismatch(
            q,
            target_q,
            mask,
            mismatch_normalization,
        ),
        **velocity,
    }


def unwrap_displacements(values: list[float], period: float) -> np.ndarray:
    radians = np.asarray(values, dtype=float) * 2.0 * np.pi / period
    return np.unwrap(radians) * period / (2.0 * np.pi)


def summarize_branch(
    alpha: float,
    rows: list[dict[str, float | int | bool]],
    target_displacement: float,
    speed_threshold: float,
) -> dict:
    valid_rows = [
        row
        for row in rows
        if row.get("finite") and np.isfinite(float(row.get("template_shift_x", np.nan)))
    ]
    all_finite = bool(valid_rows and len(valid_rows) == len(rows))
    topology_preserved = bool(
        valid_rows
        and all_finite
        and all(bool(row["loop_detected"]) for row in valid_rows)
        and all(bool(row["yz_plane_topology"]) for row in valid_rows)
    )
    topology_rows = [
        row
        for row in valid_rows
        if bool(row["loop_detected"]) and bool(row["yz_plane_topology"])
    ]
    failure_rows = [
        row
        for row in rows
        if (
            not bool(row.get("finite", False))
            or not bool(row.get("loop_detected", False))
            or not bool(row.get("yz_plane_topology", False))
        )
    ]
    speed = float("nan")
    if len(valid_rows) >= 3:
        times = np.asarray([float(row["time"]) for row in valid_rows])
        displacements = unwrap_displacements(
            [float(row["template_shift_x"]) for row in valid_rows],
            SOURCE_LENGTHS[0],
        )
        fit_start = len(times) // 2
        speed = float(np.polyfit(times[fit_start:], displacements[fit_start:], 1)[0])

    if not all_finite:
        classification = "numerical_failure"
    elif not topology_preserved:
        classification = "loop_or_topology_failure"
    elif not np.isfinite(speed) or abs(speed) < speed_threshold:
        classification = "stationary"
    elif speed * target_displacement > 0:
        final_mismatch = float(valid_rows[-1]["normalized_target_mismatch"])
        classification = (
            "moves_toward_target"
            if final_mismatch <= 1.0
            else "moves_toward_target_but_state_mismatch_worsens"
        )
    else:
        classification = "moves_away_from_target"

    estimated_time = None
    if (
        classification == "moves_toward_target"
        and abs(speed) >= speed_threshold
    ):
        estimated_time = abs(target_displacement / speed)

    initial_radius = float(valid_rows[0]["radius_rms"]) if valid_rows else float("nan")
    final_radius = float(valid_rows[-1]["radius_rms"]) if valid_rows else float("nan")
    return {
        "alpha": alpha,
        "classification": classification,
        "all_finite": all_finite,
        "topology_preserved": topology_preserved,
        "last_topology_preserved_time": (
            float(topology_rows[-1]["time"]) if topology_rows else None
        ),
        "first_topology_failure_time": (
            float(failure_rows[0]["time"]) if failure_rows else None
        ),
        "samples_completed": len(rows),
        "final_step": int(rows[-1]["step"]),
        "final_time": float(rows[-1]["time"]),
        "estimated_x_speed": speed,
        "target_displacement": target_displacement,
        "estimated_time_to_target": estimated_time,
        "maximum_displacement_while_topology_preserved": (
            max(abs(float(row["template_shift_x"])) for row in topology_rows)
            if topology_rows
            else 0.0
        ),
        "initial_radius": initial_radius,
        "final_radius": final_radius,
        "relative_radius_change": (
            (final_radius - initial_radius) / initial_radius
            if np.isfinite(initial_radius) and initial_radius > 0
            else float("nan")
        ),
        "final_template_displacement": (
            float(valid_rows[-1]["template_shift_x"]) if valid_rows else float("nan")
        ),
        "final_template_correlation": (
            float(valid_rows[-1]["template_correlation"]) if valid_rows else float("nan")
        ),
        "final_normalized_target_mismatch": (
            float(valid_rows[-1]["normalized_target_mismatch"])
            if valid_rows
            else float("nan")
        ),
        "minimum_normalized_target_mismatch_while_topology_preserved": (
            min(float(row["normalized_target_mismatch"]) for row in topology_rows)
            if topology_rows
            else float("nan")
        ),
    }


def branch_name(alpha: float) -> str:
    return f"alpha_{alpha:g}".replace("-", "minus_").replace(".", "p")


def branch_run_metadata(
    *,
    alpha: float,
    shape: tuple[int, int, int],
    args: argparse.Namespace,
    S_bulk: float,
    initial_input: dict,
    target_input: dict,
    mask_metadata: dict,
) -> dict:
    """Build canonical metadata for one constant-activity PDE branch."""

    return json_safe(
        {
            "schema_version": 1,
            "script": pathlib.Path(__file__).name,
            "solver": {
                "shape": list(shape),
                "lengths": list(SOURCE_LENGTHS),
                "dt": args.dt,
                "steps": args.steps,
                "save_interval": args.sample_every if args.save_snapshots else None,
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
                    "alpha": alpha,
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
            "calibration": {
                "activity": alpha,
                "sample_every": args.sample_every,
                "core_deficit_threshold": args.core_deficit_threshold,
                "motion_speed_threshold": args.motion_speed_threshold,
                "mask": None if args.mask is None else str(args.mask.resolve()),
                "mask_metadata": mask_metadata,
            },
        }
    )


def run_branch(
    *,
    alpha: float,
    initial_q: torch.Tensor,
    target_q: torch.Tensor,
    mask: torch.Tensor,
    shape: tuple[int, int, int],
    args: argparse.Namespace,
    device: torch.device,
    reference_profile: np.ndarray,
    S_bulk: float,
    mismatch_normalization: float,
    target_displacement: float,
    initial_input: dict,
    target_input: dict,
    mask_metadata: dict,
) -> tuple[list[dict], dict]:
    solver = build_solver(initial_q, shape, args, device)
    solver.parameters["alpha"] = alpha * mask.unsqueeze(0)
    output = args.output / branch_name(alpha)
    output.mkdir(parents=True, exist_ok=True)
    run_metadata = branch_run_metadata(
        alpha=alpha,
        shape=shape,
        args=args,
        S_bulk=S_bulk,
        initial_input=initial_input,
        target_input=target_input,
        mask_metadata=mask_metadata,
    )
    write_run_metadata(output, run_metadata, status="running")
    rows = []

    def record(step: int) -> bool:
        q = q_tensor_from_solver(solver)
        row = diagnostic_row(
            step=step,
            q=q,
            target_q=target_q,
            mask=mask,
            mismatch_normalization=mismatch_normalization,
            reference_profile=reference_profile,
            S_bulk=S_bulk,
            core_deficit_threshold=args.core_deficit_threshold,
            solver=solver,
        )
        rows.append(row)
        if args.save_snapshots:
            save_q(output / f"Q_{step}.npy", q)
        print(
            f"alpha={alpha:g} step={step:5d} t={step * args.dt:.4f} "
            f"dx={float(row.get('template_shift_x', np.nan)):+.4f} "
            f"R={float(row.get('radius_rms', np.nan)):.4f} "
            f"yz_plane_topology={row.get('yz_plane_topology', False)} "
            f"target={float(row.get('normalized_target_mismatch', np.nan)):.4f}",
            flush=True,
        )
        return bool(row.get("finite", False))

    record(0)
    current_step = 0
    while current_step < args.steps:
        next_step = min(current_step + args.sample_every, args.steps)
        solver.run(next_step - current_step)
        current_step = next_step
        if not record(current_step):
            break

    final_q = q_tensor_from_solver(solver)
    save_q(output / "Q_final.npy", final_q)
    with (output / "diagnostics.csv").open("w", newline="") as stream:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_branch(
        alpha,
        rows,
        target_displacement,
        args.motion_speed_threshold,
    )
    with (output / "summary.json").open("w") as stream:
        json.dump(json_safe(summary), stream, indent=2, allow_nan=False)
    output_summary = dynamic_q_output_summary(
        output / "Q_final.npy",
        expected_S_initial=initial_input["S_initial"],
        expected_S_bulk=S_bulk,
    )
    run_metadata["results"] = {
        "Q_final": output_summary,
        "summary": json_safe(summary),
        "snapshots_saved": bool(args.save_snapshots),
    }
    write_run_metadata(output, run_metadata, status="complete")
    return rows, summary


def run_calibration(args: argparse.Namespace) -> dict:
    validate_args(args)
    device = torch.device(args.device)
    initial_input, target_input, S_bulk = validate_canonical_q_inputs(
        args.initial,
        args.target,
    )
    initial_q, target_q, mask, shape, mask_metadata = load_problem(args, device)
    initial_values = q_numpy(initial_q)
    target_values = q_numpy(target_q)
    reference_profile = x_disturbance_profile(
        initial_values,
        S_bulk=S_bulk,
    )
    target_profile = x_disturbance_profile(
        target_values,
        S_bulk=S_bulk,
    )
    target_shift = periodic_profile_shift(
        reference_profile,
        target_profile,
        SOURCE_LENGTHS[0],
    )
    target_displacement = float(target_shift["shift_physical"])
    weighted_initial_mismatch = (
        (initial_q - target_q) * torch.sqrt(mask).unsqueeze(0).unsqueeze(0)
    )
    mismatch_normalization = float(
        torch.linalg.vector_norm(weighted_initial_mismatch).item()
    )

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "control_mask.npy", mask.detach().cpu().numpy())
    scan_metadata = branch_run_metadata(
        alpha=float(args.alphas[0]),
        shape=shape,
        args=args,
        S_bulk=S_bulk,
        initial_input=initial_input,
        target_input=target_input,
        mask_metadata=mask_metadata,
    )
    scan_metadata["model"]["parameters"]["alpha"] = {
        "type": "constant_activity_scan",
        "values": [float(alpha) for alpha in args.alphas],
    }
    scan_metadata["calibration"].pop("activity")
    scan_metadata["calibration"]["activities"] = [float(alpha) for alpha in args.alphas]
    write_run_metadata(args.output, scan_metadata, status="running")
    branch_summaries = []
    for alpha in args.alphas:
        _, summary = run_branch(
            alpha=float(alpha),
            initial_q=initial_q,
            target_q=target_q,
            mask=mask,
            shape=shape,
            args=args,
            device=device,
            reference_profile=reference_profile,
            S_bulk=S_bulk,
            mismatch_normalization=mismatch_normalization,
            target_displacement=target_displacement,
            initial_input=initial_input,
            target_input=target_input,
            mask_metadata=mask_metadata,
        )
        branch_summaries.append(summary)

    with (args.output / "calibration_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(branch_summaries[0]))
        writer.writeheader()
        writer.writerows(branch_summaries)
    result = {
        "initial": str(args.initial),
        "target": str(args.target),
        "shape": list(shape),
        "stride": list(args.stride),
        "lengths": list(SOURCE_LENGTHS),
        "steps": args.steps,
        "dt": args.dt,
        "sample_every": args.sample_every,
        "S_bulk": S_bulk,
        "target_template_displacement": target_displacement,
        "target_template_correlation": float(target_shift["correlation"]),
        "mask_metadata": mask_metadata,
        "branches": branch_summaries,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(json_safe(result), stream, indent=2, allow_nan=False)
    scan_metadata["results"] = {
        "branches": json_safe(branch_summaries),
        "summary": str((args.output / "summary.json").resolve()),
    }
    write_run_metadata(args.output, scan_metadata, status="complete")
    return result


def main() -> None:
    result = run_calibration(parse_args())
    print(json.dumps(json_safe(result), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
