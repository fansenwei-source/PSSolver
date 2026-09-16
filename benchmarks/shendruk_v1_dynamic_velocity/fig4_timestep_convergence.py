#!/usr/bin/env python3
"""Plan, execute, validate, and analyze the inertial Fig. 4 timestep study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import numpy as np

from benchmarks.shendruk_v1_dynamic_velocity import fig4_workflow as base


PROJECT_ROOT = base.PROJECT_ROOT
MODEL_SCRIPT = base.MODEL_SCRIPT
TIMESTEP_SCRIPT = Path(__file__).resolve()
DT_VALUES = (0.01, 0.005, 0.0025)
FINAL_TIME = 1.0
COMMON_OUTPUT_TIMES = (0.25, 0.5, 0.75, 1.0)
DIAGNOSTIC_TIME_INTERVAL = 0.05
FIELD_PREFIXES = ("Q", "u", "p")


def _integer_steps(time_value: float, dt: float, label: str) -> int:
    value = time_value / dt
    rounded = int(round(value))
    if not math.isclose(value, rounded, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"{label}={time_value} is not an integer multiple of dt={dt}")
    return rounded


def dt_token(dt: float) -> str:
    return f"{dt:g}".replace(".", "p")


def convergence_config(dt: float, commit: str) -> dict[str, Any]:
    if dt not in DT_VALUES:
        raise ValueError(f"unsupported timestep {dt}")
    steps = _integer_steps(FINAL_TIME, dt, "final_time")
    output_steps = [
        _integer_steps(time_value, dt, "common_output_time")
        for time_value in COMMON_OUTPUT_TIMES
    ]
    interval = output_steps[0]
    if output_steps != list(range(interval, steps + 1, interval)):
        raise RuntimeError("common output times must form one uniform step interval")
    config = base.scientific_config(
        height=20,
        nz=80,
        activity_number=18.0,
        commit=commit,
    )
    config.update(
        {
            "run_id": f"H20_A18_R320_dt{dt_token(dt)}_T1_seed24_real_first",
            "study": "inertial_timestep_convergence",
            "dt": dt,
            "steps": steps,
            "final_time": FINAL_TIME,
            "common_output_times": list(COMMON_OUTPUT_TIMES),
            "save_start_step": interval,
            "save_interval": interval,
            "diagnostic_interval": _integer_steps(
                DIAGNOSTIC_TIME_INTERVAL, dt, "diagnostic_time_interval"
            ),
            "save_hydrodynamics": True,
        }
    )
    return config


def build_rows(commit: str) -> list[dict[str, Any]]:
    rows = []
    for index, dt in enumerate(DT_VALUES):
        config = convergence_config(dt, commit)
        rows.append(
            {
                "index": index,
                "config_sha256": base.canonical_sha256(config),
                "config": config,
            }
        )
    return rows


def build_plan() -> dict[str, Any]:
    git = base.git_identity()
    return {
        "schema_version": 1,
        "workflow": "shendruk_v1_dynamic_velocity_timestep_convergence",
        "purpose": (
            "short-time first-order convergence and dt=0.005 adequacy gate; "
            "not a long-time statistical convergence claim"
        ),
        "git": git,
        "model_script": str(MODEL_SCRIPT.relative_to(PROJECT_ROOT)),
        "model_script_sha256": base.sha256_file(MODEL_SCRIPT),
        "fig4_workflow_sha256": base.sha256_file(Path(base.__file__).resolve()),
        "timestep_workflow_sha256": base.sha256_file(TIMESTEP_SCRIPT),
        "dt_values": list(DT_VALUES),
        "final_time": FINAL_TIME,
        "common_output_times": list(COMMON_OUTPUT_TIMES),
        "rows": build_rows(git["head"]),
        "acceptance": {
            "minimum_median_order_q_u_relative_l2": 0.5,
            "dt_0p005_vs_0p0025_final_relative_l2": {
                "Q": 1.0e-3,
                "u": 1.0e-2,
                "p_demeaned": 1.0e-2,
            },
            "dt_0p005_vs_0p0025_final_normalized_linf": {
                "Q": 5.0e-3,
                "u": 5.0e-2,
                "p_demeaned": 5.0e-2,
            },
            "kinetic_energy_relative_change": 1.0e-2,
        },
    }


def load_plan(path: Path, *, require_clean: bool) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported plan schema")
    if plan.get("workflow") != "shendruk_v1_dynamic_velocity_timestep_convergence":
        raise ValueError("unexpected workflow")
    git = base.git_identity()
    if plan.get("git", {}).get("head") != git["head"]:
        raise RuntimeError("current HEAD differs from the bound plan")
    if require_clean and git["dirty"]:
        raise RuntimeError("formal execution requires a clean worktree")
    expected_hashes = {
        "model_script_sha256": base.sha256_file(MODEL_SCRIPT),
        "fig4_workflow_sha256": base.sha256_file(Path(base.__file__).resolve()),
        "timestep_workflow_sha256": base.sha256_file(TIMESTEP_SCRIPT),
    }
    for name, expected in expected_hashes.items():
        if plan.get(name) != expected:
            raise RuntimeError(f"{name} differs from the bound plan")
    if plan.get("rows") != build_rows(git["head"]):
        raise RuntimeError("plan rows differ from the canonical timestep matrix")
    return plan


def selected_row(plan: dict[str, Any], index: int) -> tuple[dict[str, Any], str]:
    if not 0 <= index < len(plan["rows"]):
        raise IndexError(f"matrix index {index} is outside [0, 2]")
    row = plan["rows"][index]
    if row["config_sha256"] != base.canonical_sha256(row["config"]):
        raise RuntimeError("selected plan row has an invalid configuration hash")
    return dict(row["config"]), row["config_sha256"]


def _array_mean(array: np.ndarray, chunk_planes: int = 8) -> float:
    partials = []
    count = 0
    for start in range(0, array.shape[0], chunk_planes):
        chunk = np.asarray(array[start : start + chunk_planes], dtype=np.float64)
        partials.append(float(np.sum(chunk, dtype=np.float64)))
        count += chunk.size
    return math.fsum(partials) / count


def array_error_metrics(
    first: np.ndarray,
    second: np.ndarray,
    *,
    demean: bool = False,
    chunk_planes: int = 8,
) -> dict[str, Any]:
    if first.shape != second.shape or first.dtype != second.dtype:
        raise ValueError("compared arrays must have identical shape and dtype")
    if first.dtype != np.dtype("float64"):
        raise ValueError("convergence arrays must use float64")
    first_mean = _array_mean(first, chunk_planes) if demean else 0.0
    second_mean = _array_mean(second, chunk_planes) if demean else 0.0
    squared_error = 0.0
    squared_reference = 0.0
    maximum_error = 0.0
    maximum_reference = 0.0
    count = 0
    for start in range(0, first.shape[0], chunk_planes):
        left = np.asarray(first[start : start + chunk_planes], dtype=np.float64)
        right = np.asarray(second[start : start + chunk_planes], dtype=np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError("compared arrays must contain only finite values")
        left = left - first_mean
        right = right - second_mean
        difference = left - right
        squared_error += float(np.sum(difference * difference, dtype=np.float64))
        squared_reference += float(np.sum(right * right, dtype=np.float64))
        maximum_error = max(maximum_error, float(np.max(np.abs(difference))))
        maximum_reference = max(maximum_reference, float(np.max(np.abs(right))))
        count += difference.size
    tiny = np.finfo(np.float64).tiny
    return {
        "shape": list(first.shape),
        "dtype": first.dtype.name,
        "demeaned": demean,
        "max_abs": maximum_error,
        "rms": math.sqrt(squared_error / count),
        "relative_l2": math.sqrt(squared_error / max(squared_reference, tiny)),
        "normalized_linf": maximum_error / max(maximum_reference, tiny),
        "first_mean_removed": first_mean if demean else None,
        "second_mean_removed": second_mean if demean else None,
    }


def field_observables(
    q: np.ndarray,
    u: np.ndarray,
    p: np.ndarray,
    *,
    rho: float,
    chunk_planes: int = 8,
) -> dict[str, Any]:
    if q.shape[:-1] != u.shape[:-1] or q.shape[:-1] != p.shape:
        raise ValueError("Q, u, and p grids do not match")
    sums = {
        "q2": 0.0,
        "speed2": 0.0,
        "velocity": np.zeros(3, dtype=np.float64),
    }
    q_count = 0
    cell_count = 0
    p_mean = _array_mean(p, chunk_planes)
    p2 = 0.0
    for start in range(0, q.shape[0], chunk_planes):
        q_chunk = np.asarray(q[start : start + chunk_planes], dtype=np.float64)
        u_chunk = np.asarray(u[start : start + chunk_planes], dtype=np.float64)
        p_chunk = np.asarray(p[start : start + chunk_planes], dtype=np.float64)
        if not (
            np.isfinite(q_chunk).all()
            and np.isfinite(u_chunk).all()
            and np.isfinite(p_chunk).all()
        ):
            raise ValueError("observable inputs must contain only finite values")
        sums["q2"] += float(np.sum(q_chunk * q_chunk, dtype=np.float64))
        speed2 = np.sum(u_chunk * u_chunk, axis=-1)
        sums["speed2"] += float(np.sum(speed2, dtype=np.float64))
        sums["velocity"] += np.sum(u_chunk, axis=(0, 1, 2), dtype=np.float64)
        p2 += float(np.sum((p_chunk - p_mean) ** 2, dtype=np.float64))
        q_count += q_chunk.size
        cell_count += speed2.size
    mean_speed2 = sums["speed2"] / cell_count
    return {
        "q_rms": math.sqrt(sums["q2"] / q_count),
        "speed_rms": math.sqrt(mean_speed2),
        "kinetic_energy_density": 0.5 * rho * mean_speed2,
        "mean_velocity": (sums["velocity"] / cell_count).tolist(),
        "pressure_rms_demeaned": math.sqrt(p2 / cell_count),
        "pressure_mean_removed": p_mean,
    }


def observed_order(coarse_error: float, fine_error: float) -> float | None:
    if coarse_error < 0 or fine_error < 0:
        raise ValueError("errors must be non-negative")
    if coarse_error == 0.0 and fine_error == 0.0:
        return None
    if fine_error == 0.0:
        return None
    return math.log(coarse_error / fine_error, 2.0)


def _snapshot_path(run_dir: Path, prefix: str, step: int) -> Path:
    path = run_dir / f"{prefix}_{step}.npy"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular snapshot {path}")
    return path


def _load_snapshot(run_dir: Path, prefix: str, step: int) -> np.ndarray:
    return np.load(_snapshot_path(run_dir, prefix, step), mmap_mode="r", allow_pickle=False)


def analyze_runs(plan: dict[str, Any], output_root: Path) -> dict[str, Any]:
    entries = []
    for row in plan["rows"]:
        config = row["config"]
        run_dir = output_root / config["run_id"]
        validation = base.validate_run(run_dir, config, row["config_sha256"])
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        entries.append(
            {
                "dt": config["dt"],
                "config": config,
                "config_sha256": row["config_sha256"],
                "run_dir": run_dir,
                "validation": validation,
                "metadata": metadata,
            }
        )
    entries.sort(key=lambda item: item["dt"], reverse=True)

    raw_hashes = {
        item["metadata"].get("initial_condition", {}).get("raw_q_sha256")
        for item in entries
    }
    projected_hashes = {
        item["metadata"].get("initial_condition", {}).get("projected_q_sha256")
        for item in entries
    }
    initial_identity_passed = (
        len(raw_hashes) == 1
        and None not in raw_hashes
        and len(projected_hashes) == 1
        and None not in projected_hashes
    )

    observables: dict[str, dict[str, Any]] = {}
    for item in entries:
        dt = item["dt"]
        run_key = dt_token(dt)
        observables[run_key] = {}
        for time_value in COMMON_OUTPUT_TIMES:
            step = _integer_steps(time_value, dt, "common_output_time")
            q = _load_snapshot(item["run_dir"], "Q", step)
            u = _load_snapshot(item["run_dir"], "u", step)
            p = _load_snapshot(item["run_dir"], "p", step)
            observables[run_key][f"{time_value:g}"] = field_observables(
                q, u, p, rho=item["config"]["density"]
            )
            del q, u, p

    comparisons: dict[str, dict[str, Any]] = {}
    pair_names = ((0, 1, "dt0p01_vs_dt0p005"), (1, 2, "dt0p005_vs_dt0p0025"))
    for first_index, second_index, pair_name in pair_names:
        first = entries[first_index]
        second = entries[second_index]
        pair_result: dict[str, Any] = {
            "first_dt": first["dt"],
            "second_dt": second["dt"],
            "times": {},
        }
        for time_value in COMMON_OUTPUT_TIMES:
            first_step = _integer_steps(time_value, first["dt"], "common_output_time")
            second_step = _integer_steps(time_value, second["dt"], "common_output_time")
            time_result = {}
            for prefix in FIELD_PREFIXES:
                first_array = _load_snapshot(first["run_dir"], prefix, first_step)
                second_array = _load_snapshot(second["run_dir"], prefix, second_step)
                key = "p_demeaned" if prefix == "p" else prefix
                time_result[key] = array_error_metrics(
                    first_array,
                    second_array,
                    demean=prefix == "p",
                )
                del first_array, second_array
            pair_result["times"][f"{time_value:g}"] = time_result
        comparisons[pair_name] = pair_result

    orders: dict[str, Any] = {}
    coarse_mid = comparisons["dt0p01_vs_dt0p005"]["times"]
    mid_fine = comparisons["dt0p005_vs_dt0p0025"]["times"]
    for time_key in coarse_mid:
        orders[time_key] = {}
        for field in ("Q", "u", "p_demeaned"):
            orders[time_key][field] = {}
            for metric in ("rms", "relative_l2", "normalized_linf"):
                orders[time_key][field][metric] = observed_order(
                    coarse_mid[time_key][field][metric],
                    mid_fine[time_key][field][metric],
                )

    relative_orders = [
        orders[time_key][field]["relative_l2"]
        for time_key in orders
        for field in ("Q", "u")
        if orders[time_key][field]["relative_l2"] is not None
    ]
    median_order = float(np.median(relative_orders)) if relative_orders else None
    monotone_q_u = all(
        mid_fine[time_key][field]["relative_l2"]
        <= coarse_mid[time_key][field]["relative_l2"]
        for time_key in coarse_mid
        for field in ("Q", "u")
    )

    final_key = f"{FINAL_TIME:g}"
    final_mid_fine = mid_fine[final_key]
    acceptance = plan["acceptance"]
    relative_gate = all(
        final_mid_fine[field]["relative_l2"] <= threshold
        for field, threshold in acceptance[
            "dt_0p005_vs_0p0025_final_relative_l2"
        ].items()
    )
    linf_gate = all(
        final_mid_fine[field]["normalized_linf"] <= threshold
        for field, threshold in acceptance[
            "dt_0p005_vs_0p0025_final_normalized_linf"
        ].items()
    )
    energy_mid = observables[dt_token(0.005)][final_key]["kinetic_energy_density"]
    energy_fine = observables[dt_token(0.0025)][final_key]["kinetic_energy_density"]
    energy_relative_change = abs(energy_mid - energy_fine) / max(
        abs(energy_fine), np.finfo(np.float64).tiny
    )
    energy_gate = energy_relative_change <= acceptance[
        "kinetic_energy_relative_change"
    ]
    order_gate = (
        median_order is not None
        and median_order >= acceptance["minimum_median_order_q_u_relative_l2"]
    )
    dt_adequate = relative_gate and linf_gate and energy_gate
    eligible = (
        initial_identity_passed and monotone_q_u and order_gate and dt_adequate
    )

    return {
        "schema_version": 1,
        "analysis": "inertial_timestep_convergence",
        "validation": "passed",
        "plan_git_head": plan["git"]["head"],
        "output_root": str(output_root),
        "initial_condition_identity": {
            "passed": initial_identity_passed,
            "raw_q_sha256": sorted(value for value in raw_hashes if value is not None),
            "projected_q_sha256": sorted(
                value for value in projected_hashes if value is not None
            ),
        },
        "runs": [
            {
                "dt": item["dt"],
                "run_id": item["config"]["run_id"],
                "run_dir": str(item["run_dir"]),
                "config_sha256": item["config_sha256"],
                "metadata_sha256": item["validation"]["metadata_sha256"],
            }
            for item in entries
        ],
        "observables": observables,
        "comparisons": comparisons,
        "observed_orders": orders,
        "assessment": {
            "monotone_refinement_q_u": monotone_q_u,
            "median_observed_order_q_u_relative_l2": median_order,
            "first_order_gate_passed": order_gate,
            "dt_0p005_final_relative_l2_gate_passed": relative_gate,
            "dt_0p005_final_normalized_linf_gate_passed": linf_gate,
            "kinetic_energy_relative_change_dt0p005_vs_dt0p0025": (
                energy_relative_change
            ),
            "kinetic_energy_gate_passed": energy_gate,
            "dt_0p005_short_time_adequate": dt_adequate,
            "recommended_dt": 0.005 if eligible else 0.0025,
            "eligible_for_inertial_fig4_pilot": eligible,
            "scope_note": (
                "This T=1 deterministic trajectory test does not establish "
                "long-time statistical timestep convergence."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--output", type=Path, required=True)

    for action in ("command", "validate"):
        child = subparsers.add_parser(action)
        child.add_argument("--plan", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--index", type=int, required=True)
        if action == "command":
            child.add_argument("--python-bin", default=sys.executable)
            child.add_argument("--execute", action="store_true")
            child.add_argument("--confirm-direct-execution", action="store_true")
        else:
            child.add_argument("--report", type=Path, required=True)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--plan", type=Path, required=True)
    analyze_parser.add_argument("--output-root", type=Path, required=True)
    analyze_parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "plan":
        plan = build_plan()
        base.write_exclusive_json(args.output.resolve(), plan)
        print(
            json.dumps(
                {
                    "plan": str(args.output.resolve()),
                    "runs": len(plan["rows"]),
                    "dt_values": plan["dt_values"],
                    "git": plan["git"],
                },
                sort_keys=True,
            )
        )
        return 0

    require_clean = args.action == "command" and args.execute
    plan = load_plan(args.plan.resolve(), require_clean=require_clean)
    output_root = args.output_root.resolve()

    if args.action == "analyze":
        report = analyze_runs(plan, output_root)
        base.write_exclusive_json(args.report.resolve(), report)
        print(json.dumps(report["assessment"], indent=2, allow_nan=False))
        return 0

    config, config_sha256 = selected_row(plan, args.index)
    command, run_dir = base.command_for_config(
        config,
        config_sha256=config_sha256,
        output_root=output_root,
        python_bin=getattr(args, "python_bin", sys.executable),
    )
    if args.action == "command":
        print(
            json.dumps(
                {
                    "index": args.index,
                    "run_id": config["run_id"],
                    "config_sha256": config_sha256,
                    "run_dir": str(run_dir),
                    "command": command,
                    "shell_command": shlex.join(command),
                },
                indent=2,
            )
        )
        if not args.execute:
            return 0
        if not args.confirm_direct_execution:
            raise RuntimeError("execution requires --confirm-direct-execution")
        if run_dir.exists():
            raise FileExistsError(f"refusing existing run directory: {run_dir}")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode

    report = base.validate_run(run_dir, config, config_sha256)
    report["study"] = "inertial_timestep_convergence"
    report["dt"] = config["dt"]
    report["final_time"] = config["final_time"]
    base.write_exclusive_json(args.report.resolve(), report)
    print(json.dumps({"validation": "passed", "run_id": config["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
