#!/usr/bin/env python3
"""Plan, execute, and validate the matched V1 inertial Fig. 4 scan.

The 23-point matrix is copied from the archived 2026-09-04 V1 preview.  All
initial-condition, geometry, coefficient, resolution, time-window, precision,
and output choices are preserved.  The intentional model changes are limited
to the inertial Navier--Stokes momentum equation, dynamic velocity fields, and
evolution of the uniform tangential momentum modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SCRIPT = (
    PROJECT_ROOT
    / "benchmarks"
    / "shendruk_v1_dynamic_velocity"
    / "Plane_v1_dynamic_velocity.py"
)
ARCHIVED_REFERENCE_COMMIT = "d32008949a0024dbe5db9ce64ed5881665ecbb64"
Q_PATTERN = re.compile(r"^Q_([0-9]+)\.npy$")

ACTIVITIES = {
    10: (40, (5.0, 10.0, 15.0, 17.0, 18.0, 20.0, 22.0)),
    15: (60, (7.0, 12.0, 16.0, 17.0, 18.0, 20.0, 25.0, 33.0)),
    20: (80, (9.0, 15.0, 17.0, 18.0, 20.0, 25.0, 35.0, 44.7)),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def activity_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def git_text(*arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def git_identity() -> dict[str, Any]:
    status = git_text("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": git_text("rev-parse", "HEAD"),
        "branch": git_text("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def scientific_config(
    *,
    height: int,
    nz: int,
    activity_number: float,
    commit: str,
) -> dict[str, Any]:
    return {
        "run_id": (
            f"H{height}_A{activity_token(activity_number)}_seed24_dynamic_velocity"
        ),
        "model_script": str(MODEL_SCRIPT.relative_to(PROJECT_ROOT)),
        "pssolver_commit": commit,
        "archived_reference_commit": ARCHIVED_REFERENCE_COMMIT,
        "initialization_protocol": "v1",
        "height": height,
        "activity_number": activity_number,
        "lx": 100.0,
        "ly": 100.0,
        "nx": 320,
        "ny": 320,
        "nz": nz,
        "dt": 0.005,
        "steps": 40_000,
        "save_start_step": 20_000,
        "save_interval": 1_000,
        "diagnostic_interval": 100,
        "seed": 24,
        "parameterization": "paper-window",
        "coefficient_min": 0.01,
        "coefficient_max": 0.05,
        "ldg_a": 0.0,
        "ldg_b": -0.3,
        "ldg_c": 0.3,
        "gamma": 2.94,
        "flow_alignment": 0.3,
        "density": 1.0,
        "eta": 2.0 / 3.0,
        "fric": 0.0,
        "mean_flow_policy": "evolve",
        "beta": -1.0,
        "initial_s": 1.0 / 3.0,
        "num_defect_pairs": 6,
        "defect_min_separation": 10.0,
        "defect_core_radius": 1.5,
        "background_angle": 0.0,
        "twist_amplitude": 0.01,
        "twist_modes": [1, 2, 3],
        "q_boundary_conditions": ["periodic", "periodic", "neumann"],
        "tangential_velocity_boundary_conditions": [
            "periodic",
            "periodic",
            "neumann",
        ],
        "normal_velocity_boundary_conditions": [
            "periodic",
            "periodic",
            "dirichlet",
        ],
        "dealias_rule": "cubic_half",
        "dtype": "float64",
        "spectral_dtype": "complex128",
        "tf32": "off",
        "spectral_refresh": "disabled",
        "device": "cuda",
        "diagnostics": True,
        "save_hydrodynamics": False,
    }


def build_rows(commit: str) -> list[dict[str, Any]]:
    rows = []
    for height, (nz, activities) in ACTIVITIES.items():
        for activity in activities:
            config = scientific_config(
                height=height,
                nz=nz,
                activity_number=activity,
                commit=commit,
            )
            rows.append(
                {
                    "index": len(rows),
                    "config_sha256": canonical_sha256(config),
                    "config": config,
                }
            )
    if len(rows) != 23 or len({row["config"]["run_id"] for row in rows}) != 23:
        raise RuntimeError("the canonical Fig. 4 matrix must contain 23 runs")
    return rows


def build_plan() -> dict[str, Any]:
    git = git_identity()
    return {
        "schema_version": 1,
        "workflow": "shendruk_v1_dynamic_velocity_fig4_matched_scan",
        "claim_scope": (
            "matched repeat of the archived preliminary single-seed V1 scan; "
            "not by itself a strict or statistically converged reproduction"
        ),
        "archived_reference": {
            "commit": ARCHIVED_REFERENCE_COMMIT,
            "matrix": "2026-09-04 V1 Fig. 4 preview, 23 runs",
        },
        "intentional_changes": [
            "quasistatic Stokes replaced by inertial incompressible momentum",
            "velocity components are dynamic fields with their own L_hat",
            "uniform tangential mean momentum evolves instead of zero_mean removal",
            "initial velocity is zero",
        ],
        "git": git,
        "model_script": str(MODEL_SCRIPT.relative_to(PROJECT_ROOT)),
        "model_script_sha256": sha256_file(MODEL_SCRIPT),
        "workflow_script_sha256": sha256_file(Path(__file__).resolve()),
        "rows": build_rows(git["head"]),
    }


def write_exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_plan(path: Path, *, require_clean: bool) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported plan schema")
    if plan.get("workflow") != "shendruk_v1_dynamic_velocity_fig4_matched_scan":
        raise ValueError("unexpected workflow")
    current_git = git_identity()
    if current_git["head"] != plan["git"]["head"]:
        raise RuntimeError("current HEAD differs from the bound plan")
    if require_clean and current_git["dirty"]:
        raise RuntimeError("formal execution requires a clean worktree")
    if sha256_file(MODEL_SCRIPT) != plan["model_script_sha256"]:
        raise RuntimeError("model script differs from the bound plan")
    if sha256_file(Path(__file__).resolve()) != plan["workflow_script_sha256"]:
        raise RuntimeError("workflow script differs from the bound plan")
    expected_rows = build_rows(current_git["head"])
    if plan.get("rows") != expected_rows:
        raise RuntimeError("plan matrix differs from the canonical 23-run matrix")
    return plan


def selected_config(
    plan: dict[str, Any],
    *,
    index: int,
    preflight: bool,
) -> tuple[dict[str, Any], str]:
    if preflight:
        source = next(
            row
            for row in plan["rows"]
            if row["config"]["height"] == 20
            and row["config"]["activity_number"] == 18.0
        )
        config = dict(source["config"])
        config.update(
            {
                "run_id": "H20_A18_seed24_dynamic_velocity_preflight",
                "steps": 1,
                "save_start_step": 1,
                "save_interval": 1,
                "diagnostic_interval": 1,
                "save_hydrodynamics": True,
            }
        )
        return config, canonical_sha256(config)
    if not 0 <= index < len(plan["rows"]):
        raise IndexError(f"matrix index {index} is outside [0, 22]")
    row = plan["rows"][index]
    if row["config_sha256"] != canonical_sha256(row["config"]):
        raise RuntimeError("selected plan row has an invalid canonical SHA-256")
    return dict(row["config"]), row["config_sha256"]


def command_for_config(
    config: dict[str, Any],
    *,
    config_sha256: str,
    output_root: Path,
    python_bin: str,
) -> tuple[list[str], Path]:
    run_dir = output_root / config["run_id"]
    command = [
        python_bin,
        str(MODEL_SCRIPT),
        "--activity-number", format(config["activity_number"], ".17g"),
        "--output-dir", str(run_dir),
        "--height", format(config["height"], ".17g"),
        "--parameterization", config["parameterization"],
        "--coefficient-min", format(config["coefficient_min"], ".17g"),
        "--coefficient-max", format(config["coefficient_max"], ".17g"),
        "--lx", format(config["lx"], ".17g"),
        "--ly", format(config["ly"], ".17g"),
        "--nx", str(config["nx"]),
        "--ny", str(config["ny"]),
        "--nz", str(config["nz"]),
        "--dt", format(config["dt"], ".17g"),
        "--steps", str(config["steps"]),
        "--save-start-step", str(config["save_start_step"]),
        "--save-interval", str(config["save_interval"]),
        "--diagnostic-interval", str(config["diagnostic_interval"]),
        "--seed", str(config["seed"]),
        "--num-defect-pairs", str(config["num_defect_pairs"]),
        "--defect-min-separation", format(config["defect_min_separation"], ".17g"),
        "--defect-core-radius", format(config["defect_core_radius"], ".17g"),
        "--background-angle", format(config["background_angle"], ".17g"),
        "--twist-amplitude", format(config["twist_amplitude"], ".17g"),
        "--twist-modes", *map(str, config["twist_modes"]),
        "--initialization-protocol", config["initialization_protocol"],
        "--ldg-a", format(config["ldg_a"], ".17g"),
        "--ldg-b", format(config["ldg_b"], ".17g"),
        "--ldg-c", format(config["ldg_c"], ".17g"),
        "--gamma", format(config["gamma"], ".17g"),
        "--flow-alignment", format(config["flow_alignment"], ".17g"),
        "--density", format(config["density"], ".17g"),
        "--eta", format(config["eta"], ".17g"),
        "--fric", format(config["fric"], ".17g"),
        "--mean-flow-policy", config["mean_flow_policy"],
        "--dealias-rule", config["dealias_rule"],
        "--beta", format(config["beta"], ".17g"),
        "--initial-s", format(config["initial_s"], ".17g"),
        "--device", config["device"],
        "--dtype", config["dtype"],
        "--tf32", config["tf32"],
        "--disable-spectral-refresh",
        "--validation-config-sha256", config_sha256,
    ]
    if config["diagnostics"]:
        command.append("--diagnostics")
    if config["save_hydrodynamics"]:
        command.append("--save-hydrodynamics")
    return command, run_dir


def close(actual: Any, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-14):
        raise ValueError(f"{label}: expected {expected}, got {actual}")


def array_is_finite(array: np.ndarray) -> bool:
    planes_per_chunk = max(1, 64 * 1024**2 // array[0:1].nbytes)
    for start in range(0, array.shape[0], planes_per_chunk):
        if not np.isfinite(array[start : start + planes_per_chunk]).all():
            return False
    return True


def validate_run(
    run_dir: Path,
    config: dict[str, Any],
    config_sha256: str,
) -> dict[str, Any]:
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError(f"invalid run directory: {run_dir}")
    metadata_path = run_dir / "metadata.json"
    complete_path = run_dir / "COMPLETE"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise ValueError("missing regular metadata.json")
    if not complete_path.is_file() or complete_path.is_symlink():
        raise ValueError("missing regular COMPLETE marker")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError("run metadata is not complete")
    expected_script = str(MODEL_SCRIPT.relative_to(PROJECT_ROOT))
    if metadata.get("script") != expected_script:
        raise ValueError("run used the wrong model script")
    model = metadata.get("model", {})
    if model.get("variant") != "beris_edwards_complete_nematic_stress_navier_stokes":
        raise ValueError("run used the wrong model variant")
    if model.get("flow_dynamics", {}).get("regime") != (
        "incompressible_navier_stokes_beris_edwards"
    ):
        raise ValueError("run did not use inertial incompressible dynamics")
    if metadata.get("validation_config_sha256") != config_sha256:
        raise ValueError("metadata configuration SHA-256 mismatch")
    if metadata.get("completed_steps") != config["steps"]:
        raise ValueError("completed step count mismatch")
    if metadata.get("shape") != [config["nx"], config["ny"], config["nz"]]:
        raise ValueError("grid shape mismatch")
    close(metadata["dt"], config["dt"], "dt")
    close(metadata["density"], config["density"], "density")
    close(metadata["friction"], config["fric"], "friction")
    if metadata.get("mean_flow_policy") != "evolve":
        raise ValueError("uniform tangential momentum was not evolved")
    if metadata.get("initialization_protocol") != "v1":
        raise ValueError("run did not use V1 initialization")
    if metadata.get("dealias_rule") != "cubic_half":
        raise ValueError("dealiasing mismatch")
    if metadata.get("dtype") != "float64" or metadata.get("tf32") != "off":
        raise ValueError("precision mismatch")
    if metadata["numerics"]["spectral_refresh"]["mode"] != "disabled":
        raise ValueError("spectral refresh must be disabled")

    expected_steps = list(
        range(
            config["save_start_step"],
            config["steps"] + 1,
            config["save_interval"],
        )
    )
    if config["steps"] not in expected_steps:
        expected_steps.append(config["steps"])
    q_by_step = {}
    for path in run_dir.glob("Q_*.npy"):
        match = Q_PATTERN.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected Q snapshot name: {path.name}")
        q_by_step[int(match.group(1))] = path
    if sorted(q_by_step) != expected_steps:
        raise ValueError("Q snapshot steps do not match the bound configuration")
    inventory = []
    for step in expected_steps:
        path = q_by_step[step]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        expected_shape = (config["nx"], config["ny"], config["nz"], 5)
        if array.shape != expected_shape or array.dtype != np.dtype("float64"):
            raise ValueError(f"invalid Q snapshot layout: {path}")
        if not array_is_finite(array):
            raise ValueError(f"nonfinite Q snapshot: {path}")
        inventory.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "shape": list(array.shape),
                "dtype": array.dtype.name,
            }
        )
        del array

    hydrodynamic_inventory = []
    for prefix, components in (("u", 3), ("p", None)):
        paths = sorted(run_dir.glob(f"{prefix}_*.npy"))
        if config["save_hydrodynamics"] and len(paths) != 1:
            raise ValueError(f"preflight requires one {prefix} snapshot")
        if not config["save_hydrodynamics"] and paths:
            raise ValueError(f"formal Q-only run unexpectedly saved {prefix}")
        for path in paths:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            expected_shape = (
                (config["nx"], config["ny"], config["nz"], components)
                if components is not None
                else (config["nx"], config["ny"], config["nz"])
            )
            if array.shape != expected_shape or array.dtype != np.dtype("float64"):
                raise ValueError(f"invalid {prefix} snapshot layout: {path}")
            if not array_is_finite(array):
                raise ValueError(f"nonfinite {prefix} snapshot: {path}")
            hydrodynamic_inventory.append(
                {"path": path.name, "size_bytes": path.stat().st_size}
            )
            del array

    diagnostics = np.load(run_dir / "diagnostics.npy", allow_pickle=False)
    required_diagnostics = {
        "div_max",
        "div_rms",
        "div_rel",
        "schur_rel_residual",
        "time_discrete_momentum_residual_max",
        "time_discrete_momentum_residual_rms",
    }
    if diagnostics.dtype.names is None or not required_diagnostics.issubset(
        diagnostics.dtype.names
    ):
        raise ValueError("diagnostics use an unexpected schema")
    final = diagnostics[-1]
    for name in required_diagnostics:
        if not np.isfinite(final[name]):
            raise ValueError(f"final diagnostic {name} is nonfinite")
    if final["div_max"] > 1.0e-10:
        raise ValueError("final incompressibility residual exceeds tolerance")
    if final["schur_rel_residual"] > 1.0e-10:
        raise ValueError("final Schur residual exceeds tolerance")
    if final["time_discrete_momentum_residual_max"] > 1.0e-10:
        raise ValueError("final momentum residual exceeds tolerance")

    return {
        "schema_version": 1,
        "validation": "passed",
        "run_id": config["run_id"],
        "run_dir": str(run_dir),
        "config_sha256": config_sha256,
        "metadata_sha256": sha256_file(metadata_path),
        "complete_sha256": sha256_file(complete_path),
        "q_inventory": inventory,
        "hydrodynamic_inventory": hydrodynamic_inventory,
        "final_diagnostics": {
            name: float(final[name]) for name in sorted(required_diagnostics)
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
        child.add_argument("--index", type=int, default=0)
        child.add_argument("--preflight", action="store_true")
        if action == "command":
            child.add_argument("--python-bin", default=sys.executable)
            child.add_argument("--execute", action="store_true")
            child.add_argument("--confirm-direct-execution", action="store_true")
        else:
            child.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "plan":
        plan = build_plan()
        write_exclusive_json(args.output.resolve(), plan)
        print(
            json.dumps(
                {
                    "plan": str(args.output.resolve()),
                    "runs": len(plan["rows"]),
                    "git": plan["git"],
                },
                sort_keys=True,
            )
        )
        return 0

    plan = load_plan(args.plan.resolve(), require_clean=args.action == "command" and args.execute)
    config, config_sha256 = selected_config(
        plan,
        index=args.index,
        preflight=args.preflight,
    )
    command, run_dir = command_for_config(
        config,
        config_sha256=config_sha256,
        output_root=args.output_root.resolve(),
        python_bin=getattr(args, "python_bin", sys.executable),
    )
    if args.action == "command":
        payload = {
            "index": args.index,
            "preflight": args.preflight,
            "run_id": config["run_id"],
            "config_sha256": config_sha256,
            "run_dir": str(run_dir),
            "command": command,
            "shell_command": shlex.join(command),
        }
        print(json.dumps(payload, indent=2))
        if not args.execute:
            return 0
        if not args.confirm_direct_execution:
            raise RuntimeError("execution requires --confirm-direct-execution")
        if run_dir.exists():
            raise FileExistsError(f"refusing existing run directory: {run_dir}")
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode

    report = validate_run(run_dir, config, config_sha256)
    write_exclusive_json(args.report.resolve(), report)
    print(json.dumps({"validation": "passed", "run_id": config["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
