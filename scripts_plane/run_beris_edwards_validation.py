#!/usr/bin/env python3
"""Plan or execute staged validation for Plane_beris_edwards_stokes.py.

The default action is read-only: a machine-readable plan is printed to stdout.
Runs are started only with ``--execute``.  The short numerical matrix isolates
time step, grid, and dealiasing effects; the long pilot samples activity and
initial-condition seeds before any full Fig. 4 scan is attempted.

Execution is deliberately conservative: it is synchronous on the current
node, requires an explicit confirmation flag, reuses only completed runs whose
metadata exactly match the requested configuration and implementation hashes,
and runs the available analyses after all simulations finish.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SCRIPT = PROJECT_ROOT / "Plane_beris_edwards_stokes.py"
CONVERGENCE_ANALYZER = PROJECT_ROOT / "scripts_plane" / "analyze_fig4_convergence.py"
DEFECT_CORE_ANALYZER = (
    PROJECT_ROOT / "scripts_plane" / "analyze_fig4_defect_core_convergence.py"
)
DEFECT_CORE_OUTPUTS = (
    "defect_core_metrics.csv",
    "defect_core_summary.json",
    "initial_core_profiles.npz",
    "projected_initial_core_profiles.npz",
    "final_core_profiles.npz",
    "initial_r50_convergence.png",
    "projected_initial_profiles_by_resolution.png",
    "final_core_profiles_by_charge.png",
    "core_metric_resolution_changes.png",
    "core_zoom_positive.png",
    "core_zoom_negative.png",
    "README.md",
)
IMPLEMENTATION_SOURCE_FILES = tuple(
    PROJECT_ROOT / relative_path
    for relative_path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/Field.py",
        "pssolver/PDEmodel.py",
        "pssolver/integrator.py",
        "pssolver/transforms.py",
        "pssolver/__init__.py",
        "pssolver/models/active_nematics/__init__.py",
        "pssolver/models/active_nematics/fields.py",
        "pssolver/models/active_nematics/q_tensor.py",
        "pssolver/models/active_nematics/beris_edwards.py",
        "pssolver/models/active_nematics/initial_conditions.py",
    )
)

GRID_SHAPES = {
    256: (256, 256, 64),
    320: (320, 320, 80),
    512: (512, 512, 128),
}
NUMERICAL_STAGES = ("preflight", "time", "space", "dealias")
ALL_STAGES = (*NUMERICAL_STAGES, "long_seed")


@dataclass(frozen=True)
class RunSpec:
    """One immutable simulation configuration with one or more study roles."""

    run_id: str
    purposes: tuple[str, ...]
    activity_number: float
    seed: int
    nx: int
    ny: int
    nz: int
    dt: float
    steps: int
    save_start_step: int
    save_interval: int
    diagnostic_interval: int
    dealias_rule: str

    @property
    def final_time(self) -> float:
        return self.dt * self.steps

    def scientific_config(self) -> dict[str, Any]:
        return {
            "activity_number": self.activity_number,
            "seed": self.seed,
            "shape": [self.nx, self.ny, self.nz],
            "lengths": [100.0, 100.0, 20.0],
            "dt": self.dt,
            "steps": self.steps,
            "final_time": self.final_time,
            "save_start_step": self.save_start_step,
            "save_interval": self.save_interval,
            "diagnostic_interval": self.diagnostic_interval,
            "dealias_rule": self.dealias_rule,
            "dtype": "float64",
            "tf32": "off",
            "spectral_refresh": "disabled",
            "zero_mode_policy": "zero_mean",
            "parameterization": "paper-window",
            "model_parameters": {
                "ldg_A": 0.0,
                "ldg_B": -0.3,
                "ldg_C": 0.3,
                "gamma": 2.94,
                "flow_alignment_lambda": 0.3,
                "eta": 2.0 / 3.0,
                "active_stress_beta": -1.0,
                "S_initial": 1.0 / 3.0,
                "coefficient_min": 0.01,
                "coefficient_max": 0.05,
            },
            "outputs": {
                "diagnostics": True,
                "save_hydrodynamics": True,
            },
            "initial_condition": {
                "name": "extruded_analytic_periodic_defect_gas_2d",
                "num_defect_pairs": 6,
                "minimum_separation": 10.0,
                "core_radius": 1.5,
                "background_angle": 0.0,
                "twist_amplitude": 0.01,
                "twist_modes": [1, 2, 3],
            },
        }

    @property
    def saved_frame_count(self) -> int:
        """Number of distinct Q/u/p snapshots produced by the model script."""
        first = (
            (self.save_start_step + self.save_interval - 1)
            // self.save_interval
            * self.save_interval
        )
        in_loop = 0
        if first < self.steps:
            in_loop = (self.steps - 1 - first) // self.save_interval + 1
        # The script always writes the final state after the integration loop.
        return in_loop + 1

    def storage_estimate(self) -> dict[str, Any]:
        """Estimate primary Q/u/p bytes; logs and small sidecars are excluded."""
        grid_points = self.nx * self.ny * self.nz
        components = 5 + 3 + 1
        bytes_per_frame = grid_points * components * 8
        total_bytes = self.saved_frame_count * bytes_per_frame
        return {
            "saved_frames": self.saved_frame_count,
            "bytes_per_frame": bytes_per_frame,
            "primary_snapshot_bytes": total_bytes,
            "primary_snapshot_gib": total_bytes / 1024**3,
            "includes": "five Q, three velocity, and one pressure float64 arrays",
            "excludes": "logs, diagnostics, metadata, and initialization sidecars",
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _steps_for_time(final_time: float, dt: float) -> int:
    ratio = final_time / dt
    steps = int(round(ratio))
    if steps <= 0 or not math.isclose(ratio, steps, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"final_time={final_time} is not an integer multiple of dt={dt}")
    return steps


def _make_run(
    purpose: str,
    *,
    resolution: int,
    dt: float,
    final_time: float,
    dealias_rule: str = "cubic_half",
    activity_number: float = 18.0,
    seed: int = 24,
    preflight: bool = False,
) -> RunSpec:
    nx, ny, nz = GRID_SHAPES[resolution]
    steps = 1 if preflight else _steps_for_time(final_time, dt)
    # A one-step memory/smoke preflight needs only the mandatory final snapshot.
    # Starting at ``steps`` prevents a redundant full R512 step-0 snapshot.
    if preflight:
        save_start_step = steps
    elif purpose.startswith("long_seed"):
        save_start_step = steps // 2
    else:
        save_start_step = 0
    save_interval = steps if not purpose.startswith("long_seed") else 500
    diagnostic_interval = 1 if preflight else min(100, steps)
    config = {
        "activity_number": activity_number,
        "seed": seed,
        "shape": [nx, ny, nz],
        "dt": dt,
        "steps": steps,
        "dealias_rule": dealias_rule,
        "save_start_step": save_start_step,
        "save_interval": save_interval,
    }
    digest = _canonical_sha256(config)[:12]
    dt_label = format(dt, ".8g").replace(".", "p")
    activity_label = format(activity_number, ".8g").replace(".", "p")
    run_id = (
        f"A{activity_label}_R{resolution}_dt{dt_label}_"
        f"{dealias_rule}_seed{seed}_{digest}"
    )
    return RunSpec(
        run_id=run_id,
        purposes=(purpose,),
        activity_number=activity_number,
        seed=seed,
        nx=nx,
        ny=ny,
        nz=nz,
        dt=dt,
        steps=steps,
        save_start_step=save_start_step,
        save_interval=save_interval,
        diagnostic_interval=diagnostic_interval,
        dealias_rule=dealias_rule,
    )


def _deduplicate(runs: Iterable[RunSpec]) -> list[RunSpec]:
    by_id: dict[str, RunSpec] = {}
    for run in runs:
        previous = by_id.get(run.run_id)
        if previous is None:
            by_id[run.run_id] = run
            continue
        if previous.scientific_config() != run.scientific_config():
            raise RuntimeError(f"run-id collision for {run.run_id}")
        purposes = tuple(dict.fromkeys((*previous.purposes, *run.purposes)))
        by_id[run.run_id] = replace(previous, purposes=purposes)
    return list(by_id.values())


def build_runs(
    stages: Iterable[str],
    *,
    include_r512_time_control: bool = False,
    long_final_time: float = 100.0,
) -> list[RunSpec]:
    """Build the requested validation matrix without touching the filesystem."""
    requested = tuple(dict.fromkeys(stages))
    unknown = set(requested) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"unknown validation stages: {sorted(unknown)}")
    runs: list[RunSpec] = []

    if "preflight" in requested or set(requested).intersection(
        ("space", "dealias", "long_seed")
    ):
        runs.append(
            _make_run(
                "preflight",
                resolution=512,
                dt=0.005,
                final_time=0.005,
                preflight=True,
            )
        )
    if "time" in requested:
        for dt in (0.01, 0.005, 0.0025):
            runs.append(_make_run("time", resolution=256, dt=dt, final_time=1.0))
    if "space" in requested:
        for resolution in (256, 320, 512):
            runs.append(
                _make_run("space", resolution=resolution, dt=0.005, final_time=1.0)
            )
    if "dealias" in requested:
        for resolution in (320, 512):
            for dealias_rule in ("cubic_half", "two_thirds"):
                runs.append(
                    _make_run(
                        "dealias",
                        resolution=resolution,
                        dt=0.005,
                        final_time=1.0,
                        dealias_rule=dealias_rule,
                    )
                )
    if include_r512_time_control:
        runs.append(
            _make_run(
                "r512_time_baseline",
                resolution=512,
                dt=0.005,
                final_time=1.0,
            )
        )
        runs.append(
            _make_run(
                "r512_time_control",
                resolution=512,
                dt=0.0025,
                final_time=1.0,
            )
        )
    if "long_seed" in requested:
        for activity_number in (15.0, 18.0, 25.0):
            for seed in (24, 41, 73):
                runs.append(
                    _make_run(
                        "long_seed",
                        resolution=320,
                        dt=0.005,
                        final_time=long_final_time,
                        activity_number=activity_number,
                        seed=seed,
                    )
                )
    return _deduplicate(runs)


def command_for_run(
    run: RunSpec,
    *,
    python_bin: str,
    output_root: Path,
    device: str,
    validation_config_sha256: str | None = None,
) -> list[str]:
    output_dir = output_root / run.run_id
    command = [
        python_bin,
        str(MODEL_SCRIPT),
        "--activity-number",
        format(run.activity_number, ".17g"),
        "--output-dir",
        str(output_dir),
        "--height",
        "20",
        "--parameterization",
        "paper-window",
        "--coefficient-min",
        "0.01",
        "--coefficient-max",
        "0.05",
        "--lx",
        "100",
        "--ly",
        "100",
        "--nx",
        str(run.nx),
        "--ny",
        str(run.ny),
        "--nz",
        str(run.nz),
        "--dt",
        format(run.dt, ".17g"),
        "--steps",
        str(run.steps),
        "--save-start-step",
        str(run.save_start_step),
        "--save-interval",
        str(run.save_interval),
        "--diagnostic-interval",
        str(run.diagnostic_interval),
        "--seed",
        str(run.seed),
        "--num-defect-pairs",
        "6",
        "--defect-min-separation",
        "10",
        "--defect-core-radius",
        "1.5",
        "--background-angle",
        "0",
        "--twist-amplitude",
        "0.01",
        "--twist-modes",
        "1",
        "2",
        "3",
        "--ldg-a",
        "0",
        "--ldg-b",
        "-0.3",
        "--ldg-c",
        "0.3",
        "--gamma",
        "2.94",
        "--flow-alignment",
        "0.3",
        "--eta",
        "0.6666666666666666",
        "--zero-mode-policy",
        "zero_mean",
        "--dealias-rule",
        run.dealias_rule,
        "--beta",
        "-1",
        "--initial-s",
        "0.3333333333333333",
        "--device",
        device,
        "--dtype",
        "float64",
        "--tf32",
        "off",
        "--disable-spectral-refresh",
        "--diagnostics",
        "--save-hydrodynamics",
    ]
    if validation_config_sha256 is not None:
        command.extend(
            ["--validation-config-sha256", validation_config_sha256]
        )
    return command


def _runs_with_purpose(runs: Iterable[RunSpec], purpose: str) -> list[RunSpec]:
    return [run for run in runs if purpose in run.purposes]


def _global_analysis_outputs(mode: str) -> tuple[str, str, str]:
    prefix = (
        "fig4_dealias_sensitivity"
        if mode == "dealias"
        else f"fig4_{mode}_convergence"
    )
    return (f"{prefix}_runs.csv", f"{prefix}_pairs.csv", f"{prefix}.json")


def analysis_commands(
    runs: list[RunSpec],
    *,
    python_bin: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    time_runs = sorted(_runs_with_purpose(runs, "time"), key=lambda run: run.dt)
    if len(time_runs) >= 3:
        output_dir = output_root / "analysis_time"
        commands.append(
            {
                "name": "time_convergence",
                "analyzer": str(CONVERGENCE_ANALYZER),
                "input_run_ids": [run.run_id for run in time_runs],
                "dependency_files": [],
                "required_outputs": list(_global_analysis_outputs("time")),
                "command": [
                    python_bin,
                    str(CONVERGENCE_ANALYZER),
                    "--mode",
                    "time",
                    "--output-dir",
                    str(output_dir),
                    "--order-dt",
                    "0.01",
                    "0.005",
                    "0.0025",
                    *[str(output_root / run.run_id) for run in time_runs],
                ],
            }
        )

    r512_baselines = _runs_with_purpose(runs, "r512_time_baseline")
    r512_controls = _runs_with_purpose(runs, "r512_time_control")
    r512_time_runs: list[RunSpec] = []
    if len(r512_baselines) == 1 and len(r512_controls) == 1:
        r512_time_runs = sorted(
            (*r512_baselines, *r512_controls),
            key=lambda run: run.dt,
        )
        output_dir = output_root / "analysis_time_R512"
        commands.append(
            {
                "name": "r512_time_control",
                "analyzer": str(CONVERGENCE_ANALYZER),
                "input_run_ids": [run.run_id for run in r512_time_runs],
                "dependency_files": [],
                "required_outputs": list(_global_analysis_outputs("time")),
                "command": [
                    python_bin,
                    str(CONVERGENCE_ANALYZER),
                    "--mode",
                    "time",
                    "--output-dir",
                    str(output_dir),
                    *[str(output_root / run.run_id) for run in r512_time_runs],
                ],
                "limitation": "Two-level sensitivity check; no order estimate.",
            }
        )

    space_runs = sorted(
        _runs_with_purpose(runs, "space"),
        key=lambda run: run.nx,
    )
    space_core_summary: Path | None = None
    if len(space_runs) >= 3:
        global_output_dir = output_root / "analysis_space"
        commands.append(
            {
                "name": "space_global_scalars",
                "analyzer": str(CONVERGENCE_ANALYZER),
                "input_run_ids": [run.run_id for run in space_runs],
                "dependency_files": [],
                "required_outputs": list(_global_analysis_outputs("space")),
                "command": [
                    python_bin,
                    str(CONVERGENCE_ANALYZER),
                    "--mode",
                    "space",
                    "--output-dir",
                    str(global_output_dir),
                    *[str(output_root / run.run_id) for run in space_runs],
                ],
                "limitation": (
                    "Global scalar convergence; complemented by the independent "
                    "defect-core space analysis below."
                ),
            }
        )
        core_output_dir = output_root / "analysis_defect_core_space"
        space_core_summary = core_output_dir / "defect_core_summary.json"
        commands.append(
            {
                "name": "space_defect_core",
                "analyzer": str(DEFECT_CORE_ANALYZER),
                "input_run_ids": [run.run_id for run in space_runs],
                "dependency_files": [],
                "required_outputs": list(DEFECT_CORE_OUTPUTS),
                "command": [
                    python_bin,
                    str(DEFECT_CORE_ANALYZER),
                    "--mode",
                    "space",
                    "--output-dir",
                    str(core_output_dir),
                    *[str(output_root / run.run_id) for run in space_runs],
                ],
                "limitation": (
                    "T=1 defect-core space gate; not a long-time statistical "
                    "convergence claim."
                ),
            }
        )

    if r512_time_runs and space_core_summary is not None:
        core_time_output_dir = output_root / "analysis_defect_core_time_R512"
        commands.append(
            {
                "name": "time_defect_core_R512",
                "analyzer": str(DEFECT_CORE_ANALYZER),
                "input_run_ids": [run.run_id for run in r512_time_runs],
                "dependency_files": [str(space_core_summary)],
                "required_outputs": list(DEFECT_CORE_OUTPUTS),
                "command": [
                    python_bin,
                    str(DEFECT_CORE_ANALYZER),
                    "--mode",
                    "time",
                    "--output-dir",
                    str(core_time_output_dir),
                    "--space-core-summary",
                    str(space_core_summary),
                    *[str(output_root / run.run_id) for run in r512_time_runs],
                ],
                "limitation": (
                    "Two-level R512 core sensitivity check constrained by the "
                    "space-core error floor; no time-order estimate."
                ),
            }
        )

    dealias_runs = _runs_with_purpose(runs, "dealias")
    for resolution in sorted({run.nx for run in dealias_runs}):
        candidates = [run for run in dealias_runs if run.nx == resolution]
        by_rule = {run.dealias_rule: run for run in candidates}
        if not {"cubic_half", "two_thirds"}.issubset(by_rule):
            continue
        comparison_runs = [
            by_rule["cubic_half"],
            by_rule["two_thirds"],
        ]
        output_dir = output_root / f"analysis_dealias_R{resolution}"
        commands.append(
            {
                "name": f"dealias_sensitivity_R{resolution}",
                "analyzer": str(CONVERGENCE_ANALYZER),
                "input_run_ids": [run.run_id for run in comparison_runs],
                "dependency_files": [],
                "required_outputs": list(_global_analysis_outputs("dealias")),
                "command": [
                    python_bin,
                    str(CONVERGENCE_ANALYZER),
                    "--mode",
                    "dealias",
                    "--output-dir",
                    str(output_dir),
                    *[
                        str(output_root / run.run_id)
                        for run in comparison_runs
                    ],
                ],
                "limitation": "Sensitivity comparison, not a convergence order.",
            }
        )
    return commands


def _git_provenance() -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        result = subprocess.run(
            ("git", *arguments),
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "head": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": (
            None if status is None else hashlib.sha256(status.encode()).hexdigest()
        ),
    }


def build_plan(
    runs: list[RunSpec],
    *,
    python_bin: str,
    output_root: Path,
    device: str,
    expected_gpu_name: str | None = None,
) -> dict[str, Any]:
    implementation_files = {
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in IMPLEMENTATION_SOURCE_FILES
    }
    validation_tool_files = {
        str(path.relative_to(PROJECT_ROOT)): _sha256_file(path)
        for path in (
            Path(__file__).resolve(),
            CONVERGENCE_ANALYZER,
            DEFECT_CORE_ANALYZER,
        )
    }
    run_rows = []
    for run in runs:
        config = run.scientific_config()
        config["device"] = device
        config["expected_gpu_name"] = expected_gpu_name
        config_sha256 = _canonical_sha256(config)
        command = command_for_run(
            run,
            python_bin=python_bin,
            output_root=output_root,
            device=device,
            validation_config_sha256=config_sha256,
        )
        run_rows.append(
            {
                "run_id": run.run_id,
                "purposes": list(run.purposes),
                "config": config,
                "config_sha256": config_sha256,
                "output_dir": str(output_root / run.run_id),
                "command": command,
                "shell_command": shlex.join(command),
                "storage_estimate": run.storage_estimate(),
            }
        )
    total_primary_snapshot_bytes = sum(
        row["storage_estimate"]["primary_snapshot_bytes"] for row in run_rows
    )
    analyses = analysis_commands(
        runs,
        python_bin=python_bin,
        output_root=output_root,
    )
    config_sha256_by_run_id = {
        row["run_id"]: row["config_sha256"] for row in run_rows
    }
    for analysis in analyses:
        analysis["input_config_sha256"] = {
            run_id: config_sha256_by_run_id[run_id]
            for run_id in analysis["input_run_ids"]
        }
    plan = {
        "schema_version": 1,
        "validation": "beris_edwards_stokes_issue6",
        "model_script": MODEL_SCRIPT.name,
        "output_root": str(output_root),
        "git": _git_provenance(),
        "implementation_sha256": implementation_files,
        "validation_tools_sha256": validation_tool_files,
        "fixed_numerics": {
            "dtype": "float64",
            "tf32": "off",
            "spectral_refresh": "disabled",
            "zero_mode_policy": "zero_mean",
        },
        "storage_estimate": {
            "primary_snapshot_bytes": total_primary_snapshot_bytes,
            "primary_snapshot_gib": total_primary_snapshot_bytes / 1024**3,
            "scope": "Q/u/p arrays only; filesystem overhead and sidecars excluded",
        },
        "execution_safety": {
            "execution_model": "synchronous_on_current_node_not_a_scheduler",
            "requires_confirm_direct_execution": True,
            "large_output_threshold_gib": 50.0,
            "long_seed_requires_allow_large_output": True,
        },
        "issue_6_component_status": {
            "time_step": "implemented analysis; scientific runs pending",
            "space_global": "implemented analysis; scientific runs pending",
            "dealias": "implemented sensitivity analysis; scientific runs pending",
            "defect_core": (
                "implemented space analysis and optional R512 time sensitivity; "
                "scientific runs pending"
            ),
            "seed_activity_pilot": "plan only; not initial-condition convergence",
        },
        "known_scope": {
            "issue_1": (
                "deferred by modeling choice: first attempt the quasistatic "
                "Stokes reproduction; revisit inertia only if results disagree"
            ),
            "issue_6": (
                "validate pseudospectral/semi-implicit results through staged "
                "time, grid, dealiasing, and seed studies"
            ),
            "issue_7": (
                "independent manufactured-solution and energy tests remain open"
            ),
        },
        "interpretation_limits": [
            "T=1 is a short numerical gate, not long-time Fig. 4 convergence.",
            "Different seeds must be compared statistically, not pointwise.",
            "The seed pilot does not make the initializer paper-identical.",
            (
                "Varying seed alone is not initial-condition convergence; core radius, "
                "twist, and initializer-family sensitivity remain separate."
            ),
            "dealias-rule=none is excluded until its highest-DST-mode test exists.",
            "No automatic convergence claim is made by this plan.",
        ],
        "advisory_gates": {
            "divergence_and_schur_relative_residual_max": 1e-10,
            "time_observed_order_expected": [0.8, 1.2],
            "defect_core_adjacent_median_relative_change_max": 0.05,
            "defect_core_individual_relative_change_max": 0.10,
        },
        "runs": run_rows,
        "analysis_commands": analyses,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--expected-gpu-name",
        default=None,
        help="Required GPU-name substring when a CUDA preflight is executed.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        choices=ALL_STAGES,
        dest="stages",
        help=(
            "Stage to include; repeat as needed. Planning defaults to all four "
            "short numerical stages. --execute requires at least one explicit stage."
        ),
    )
    parser.add_argument("--include-r512-time-control", action="store_true")
    parser.add_argument("--long-final-time", type=float, default=100.0)
    parser.add_argument("--write-plan", type=Path, default=None)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run synchronously on the current node after explicit confirmation.",
    )
    parser.add_argument("--confirm-direct-execution", action="store_true")
    parser.add_argument(
        "--allow-large-output",
        action="store_true",
        help="Required for a plan whose primary snapshot estimate exceeds 50 GiB.",
    )
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    if args.long_final_time <= 0 or not math.isfinite(args.long_final_time):
        parser.error("--long-final-time must be positive and finite")
    if args.execute and not args.stages:
        parser.error("--execute requires at least one explicit --stage")
    if args.execute and not args.confirm_direct_execution:
        parser.error(
            "--execute runs synchronously on the current node and requires "
            "--confirm-direct-execution"
        )
    if args.expected_gpu_name is not None:
        args.expected_gpu_name = args.expected_gpu_name.strip()
        if not args.expected_gpu_name:
            parser.error("--expected-gpu-name must not be empty")
    cuda_preflight_requested = (
        args.execute
        and args.device.split(":", 1)[0] in {"cuda", "auto"}
        and bool(
            set(args.stages or ()).intersection(
                {"preflight", "space", "dealias", "long_seed"}
            )
        )
    )
    if cuda_preflight_requested and args.expected_gpu_name is None:
        parser.error("CUDA preflight execution requires --expected-gpu-name")
    if not args.stages:
        args.stages = list(NUMERICAL_STAGES)
    return args


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return False
        return math.isclose(
            float(actual),
            float(expected),
            rel_tol=1.0e-12,
            abs_tol=1.0e-14,
        )
    return actual == expected


def _expected_metadata_fields(run_row: dict[str, Any]) -> dict[tuple[str, ...], Any]:
    config = run_row["config"]
    model = config["model_parameters"]
    initial = config["initial_condition"]
    activity_ratio = (config["activity_number"] / config["lengths"][2]) ** 2
    if activity_ratio <= 1.0:
        zeta = model["coefficient_min"]
        frank_k = zeta / activity_ratio
    else:
        frank_k = model["coefficient_min"]
        zeta = frank_k * activity_ratio
    q_equilibrium_amplitude = 0.5
    ldg_l1 = frank_k / (2.0 * q_equilibrium_amplitude**2)
    return {
        ("schema_version",): 1,
        ("script",): MODEL_SCRIPT.name,
        ("validation_config_sha256",): run_row["config_sha256"],
        ("status",): "complete",
        ("runtime_environment", "device_type"): (
            "cuda" if config["device"].split(":", 1)[0] in {"cuda", "auto"}
            else "cpu"
        ),
        ("completed_steps",): config["steps"],
        ("solver", "shape"): config["shape"],
        ("solver", "lengths"): config["lengths"],
        ("solver", "dt"): config["dt"],
        ("solver", "steps"): config["steps"],
        ("solver", "save_interval"): config["save_interval"],
        ("solver", "real_dtype"): config["dtype"],
        ("model", "name"): "active_nematics",
        ("model", "variant"): "beris_edwards_complete_nematic_stress_stokes",
        ("model", "parameters", "activity_number"): config["activity_number"],
        ("model", "parameters", "zeta"): zeta,
        ("model", "parameters", "frank_K"): frank_k,
        ("model", "parameters", "ldg_A"): model["ldg_A"],
        ("model", "parameters", "ldg_B"): model["ldg_B"],
        ("model", "parameters", "ldg_C"): model["ldg_C"],
        ("model", "parameters", "ldg_L1"): ldg_l1,
        ("model", "parameters", "rotational_viscosity_gamma"): model["gamma"],
        ("model", "parameters", "flow_alignment_lambda"): model[
            "flow_alignment_lambda"
        ],
        ("model", "parameters", "alpha"): zeta,
        ("model", "parameters", "beta"): model["active_stress_beta"],
        ("model", "parameters", "S_initial"): model["S_initial"],
        ("model", "parameters", "fric"): 0.0,
        ("model", "parameters", "eta"): model["eta"],
        ("numerics", "dealiasing", "rule"): config["dealias_rule"],
        ("numerics", "velocity_zero_mode"): config["zero_mode_policy"],
        ("numerics", "precision", "real_dtype"): config["dtype"],
        ("numerics", "precision", "tf32_requested"): config["tf32"],
        ("numerics", "precision", "tf32_effective"): False,
        ("numerics", "spectral_refresh", "mode"): config["spectral_refresh"],
        ("numerics", "spectral_refresh", "actual_count"): 0,
        ("activity_number",): config["activity_number"],
        ("domain",): config["lengths"],
        ("shape",): config["shape"],
        ("dt",): config["dt"],
        ("steps",): config["steps"],
        ("save_start_step",): config["save_start_step"],
        ("save_interval",): config["save_interval"],
        ("diagnostic_interval",): config["diagnostic_interval"],
        ("seed",): config["seed"],
        ("device",): config["device"],
        ("dtype",): config["dtype"],
        ("tf32",): config["tf32"],
        ("zero_mode_policy",): config["zero_mode_policy"],
        ("dealias_rule",): config["dealias_rule"],
        ("parameterization",): config["parameterization"],
        ("save_hydrodynamics",): config["outputs"]["save_hydrodynamics"],
        ("initial_condition", "name"): initial["name"],
        ("initial_condition", "seed"): config["seed"],
        ("initial_condition", "S_initial"): model["S_initial"],
        ("initial_condition", "twist_amplitude"): initial["twist_amplitude"],
        ("initial_condition", "twist_modes"): initial["twist_modes"],
        ("initial_defect_gas", "num_pairs"): initial["num_defect_pairs"],
        ("initial_defect_gas", "minimum_separation"): initial[
            "minimum_separation"
        ],
        ("initial_defect_gas", "core_radius"): initial["core_radius"],
        ("initial_defect_gas", "background_angle"): initial[
            "background_angle"
        ],
    }


def _validate_completed_run(
    run_row: dict[str, Any],
    *,
    implementation_sha256: dict[str, str],
) -> bool:
    directory = Path(run_row["output_dir"])
    if not directory.exists():
        return False
    if not directory.is_dir():
        raise FileExistsError(f"run output path is not a directory: {directory}")
    if not any(directory.iterdir()):
        return False

    complete_path = directory / "COMPLETE"
    metadata_path = directory / "metadata.json"
    if not complete_path.is_file() or not metadata_path.is_file():
        raise RuntimeError(
            f"refusing incomplete existing run directory {directory}; "
            "both COMPLETE and metadata.json are required"
        )
    if complete_path.read_text(encoding="utf-8").strip() != "complete":
        raise RuntimeError(f"invalid COMPLETE marker in {directory}")

    metadata = _read_json(metadata_path)
    mismatches: list[str] = []
    for path, expected in _expected_metadata_fields(run_row).items():
        label = ".".join(path)
        try:
            actual = _nested_value(metadata, path)
        except KeyError:
            mismatches.append(f"{label}=<missing>, expected {expected!r}")
            continue
        if not _values_match(actual, expected):
            mismatches.append(f"{label}={actual!r}, expected {expected!r}")

    try:
        recorded_files = _nested_value(
            metadata,
            ("implementation_provenance", "files"),
        )
    except KeyError:
        recorded_files = None
    if recorded_files != implementation_sha256:
        mismatches.append("implementation_provenance.files differs from this plan")

    expected_gpu_name = run_row["config"].get("expected_gpu_name")
    if expected_gpu_name is not None:
        try:
            actual_gpu_name = _nested_value(
                metadata,
                ("runtime_environment", "cuda_device_name"),
            )
        except KeyError:
            actual_gpu_name = None
        if not isinstance(actual_gpu_name, str) or (
            expected_gpu_name.casefold() not in actual_gpu_name.casefold()
        ):
            mismatches.append(
                f"GPU name {actual_gpu_name!r} does not contain {expected_gpu_name!r}"
            )

    final_step = run_row["config"]["steps"]
    required_outputs = (
        directory / f"Q_{final_step}.npy",
        directory / f"u_{final_step}.npy",
        directory / f"p_{final_step}.npy",
        directory / "diagnostics.npy",
    )
    missing_outputs = [path.name for path in required_outputs if not path.is_file()]
    if missing_outputs:
        mismatches.append(f"missing required outputs: {missing_outputs}")

    if mismatches:
        preview = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            preview += f"; ... ({len(mismatches)} total mismatches)"
        raise ValueError(
            f"completed run {directory} is not reusable for this plan: {preview}"
        )
    return True


def _write_json_if_absent_or_identical(path: Path, value: Any) -> None:
    if path.exists():
        if _read_json(path) != value:
            raise FileExistsError(f"refusing to replace a different manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required validation file is missing: {path}")
    return {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _analysis_output_manifest(
    analysis_dir: Path,
    required_outputs: Iterable[str],
) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for name in required_outputs:
        relative_path = Path(name)
        if (
            relative_path.is_absolute()
            or len(relative_path.parts) != 1
            or relative_path.name in {"", ".", ".."}
        ):
            raise ValueError(f"unsafe required analysis output name: {name!r}")
        path = analysis_dir / relative_path
        identity = _file_identity(path)
        if identity["size_bytes"] <= 0:
            raise RuntimeError(f"required analysis output is empty: {path}")
        outputs[name] = identity
    return outputs


def _analysis_input_manifest(
    analysis: dict[str, Any],
    run_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    analyzer = Path(analysis["analyzer"]).resolve()
    is_core_analysis = analyzer == DEFECT_CORE_ANALYZER.resolve()
    inputs: dict[str, Any] = {}
    for run_id in analysis["input_run_ids"]:
        if run_id not in run_by_id:
            raise RuntimeError(
                "analysis {} references unknown run {}".format(analysis["name"], run_id)
            )
        run_row = run_by_id[run_id]
        directory = Path(run_row["output_dir"])
        final_step = int(run_row["config"]["steps"])
        filenames = [
            "metadata.json",
            f"Q_{final_step}.npy",
            f"u_{final_step}.npy",
            f"p_{final_step}.npy",
            "diagnostics.npy",
        ]
        if is_core_analysis:
            filenames.extend(
                ("Q_0.npy", "Q2D_initial.npy", "Q2D_defects.csv")
            )
        inputs[run_id] = {
            "config_sha256": run_row["config_sha256"],
            "files": {
                name: _file_identity(directory / name) for name in filenames
            },
        }
    return inputs


def _analysis_manifest_base(
    plan: dict[str, Any],
    analysis: dict[str, Any],
    run_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    command = analysis["command"]
    analyzer = Path(analysis["analyzer"]).resolve()
    if len(command) < 2 or Path(command[1]).resolve() != analyzer:
        raise RuntimeError(
            "analysis {} command/analyzer mismatch".format(analysis["name"])
        )
    try:
        analyzer_relative = str(analyzer.relative_to(PROJECT_ROOT.resolve()))
    except ValueError as error:
        raise RuntimeError(
            f"analysis analyzer is outside the project: {analyzer}"
        ) from error
    planned_hash = plan["validation_tools_sha256"].get(analyzer_relative)
    actual_hash = _sha256_file(analyzer)
    if planned_hash is None or actual_hash != planned_hash:
        raise RuntimeError(
            f"analysis tool changed after plan creation: {analyzer_relative}"
        )
    dependency_files = {
        str(Path(value).resolve()): _file_identity(Path(value))
        for value in analysis.get("dependency_files", ())
    }
    return {
        "schema_version": 2,
        "status": "complete",
        "name": analysis["name"],
        "command_sha256": _canonical_sha256(command),
        "analyzer": analyzer_relative,
        "analyzer_sha256": actual_hash,
        "validation_tools_sha256": plan["validation_tools_sha256"],
        "implementation_sha256": plan["implementation_sha256"],
        "inputs": _analysis_input_manifest(analysis, run_by_id),
        "dependencies": dependency_files,
    }


def execute_plan(
    plan: dict[str, Any],
    *,
    output_root: Path,
    skip_analysis: bool = False,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / (
        f"validation_plan_{plan['plan_sha256'][:12]}.json"
    )
    _write_json_if_absent_or_identical(plan_path, plan)

    implementation_sha256 = plan["implementation_sha256"]
    for index, run in enumerate(plan["runs"], start=1):
        run_label = f"[{index}/{len(plan['runs'])}] {run['run_id']}"
        if _validate_completed_run(
            run,
            implementation_sha256=implementation_sha256,
        ):
            print(f"{run_label}: reuse validated COMPLETE run", flush=True)
            continue

        command = run["command"]
        log_path = output_root / f"{run['run_id']}.log"
        if log_path.exists():
            raise FileExistsError(
                f"refusing to replace an existing log for a new run: {log_path}"
            )
        print(f"{run_label}: execute", flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"run {run['run_id']} failed with exit code {result.returncode}; "
                f"see {log_path}"
            )
        _validate_completed_run(
            run,
            implementation_sha256=implementation_sha256,
        )

    if skip_analysis:
        print("Analysis skipped by explicit --skip-analysis.", flush=True)
        return

    run_by_id = {row["run_id"]: row for row in plan["runs"]}
    for analysis in plan["analysis_commands"]:
        command = analysis["command"]
        try:
            output_index = command.index("--output-dir") + 1
        except ValueError as error:
            raise RuntimeError(
                f"analysis {analysis['name']} has no --output-dir"
            ) from error
        analysis_dir = Path(command[output_index])
        manifest_path = analysis_dir / "validation_analysis_manifest.json"
        log_path = output_root / f"analysis_{analysis['name']}.log"
        manifest_base = _analysis_manifest_base(plan, analysis, run_by_id)

        if analysis_dir.exists():
            if not analysis_dir.is_dir():
                raise FileExistsError(
                    f"analysis output path is not a directory: {analysis_dir}"
                )
            if manifest_path.is_file():
                completed_manifest = {
                    **manifest_base,
                    "outputs": _analysis_output_manifest(
                        analysis_dir, analysis["required_outputs"]
                    ),
                    "log": _file_identity(log_path),
                }
                if _read_json(manifest_path) == completed_manifest:
                    print(
                        f"analysis {analysis['name']}: reuse validated output",
                        flush=True,
                    )
                    continue
            raise FileExistsError(
                "refusing to mix or replace an existing analysis directory "
                f"without a matching manifest and file hashes: {analysis_dir}"
            )

        if log_path.exists():
            raise FileExistsError(
                f"refusing to replace an existing analysis log: {log_path}"
            )
        print(f"analysis {analysis['name']}: execute", flush=True)
        with log_path.open("x", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"analysis {analysis['name']} failed with exit code "
                f"{result.returncode}; see {log_path}"
            )
        manifest = {
            **manifest_base,
            "outputs": _analysis_output_manifest(
                analysis_dir, analysis["required_outputs"]
            ),
            "log": _file_identity(log_path),
        }
        _write_json_if_absent_or_identical(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    runs = build_runs(
        args.stages,
        include_r512_time_control=args.include_r512_time_control,
        long_final_time=args.long_final_time,
    )
    plan = build_plan(
        runs,
        python_bin=args.python_bin,
        output_root=output_root,
        device=args.device,
        expected_gpu_name=args.expected_gpu_name,
    )
    if args.write_plan is not None:
        _write_json(args.write_plan.expanduser().resolve(), plan)
    print(json.dumps(plan, indent=2, allow_nan=False))
    if args.execute:
        estimated_gib = plan["storage_estimate"]["primary_snapshot_gib"]
        requires_large_output_confirmation = (
            "long_seed" in args.stages or estimated_gib > 50.0
        )
        if requires_large_output_confirmation and not args.allow_large_output:
            raise SystemExit(
                "This plan estimates "
                f"{estimated_gib:.3f} GiB of primary snapshots and requires "
                "--allow-large-output in addition to --confirm-direct-execution."
            )
        execute_plan(
            plan,
            output_root=output_root,
            skip_analysis=args.skip_analysis,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
